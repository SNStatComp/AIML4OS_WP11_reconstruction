from __future__ import annotations

from pathlib import Path
import argparse

import ibis
import numpy as np
import pandas as pd

from network_builder.io import (
    pandas_to_table,
    read_parquet_table,
    table_to_pandas,
    write_parquet_table,
)


def sample_edges(
    calibrated_edges: ibis.Table,
    expected_suppliers: ibis.Table,
    selection_mode: str = "probabilistic",
    random_state: int = 42,
) -> ibis.Table:
    """
    Select candidate edges per user.

    selection_mode:
    - probabilistic: weighted sampling without replacement (default)
    - topk: deterministic selection of highest-probability suppliers
    """
    calibrated_edges_df = table_to_pandas(calibrated_edges)
    expected_suppliers_df = table_to_pandas(expected_suppliers)

    required_edges = ["user_id", "supplier_id", "calibrated_probability"]
    required_expected = ["user_id", "expected_num_suppliers"]

    missing_edges = [c for c in required_edges if c not in calibrated_edges_df.columns]
    missing_expected = [c for c in required_expected if c not in expected_suppliers_df.columns]
    if missing_edges:
        raise ValueError(f"Calibrated edge table missing columns: {missing_edges}")
    if missing_expected:
        raise ValueError(f"Expected supplier table missing columns: {missing_expected}")

    edges = calibrated_edges_df[required_edges].copy()
    edges["calibrated_probability"] = pd.to_numeric(
        edges["calibrated_probability"], errors="coerce"
    ).fillna(0.0).clip(0.0, 1.0)

    expected = expected_suppliers_df[required_expected].copy()
    expected["expected_num_suppliers"] = pd.to_numeric(
        expected["expected_num_suppliers"], errors="coerce"
    ).fillna(1.0).astype(int)

    merged = edges.merge(expected, on="user_id", how="left")
    merged["expected_num_suppliers"] = merged["expected_num_suppliers"].fillna(1).astype(int)

    if selection_mode not in {"probabilistic", "topk"}:
        raise ValueError("selection_mode must be one of {'probabilistic', 'topk'}")

    rng = np.random.default_rng(random_state)
    sampled_parts: list[pd.DataFrame] = []

    for user_id, group in merged.groupby("user_id", sort=False):
        if group.empty:
            continue

        n = int(group["expected_num_suppliers"].iloc[0])
        n = max(0, min(n, len(group)))
        if n == 0:
            continue

        if selection_mode == "topk":
            chosen = group.nlargest(n, "calibrated_probability")
            sampled_parts.append(chosen[["user_id", "supplier_id", "calibrated_probability"]])
        else:
            weights = group["calibrated_probability"].to_numpy(dtype=float)
            if weights.sum() <= 0:
                probs = None
            else:
                probs = weights / weights.sum()

            chosen_idx = rng.choice(group.index.to_numpy(), size=n, replace=False, p=probs)
            sampled_parts.append(group.loc[chosen_idx, ["user_id", "supplier_id", "calibrated_probability"]])

    if not sampled_parts:
        return pandas_to_table(pd.DataFrame(columns=["user_id", "supplier_id", "selection_probability"]))

    sampled = pd.concat(sampled_parts, ignore_index=True)
    sampled = sampled.rename(columns={"calibrated_probability": "selection_probability"})
    result = sampled.sort_values(["user_id", "selection_probability"], ascending=[True, False]).reset_index(drop=True)
    return pandas_to_table(result)


def run_step(
    calibrated_path: str | Path = "data/calibrated_probabilities.parquet",
    expected_path: str | Path = "data/expected_suppliers.parquet",
    output_path: str | Path = "data/sampled_edges.parquet",
    selection_mode: str = "probabilistic",
    random_state: int = 42,
) -> ibis.Table:
    calibrated = read_parquet_table(calibrated_path)
    expected = read_parquet_table(expected_path)
    sampled = sample_edges(
        calibrated,
        expected,
        selection_mode=selection_mode,
        random_state=random_state,
    )
    write_parquet_table(sampled, output_path)
    return sampled


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 05a - Sample edges.")
    parser.add_argument("--calibrated", default="data/calibrated_probabilities.parquet")
    parser.add_argument("--expected", default="data/expected_suppliers.parquet")
    parser.add_argument("--output", default="data/sampled_edges.parquet")
    parser.add_argument("--selection-mode", choices=["probabilistic", "topk"], default="probabilistic")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_step(
        calibrated_path=args.calibrated,
        expected_path=args.expected,
        output_path=args.output,
        selection_mode=args.selection_mode,
        random_state=args.random_state,
    )