from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from network_builder._io import read_parquet_df, write_parquet_df


def calibrate_probabilities(
    probabilities: pd.DataFrame,
    expected_degrees: pd.DataFrame,
) -> pd.DataFrame:
    """Scale raw probabilities per user so expected sum matches expected degree."""
    required_probs = ["user_id", "supplier_id", "raw_probability"]
    required_exp = ["user_id", "expected_num_suppliers"]

    missing_probs = [c for c in required_probs if c not in probabilities.columns]
    missing_exp = [c for c in required_exp if c not in expected_degrees.columns]
    if missing_probs:
        raise ValueError(f"Raw probabilities missing columns: {missing_probs}")
    if missing_exp:
        raise ValueError(f"Expected degree table missing columns: {missing_exp}")

    probs = probabilities.copy()
    probs["raw_probability"] = pd.to_numeric(probs["raw_probability"], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    merged = probs.merge(expected_degrees[required_exp], on="user_id", how="left")
    merged["expected_num_suppliers"] = pd.to_numeric(
        merged["expected_num_suppliers"], errors="coerce"
    ).fillna(1.0)

    sum_raw = merged.groupby("user_id")["raw_probability"].transform("sum").replace(0.0, np.nan)
    scale = (merged["expected_num_suppliers"] / sum_raw).fillna(0.0)

    merged["calibrated_probability"] = (merged["raw_probability"] * scale).clip(0.0, 1.0)
    return merged[["user_id", "supplier_id", "raw_probability", "calibrated_probability"]]


def run_step(
    raw_probabilities_path: str | Path = "data/raw_probabilities.parquet",
    expected_path: str | Path = "data/expected_suppliers.parquet",
    output_path: str | Path = "data/calibrated_probabilities.parquet",
) -> pd.DataFrame:
    raw_probs = read_parquet_df(raw_probabilities_path)
    expected = read_parquet_df(expected_path)
    calibrated = calibrate_probabilities(raw_probs, expected)
    write_parquet_df(calibrated, output_path)
    return calibrated


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 04 - Calibrate link probabilities.")
    parser.add_argument("--raw-probabilities", default="data/raw_probabilities.parquet")
    parser.add_argument("--expected", default="data/expected_suppliers.parquet")
    parser.add_argument("--output", default="data/calibrated_probabilities.parquet")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_step(
        raw_probabilities_path=args.raw_probabilities,
        expected_path=args.expected,
        output_path=args.output,
    )