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


FEATURE_COLUMNS = ["diff_TO", "diff_NPE", "same_sector", "same_region", "diff_WAGES"]


def derive_dyadic_properties(candidates: ibis.Table) -> ibis.Table:
    """Ensure model-required dyadic feature columns exist and are numerically typed."""
    candidates_df = table_to_pandas(candidates)
    result = candidates_df.copy()
    for col in FEATURE_COLUMNS:
        if col not in result.columns:
            result[col] = 0.0

    result["same_sector"] = result["same_sector"].astype(float)
    result["same_region"] = result["same_region"].astype(float)

    for col in ["diff_TO", "diff_NPE", "diff_WAGES"]:
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0.0)

    return pandas_to_table(result)


def _standardize_pairs_schema(pairs: ibis.Table) -> ibis.Table:
    pairs_df = table_to_pandas(pairs)
    rename_map = {
        "id_suplier": "supplier_id",
        "id_supplier": "supplier_id",
        "id_user": "user_id",
        "TO_DIFF": "diff_TO",
        "WAGES_DIFF": "diff_WAGES",
        "NACE_same": "same_sector",
        "NUTS3_same": "same_region",
    }
    standardized = pairs_df.rename(columns=rename_map)
    return derive_dyadic_properties(pandas_to_table(standardized))


def generate_candidates_from_pairs(pairs: ibis.Table) -> ibis.Table:
    """Generate candidates from precomputed pair table in data-raw/pairs.parquet."""
    candidates = table_to_pandas(_standardize_pairs_schema(pairs))

    required = ["user_id", "supplier_id", *FEATURE_COLUMNS]
    missing = [c for c in required if c not in candidates.columns]
    if missing:
        raise ValueError(f"Missing required candidate columns: {missing}")

    candidates = candidates[required].copy()
    candidates = candidates.dropna(subset=["user_id", "supplier_id"])
    candidates["user_id"] = candidates["user_id"].astype(int)
    candidates["supplier_id"] = candidates["supplier_id"].astype(int)
    candidates = candidates[candidates["user_id"] != candidates["supplier_id"]]
    candidates = candidates.drop_duplicates(["user_id", "supplier_id"])
    return pandas_to_table(candidates)


def generate_candidates_probabilistic(
    enterprises: ibis.Table,
    max_candidates_per_user: int = 30,
    random_state: int = 42,
) -> ibis.Table:
    """
    Fallback candidate generator based on sector/region blocks.

    This is intentionally conservative to avoid combinatorial blow-up when no
    precomputed pair table is available.
    """
    rng = np.random.default_rng(random_state)

    enterprises_df = table_to_pandas(enterprises)

    required = ["id", "NACE", "NUTS3", "TO", "WAGES"]
    missing = [c for c in required if c not in enterprises_df.columns]
    if missing:
        raise ValueError(f"Enterprise table is missing required columns: {missing}")

    users = enterprises_df[required].rename(
        columns={"id": "user_id", "NACE": "user_nace", "NUTS3": "user_nuts3", "TO": "user_to", "WAGES": "user_wages"}
    )
    suppliers = enterprises_df[required].rename(
        columns={"id": "supplier_id", "NACE": "supplier_nace", "NUTS3": "supplier_nuts3", "TO": "supplier_to", "WAGES": "supplier_wages"}
    )

    sampled = []
    suppliers_by_nace = {k: g for k, g in suppliers.groupby("supplier_nace")}
    for _, user in users.iterrows():
        pool = suppliers_by_nace.get(user["user_nace"])
        if pool is None or pool.empty:
            continue

        pool = pool[pool["supplier_id"] != user["user_id"]]
        if pool.empty:
            continue

        sample_n = min(max_candidates_per_user, len(pool))
        sampled_idx = rng.choice(pool.index.to_numpy(), size=sample_n, replace=False)
        block = pool.loc[sampled_idx].copy()
        block["user_id"] = int(user["user_id"])
        block["user_nace"] = user["user_nace"]
        block["user_nuts3"] = user["user_nuts3"]
        block["user_to"] = float(user["user_to"])
        block["user_wages"] = float(user["user_wages"])
        sampled.append(block)

    if not sampled:
        return pandas_to_table(pd.DataFrame(columns=["user_id", "supplier_id", *FEATURE_COLUMNS]))

    candidates = pd.concat(sampled, ignore_index=True)
    candidates["diff_TO"] = candidates["user_to"] - candidates["supplier_to"]
    candidates["diff_NPE"] = 0.0
    candidates["same_sector"] = (candidates["user_nace"] == candidates["supplier_nace"]).astype(float)
    candidates["same_region"] = (candidates["user_nuts3"] == candidates["supplier_nuts3"]).astype(float)
    candidates["diff_WAGES"] = candidates["user_wages"] - candidates["supplier_wages"]
    result = candidates[["user_id", "supplier_id", *FEATURE_COLUMNS]].drop_duplicates(["user_id", "supplier_id"])
    return pandas_to_table(result)


def run_step(
    data_raw_dir: str | Path = "data-raw",
    output_path: str | Path = "data/candidates.parquet",
    max_candidates_per_user: int = 30,
    random_state: int = 42,
) -> ibis.Table:
    data_raw_dir = Path(data_raw_dir)
    pairs_path = data_raw_dir / "pairs.parquet"
    enterprises_path = data_raw_dir / "data.parquet"

    if pairs_path.exists():
        pairs = read_parquet_table(pairs_path)
        candidates = generate_candidates_from_pairs(pairs)
    else:
        enterprises = read_parquet_table(enterprises_path)
        candidates = generate_candidates_probabilistic(
            enterprises,
            max_candidates_per_user=max_candidates_per_user,
            random_state=random_state,
        )

    write_parquet_table(candidates, output_path)
    return candidates


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 01 - Generate candidate links.")
    parser.add_argument("--data-raw-dir", default="data-raw")
    parser.add_argument("--output", default="data/candidates.parquet")
    parser.add_argument("--max-candidates-per-user", type=int, default=30)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_step(
        data_raw_dir=args.data_raw_dir,
        output_path=args.output,
        max_candidates_per_user=args.max_candidates_per_user,
        random_state=args.random_state,
    )

