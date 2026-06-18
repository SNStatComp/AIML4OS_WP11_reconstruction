from __future__ import annotations

from pathlib import Path
import argparse
from collections import defaultdict

import ibis
import pandas as pd

from network_builder.io import (
    pandas_to_table,
    read_parquet_table,
    table_to_pandas,
    write_parquet_table,
)


def _drop_reciprocal_duplicates(edges: pd.DataFrame) -> pd.DataFrame:
    """If both (u,v) and (v,u) exist, keep only the one with higher probability."""
    edges = edges.copy()
    low = edges[["user_id", "supplier_id"]].min(axis=1)
    high = edges[["user_id", "supplier_id"]].max(axis=1)
    edges["pair_key"] = low.astype(str) + "::" + high.astype(str)
    edges = edges.sort_values("selection_probability", ascending=False)
    edges = edges.drop_duplicates(subset=["pair_key"], keep="first")
    return edges.drop(columns=["pair_key"])


def prune_links(
    sampled_edges: ibis.Table,
    enterprises: ibis.Table | None = None,
    nace_exclusions: ibis.Table | None = None,
    max_suppliers_per_user: int = 25,
    max_users_per_supplier: int = 500,
) -> ibis.Table:
    """Prune sampled edges while enforcing degree constraints and optional NACE exclusions."""
    sampled_edges_df = table_to_pandas(sampled_edges)
    required = ["user_id", "supplier_id", "selection_probability"]
    missing = [c for c in required if c not in sampled_edges_df.columns]
    if missing:
        raise ValueError(f"Sampled edge table missing columns: {missing}")

    edges = sampled_edges_df[required].copy()
    edges = edges.dropna(subset=["user_id", "supplier_id"])
    edges["user_id"] = edges["user_id"].astype(int)
    edges["supplier_id"] = edges["supplier_id"].astype(int)
    edges["selection_probability"] = pd.to_numeric(
        edges["selection_probability"], errors="coerce"
    ).fillna(0.0)

    edges = edges[edges["user_id"] != edges["supplier_id"]]
    edges = edges.drop_duplicates(subset=["user_id", "supplier_id"])

    if enterprises is not None and nace_exclusions is not None:
        enterprises_df = table_to_pandas(enterprises)
        nace_exclusions_df = table_to_pandas(nace_exclusions)

        required_enterprises = ["id", "NACE"]
        required_exclusions = ["user_nace", "supplier_nace"]

        missing_enterprises = [c for c in required_enterprises if c not in enterprises_df.columns]
        missing_exclusions = [c for c in required_exclusions if c not in nace_exclusions_df.columns]
        if missing_enterprises:
            raise ValueError(f"Enterprise table missing columns for NACE filtering: {missing_enterprises}")
        if missing_exclusions:
            raise ValueError(f"NACE exclusion table missing columns: {missing_exclusions}")

        nace_map = enterprises_df[required_enterprises].rename(columns={"id": "node_id", "NACE": "nace"})
        user_nace = nace_map.rename(columns={"node_id": "user_id", "nace": "user_nace"})
        supplier_nace = nace_map.rename(columns={"node_id": "supplier_id", "nace": "supplier_nace"})

        if not nace_exclusions_df.empty:
            edges = edges.merge(user_nace, on="user_id", how="left")
            edges = edges.merge(supplier_nace, on="supplier_id", how="left")

            exclusion_pairs = set(
                zip(
                    nace_exclusions_df["user_nace"].astype(str),
                    nace_exclusions_df["supplier_nace"].astype(str),
                )
            )
            blocked = edges.apply(
                lambda r: (str(r["user_nace"]), str(r["supplier_nace"])) in exclusion_pairs,
                axis=1,
            )
            edges = edges.loc[~blocked].drop(columns=["user_nace", "supplier_nace"])

    edges = _drop_reciprocal_duplicates(edges)
    edges = edges.sort_values("selection_probability", ascending=False)

    out_degree = defaultdict(int)
    in_degree = defaultdict(int)
    kept_rows = []

    for row in edges.itertuples(index=False):
        if out_degree[row.user_id] >= max_suppliers_per_user:
            continue
        if in_degree[row.supplier_id] >= max_users_per_supplier:
            continue

        out_degree[row.user_id] += 1
        in_degree[row.supplier_id] += 1
        kept_rows.append((row.user_id, row.supplier_id, row.selection_probability))

    pruned = pd.DataFrame(kept_rows, columns=required)
    result = pruned.sort_values(["user_id", "selection_probability"], ascending=[True, False]).reset_index(drop=True)
    return pandas_to_table(result)


def run_step(
    sampled_path: str | Path = "data/sampled_edges.parquet",
    output_path: str | Path = "data/reconstructed_network.parquet",
    enterprises_path: str | Path | None = "data-raw/data.parquet",
    nace_exclusions_path: str | Path | None = None,
    max_suppliers_per_user: int = 25,
    max_users_per_supplier: int = 500,
) -> ibis.Table:
    sampled = read_parquet_table(sampled_path)

    enterprises = None
    nace_exclusions = None
    if enterprises_path is not None:
        enterprises_candidate = Path(enterprises_path)
        if enterprises_candidate.exists():
            enterprises = read_parquet_table(enterprises_candidate)
    if nace_exclusions_path is not None:
        exclusions_candidate = Path(nace_exclusions_path)
        if not exclusions_candidate.exists():
            raise FileNotFoundError(f"NACE exclusions file not found: {exclusions_candidate}")
        if exclusions_candidate.suffix.lower() in {".csv", ".txt"}:
            nace_exclusions = pandas_to_table(pd.read_csv(exclusions_candidate))
        else:
            nace_exclusions = read_parquet_table(exclusions_candidate)

    pruned = prune_links(
        sampled,
        enterprises=enterprises,
        nace_exclusions=nace_exclusions,
        max_suppliers_per_user=max_suppliers_per_user,
        max_users_per_supplier=max_users_per_supplier,
    )
    write_parquet_table(pruned, output_path)
    return pruned


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 05b - Prune sampled edges.")
    parser.add_argument("--sampled", default="data/sampled_edges.parquet")
    parser.add_argument("--output", default="data/reconstructed_network.parquet")
    parser.add_argument("--enterprises", default="data-raw/data.parquet")
    parser.add_argument("--nace-exclusions", default=None)
    parser.add_argument("--max-suppliers-per-user", type=int, default=25)
    parser.add_argument("--max-users-per-supplier", type=int, default=500)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_step(
        sampled_path=args.sampled,
        output_path=args.output,
        enterprises_path=args.enterprises,
        nace_exclusions_path=args.nace_exclusions,
        max_suppliers_per_user=args.max_suppliers_per_user,
        max_users_per_supplier=args.max_users_per_supplier,
    )