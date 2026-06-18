from __future__ import annotations

from pathlib import Path
import argparse

from network_builder._05a_sample_edges import run_step as run_sample_edges
from network_builder._05b_prune_links import run_step as run_prune_links


def reconstruct(
    calibrated_path: str | Path = "data/calibrated_probabilities.parquet",
    expected_path: str | Path = "data/expected_suppliers.parquet",
    sampled_path: str | Path = "data/sampled_edges.parquet",
    output_path: str | Path = "data/reconstructed_network.parquet",
    enterprises_path: str | Path | None = "data-raw/data.parquet",
    nace_exclusions_path: str | Path | None = None,
    selection_mode: str = "probabilistic",
    random_state: int = 42,
    max_suppliers_per_user: int = 25,
    max_users_per_supplier: int = 500,
):
    run_sample_edges(
        calibrated_path=calibrated_path,
        expected_path=expected_path,
        output_path=sampled_path,
        selection_mode=selection_mode,
        random_state=random_state,
    )
    return run_prune_links(
        sampled_path=sampled_path,
        output_path=output_path,
        enterprises_path=enterprises_path,
        nace_exclusions_path=nace_exclusions_path,
        max_suppliers_per_user=max_suppliers_per_user,
        max_users_per_supplier=max_users_per_supplier,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 05 - Reconstruct edge set from calibrated probabilities.")
    parser.add_argument("--calibrated", default="data/calibrated_probabilities.parquet")
    parser.add_argument("--expected", default="data/expected_suppliers.parquet")
    parser.add_argument("--sampled", default="data/sampled_edges.parquet")
    parser.add_argument("--output", default="data/reconstructed_network.parquet")
    parser.add_argument("--enterprises", default="data-raw/data.parquet")
    parser.add_argument("--nace-exclusions", default=None)
    parser.add_argument("--selection-mode", choices=["probabilistic", "topk"], default="probabilistic")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-suppliers-per-user", type=int, default=25)
    parser.add_argument("--max-users-per-supplier", type=int, default=500)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    reconstruct(
        calibrated_path=args.calibrated,
        expected_path=args.expected,
        sampled_path=args.sampled,
        output_path=args.output,
        enterprises_path=args.enterprises,
        nace_exclusions_path=args.nace_exclusions,
        selection_mode=args.selection_mode,
        random_state=args.random_state,
        max_suppliers_per_user=args.max_suppliers_per_user,
        max_users_per_supplier=args.max_users_per_supplier,
    )