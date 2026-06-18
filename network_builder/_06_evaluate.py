from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

from network_builder._io import read_parquet_df, write_parquet_df


def evaluate_reconstruction(
	reconstructed_edges: pd.DataFrame,
	expected_suppliers: pd.DataFrame,
) -> dict:
	required_edges = ["user_id", "supplier_id"]
	required_expected = ["user_id", "expected_num_suppliers"]

	missing_edges = [c for c in required_edges if c not in reconstructed_edges.columns]
	missing_expected = [c for c in required_expected if c not in expected_suppliers.columns]
	if missing_edges:
		raise ValueError(f"Reconstructed edges missing columns: {missing_edges}")
	if missing_expected:
		raise ValueError(f"Expected suppliers missing columns: {missing_expected}")

	edges = reconstructed_edges.copy()
	edges = edges.drop_duplicates(["user_id", "supplier_id"])

	user_degree = edges.groupby("user_id").size().rename("actual_num_suppliers").reset_index()
	supplier_degree = edges.groupby("supplier_id").size().rename("actual_num_users").reset_index()

	fit_df = expected_suppliers[["user_id", "expected_num_suppliers"]].merge(
		user_degree,
		on="user_id",
		how="left",
	)
	fit_df["actual_num_suppliers"] = fit_df["actual_num_suppliers"].fillna(0)
	fit_df["abs_error"] = (fit_df["actual_num_suppliers"] - fit_df["expected_num_suppliers"]).abs()
	fit_df["sq_error"] = (fit_df["actual_num_suppliers"] - fit_df["expected_num_suppliers"]) ** 2

	result = {
		"num_edges": int(len(edges)),
		"num_users_with_outgoing_edges": int(user_degree["user_id"].nunique()),
		"num_suppliers_with_incoming_edges": int(supplier_degree["supplier_id"].nunique()),
		"avg_suppliers_per_user": float(user_degree["actual_num_suppliers"].mean() if not user_degree.empty else 0.0),
		"avg_users_per_supplier": float(supplier_degree["actual_num_users"].mean() if not supplier_degree.empty else 0.0),
		"mae_user_degree_vs_expected": float(fit_df["abs_error"].mean() if not fit_df.empty else 0.0),
		"rmse_user_degree_vs_expected": float(np.sqrt(fit_df["sq_error"].mean()) if not fit_df.empty else 0.0),
	}

	return result


def run_step(
	reconstructed_path: str | Path = "data/reconstructed_network.parquet",
	expected_path: str | Path = "data/expected_suppliers.parquet",
	output_json_path: str | Path = "data/evaluation_summary.json",
	output_user_fit_path: str | Path = "data/evaluation_user_degree_fit.parquet",
) -> dict:
	reconstructed = read_parquet_df(reconstructed_path)
	expected = read_parquet_df(expected_path)

	summary = evaluate_reconstruction(reconstructed, expected)

	fit_df = expected[["user_id", "expected_num_suppliers"]].merge(
		reconstructed.groupby("user_id").size().rename("actual_num_suppliers").reset_index(),
		on="user_id",
		how="left",
	)
	fit_df["actual_num_suppliers"] = fit_df["actual_num_suppliers"].fillna(0).astype(int)
	fit_df["abs_error"] = (fit_df["actual_num_suppliers"] - fit_df["expected_num_suppliers"]).abs()

	output_json_path = Path(output_json_path)
	output_json_path.parent.mkdir(parents=True, exist_ok=True)
	output_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
	write_parquet_df(fit_df, output_user_fit_path)

	return summary


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Step 06 - Evaluate reconstructed network.")
	parser.add_argument("--reconstructed", default="data/reconstructed_network.parquet")
	parser.add_argument("--expected", default="data/expected_suppliers.parquet")
	parser.add_argument("--output-json", default="data/evaluation_summary.json")
	parser.add_argument("--output-user-fit", default="data/evaluation_user_degree_fit.parquet")
	return parser.parse_args()


if __name__ == "__main__":
	args = _parse_args()
	run_step(
		reconstructed_path=args.reconstructed,
		expected_path=args.expected,
		output_json_path=args.output_json,
		output_user_fit_path=args.output_user_fit,
	)
