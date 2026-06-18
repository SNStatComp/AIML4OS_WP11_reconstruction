from __future__ import annotations

from pathlib import Path
import argparse
import json

from network_builder._01_generate_candidates import run_step as run_generate_candidates
from network_builder._02_predict_raw_probs import run_step as run_predict_raw_probs
from network_builder._03_expected_suppliers_per_user import run_step as run_expected_suppliers
from network_builder._04_calibrate_probs import run_step as run_calibrate_probs
from network_builder._05_reconstruct import reconstruct
from network_builder._06_evaluate import run_step as run_evaluate


def run_pipeline(
    data_raw_dir: str | Path = "data-raw",
    data_dir: str | Path = "data",
    model_path: str | Path = "models/model_LightGBM.pkl",
    selection_mode: str = "probabilistic",
    nace_exclusions_path: str | Path | None = None,
    random_state: int = 42,
) -> dict:
    data_raw_dir = Path(data_raw_dir)
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    candidates_path = data_dir / "candidates.parquet"
    raw_probs_path = data_dir / "raw_probabilities.parquet"
    expected_path = data_dir / "expected_suppliers.parquet"
    calibrated_path = data_dir / "calibrated_probabilities.parquet"
    sampled_path = data_dir / "sampled_edges.parquet"
    reconstructed_path = data_dir / "reconstructed_network.parquet"
    eval_json_path = data_dir / "evaluation_summary.json"
    eval_user_fit_path = data_dir / "evaluation_user_degree_fit.parquet"

    run_generate_candidates(data_raw_dir=data_raw_dir, output_path=candidates_path, random_state=random_state)
    run_predict_raw_probs(candidates_path=candidates_path, output_path=raw_probs_path, model_path=model_path)
    run_expected_suppliers(enterprises_path=data_raw_dir / "data.parquet", output_path=expected_path)
    run_calibrate_probs(raw_probabilities_path=raw_probs_path, expected_path=expected_path, output_path=calibrated_path)
    reconstruct(
        calibrated_path=calibrated_path,
        expected_path=expected_path,
        sampled_path=sampled_path,
        output_path=reconstructed_path,
        enterprises_path=data_raw_dir / "data.parquet",
        nace_exclusions_path=nace_exclusions_path,
        selection_mode=selection_mode,
        random_state=random_state,
    )
    summary = run_evaluate(
        reconstructed_path=reconstructed_path,
        expected_path=expected_path,
        output_json_path=eval_json_path,
        output_user_fit_path=eval_user_fit_path,
    )

    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run network reconstruction pipeline end-to-end.")
    parser.add_argument("--data-raw-dir", default="data-raw")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model", default="models/model_LightGBM.pkl")
    parser.add_argument("--selection-mode", choices=["probabilistic", "topk"], default="probabilistic")
    parser.add_argument("--nace-exclusions", default=None)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_pipeline(
        data_raw_dir=args.data_raw_dir,
        data_dir=args.data_dir,
        model_path=args.model,
        selection_mode=args.selection_mode,
        nace_exclusions_path=args.nace_exclusions,
        random_state=args.random_state,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
