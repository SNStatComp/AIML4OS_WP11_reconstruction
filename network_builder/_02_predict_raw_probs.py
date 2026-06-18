from __future__ import annotations

from pathlib import Path
import argparse

import ibis
import joblib
import lightgbm as lgb
import pandas as pd

from network_builder.io import (
    pandas_to_table,
    read_parquet_table,
    table_to_pandas,
    write_parquet_table,
)


DEFAULT_FEATURE_COLUMNS = ["diff_TO", "diff_NPE", "same_sector", "same_region", "diff_WAGES"]


def load_lightgbm_model(model_path: str | Path = "models/model_LightGBM.pkl") -> lgb.LGBMClassifier:
    return joblib.load(model_path, mmap_mode="r")


def _get_feature_columns(model: lgb.LGBMClassifier) -> list[str]:
    if hasattr(model, "feature_name_") and len(getattr(model, "feature_name_")) > 0:
        return list(model.feature_name_)
    if hasattr(model, "feature_names_in_") and len(getattr(model, "feature_names_in_")) > 0:
        return list(model.feature_names_in_)
    return DEFAULT_FEATURE_COLUMNS


def predict_raw_probabilities(candidates: ibis.Table, model: lgb.LGBMClassifier) -> ibis.Table:
    candidates_df = table_to_pandas(candidates)
    feature_columns = _get_feature_columns(model)
    scored = candidates_df.copy()

    for col in feature_columns:
        if col not in scored.columns:
            scored[col] = 0.0
    x = scored[feature_columns].astype(float)

    raw_probabilities = model.predict_proba(x)[:, 1]
    result = scored[["user_id", "supplier_id"]].copy()
    result["raw_probability"] = raw_probabilities
    return pandas_to_table(result)


def run_step(
    candidates_path: str | Path = "data/candidates.parquet",
    output_path: str | Path = "data/raw_probabilities.parquet",
    model_path: str | Path = "models/model_LightGBM.pkl",
) -> ibis.Table:
    candidates = read_parquet_table(candidates_path)
    model = load_lightgbm_model(model_path)
    raw_probs = predict_raw_probabilities(candidates, model)
    write_parquet_table(raw_probs, output_path)
    return raw_probs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 02 - Predict raw edge probabilities.")
    parser.add_argument("--candidates", default="data/candidates.parquet")
    parser.add_argument("--output", default="data/raw_probabilities.parquet")
    parser.add_argument("--model", default="models/model_LightGBM.pkl")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_step(candidates_path=args.candidates, output_path=args.output, model_path=args.model)