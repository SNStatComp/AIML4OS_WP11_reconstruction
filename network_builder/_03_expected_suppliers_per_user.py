from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from network_builder.io import read_parquet_df, write_parquet_df


def _safe_percentile(series: pd.Series) -> pd.Series:
    if len(series) <= 1:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return series.rank(method="average", pct=True)


def estimate_expected_suppliers_per_user(
    enterprises: pd.DataFrame,
    max_expected_suppliers: int = 25,
) -> pd.DataFrame:
    """
    Estimate expected number of suppliers per enterprise from size-like signals.

    Uses a stable heuristic based on within-sector turnover/wage percentiles.
    """
    required = ["id", "NACE", "TO", "WAGES"]
    missing = [c for c in required if c not in enterprises.columns]
    if missing:
        raise ValueError(f"Enterprise table is missing required columns: {missing}")

    df = enterprises[required].copy()
    df = df.rename(columns={"id": "user_id", "NACE": "sector"})

    df["TO"] = pd.to_numeric(df["TO"], errors="coerce").fillna(0.0)
    df["WAGES"] = pd.to_numeric(df["WAGES"], errors="coerce").fillna(0.0)

    df["to_pct"] = df.groupby("sector", group_keys=False)["TO"].apply(_safe_percentile)
    df["wages_pct"] = df.groupby("sector", group_keys=False)["WAGES"].apply(_safe_percentile)

    # Heuristic degree target from economic size proxies.
    raw_expected = 1.0 + 8.0 * df["to_pct"] + 4.0 * df["wages_pct"]
    df["expected_num_suppliers"] = np.clip(np.round(raw_expected), 1, max_expected_suppliers).astype(int)

    bins = [-np.inf, 2, 6, 12, np.inf]
    labels = ["S", "M", "L", "XL"]
    df["size_class"] = pd.cut(df["expected_num_suppliers"], bins=bins, labels=labels).astype(str)

    return df[["user_id", "sector", "size_class", "expected_num_suppliers"]]


def run_step(
    enterprises_path: str | Path = "data-raw/data.parquet",
    output_path: str | Path = "data/expected_suppliers.parquet",
    max_expected_suppliers: int = 25,
) -> pd.DataFrame:
    enterprises = read_parquet_df(enterprises_path)
    expected = estimate_expected_suppliers_per_user(
        enterprises,
        max_expected_suppliers=max_expected_suppliers,
    )
    write_parquet_df(expected, output_path)
    return expected


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 03 - Estimate expected suppliers per user.")
    parser.add_argument("--enterprises", default="data-raw/data.parquet")
    parser.add_argument("--output", default="data/expected_suppliers.parquet")
    parser.add_argument("--max-expected-suppliers", type=int, default=25)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_step(
        enterprises_path=args.enterprises,
        output_path=args.output,
        max_expected_suppliers=args.max_expected_suppliers,
    )
