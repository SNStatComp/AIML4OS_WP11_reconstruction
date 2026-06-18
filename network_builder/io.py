from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


def read_parquet_df(path: str | Path) -> pd.DataFrame:
    """Read parquet with pandas and fall back to DuckDB when parquet engines are missing."""
    path = Path(path)
    try:
        return pd.read_parquet(path)
    except Exception:
        return duckdb.sql("SELECT * FROM read_parquet(?)", params=[str(path)]).df()


def write_parquet_df(df: pd.DataFrame, path: str | Path) -> None:
    """Write parquet with pandas and fall back to DuckDB when parquet engines are missing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_parquet(path, index=False)
        return
    except Exception:
        pass

    con = duckdb.connect()
    con.register("tmp_df", df)
    con.execute("COPY tmp_df TO ? (FORMAT PARQUET)", [str(path)])
