from __future__ import annotations

from pathlib import Path

import duckdb
import ibis
import pandas as pd


def read_parquet_df(path: str | Path) -> pd.DataFrame:
    """Read parquet with pandas and fall back to DuckDB when parquet engines are missing."""
    path = Path(path)
    try:
        return pd.read_parquet(path)
    except Exception:
        return duckdb.sql("SELECT * FROM read_parquet(?)", params=[str(path)]).df()


def read_parquet_table(path: str | Path) -> ibis.Table:
    """Read parquet as an ibis table with robust pandas/duckdb fallback."""
    path = Path(path)
    try:
        return ibis.read_parquet(str(path))
    except Exception:
        return ibis.memtable(read_parquet_df(path))


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


def table_to_pandas(table: ibis.Table) -> pd.DataFrame:
    """Materialize an ibis table to pandas regardless of backend."""
    if hasattr(table, "to_pandas"):
        return table.to_pandas()
    return table.execute()


def pandas_to_table(df: pd.DataFrame) -> ibis.Table:
    """Create an ibis table from pandas."""
    return ibis.memtable(df)


def write_parquet_table(table: ibis.Table, path: str | Path) -> None:
    """Write an ibis table to parquet using pandas/duckdb fallback path."""
    write_parquet_df(table_to_pandas(table), path)
