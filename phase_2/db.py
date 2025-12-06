# phase_2/db.py

import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Optional

import pandas as pd

from .config import DB_PATH


@contextmanager
def get_connection(readonly: bool = True):
    """
    Context manager yielding a SQLite connection.

    For now we keep it simple and open a new connection per operation.
    If performance ever becomes an issue, we can add connection pooling.
    """
    # SQLite URI for optional read-only mode
    if readonly:
        uri = f"file:{DB_PATH.as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
    else:
        conn = sqlite3.connect(DB_PATH)

    try:
        yield conn
    finally:
        conn.close()


def read_sql(
    query: str,
    params: Optional[Iterable[Any]] = None,
    index_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run a SELECT query and return a pandas DataFrame.
    """
    with get_connection(readonly=True) as conn:
        df = pd.read_sql_query(query, conn, params=params or [])
    if index_col is not None and index_col in df.columns:
        df = df.set_index(index_col)
    return df


def write_dataframe(
    df: pd.DataFrame,
    table_name: str,
    if_exists: str = "replace",
    dtype: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Write a DataFrame to the SQLite database.

    Parameters
    ----------
    df : pd.DataFrame
        Data to write.
    table_name : str
        Name of the table in SQLite.
    if_exists : {"fail", "replace", "append"}
        Behavior when the table already exists.
    dtype : Optional[Dict[str, Any]]
        Optional SQL column types mapping.
    """
    if df.empty:
        return

    with get_connection(readonly=False) as conn:
        df.to_sql(
            table_name,
            conn,
            if_exists=if_exists,
            index=False,
            dtype=dtype,
        )
