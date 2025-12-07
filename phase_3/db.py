# phase_3/db.py
"""Database utilities for Phase 3."""

from contextlib import contextmanager
from pathlib import Path
import sqlite3
from typing import Generator, Optional, List, Any

import pandas as pd

# Database path
PHASE1_DIR = Path(__file__).parent.parent / "phase_1"
DATABASE_PATH = PHASE1_DIR / "database" / "nfl_data.db"


@contextmanager
def get_connection(readonly: bool = True) -> Generator[sqlite3.Connection, None, None]:
    """Get a database connection."""
    uri = f"file:{DATABASE_PATH}"
    if readonly:
        uri += "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        yield conn
    finally:
        conn.close()


def read_sql(query: str, params: Optional[List[Any]] = None) -> pd.DataFrame:
    """Execute a SQL query and return results as DataFrame."""
    with get_connection(readonly=True) as conn:
        if params:
            return pd.read_sql_query(query, conn, params=params)
        return pd.read_sql_query(query, conn)


def execute_sql(query: str, params: Optional[List[Any]] = None) -> None:
    """Execute a SQL statement."""
    with get_connection(readonly=False) as conn:
        if params:
            conn.execute(query, params)
        else:
            conn.execute(query)
        conn.commit()
