"""
db_loader.py

PostgreSQL and SQLAlchemy-compatible database data loader.
Safely extracts wide-format time series metrics via parameterized connection strings.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DatabaseLoadError(Exception):
    """Raised when database connection or query execution fails."""


def _mask_connection_string(conn_str: str) -> str:
    """Sanitizes passwords from connection strings for safe logging."""
    return re.sub(r":([^:@]+)@", r":***@", conn_str)


def load_from_database(
    connection_string: str,
    sql_query: str,
    timestamp_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Executes a SQL query via SQLAlchemy and normalizes results into a wide DataFrame.
    """
    if not connection_string:
        raise DatabaseLoadError("Database connection string is empty.")
    if not sql_query:
        raise DatabaseLoadError("Database query string is empty.")

    try:
        from sqlalchemy import create_engine, text
    except ImportError as exc:
        raise DatabaseLoadError("SQLAlchemy is required for database mode. Install with `pip install SQLAlchemy psycopg2-binary`.") from exc

    masked = _mask_connection_string(connection_string)
    logger.info(f"Connecting to database ({masked})...")

    try:
        engine = create_engine(connection_string, future=True, pool_pre_ping=True)
        with engine.connect() as conn:
            df = pd.read_sql_query(text(sql_query), con=conn)
        engine.dispose()
    except Exception as exc:
        raise DatabaseLoadError(f"Database query failed: {exc}") from exc

    if df.empty:
        raise DatabaseLoadError("Database query returned 0 rows.")

    # Resolve timestamp
    if timestamp_column and timestamp_column in df.columns:
        ts_col = timestamp_column
    else:
        candidates = ["date", "timestamp", "datetime", "time"]
        matched = [c for c in df.columns if c.lower() in candidates]
        if matched:
            ts_col = matched[0]
        else:
            non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
            if not non_numeric:
                raise DatabaseLoadError("No timestamp column detected in database query results.")
            ts_col = non_numeric[0]

    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    if df[ts_col].isna().all():
        raise DatabaseLoadError(f"Column '{ts_col}' could not be parsed as datetimes.")

    df = df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)

    numeric_cols = [c for c in df.columns if c != ts_col and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise DatabaseLoadError("Database query results contained no numeric metric columns.")

    logger.info(f"Loaded {len(df)} rows and {len(numeric_cols)} metrics from database.")
    return df
