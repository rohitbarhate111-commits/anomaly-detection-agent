"""
db_loader.py

Optional Postgres input source for v2.

Activated only when config.yaml sets `input_mode: database`. The connection
string and SQL query are read from config (or env vars) and never hardcoded.

Returns the same wide-format DataFrame the rest of the agent expects:
    one column = timestamp (configurable name), the rest = numeric metrics.
"""

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DatabaseLoadError(Exception):
    """Raised when the database query cannot be executed."""


def load_from_database(
    connection_string: str,
    sql_query: str,
    timestamp_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Execute a SQL query via SQLAlchemy and return the result as a DataFrame.

    Args:
        connection_string: full SQLAlchemy URL, e.g.
            postgresql+psycopg2://user:pass@host:5432/dbname
        sql_query: SELECT statement expected to return wide-format rows
            (one timestamp column + N numeric metric columns).
        timestamp_column: optional explicit name of the timestamp column;
            if None, the first non-numeric column is used.

    Raises:
        DatabaseLoadError: on missing config, missing driver, or query failure.
    """
    if not connection_string:
        raise DatabaseLoadError(
            "db_connection_string is empty. Set it in config.yaml or the "
            "DB_CONNECTION_STRING environment variable."
        )
    if not sql_query:
        raise DatabaseLoadError(
            "db_query is empty. Configure the SQL to run for metric extraction."
        )

    try:
        from sqlalchemy import create_engine
    except ImportError as exc:
        raise DatabaseLoadError(
            "SQLAlchemy is required for database input mode. "
            "Install with `pip install sqlalchemy psycopg2-binary`."
        ) from exc

    logger.info(
        f"Connecting to database and running query "
        f"({len(sql_query)} chars)..."
    )

    try:
        engine = create_engine(connection_string, future=True)
        df = pd.read_sql_query(sql_query, con=engine)
        engine.dispose()
    except Exception as exc:
        raise DatabaseLoadError(f"Database query failed: {exc}") from exc

    if df.empty:
        raise DatabaseLoadError("Database query returned no rows.")

    # Resolve timestamp column.
    if timestamp_column and timestamp_column in df.columns:
        ts_col = timestamp_column
    else:
        non_numeric = [
            c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])
        ]
        if not non_numeric:
            raise DatabaseLoadError(
                "No non-numeric column found in query result for timestamp."
            )
        ts_col = non_numeric[0]

    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    if df[ts_col].isna().all():
        raise DatabaseLoadError(
            f"Column '{ts_col}' could not be parsed as datetime."
        )
    df = df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)

    # Drop rows where all metric columns are null.
    metric_cols = [
        c for c in df.columns
        if c != ts_col and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not metric_cols:
        raise DatabaseLoadError("Query result has no numeric metric columns.")
    df = df.dropna(subset=metric_cols, how="all").reset_index(drop=True)

    logger.info(
        f"Loaded {len(df)} rows from database "
        f"({len(metric_cols)} numeric columns, timestamp='{ts_col}')."
    )
    return df
