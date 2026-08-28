"""Load time-series metrics from a SQLAlchemy-compatible database."""

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DatabaseLoadError(Exception):
    """Raised when the database query cannot be executed or validated."""


def load_from_database(
    connection_string: str,
    sql_query: str,
    timestamp_column: Optional[str] = None,
) -> pd.DataFrame:
    """Execute a configured SQL query and normalize its result."""
    if not connection_string:
        raise DatabaseLoadError("Database connection string is empty.")
    if not sql_query:
        raise DatabaseLoadError("Database query is empty.")

    try:
        from sqlalchemy import create_engine
    except ImportError as exc:
        raise DatabaseLoadError(
            "SQLAlchemy is required for database input mode. "
            "Install the database dependencies from requirements.txt."
        ) from exc

    logger.info("Running configured database query (%d characters).", len(sql_query))
    engine = None
    try:
        engine = create_engine(connection_string, future=True)
        with engine.connect() as connection:
            df = pd.read_sql_query(sql_query, con=connection)
    except Exception as exc:
        raise DatabaseLoadError(f"Database query failed: {exc}") from exc
    finally:
        if engine is not None:
            engine.dispose()

    if df.empty:
        raise DatabaseLoadError("Database query returned no rows.")

    if timestamp_column and timestamp_column in df.columns:
        timestamp = timestamp_column
    else:
        non_numeric = [
            column for column in df.columns
            if not pd.api.types.is_numeric_dtype(df[column])
        ]
        if not non_numeric:
            raise DatabaseLoadError(
                "No non-numeric column found in query result for timestamp."
            )
        timestamp = non_numeric[0]

    df[timestamp] = pd.to_datetime(df[timestamp], errors="coerce")
    if df[timestamp].isna().all():
        raise DatabaseLoadError(
            f"Column '{timestamp}' could not be parsed as datetime."
        )

    df = df.dropna(subset=[timestamp]).sort_values(timestamp).reset_index(drop=True)
    metric_columns = [
        column for column in df.columns
        if column != timestamp and pd.api.types.is_numeric_dtype(df[column])
    ]
    if not metric_columns:
        raise DatabaseLoadError("Query result has no numeric metric columns.")

    df = df.dropna(subset=metric_columns, how="all").reset_index(drop=True)
    logger.info(
        "Loaded %d rows from database (%d numeric columns, timestamp='%s').",
        len(df),
        len(metric_columns),
        timestamp,
    )
    return df
