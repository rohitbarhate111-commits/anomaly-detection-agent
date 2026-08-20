"""
data_loader.py

Responsible for reading the Excel file from disk and turning it into a clean
pandas DataFrame ready for anomaly detection.

Failures are surfaced as explicit exceptions with a clear message; the caller
logs them and exits cleanly rather than crashing silently on bad data.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DataLoadError(Exception):
    """Raised when the input Excel file cannot be loaded or is invalid."""


def load_excel(
    file_path: str,
    timestamp_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load a wide-format Excel file into a pandas DataFrame.

    Expected shape:
      - One sheet.
      - One column holds the date/timestamp (auto-detected if not provided).
      - All remaining columns are numeric metrics.

    Args:
        file_path: Absolute or relative path to the .xlsx file.
        timestamp_column: Optional explicit name of the date column. If None,
            the loader will pick the first non-numeric column.

    Raises:
        DataLoadError: If the file is missing, unreadable, has no numeric
            columns, or has no parseable timestamp column.
    """
    path = Path(file_path)

    # 1. Existence check - fail loudly if missing.
    if not path.is_file():
        raise DataLoadError(f"Excel file not found at: {path.resolve()}")

    logger.info(f"Loading Excel file: {path.resolve()}")

    # 2. Try to read. openpyxl handles .xlsx; pandas raises on corruption.
    try:
        df = pd.read_excel(path)
    except Exception as exc:
        raise DataLoadError(f"Failed to read Excel file '{path}': {exc}") from exc

    if df.empty:
        raise DataLoadError(f"Excel file '{path}' contains no rows.")

    # 3. Identify the timestamp column.
    timestamp_col = _resolve_timestamp_column(df, timestamp_column)
    logger.info(f"Using '{timestamp_col}' as the timestamp column.")

    # 4. Parse timestamps to datetime. Bad rows become NaT, not crashes.
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    bad_ts = df[timestamp_col].isna().sum()
    if bad_ts == len(df):
        raise DataLoadError(
            f"Column '{timestamp_col}' could not be parsed as dates. "
            "Check the timestamp format in the Excel file."
        )
    if bad_ts > 0:
        logger.warning(f"Dropped {bad_ts} rows with unparseable timestamps.")
        df = df.dropna(subset=[timestamp_col]).reset_index(drop=True)

    # 5. Keep only numeric metric columns (plus the timestamp).
    numeric_cols = [
        col for col in df.columns
        if col != timestamp_col and pd.api.types.is_numeric_dtype(df[col])
    ]

    if not numeric_cols:
        raise DataLoadError(
            "No numeric metric columns found. Expected one date/timestamp "
            "column and the rest numeric."
        )

    logger.info(f"Detected {len(numeric_cols)} numeric metric columns: {numeric_cols}")

    # Coerce non-numeric values in metric columns to NaN so detection skips them.
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 6. Sort by timestamp for a sensible rolling baseline.
    df = df.sort_values(timestamp_col).reset_index(drop=True)

    return df


def _resolve_timestamp_column(df: pd.DataFrame, explicit: Optional[str]) -> str:
    """Pick the timestamp column: explicit name, else first non-numeric column."""
    if explicit and explicit in df.columns:
        return explicit

    non_numeric = [
        col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])
    ]
    if not non_numeric:
        raise DataLoadError(
            "No non-numeric column found to use as timestamp. "
            "Pass `timestamp_column` in config or add a date column to the file."
        )
    return non_numeric[0]
