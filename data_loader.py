"""
data_loader.py

Data ingestion pipeline supporting:
  1. Single Excel workbook (.xlsx)
  2. Directory of Excel files (concatenated and aligned)
  3. Relational databases via SQLAlchemy (delegated to db_loader)

Normalizes inputs into a consistent wide DataFrame:
  [timestamp_column, metric_1, metric_2, ...]
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class DataLoadError(Exception):
    """Raised when data source loading or schema validation fails."""


def resolve_path(path_str: str, base_dir: Optional[Path] = None) -> Path:
    """Resolves relative paths against base_dir or project root."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    if base_dir:
        return (base_dir / p).resolve()
    return (Path(__file__).parent / p).resolve()


def load_excel(
    file_path: str | Path,
    timestamp_column: Optional[str] = None,
    base_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Load and normalize a single Excel workbook."""
    path = resolve_path(str(file_path), base_dir=base_dir)

    if not path.is_file():
        raise DataLoadError(f"Excel file not found: {path}")

    logger.info(f"Loading Excel file: {path}")
    try:
        df = pd.read_excel(path)
    except Exception as exc:
        raise DataLoadError(f"Failed reading Excel file '{path}': {exc}") from exc

    if df.empty:
        raise DataLoadError(f"Excel file '{path}' is empty.")

    ts_col = _resolve_timestamp_column(df, timestamp_column)
    logger.info(f"Timestamp column resolved as: '{ts_col}'")

    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    bad_ts_count = df[ts_col].isna().sum()
    if bad_ts_count > 0:
        logger.warning(f"Dropping {bad_ts_count} row(s) with invalid timestamps.")
        df = df.dropna(subset=[ts_col])

    if df.empty:
        raise DataLoadError("All rows contained unparseable timestamps.")

    df = df.sort_values(ts_col).reset_index(drop=True)

    numeric_cols = [
        c for c in df.columns
        if c != ts_col and pd.api.types.is_numeric_dtype(df[c])
    ]

    if not numeric_cols:
        raise DataLoadError(f"No numeric metric columns detected in '{path.name}'.")

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info(f"Successfully loaded {len(df)} rows and {len(numeric_cols)} metric columns.")
    return df


def load_excel_folder(
    folder_path: str | Path,
    timestamp_column: Optional[str] = None,
    base_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Load and concatenate all .xlsx workbooks in a directory."""
    folder = resolve_path(str(folder_path), base_dir=base_dir)
    if not folder.is_dir():
        raise DataLoadError(f"Folder not found: {folder}")

    files = sorted(folder.glob("*.xlsx"))
    if not files:
        raise DataLoadError(f"No .xlsx files found in: {folder}")

    logger.info(f"Aggregating {len(files)} workbook(s) from {folder}")
    frames = []

    for fp in files:
        try:
            sub = load_excel(fp, timestamp_column=timestamp_column, base_dir=folder)
            sub["__source_file"] = fp.name
            frames.append(sub)
        except DataLoadError as exc:
            logger.warning(f"Skipping '{fp.name}': {exc}")

    if not frames:
        raise DataLoadError(f"None of the workbooks in '{folder}' could be parsed.")

    all_cols = sorted({c for f in frames for c in f.columns})
    aligned = [f.reindex(columns=all_cols) for f in frames]
    merged = pd.concat(aligned, ignore_index=True)

    ts_col = _resolve_timestamp_column(merged, timestamp_column)
    merged[ts_col] = pd.to_datetime(merged[ts_col], errors="coerce")
    merged = merged.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)

    metric_cols = [
        c for c in merged.columns
        if c != ts_col and c != "__source_file" and pd.api.types.is_numeric_dtype(merged[c])
    ]

    if not metric_cols:
        raise DataLoadError("No numeric metric columns found across loaded files.")

    logger.info(f"Aggregated {len(merged)} total rows across {len(frames)} file(s).")
    return merged


def load_data(config: dict, base_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, str]:
    """
    Dispatcher loading data based on config['input_mode'].
    Returns (DataFrame, timestamp_column_name).
    """
    mode = config.get("input_mode", "excel_file")
    data_cfg = config.get("data", {})
    ts_col_pref = data_cfg.get("timestamp_column")

    if mode == "excel_file":
        file_path = data_cfg.get("file_path", "data/sample_metrics.xlsx")
        df = load_excel(file_path, timestamp_column=ts_col_pref, base_dir=base_dir)
        ts_col = _resolve_timestamp_column(df, ts_col_pref)
        return df, ts_col

    if mode == "excel_folder":
        folder_path = data_cfg.get("folder_path", "data/folder_in")
        df = load_excel_folder(folder_path, timestamp_column=ts_col_pref, base_dir=base_dir)
        ts_col = _resolve_timestamp_column(df, ts_col_pref)
        return df, ts_col

    if mode == "database":
        from db_loader import load_from_database
        db_cfg = config.get("db", {})
        conn_str = _resolve_env_value(db_cfg.get("connection_string", ""))
        query = _resolve_env_value(db_cfg.get("query", ""))
        df = load_from_database(conn_str, query, timestamp_column=ts_col_pref)
        ts_col = _resolve_timestamp_column(df, ts_col_pref)
        return df, ts_col

    raise DataLoadError(f"Unsupported input_mode: {mode!r}. Expected 'excel_file', 'excel_folder', or 'database'.")


def _resolve_timestamp_column(df: pd.DataFrame, explicit: Optional[str]) -> str:
    """Finds or auto-detects the timestamp column."""
    if explicit and explicit in df.columns:
        return explicit

    # Common candidates
    candidates = ["date", "timestamp", "datetime", "time", "day", "ts"]
    for col in df.columns:
        if col.lower() in candidates:
            return col

    # Fall back to first non-numeric column
    non_numeric = [
        c for c in df.columns
        if c != "__source_file" and not pd.api.types.is_numeric_dtype(df[c])
    ]
    if non_numeric:
        return non_numeric[0]

    raise DataLoadError("Could not identify a timestamp column in data.")


def _resolve_env_value(value: str) -> str:
    """Resolves 'ENV:VAR_NAME' references from environment variables."""
    if isinstance(value, str) and value.startswith("ENV:"):
        var_name = value[4:].strip()
        val = os.environ.get(var_name, "")
        if not val:
            logger.warning(f"Environment variable '{var_name}' referenced in config is empty.")
        return val
    return value or ""
