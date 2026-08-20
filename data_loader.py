"""
data_loader.py

Responsible for reading input data and turning it into a clean pandas
DataFrame ready for anomaly detection.

v1: load_excel(path) — single file.
v2: load_excel_folder(path) — concatenate all .xlsx in a directory, tagging
    each row with its source filename.
v2: load_data(config) — dispatcher that picks the right loader based on
    config.input_mode. Returns (df, timestamp_column).

All loaders normalize into the same wide shape:
    one timestamp column + N numeric metric columns.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class DataLoadError(Exception):
    """Raised when the input data cannot be loaded or is invalid."""


# ---------- v1 (unchanged) ----------

def load_excel(
    file_path: str,
    timestamp_column: Optional[str] = None,
) -> pd.DataFrame:
    """Load a single wide-format .xlsx file into a DataFrame."""
    path = Path(file_path)

    if not path.is_file():
        raise DataLoadError(f"Excel file not found at: {path.resolve()}")

    logger.info(f"Loading Excel file: {path.resolve()}")

    try:
        df = pd.read_excel(path)
    except Exception as exc:
        raise DataLoadError(f"Failed to read Excel file '{path}': {exc}") from exc

    if df.empty:
        raise DataLoadError(f"Excel file '{path}' contains no rows.")

    timestamp_col = _resolve_timestamp_column(df, timestamp_column)
    logger.info(f"Using '{timestamp_col}' as the timestamp column.")

    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    if df[timestamp_col].isna().all():
        raise DataLoadError(
            f"Column '{timestamp_col}' could not be parsed as dates."
        )
    bad_ts = df[timestamp_col].isna().sum()
    if bad_ts:
        logger.warning(f"Dropped {bad_ts} rows with unparseable timestamps.")
        df = df.dropna(subset=[timestamp_col]).reset_index(drop=True)

    numeric_cols = [
        c for c in df.columns
        if c != timestamp_col and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not numeric_cols:
        raise DataLoadError(
            "No numeric metric columns found. Expected one date column + numeric metrics."
        )

    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values(timestamp_col).reset_index(drop=True)
    logger.info(f"Loaded {len(df)} rows, {len(numeric_cols)} metric columns.")
    return df


# ---------- v2: folder ----------

def load_excel_folder(
    folder_path: str,
    timestamp_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Read every .xlsx in `folder_path`, concatenate them, tag each row with its
    source filename in a new '__source_file' column. Returns the same wide
    format as load_excel; the source column is preserved but non-numeric, so
    it is ignored by anomaly_detector.

    All files MUST share the same timestamp column name and metric columns.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise DataLoadError(f"Excel folder not found: {folder.resolve()}")

    files = sorted(folder.glob("*.xlsx"))
    if not files:
        raise DataLoadError(f"No .xlsx files found in {folder.resolve()}")

    logger.info(f"Loading {len(files)} .xlsx file(s) from {folder.resolve()}")

    frames = []
    for fp in files:
        try:
            sub = load_excel(str(fp), timestamp_column=timestamp_column)
            sub["__source_file"] = fp.name
            frames.append(sub)
        except DataLoadError as exc:
            logger.warning(f"Skipping {fp.name}: {exc}")

    if not frames:
        raise DataLoadError(
            f"None of the .xlsx files in {folder} could be loaded."
        )

    # Ensure all frames share the same columns; align on the union, filling gaps with NaN.
    all_cols = sorted({c for f in frames for c in f.columns})
    aligned = [f.reindex(columns=all_cols) for f in frames]
    df = pd.concat(aligned, ignore_index=True)

    # Resolve timestamp column (union may have included it from any frame).
    ts_col = _resolve_timestamp_column(df, timestamp_column)
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)

    metric_cols = [
        c for c in df.columns
        if c != ts_col and c != "__source_file"
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not metric_cols:
        raise DataLoadError("No numeric metric columns found across the loaded files.")

    logger.info(
        f"Concatenated {len(df)} rows from {len(frames)} file(s); "
        f"{len(metric_cols)} metric column(s)."
    )
    return df


# ---------- v2: dispatcher ----------

def load_data(config: dict) -> Tuple[pd.DataFrame, str]:
    """
    Dispatcher driven by config['input_mode']:
        - 'excel_file'  : config['data']['file_path']
        - 'excel_folder': config['data']['folder_path']
        - 'database'    : config['db']['connection_string'] + ['db']['query']

    Returns (df, timestamp_column).
    """
    from db_loader import DatabaseLoadError, load_from_database  # local to avoid hard dep

    mode = config.get("input_mode", "excel_file")
    ts_col = config.get("data", {}).get("timestamp_column")

    if mode == "excel_file":
        file_path = config["data"]["file_path"]
        df = load_excel(file_path, timestamp_column=ts_col)
        return df, _resolve_timestamp_column(df, ts_col)

    if mode == "excel_folder":
        folder_path = config["data"].get("folder_path")
        if not folder_path:
            raise DataLoadError(
                "input_mode='excel_folder' requires data.folder_path in config."
            )
        df = load_excel_folder(folder_path, timestamp_column=ts_col)
        return df, _resolve_timestamp_column(df, ts_col)

    if mode == "database":
        db_cfg = config.get("db", {})
        conn_str = _resolve_env_value(db_cfg.get("connection_string", ""))
        sql = _resolve_env_value(db_cfg.get("query", ""))
        df = load_from_database(conn_str, sql, timestamp_column=ts_col)
        return df, _resolve_timestamp_column(df, ts_col)

    raise DataLoadError(f"Unknown input_mode: {mode!r}")


# ---------- helpers ----------

def _resolve_timestamp_column(df: pd.DataFrame, explicit: Optional[str]) -> str:
    if explicit and explicit in df.columns:
        return explicit
    non_numeric = [
        c for c in df.columns
        if c != "__source_file" and not pd.api.types.is_numeric_dtype(df[c])
    ]
    if not non_numeric:
        raise DataLoadError(
            "No non-numeric column found to use as timestamp."
        )
    return non_numeric[0]


def _resolve_env_value(value: str) -> str:
    """
    If `value` looks like an env-var reference ('ENV:NAME'), look it up.
    Otherwise return as-is. Lets users keep secrets out of config.yaml.
    """
    if isinstance(value, str) and value.startswith("ENV:"):
        import os
        name = value[4:].strip()
        resolved = os.environ.get(name, "")
        if not resolved:
            logger.warning(f"Environment variable {name} referenced in config is empty.")
        return resolved
    return value or ""
