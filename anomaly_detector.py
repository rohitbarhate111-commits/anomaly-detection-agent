"""
anomaly_detector.py

Core anomaly detection algorithms for time-series metrics.
Supports:
  1. Rolling Z-score (standard statistical process control)
  2. Seasonal STL Decomposition (robust LOESS decomposition + residual z-scoring)

Mathematical formulation:
  - Rolling Z-Score:
      z_t = (y_t - mean_{W}(y)) / std_{W}(y)
      where W is the trailing window size.
  - Seasonal STL:
      y_t = Trend_t + Seasonal_t + Residual_t
      Expected Baseline: Baseline_t = Trend_t + Seasonal_t
      Residual_t = y_t - Baseline_t
      z_t = (Residual_t - mean_{W}(Residual)) / std_{W}(Residual)
      Expected value is Baseline_t + mean_{W}(Residual).

Handles insufficient history, zero-variance edge cases, and directional classification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_SEASONAL_PERIOD = 7   # e.g., weekly cycles for daily data
DEFAULT_MIN_CYCLES = 2        # requires at least 2 full cycles


@dataclass(frozen=True)
class Anomaly:
    """Structured, immutable representation of a single detected anomaly."""
    metric: str
    timestamp: pd.Timestamp
    actual: float
    rolling_mean: float           # Expected / baseline value of the metric
    rolling_std: float            # Local standard deviation
    z_score: float                # Signed z-score deviation
    direction: str                # "up" or "down"
    detection_mode_used: str = "zscore"
    seasonal_period: Optional[int] = None

    def to_dict(self) -> dict:
        """Serialize anomaly record to dictionary."""
        return {
            "metric": self.metric,
            "timestamp": self.timestamp.isoformat() if hasattr(self.timestamp, "isoformat") else str(self.timestamp),
            "actual": float(self.actual),
            "rolling_mean": float(self.rolling_mean),
            "rolling_std": float(self.rolling_std),
            "z_score": float(self.z_score),
            "direction": self.direction,
            "detection_mode_used": self.detection_mode_used,
            "seasonal_period": self.seasonal_period,
        }


def detect_anomalies(
    df: pd.DataFrame,
    timestamp_column: str,
    window_size: int = 30,
    z_threshold: float = 3.0,
    min_periods: Optional[int] = None,
    detection_mode: str = "seasonal",
    seasonal_period: int = DEFAULT_SEASONAL_PERIOD,
    min_cycles: int = DEFAULT_MIN_CYCLES,
) -> List[Anomaly]:
    """
    Detect statistical and seasonal anomalies across all numeric columns in df.

    Args:
        df: Wide-format DataFrame containing timestamp_column and numeric metrics.
        timestamp_column: Name of the timestamp / date column.
        window_size: Trailing window size for baseline computation (>= 2).
        z_threshold: Absolute z-score cutoff beyond which a point is anomalous (> 0).
        min_periods: Minimum non-null observations in a window. Defaults to window_size.
        detection_mode: "zscore" or "seasonal".
        seasonal_period: Seasonality cycle length (default 7).
        min_cycles: Minimum full seasonal cycles required for STL (default 2).

    Returns:
        List[Anomaly] sorted chronologically by (timestamp, metric).
    """
    if window_size < 2:
        raise ValueError(f"window_size must be >= 2 to compute standard deviation; received {window_size}.")
    if z_threshold <= 0:
        raise ValueError(f"z_threshold must be strictly positive; received {z_threshold}.")
    if detection_mode not in ("zscore", "seasonal"):
        raise ValueError(f"Unknown detection_mode: {detection_mode!r}. Expected 'zscore' or 'seasonal'.")
    if seasonal_period < 2:
        raise ValueError(f"seasonal_period must be >= 2; received {seasonal_period}.")

    if min_periods is None:
        min_periods = window_size

    if timestamp_column not in df.columns:
        raise KeyError(f"Timestamp column '{timestamp_column}' not found in DataFrame.")

    # Work on a copy sorted chronologically
    work_df = df.sort_values(timestamp_column).reset_index(drop=True)
    timestamps = work_df[timestamp_column]

    metric_cols = [
        col for col in work_df.columns
        if col != timestamp_column and col != "__source_file" and pd.api.types.is_numeric_dtype(work_df[col])
    ]

    if not metric_cols:
        logger.warning("No numeric metric columns found for anomaly detection.")
        return []

    logger.info(
        f"Scanning {len(metric_cols)} metric(s) | mode={detection_mode} | "
        f"window={window_size} | z_threshold={z_threshold} | seasonal_period={seasonal_period}"
    )

    anomalies: List[Anomaly] = []

    for metric in metric_cols:
        series = work_df[metric]

        if detection_mode == "seasonal":
            if not _has_enough_history(series, seasonal_period, min_cycles):
                logger.warning(
                    f"[{metric}] Insufficient history for seasonal STL "
                    f"(requires >= {min_cycles * seasonal_period} points, found {series.dropna().size}). "
                    f"Falling back to rolling z-score."
                )
                metric_anomalies = _detect_zscore(
                    series, timestamps, metric, window_size, z_threshold, min_periods
                )
            else:
                metric_anomalies = _detect_seasonal(
                    series, timestamps, metric, window_size, z_threshold, min_periods, seasonal_period
                )
        else:
            metric_anomalies = _detect_zscore(
                series, timestamps, metric, window_size, z_threshold, min_periods
            )

        anomalies.extend(metric_anomalies)

    anomalies.sort(key=lambda a: (a.timestamp, a.metric))
    logger.info(f"Anomaly detection complete: {len(anomalies)} anomalies flagged across {len(metric_cols)} metrics.")
    return anomalies


def _detect_zscore(
    series: pd.Series,
    timestamps: pd.Series,
    metric: str,
    window_size: int,
    z_threshold: float,
    min_periods: int,
) -> List[Anomaly]:
    """Pure rolling z-score outlier detection."""
    rolling_mean = series.rolling(window=window_size, min_periods=min_periods).mean()
    rolling_std = series.rolling(window=window_size, min_periods=min_periods).std(ddof=0)

    # Protect against zero standard deviation (constant series)
    safe_std = rolling_std.replace(0.0, np.nan)
    safe_std[safe_std < 1e-9] = np.nan
    z_scores = (series - rolling_mean) / safe_std

    return _collect_anomalies(
        series=series,
        timestamps=timestamps,
        metric=metric,
        expected_series=rolling_mean,
        std_series=rolling_std,
        z_scores=z_scores,
        z_threshold=z_threshold,
        mode="zscore",
        seasonal_period=None,
    )


def _detect_seasonal(
    series: pd.Series,
    timestamps: pd.Series,
    metric: str,
    window_size: int,
    z_threshold: float,
    min_periods: int,
    seasonal_period: int,
) -> List[Anomaly]:
    """
    Robust STL seasonal decomposition + residual outlier scoring.

    Model:
      Actual = Trend + Seasonal + Residual
      Expected baseline = Trend + Seasonal
      Residual = Actual - (Trend + Seasonal)
    """
    from statsmodels.tsa.seasonal import STL

    # Forward/backward fill small missing gaps for STL decomposition
    filled = series.ffill().bfill()
    clean = filled.dropna()

    if clean.size < 2 * seasonal_period:
        return _detect_zscore(series, timestamps, metric, window_size, z_threshold, min_periods)

    try:
        stl = STL(clean.values, period=seasonal_period, robust=True)
        res = stl.fit()
    except Exception as exc:
        logger.warning(f"[{metric}] STL fitting failed ({exc}). Falling back to rolling z-score.")
        return _detect_zscore(series, timestamps, metric, window_size, z_threshold, min_periods)

    trend_seasonal = pd.Series(res.trend + res.seasonal, index=clean.index)
    residuals = pd.Series(res.resid, index=clean.index)

    # Rolling baseline over residuals
    r_mean = residuals.rolling(window=window_size, min_periods=min_periods).mean().fillna(0.0)
    r_std = residuals.rolling(window=window_size, min_periods=min_periods).std(ddof=0)
    safe_r_std = r_std.replace(0.0, np.nan)
    safe_r_std[safe_r_std < 1e-9] = np.nan

    # Z-score of residuals relative to residual baseline
    z_scores_clean = (residuals - r_mean) / safe_r_std

    # Model expected value for the original series = (Trend + Seasonal) + rolling residual mean
    expected_clean = trend_seasonal + r_mean

    # Align onto full series length
    z_scores_full = pd.Series(np.nan, index=series.index, dtype=float)
    z_scores_full.loc[clean.index] = z_scores_clean

    expected_full = pd.Series(np.nan, index=series.index, dtype=float)
    expected_full.loc[clean.index] = expected_clean

    std_full = pd.Series(np.nan, index=series.index, dtype=float)
    std_full.loc[clean.index] = r_std

    return _collect_anomalies(
        series=series,
        timestamps=timestamps,
        metric=metric,
        expected_series=expected_full,
        std_series=std_full,
        z_scores=z_scores_full,
        z_threshold=z_threshold,
        mode="seasonal",
        seasonal_period=seasonal_period,
    )


def _collect_anomalies(
    series: pd.Series,
    timestamps: pd.Series,
    metric: str,
    expected_series: pd.Series,
    std_series: pd.Series,
    z_scores: pd.Series,
    z_threshold: float,
    mode: str,
    seasonal_period: Optional[int],
) -> List[Anomaly]:
    """
    Safely extracts Anomaly objects using positional integer indexing to eliminate index mismatches.
    """
    out: List[Anomaly] = []
    n = len(series)

    for i in range(n):
        z_val = z_scores.iloc[i]
        actual_val = series.iloc[i]

        if pd.isna(z_val) or pd.isna(actual_val):
            continue

        if abs(z_val) <= z_threshold:
            continue

        exp_val = expected_series.iloc[i]
        std_val = std_series.iloc[i]

        ts = timestamps.iloc[i]
        if not isinstance(ts, pd.Timestamp):
            ts = pd.Timestamp(ts)

        direction = "up" if z_val > 0 else "down"

        out.append(
            Anomaly(
                metric=metric,
                timestamp=ts,
                actual=float(actual_val),
                rolling_mean=float(exp_val) if not pd.isna(exp_val) else float(actual_val),
                rolling_std=float(std_val) if not pd.isna(std_val) else 0.0,
                z_score=float(z_val),
                direction=direction,
                detection_mode_used=mode,
                seasonal_period=seasonal_period,
            )
        )

    return out


def _has_enough_history(series: pd.Series, period: int, min_cycles: int) -> bool:
    """Verifies that a series contains enough non-null points for reliable cycle decomposition."""
    needed = period * min_cycles
    have = int(series.dropna().size)
    return have >= needed
