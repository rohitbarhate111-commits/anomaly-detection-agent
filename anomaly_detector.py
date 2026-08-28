"""Detect anomalies in numeric time-series metrics."""

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_SEASONAL_PERIOD = 7
DEFAULT_MIN_CYCLES = 2


@dataclass
class Anomaly:
    """Structured representation of a detected anomaly."""

    metric: str
    timestamp: pd.Timestamp
    actual: float
    rolling_mean: float
    rolling_std: float
    z_score: float
    direction: str
    detection_mode_used: str = "zscore"
    seasonal_period: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "timestamp": self.timestamp.isoformat(),
            "actual": self.actual,
            "rolling_mean": self.rolling_mean,
            "rolling_std": self.rolling_std,
            "z_score": self.z_score,
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
    detection_mode: str = "zscore",
    seasonal_period: int = DEFAULT_SEASONAL_PERIOD,
    min_cycles: int = DEFAULT_MIN_CYCLES,
) -> List[Anomaly]:
    """Detect anomalies across all numeric metric columns."""
    if timestamp_column not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_column}' is not present.")
    if window_size < 2:
        raise ValueError("window_size must be >= 2.")
    if z_threshold <= 0:
        raise ValueError("z_threshold must be positive.")
    if detection_mode not in {"zscore", "seasonal"}:
        raise ValueError(f"Unknown detection_mode: {detection_mode}")
    if seasonal_period < 2:
        raise ValueError("seasonal_period must be >= 2.")
    if min_cycles < 1:
        raise ValueError("min_cycles must be >= 1.")

    if min_periods is None:
        min_periods = window_size
    if not 1 <= min_periods <= window_size:
        raise ValueError("min_periods must be between 1 and window_size.")

    metric_cols = [
        column for column in df.columns
        if column != timestamp_column and pd.api.types.is_numeric_dtype(df[column])
    ]
    if not metric_cols:
        return []

    anomalies: List[Anomaly] = []
    logger.info(
        "Detection mode: %s | window=%d, z_threshold=±%s, %d metric(s).",
        detection_mode,
        window_size,
        z_threshold,
        len(metric_cols),
    )

    for metric in metric_cols:
        series = df[metric]
        if detection_mode == "seasonal" and _has_enough_history(series, seasonal_period, min_cycles):
            detected = _detect_seasonal(
                series, df[timestamp_column], metric, window_size,
                z_threshold, min_periods, seasonal_period,
            )
        else:
            if detection_mode == "seasonal":
                logger.warning(
                    "[%s] Insufficient history for seasonal mode; falling back to zscore.",
                    metric,
                )
            detected = _detect_zscore(
                series, df[timestamp_column], metric,
                window_size, z_threshold, min_periods,
            )
        anomalies.extend(detected)

    anomalies.sort(key=lambda anomaly: (anomaly.timestamp, anomaly.metric))
    logger.info("Detection complete. %d anomalies flagged.", len(anomalies))
    return anomalies


def _detect_zscore(series, timestamps, metric, window_size, z_threshold, min_periods):
    rolling_mean = series.rolling(window=window_size, min_periods=min_periods).mean()
    rolling_std = series.rolling(window=window_size, min_periods=min_periods).std(ddof=0)
    z_scores = (series - rolling_mean) / rolling_std.replace(0, np.nan)
    return _collect(
        series, timestamps, metric, rolling_mean, rolling_std, z_scores,
        z_threshold, mode="zscore", seasonal_period=None,
    )


def _detect_seasonal(series, timestamps, metric, window_size, z_threshold, min_periods, seasonal_period):
    from statsmodels.tsa.seasonal import STL

    filled = series.ffill().bfill()
    clean = filled.dropna()
    if clean.size < 2 * seasonal_period:
        return _detect_zscore(series, timestamps, metric, window_size, z_threshold, min_periods)

    result = STL(clean.values, period=seasonal_period, robust=True).fit()
    residuals = pd.Series(result.resid, index=clean.index)
    residual_mean = residuals.rolling(window=window_size, min_periods=min_periods).mean()
    residual_std = residuals.rolling(window=window_size, min_periods=min_periods).std(ddof=0)
    z_scores = (residuals - residual_mean) / residual_std.replace(0, np.nan)

    z_full = pd.Series(index=series.index, dtype=float)
    mean_full = pd.Series(index=series.index, dtype=float)
    std_full = pd.Series(index=series.index, dtype=float)
    z_full.loc[z_scores.index] = z_scores.values
    mean_full.loc[residual_mean.index] = residual_mean.values
    std_full.loc[residual_std.index] = residual_std.values

    return _collect(
        series, timestamps, metric, mean_full, std_full, z_full,
        z_threshold, mode="seasonal", seasonal_period=seasonal_period,
    )


def _collect(series, timestamps, metric, rolling_mean, rolling_std, z_scores, z_threshold, mode, seasonal_period):
    anomalies = []
    for position in range(len(series)):
        z = z_scores.iloc[position]
        value = series.iloc[position]
        if pd.isna(z) or pd.isna(value) or abs(z) <= z_threshold:
            continue
        mean = rolling_mean.iloc[position]
        std = rolling_std.iloc[position]
        anomalies.append(
            Anomaly(
                metric=metric,
                timestamp=pd.Timestamp(timestamps.iloc[position]),
                actual=float(value),
                rolling_mean=float(mean) if not pd.isna(mean) else 0.0,
                rolling_std=float(std) if not pd.isna(std) else 0.0,
                z_score=float(z),
                direction="up" if z > 0 else "down",
                detection_mode_used=mode,
                seasonal_period=seasonal_period,
            )
        )
    return anomalies


def _has_enough_history(series: pd.Series, period: int, min_cycles: int) -> bool:
    return int(series.dropna().size) >= period * min_cycles
