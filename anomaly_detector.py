"""
anomaly_detector.py

Detects anomalies in each numeric metric column.

v1 (zscore):
    z = (x - rolling_mean) / rolling_std
    flag points where |z| > threshold

v2 (seasonal):
    STL decomposition per metric -> trend + seasonal + residual.
    flag residual points where |residual_z| > threshold, where
    residual_z = (residual - rolling_mean(residual)) / rolling_std(residual)
    Requires >= 2 full seasonal cycles of history; otherwise falls back to
    zscore with a logged warning.

The output schema is intentionally identical to v1 (metric, timestamp, actual,
rolling_mean, rolling_std, z_score, direction). v2 adds:
    detection_mode_used  : "zscore" | "seasonal"
    seasonal_period      : int or None

The downstream modules (summary_generator, email_alerter) read only the v1
fields, so this change is fully backward-compatible.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Seasonal mode defaults — exposed so main.py / tests can inspect them.
DEFAULT_SEASONAL_PERIOD = 7           # weekly for daily data
DEFAULT_MIN_CYCLES = 2                # >= 2 cycles required


@dataclass
class Anomaly:
    """Structured representation of a single detected anomaly."""
    metric: str
    timestamp: pd.Timestamp
    actual: float
    rolling_mean: float
    rolling_std: float
    z_score: float
    direction: str  # "up" or "down"
    # v2 additions (backward-compatible: defaults match v1 behaviour)
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


# ---------- public API ----------

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
    """
    Detect anomalies across all numeric metric columns in `df`.

    Args:
        df: wide-format DataFrame (timestamp column + N metric columns).
        timestamp_column: name of the timestamp column.
        window_size: trailing window length for the rolling baseline.
        z_threshold: |z| beyond this is anomalous.
        min_periods: min non-NaN observations in a window. Defaults to window_size.
        detection_mode: "zscore" (v1) or "seasonal" (v2 STL).
        seasonal_period: period for STL decomposition (default 7 = weekly).
        min_cycles: minimum full cycles of history required for seasonal mode.

    Returns:
        List[Anomaly] sorted by (timestamp, metric).
    """
    if window_size < 2:
        raise ValueError("window_size must be >= 2 to compute a std deviation.")
    if z_threshold <= 0:
        raise ValueError("z_threshold must be positive.")
    if detection_mode not in ("zscore", "seasonal"):
        raise ValueError(f"Unknown detection_mode: {detection_mode}")

    if min_periods is None:
        min_periods = window_size

    metric_cols = [
        col for col in df.columns
        if col != timestamp_column and pd.api.types.is_numeric_dtype(df[col])
    ]

    anomalies: List[Anomaly] = []
    logger.info(
        f"Detection mode: {detection_mode} | "
        f"window={window_size}, z_threshold=±{z_threshold}, "
        f"{len(metric_cols)} metric(s)."
    )

    for metric in metric_cols:
        series = df[metric]

        if detection_mode == "seasonal":
            enough = _has_enough_history(series, seasonal_period, min_cycles)
            if not enough:
                logger.warning(
                    f"[{metric}] Insufficient history for seasonal mode "
                    f"(need >= {min_cycles * seasonal_period} points, have "
                    f"{series.dropna().size}). Falling back to zscore."
                )
                metric_anomalies = _detect_zscore(
                    series, df[timestamp_column], metric,
                    window_size, z_threshold, min_periods,
                )
            else:
                metric_anomalies = _detect_seasonal(
                    series, df[timestamp_column], metric,
                    window_size, z_threshold, min_periods,
                    seasonal_period,
                )
        else:
            metric_anomalies = _detect_zscore(
                series, df[timestamp_column], metric,
                window_size, z_threshold, min_periods,
            )

        anomalies.extend(metric_anomalies)

    anomalies.sort(key=lambda a: (a.timestamp, a.metric))
    logger.info(f"Detection complete. {len(anomalies)} anomalies flagged.")
    return anomalies


# ---------- mode implementations ----------

def _detect_zscore(
    series: pd.Series,
    timestamps: pd.Series,
    metric: str,
    window_size: int,
    z_threshold: float,
    min_periods: int,
) -> List[Anomaly]:
    rolling_mean = series.rolling(window=window_size, min_periods=min_periods).mean()
    rolling_std = series.rolling(window=window_size, min_periods=min_periods).std(ddof=0)

    safe_std = rolling_std.replace(0, np.nan)
    z_scores = (series - rolling_mean) / safe_std

    return _collect(
        series, timestamps, metric, rolling_mean, rolling_std, z_scores,
        z_threshold, mode="zscore", seasonal_period=None,
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
    STL on the filled series, then z-score the residuals.

    We forward-fill small gaps and drop remaining NaNs before STL because STL
    cannot handle NaN. The returned anomalies are re-indexed back onto the
    original timestamps so output alignment is identical to v1.
    """
    from statsmodels.tsa.seasonal import STL  # local import: heavy, optional

    filled = series.ffill().bfill()
    clean = filled.dropna()

    if clean.size < 2 * seasonal_period:
        # Defensive — _has_enough_history should have caught this, but guard anyway.
        return _detect_zscore(series, timestamps, metric, window_size,
                              z_threshold, min_periods)

    stl = STL(clean.values, period=seasonal_period, robust=True)
    result = stl.fit()
    residuals = pd.Series(result.resid, index=clean.index)

    # Rolling baseline over the RESIDUALS — same z-score machinery as v1.
    r_mean = residuals.rolling(window=window_size, min_periods=min_periods).mean()
    r_std = residuals.rolling(window=window_size, min_periods=min_periods).std(ddof=0)
    safe_std = r_std.replace(0, np.nan)
    z_scores = (residuals - r_mean) / safe_std

    # Re-emit a full-length series so _collect can iterate the original index.
    z_full = pd.Series(index=series.index, dtype=float)
    z_full.loc[z_scores.index] = z_scores.values

    r_mean_full = pd.Series(index=series.index, dtype=float)
    r_mean_full.loc[r_mean.index] = r_mean.values
    r_std_full = pd.Series(index=series.index, dtype=float)
    r_std_full.loc[r_std.index] = r_std.values

    return _collect(
        series, timestamps, metric, r_mean_full, r_std_full, z_full,
        z_threshold, mode="seasonal", seasonal_period=seasonal_period,
    )


def _collect(
    series: pd.Series,
    timestamps: pd.Series,
    metric: str,
    rolling_mean: pd.Series,
    rolling_std: pd.Series,
    z_scores: pd.Series,
    z_threshold: float,
    mode: str,
    seasonal_period: Optional[int],
) -> List[Anomaly]:
    out: List[Anomaly] = []
    for idx in series.index:
        z = z_scores.iloc[idx]
        value = series.iloc[idx]
        if pd.isna(z) or pd.isna(value):
            continue
        if abs(z) <= z_threshold:
            continue
        out.append(
            Anomaly(
                metric=metric,
                timestamp=timestamps.iloc[idx],
                actual=float(value),
                rolling_mean=float(rolling_mean.iloc[idx]) if not pd.isna(rolling_mean.iloc[idx]) else 0.0,
                rolling_std=float(rolling_std.iloc[idx]) if not pd.isna(rolling_std.iloc[idx]) else 0.0,
                z_score=float(z),
                direction="up" if z > 0 else "down",
                detection_mode_used=mode,
                seasonal_period=seasonal_period,
            )
        )
    return out


# ---------- helpers ----------

def _has_enough_history(series: pd.Series, period: int, min_cycles: int) -> bool:
    needed = period * min_cycles
    have = int(series.dropna().size)
    return have >= needed
