"""
anomaly_detector.py

Detects anomalies in each numeric metric column using a rolling z-score:
  z = (x - rolling_mean) / rolling_std

A point is anomalous if |z| > threshold. Direction is tracked explicitly so
upward/downward spikes can be reported separately.

The first `window_size` rows per metric are skipped (insufficient baseline).
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "timestamp": self.timestamp.isoformat(),
            "actual": self.actual,
            "rolling_mean": self.rolling_mean,
            "rolling_std": self.rolling_std,
            "z_score": self.z_score,
            "direction": self.direction,
        }


def detect_anomalies(
    df: pd.DataFrame,
    timestamp_column: str,
    window_size: int = 30,
    z_threshold: float = 3.0,
    min_periods: Optional[int] = None,
) -> List[Anomaly]:
    """
    Run rolling z-score anomaly detection on every numeric metric column.

    Args:
        df: Wide-format DataFrame with one timestamp column and N metric columns.
        timestamp_column: Name of the timestamp column.
        window_size: Trailing window length for mean/std baseline.
        z_threshold: Points beyond ±z_threshold std-devs are flagged.
        min_periods: Minimum non-NaN observations required in the window. Defaults
            to window_size (so the first window rows are skipped by design).

    Returns:
        List of Anomaly records, sorted by timestamp then metric.
    """
    if window_size < 2:
        raise ValueError("window_size must be >= 2 to compute a std deviation.")
    if z_threshold <= 0:
        raise ValueError("z_threshold must be positive.")

    if min_periods is None:
        min_periods = window_size

    metric_cols = [
        col for col in df.columns
        if col != timestamp_column and pd.api.types.is_numeric_dtype(df[col])
    ]

    anomalies: List[Anomaly] = []
    logger.info(
        f"Running anomaly detection on {len(metric_cols)} metrics "
        f"(window={window_size}, z_threshold=±{z_threshold})."
    )

    for metric in metric_cols:
        series = df[metric]
        rolling_mean = series.rolling(window=window_size, min_periods=min_periods).mean()
        rolling_std = series.rolling(window=window_size, min_periods=min_periods).std(ddof=0)

        # Avoid divide-by-zero: when std is 0 (flat signal) z is undefined; skip.
        safe_std = rolling_std.replace(0, np.nan)
        z_scores = (series - rolling_mean) / safe_std

        for idx in df.index:
            z = z_scores.iloc[idx]
            mean = rolling_mean.iloc[idx]
            std = rolling_std.iloc[idx]
            value = series.iloc[idx]

            if pd.isna(z) or pd.isna(value):
                continue
            if abs(z) <= z_threshold:
                continue

            direction = "up" if z > 0 else "down"
            anomalies.append(
                Anomaly(
                    metric=metric,
                    timestamp=df[timestamp_column].iloc[idx],
                    actual=float(value),
                    rolling_mean=float(mean),
                    rolling_std=float(std),
                    z_score=float(z),
                    direction=direction,
                )
            )

    anomalies.sort(key=lambda a: (a.timestamp, a.metric))
    logger.info(f"Detection complete. {len(anomalies)} anomalies flagged.")
    return anomalies
