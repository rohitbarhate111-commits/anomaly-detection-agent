import numpy as np
import pandas as pd
import pytest

from anomaly_detector import Anomaly, detect_anomalies


def make_clean_series(n=100, baseline=100.0, std=2.0, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=n, freq="D")
    values = rng.normal(loc=baseline, scale=std, size=n)
    return pd.DataFrame({"Date": dates, "MetricA": values})


def test_validation_errors():
    df = make_clean_series(20)
    with pytest.raises(ValueError, match="window_size"):
        detect_anomalies(df, "Date", window_size=1)
    with pytest.raises(ValueError, match="z_threshold"):
        detect_anomalies(df, "Date", z_threshold=-1.0)
    with pytest.raises(ValueError, match="Unknown detection_mode"):
        detect_anomalies(df, "Date", detection_mode="magic")
    with pytest.raises(KeyError):
        detect_anomalies(df, "MissingDateColumn")


def test_zscore_detection_direction():
    df = make_clean_series(60, baseline=50.0, std=1.0)
    # Inject one upward spike and one downward drop
    df.loc[40, "MetricA"] = 150.0   # spike
    df.loc[50, "MetricA"] = -50.0   # drop

    anomalies = detect_anomalies(
        df,
        timestamp_column="Date",
        window_size=20,
        z_threshold=3.0,
        detection_mode="zscore",
    )

    metrics = [a.metric for a in anomalies]
    assert all(m == "MetricA" for m in metrics)
    assert len(anomalies) == 2

    spike = next(a for a in anomalies if a.timestamp == df.loc[40, "Date"])
    assert spike.direction == "up"
    assert spike.z_score > 3.0
    assert spike.actual == 150.0

    drop = next(a for a in anomalies if a.timestamp == df.loc[50, "Date"])
    assert drop.direction == "down"
    assert drop.z_score < -3.0
    assert drop.actual == -50.0


def test_seasonal_detection_with_stl():
    days = 120
    period = 7
    dates = pd.date_range("2026-01-01", periods=days, freq="D")
    t = np.arange(days)
    season = 10.0 * np.sin(2 * np.pi * t / period)
    values = 100.0 + season

    # Inject an anomaly that violates the seasonal pattern
    values[80] += 50.0

    df = pd.DataFrame({"Date": dates, "SeasonalMetric": values})
    anomalies = detect_anomalies(
        df,
        timestamp_column="Date",
        window_size=20,
        z_threshold=3.0,
        detection_mode="seasonal",
        seasonal_period=7,
        min_cycles=2,
    )

    assert len(anomalies) >= 1
    flagged = next((a for a in anomalies if a.timestamp == dates[80]), None)
    assert flagged is not None
    assert flagged.direction == "up"
    assert flagged.detection_mode_used == "seasonal"
    # Ensure expected baseline is meaningful (close to 100 + seasonal component)
    assert abs(flagged.rolling_mean - (100.0 + season[80])) < 15.0


def test_insufficient_history_seasonal_fallback():
    # Only 10 points (needs >= 14 for 2 cycles of 7)
    df = make_clean_series(10)
    anomalies = detect_anomalies(
        df,
        timestamp_column="Date",
        window_size=5,
        detection_mode="seasonal",
        seasonal_period=7,
        min_cycles=2,
    )
    # Should fall back to zscore without raising an exception
    assert isinstance(anomalies, list)
