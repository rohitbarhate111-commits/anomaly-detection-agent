import pandas as pd
from correlation import attach_correlation_notes, find_co_occurrences
from anomaly_detector import Anomaly


def test_exact_day_correlation():
    d1 = pd.Timestamp("2026-06-01")
    d2 = pd.Timestamp("2026-06-02")

    a1 = Anomaly("Revenue", d1, 500, 100, 10, 4.0, "up")
    a2 = Anomaly("Latency_ms", d1, 300, 50, 5, 5.0, "up")
    a3 = Anomaly("Error_Rate", d2, 10, 1, 0.5, 3.5, "up")

    co = find_co_occurrences([a1, a2, a3], window_days=0)
    assert "Revenue" in co
    assert co["Revenue"] == ["Latency_ms"]
    assert "Latency_ms" in co
    assert co["Latency_ms"] == ["Revenue"]
    assert "Error_Rate" not in co


def test_sliding_window_correlation():
    d1 = pd.Timestamp("2026-06-01")
    d2 = pd.Timestamp("2026-06-02")  # 1 day apart

    a1 = Anomaly("MetricA", d1, 100, 50, 5, 4.0, "up")
    a2 = Anomaly("MetricB", d2, 200, 50, 5, 4.5, "up")

    # window_days=0 -> no correlation
    assert find_co_occurrences([a1, a2], window_days=0) == {}

    # window_days=1 -> correlates MetricA and MetricB
    co = find_co_occurrences([a1, a2], window_days=1)
    assert co["MetricA"] == ["MetricB"]
    assert co["MetricB"] == ["MetricA"]


def test_attach_correlation_notes_non_causal_language():
    summaries = [
        {"metric": "Revenue", "co_occurrences": [], "correlation_note": ""},
        {"metric": "Users", "co_occurrences": [], "correlation_note": ""},
    ]
    co = {"Revenue": ["Latency_ms", "Error_Rate"]}
    attach_correlation_notes(summaries, co)

    s_rev = summaries[0]
    assert s_rev["co_occurrences"] == ["Latency_ms", "Error_Rate"]
    assert "may be coincidental - not confirmed causation" in s_rev["correlation_note"]
    assert "Latency_ms, Error_Rate" in s_rev["correlation_note"]

    s_user = summaries[1]
    assert s_user["co_occurrences"] == []
    assert s_user["correlation_note"] == ""
