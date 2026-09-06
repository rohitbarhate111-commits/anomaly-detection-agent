"""
summary_generator.py

Translates quantitative anomaly events into human-readable, domain-aware summaries.
Calculates percentage deviations relative to model baselines and categorizes severity.
"""

from __future__ import annotations

import logging
from typing import List, Sequence

from anomaly_detector import Anomaly

logger = logging.getLogger(__name__)

# Business impact heuristics mapped to metric name keywords
_IMPACT_HINTS = [
    (("revenue", "sales", "income", "profit", "mrr", "arr"), "financial performance and revenue capture"),
    (("cost", "expense", "spend", "burn"), "operating expenditure and budget run rate"),
    (("error", "fail", "crash", "5xx", "500", "exception"), "system stability and service reliability"),
    (("latency", "response_time", "p99", "p95", "duration", "lag"), "user experience and application performance"),
    (("user", "active", "signup", "session", "dau", "mau"), "customer adoption and user engagement"),
    (("cpu", "memory", "disk", "io", "load", "network"), "infrastructure capacity and resource health"),
    (("conversion", "click", "impression", "ctr"), "marketing funnel and conversion efficiency"),
    (("order", "cart", "checkout", "transaction"), "e-commerce order pipeline and checkout throughput"),
]


def _infer_impact(metric_name: str) -> str:
    name = metric_name.lower()
    for needles, label in _IMPACT_HINTS:
        if any(n in name for n in needles):
            return label
    return "operational telemetry (verify against domain-specific glossary)"


def _severity_label(z_score: float) -> str:
    a = abs(z_score)
    if a >= 6.0:
        return "extreme outlier"
    if a >= 4.5:
        return "critical outlier"
    if a >= 3.0:
        return "noticeable outlier"
    return "moderate deviation"


def generate_summary(anomaly: Anomaly, is_escalation: bool = False) -> dict:
    """
    Constructs a structured plain-language summary for an Anomaly.
    """
    direction_word = "spiked" if anomaly.direction == "up" else "dropped"

    expected = anomaly.rolling_mean
    actual = anomaly.actual

    if abs(expected) > 1e-6:
        pct_change = ((actual - expected) / abs(expected)) * 100.0
        pct_str = f"{pct_change:+.1f}% vs expected baseline"
    else:
        pct_str = "baseline near zero"

    ts_str = anomaly.timestamp.strftime("%Y-%m-%d") if hasattr(anomaly.timestamp, "strftime") else str(anomaly.timestamp)

    what_changed = (
        f"{anomaly.metric} {direction_word} from an expected baseline of "
        f"{expected:.2f} to {actual:.2f} ({pct_str}) on {ts_str}."
    )

    mode_tag = f" [{anomaly.detection_mode_used}]" if anomaly.detection_mode_used != "zscore" else ""
    significance = (
        f"Z-score of {anomaly.z_score:+.2f}{mode_tag} - classified as "
        f"{_severity_label(anomaly.z_score)} (threshold: |z| >= 3.0)."
    )

    impact = _infer_impact(anomaly.metric)
    possible_impact = (
        f"Metric '{anomaly.metric}' relates to {impact}. "
        f"Corroborate with system logs and relevant alerts before escalation."
    )

    return {
        "metric": anomaly.metric,
        "timestamp": ts_str,
        "direction": anomaly.direction,
        "severity": _severity_label(anomaly.z_score),
        "z_score": anomaly.z_score,
        "actual": actual,
        "rolling_mean": expected,
        "what_changed": what_changed,
        "significance": significance,
        "possible_impact": possible_impact,
        "co_occurrences": [],
        "correlation_note": "",
        "is_escalation": is_escalation,
    }


def generate_summaries(anomalies: Sequence[Anomaly]) -> List[dict]:
    """Generates structured summaries for all detected anomalies."""
    summaries = [generate_summary(a) for a in anomalies]
    logger.info(f"Generated {len(summaries)} business-context summaries.")
    return summaries
