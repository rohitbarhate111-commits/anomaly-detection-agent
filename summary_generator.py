"""
summary_generator.py

Turns raw Anomaly records into short, human-readable summaries.

Caveat: we don't have a real business glossary, so the "Possible Impact" section
is inferred from the column name using generic heuristics (e.g. "revenue" ->
financial impact, "error" -> reliability impact). This is intentionally generic
and should be replaced with a proper glossary mapping for production use.
"""

import logging
from typing import List

from anomaly_detector import Anomaly

logger = logging.getLogger(__name__)


# Generic impact hints keyed by lowercased metric substring.
# NOTE: inferential only - no business glossary provided.
_IMPACT_HINTS = [
    (("revenue", "sales", "income", "profit"), "financial performance"),
    (("cost", "expense", "spend"), "operating costs"),
    (("error", "fail", "crash", "5xx"), "system reliability"),
    (("latency", "response_time", "p99", "p95"), "user experience / performance"),
    (("user", "active", "signup", "session"), "user engagement"),
    (("cpu", "memory", "disk", "io"), "infrastructure health"),
    (("conversion", "click", "impression"), "marketing effectiveness"),
    (("order", "cart", "checkout"), "purchase funnel"),
]


def _infer_impact(metric_name: str) -> str:
    name = metric_name.lower()
    for needles, label in _IMPACT_HINTS:
        if any(n in name for n in needles):
            return label
    return "operational metric (impact unclear without business context)"


def _severity_label(z_score: float) -> str:
    """Translate raw z-score into plain language."""
    a = abs(z_score)
    if a >= 6:
        return "extreme outlier"
    if a >= 4.5:
        return "well outside the typical range"
    if a >= 3:
        return "noticeably outside the typical range"
    return "slightly outside the typical range"


def generate_summary(anomaly: Anomaly) -> dict:
    """
    Build a structured summary block for one anomaly.

    Returns:
        dict with keys: metric, timestamp, direction, severity, what_changed,
        significance, possible_impact, pct_change.
    """
    direction_word = "spiked" if anomaly.direction == "up" else "dropped"

    # Percent change vs the rolling mean (the "expected" baseline).
    if anomaly.rolling_mean and not _is_zero(anomaly.rolling_mean):
        pct_change = ((anomaly.actual - anomaly.rolling_mean) / anomaly.rolling_mean) * 100.0
        pct_str = f"{pct_change:+.1f}% vs baseline"
    else:
        pct_str = "baseline near zero (percent change not meaningful)"

    what_changed = (
        f"{anomaly.metric} {direction_word} from a baseline of "
        f"{anomaly.rolling_mean:.2f} to {anomaly.actual:.2f} ({pct_str}) "
        f"on {anomaly.timestamp.strftime('%Y-%m-%d')}."
    )

    significance = (
        f"Z-score of {anomaly.z_score:+.2f} - this is "
        f"{_severity_label(anomaly.z_score)} "
        f"(threshold: ±3.0 std deviations)."
    )

    impact = _infer_impact(anomaly.metric)

    possible_impact = (
        f"Given the column name '{anomaly.metric}', this likely affects "
        f"{impact}. Confirm against your internal business glossary before "
        f"acting on it."
    )

    return {
        "metric": anomaly.metric,
        "timestamp": anomaly.timestamp.strftime("%Y-%m-%d"),
        "direction": anomaly.direction,
        "severity": _severity_label(anomaly.z_score),
        "z_score": anomaly.z_score,
        "actual": anomaly.actual,
        "rolling_mean": anomaly.rolling_mean,
        "what_changed": what_changed,
        "significance": significance,
        "possible_impact": possible_impact,
    }


def generate_summaries(anomalies: List[Anomaly]) -> List[dict]:
    """Generate summaries for every anomaly in the list."""
    summaries = [generate_summary(a) for a in anomalies]
    logger.info(f"Generated {len(summaries)} business-context summaries.")
    return summaries


def _is_zero(x: float, tol: float = 1e-9) -> bool:
    return abs(x) < tol
