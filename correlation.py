"""Identify anomalies that occur within a configurable time window."""

import logging
from collections import defaultdict
from datetime import timedelta
from typing import Dict, List

logger = logging.getLogger(__name__)

CO_OCCURRENCE_NOTE = (
    "Note: this anomaly coincided with unusual movement in {others} within "
    "the configured time window. This may indicate a related cause, or may "
    "be coincidental - not confirmed causation."
)


def find_co_occurrences(anomalies: List, window_days: int = 0) -> Dict[str, List[str]]:
    """Return other metrics with anomalies within ``window_days`` of each anomaly.

    A zero-day window means the same calendar date. For a positive window,
    timestamps are compared pairwise rather than bucketed, so anomalies near
    opposite edges of a bucket cannot be incorrectly grouped together.
    """
    if window_days < 0:
        raise ValueError("window_days must be >= 0")
    if not anomalies:
        return {}

    out = defaultdict(set)
    delta = timedelta(days=window_days)

    for index, anomaly in enumerate(anomalies):
        for other in anomalies[index + 1 :]:
            if _within_window(anomaly.timestamp, other.timestamp, window_days, delta):
                if anomaly.metric != other.metric:
                    out[anomaly.metric].add(other.metric)
                    out[other.metric].add(anomaly.metric)

    result = {metric: sorted(metrics) for metric, metrics in out.items()}
    logger.info(
        "Correlation scan: %d metric(s) had co-occurring anomalies "
        "(window_days=%d).",
        len(result),
        window_days,
    )
    return result


def attach_correlation_notes(
    summaries: List,
    co_occurrences: Dict[str, List[str]],
) -> List:
    """Attach non-causal co-occurrence notes to summary dictionaries."""
    for summary in summaries:
        others = co_occurrences.get(summary["metric"], [])
        summary["co_occurrences"] = others
        summary["correlation_note"] = (
            CO_OCCURRENCE_NOTE.format(others=", ".join(others)) if others else ""
        )
    return summaries


def _within_window(left, right, window_days: int, delta: timedelta) -> bool:
    """Compare timestamps using calendar dates for zero-day windows."""
    if window_days == 0:
        return left.date() == right.date()
    return abs(left - right) <= delta
