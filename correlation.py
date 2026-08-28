"""
correlation.py

Groups anomalies that occurred on the same date (or within a configured window)
and attaches a co-occurrence hint to each affected summary.

Important: this is NOT causal inference. We only observe that two metrics moved
on the same day. The user-facing language must reflect that uncertainty.
"""

import logging
from collections import defaultdict
from datetime import timedelta
from typing import Dict, List

logger = logging.getLogger(__name__)


CO_OCCURRENCE_NOTE = (
    "Note: this anomaly coincided with unusual movement in {others} on the "
    "same date. This may indicate a related cause, or may be coincidental "
    "- not confirmed causation."
)


def find_co_occurrences(
    anomalies: List,
    window_days: int = 0,
) -> Dict[str, List[str]]:
    """
    Return a dict: metric_name -> ordered list of OTHER metric names that
    also had anomalies within the same time window.

    Args:
        anomalies: output of anomaly_detector.detect_anomalies (or filtered subset).
        window_days: 0 means "same calendar date"; N>0 expands the window on
            either side of each anomaly's timestamp by N days.

    Behaviour:
        - Timestamps are compared on their calendar date, not exact moment.
        - A metric never lists itself.
        - Within each cluster, each metric sees the same set of others.
    """
    if not anomalies:
        return {}

    by_date: Dict[object, List] = defaultdict(list)
    for a in anomalies:
        ts = a.timestamp
        if hasattr(ts, "date"):
            key = ts.date() if window_days == 0 else _window_key(ts, window_days)
        else:
            key = ts
        by_date[key].append(a)

    out: Dict[str, List[str]] = {}
    for cluster in by_date.values():
        if len(cluster) < 2:
            continue
        names = sorted({a.metric for a in cluster})
        for n in names:
            others = [o for o in names if o != n]
            if others:
                out[n] = others

    n_clusters = sum(1 for v in by_date.values() if len({a.metric for a in v}) > 1)
    logger.info(
        f"Correlation scan: {n_clusters} co-occurring cluster(s) found "
        f"(window_days={window_days})."
    )
    return out


def attach_correlation_notes(
    summaries: List,
    co_occurrences: Dict[str, List[str]],
) -> List:
    """
    Mutates each summary dict in place to add:
      - 'co_occurrences': List[str] of other metric names (always set, may be empty)
      - 'correlation_note': the formatted note, or "" when no co-occurrence.

    Returns the same list for convenience.
    """
    for s in summaries:
        others = co_occurrences.get(s["metric"], [])
        s["co_occurrences"] = others
        if others:
            joined = ", ".join(others)
            s["correlation_note"] = CO_OCCURRENCE_NOTE.format(others=joined)
        else:
            s["correlation_note"] = ""
    return summaries


def _window_key(ts, window_days: int):
    """Bucket a timestamp into a window key of width 2*window_days+1 days."""
    base = ts.date() if hasattr(ts, "date") else ts
    return base - timedelta(days=window_days)
