"""
correlation.py

Identifies temporal co-occurrences of anomalies across distinct metrics.
Provides root-cause hints while maintaining rigorous non-causal language.

Language specification:
  - Must never claim verified causality.
  - Phrasing: "Note: this anomaly coincided with unusual movement in {others} on the
    same date. This may indicate a related cause, or may be coincidental - not confirmed causation."
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date, datetime
from typing import Dict, List, Sequence, Set

import pandas as pd

logger = logging.getLogger(__name__)

CO_OCCURRENCE_NOTE = (
    "Note: this anomaly coincided with unusual movement in {others} on the "
    "same date. This may indicate a related cause, or may be coincidental "
    "- not confirmed causation."
)


def _to_date(ts: object) -> date:
    """Converts a timestamp object to a standard calendar date."""
    if isinstance(ts, (datetime, date)):
        return ts.date() if hasattr(ts, "date") else ts
    parsed = pd.to_datetime(ts)
    return parsed.date()


def find_co_occurrences(
    anomalies: Sequence[object],
    window_days: int = 0,
) -> Dict[str, List[str]]:
    """
    Finds co-occurring anomalies across metrics within a sliding time window.

    Args:
        anomalies: Collection of Anomaly objects or dicts with 'metric' and 'timestamp'.
        window_days: 0 = exact same calendar day; N > 0 = within N days of each other.

    Returns:
        Dict mapping metric_name -> sorted list of other metric names co-occurring with it.
    """
    if not anomalies:
        return {}

    # Extract clean (metric, date) tuples
    items: List[tuple[str, date]] = []
    for a in anomalies:
        metric = getattr(a, "metric", None) or (a.get("metric") if isinstance(a, dict) else None)
        ts = getattr(a, "timestamp", None) or (a.get("timestamp") if isinstance(a, dict) else None)
        if metric is not None and ts is not None:
            items.append((str(metric), _to_date(ts)))

    co_occurring_map: Dict[str, Set[str]] = defaultdict(set)

    n = len(items)
    for i in range(n):
        metric_a, date_a = items[i]
        for j in range(i + 1, n):
            metric_b, date_b = items[j]
            if metric_a == metric_b:
                continue

            day_diff = abs((date_a - date_b).days)
            if day_diff <= window_days:
                co_occurring_map[metric_a].add(metric_b)
                co_occurring_map[metric_b].add(metric_a)

    # Convert sets to sorted lists
    result: Dict[str, List[str]] = {
        m: sorted(others) for m, others in sorted(co_occurring_map.items()) if others
    }

    n_co = len(result)
    logger.info(
        f"Correlation scan: {n_co} metric(s) associated with co-occurring events "
        f"(window_days={window_days})."
    )
    return result


def attach_correlation_notes(
    summaries: List[dict],
    co_occurrences: Dict[str, List[str]],
) -> List[dict]:
    """
    Enriches summary dictionaries in-place with co-occurrence hints and non-causal phrasing.

    Adds:
      - 'co_occurrences': List[str]
      - 'correlation_note': str
    """
    for s in summaries:
        metric = s.get("metric", "")
        others = co_occurrences.get(metric, [])
        s["co_occurrences"] = others
        if others:
            joined = ", ".join(others)
            s["correlation_note"] = CO_OCCURRENCE_NOTE.format(others=joined)
        else:
            s["correlation_note"] = ""
    return summaries
