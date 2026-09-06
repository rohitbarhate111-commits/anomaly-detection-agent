"""
state_store.py

SQLite-backed state persistence for anomaly tracking, alert suppression,
and escalation detection across pipeline execution runs.

States:
  - normal: The metric is operating within expected baseline.
  - active_anomaly: The metric has breached threshold; repeat alerts are suppressed
    unless an escalation (significant |z| jump) occurs.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Generator, Optional

logger = logging.getLogger(__name__)

STATE_NORMAL = "normal"
STATE_ACTIVE = "active_anomaly"


@dataclass(frozen=True)
class MetricState:
    """Snapshot of a metric's alert state."""
    metric: str
    state: str
    last_alert_z: Optional[float]
    last_alert_at: Optional[str]
    last_value: Optional[float]


class StateStore:
    """Persistent SQLite store for alert suppression states."""

    def __init__(self, db_path: str = "anomaly_state.db") -> None:
        self.db_path = Path(db_path)
        if self.db_path.parent:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS metric_state (
                    metric        TEXT PRIMARY KEY,
                    state         TEXT NOT NULL,
                    last_alert_z  REAL,
                    last_alert_at TEXT,
                    last_value    REAL,
                    updated_at    TEXT NOT NULL
                )
                """
            )
            conn.commit()
        logger.debug(f"StateStore initialized at {self.db_path.resolve()}")

    def get(self, metric: str) -> MetricState:
        """Fetch state for a single metric. Returns normal defaults if unrecorded."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT metric, state, last_alert_z, last_alert_at, last_value "
                "FROM metric_state WHERE metric = ?",
                (metric,),
            ).fetchone()

        if row is None:
            return MetricState(metric, STATE_NORMAL, None, None, None)

        return MetricState(
            metric=row[0],
            state=row[1],
            last_alert_z=row[2],
            last_alert_at=row[3],
            last_value=row[4],
        )

    def get_all(self) -> Dict[str, MetricState]:
        """Fetch all recorded metric states."""
        out: Dict[str, MetricState] = {}
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT metric, state, last_alert_z, last_alert_at, last_value "
                "FROM metric_state"
            ).fetchall()
        for row in rows:
            out[row[0]] = MetricState(
                metric=row[0],
                state=row[1],
                last_alert_z=row[2],
                last_alert_at=row[3],
                last_value=row[4],
            )
        return out

    def upsert(self, state: MetricState) -> None:
        """Persist or update state for a metric."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO metric_state
                    (metric, state, last_alert_z, last_alert_at, last_value, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(metric) DO UPDATE SET
                    state         = excluded.state,
                    last_alert_z  = excluded.last_alert_z,
                    last_alert_at = excluded.last_alert_at,
                    last_value    = excluded.last_value,
                    updated_at    = excluded.updated_at
                """,
                (
                    state.metric,
                    state.state,
                    state.last_alert_z,
                    state.last_alert_at,
                    state.last_value,
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )
            conn.commit()

    def reset(self, metric: str) -> None:
        """Reset an active metric back to normal state."""
        current = self.get(metric)
        if current.state == STATE_NORMAL and current.last_alert_z is None:
            return
        self.upsert(
            MetricState(
                metric=metric,
                state=STATE_NORMAL,
                last_alert_z=None,
                last_alert_at=None,
                last_value=current.last_value,
            )
        )

    def clear_all(self) -> None:
        """Clear all stored state rows."""
        with self._connect() as conn:
            conn.execute("DELETE FROM metric_state")
            conn.commit()
