"""
state_store.py

SQLite-backed per-metric anomaly state tracking for v2 alert suppression.

Two states per metric:
  - normal         : last observed value was in-range; next anomaly WILL alert.
  - active_anomaly : last observed value was anomalous; suppress until it returns.

Transitions:
  - First anomaly seen for a metric: state -> active_anomaly, last_alert_z recorded.
  - Subsequent anomalies while active_anomaly: no alert, BUT escalation is possible
    if abs(z) - abs(last_alert_z) >= escalation_z_delta.
  - In-range observation: state -> normal, history cleared.

This module has zero opinion on how data flows — it just owns the persistence.
"""

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


STATE_NORMAL = "normal"
STATE_ACTIVE = "active_anomaly"


@dataclass
class MetricState:
    """In-memory snapshot of one metric's state."""
    metric: str
    state: str
    last_alert_z: Optional[float]  # absolute value of |z| at the last alert sent
    last_alert_at: Optional[str]   # ISO timestamp of the last alert sent
    last_value: Optional[float]


class StateStore:
    """Thin SQLite wrapper. One file, one table, one upsert per metric."""

    def __init__(self, db_path: str = "anomaly_state.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        # check_same_thread=False keeps things simple if main.py is ever threaded.
        return sqlite3.connect(self.db_path, check_same_thread=False)

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
        logger.debug(f"StateStore ready at {self.db_path.resolve()}")

    # ---------- reads ----------

    def get(self, metric: str) -> MetricState:
        """Read the state for one metric. Returns normal-state default if absent."""
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
        """Bulk read for logging at run start."""
        out: Dict[str, MetricState] = {}
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT metric, state, last_alert_z, last_alert_at, last_value "
                "FROM metric_state"
            ).fetchall()
        for row in rows:
            out[row[0]] = MetricState(
                metric=row[0], state=row[1], last_alert_z=row[2],
                last_alert_at=row[3], last_value=row[4],
            )
        return out

    # ---------- writes ----------

    def upsert(self, state: MetricState) -> None:
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
                    _now_iso(),
                ),
            )
            conn.commit()

    # ---------- helpers used by main.py ----------

    def reset(self, metric: str) -> None:
        """Force a metric back to normal (e.g. when observed value is in-range)."""
        current = self.get(metric)
        if current.state == STATE_NORMAL and current.last_alert_z is None:
            return  # already clean
        self.upsert(MetricState(
            metric=metric,
            state=STATE_NORMAL,
            last_alert_z=None,
            last_alert_at=None,
            last_value=current.last_value,
        ))


def _now_iso() -> str:
    from datetime import datetime
    return datetime.now().isoformat(timespec="seconds")
