import tempfile
from pathlib import Path
import pandas as pd

from state_store import STATE_ACTIVE, STATE_NORMAL, MetricState, StateStore
from main import apply_suppression
from anomaly_detector import Anomaly


def test_state_store_persistence():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = Path(tmpdir) / "test_state.db"
        store = StateStore(str(db_file))

        # Initial state should be normal
        init_state = store.get("Revenue")
        assert init_state.state == STATE_NORMAL
        assert init_state.last_alert_z is None

        # Upsert active state
        store.upsert(MetricState("Revenue", STATE_ACTIVE, 3.5, "2026-06-01", 100.0))
        fetched = store.get("Revenue")
        assert fetched.state == STATE_ACTIVE
        assert fetched.last_alert_z == 3.5
        assert fetched.last_value == 100.0

        # Reset
        store.reset("Revenue")
        after_reset = store.get("Revenue")
        assert after_reset.state == STATE_NORMAL
        assert after_reset.last_alert_z is None


def test_suppression_and_escalation_lifecycle():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = Path(tmpdir) / "test_supp.db"
        store = StateStore(str(db_file))

        ts1 = pd.Timestamp("2026-06-01")
        ts2 = pd.Timestamp("2026-06-02")
        ts3 = pd.Timestamp("2026-06-03")

        # Run 1: First anomaly -> should be kept (alert emitted)
        a1 = Anomaly("Revenue", ts1, 200, 100, 10, 3.2, "up")
        s1 = {"metric": "Revenue", "timestamp": "2026-06-01", "is_escalation": False}
        kept1 = apply_suppression([a1], [s1], store, enabled=True, observed_metrics={"Revenue"})
        assert len(kept1) == 1
        assert kept1[0]["is_escalation"] is False

        # Run 2: Anomaly continues at |z| = 3.8 (delta = 0.6 < 2.0) -> SUPPRESSED
        a2 = Anomaly("Revenue", ts2, 210, 100, 10, 3.8, "up")
        s2 = {"metric": "Revenue", "timestamp": "2026-06-02", "is_escalation": False}
        kept2 = apply_suppression([a2], [s2], store, enabled=True, observed_metrics={"Revenue"})
        assert len(kept2) == 0

        # Run 3: Anomaly escalates to |z| = 5.5 (delta = 5.5 - 3.2 = 2.3 >= 2.0) -> ESCALATION
        a3 = Anomaly("Revenue", ts3, 350, 100, 10, 5.5, "up")
        s3 = {"metric": "Revenue", "timestamp": "2026-06-03", "is_escalation": False}
        kept3 = apply_suppression([a3], [s3], store, enabled=True, observed_metrics={"Revenue"})
        assert len(kept3) == 1
        assert kept3[0]["is_escalation"] is True

        # Run 4: Metric returns to normal (no anomalies observed) -> reset
        kept4 = apply_suppression([], [], store, enabled=True, observed_metrics={"Revenue"})
        assert len(kept4) == 0
        state = store.get("Revenue")
        assert state.state == STATE_NORMAL
