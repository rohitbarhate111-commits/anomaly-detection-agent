"""
main.py

Orchestrator. v2 superset of v1; defaults preserve the v1 flow.

Run modes (from config.yaml):
  input_mode       : excel_file | excel_folder | database
  detection_mode   : zscore | seasonal
  suppression_enabled: true | false

Run flow:
  1. Load config (+ optional .env).
  2. Load data via the dispatcher in data_loader.
  3. Detect anomalies (zscore or seasonal).
  4. Generate summaries.
  5. Find co-occurring anomalies and attach correlation notes.
  6. Apply suppression state machine -> which items get sent in the email.
  7. Send ONE summary email (with escalations merged in).
  8. Write the HTML report (always, when there are anomalies to display).

CLI:
    python main.py
    python main.py --file path/to/file.xlsx
    python main.py --input-mode excel_folder --folder path/to/dir
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

import yaml

from data_loader import DataLoadError, load_data
from anomaly_detector import detect_anomalies
from summary_generator import generate_summaries
from email_alerter import send_alert
from correlation import attach_correlation_notes, find_co_occurrences
from state_store import STATE_ACTIVE, STATE_NORMAL, StateStore
from report_generator import generate_report

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def setup_logging(level: str, fmt: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("openpyxl").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI Anomaly Detection Agent - scans metrics for outliers.",
    )
    parser.add_argument("--file", help="Override config data.file_path (input_mode=excel_file).", default=None)
    parser.add_argument("--folder", help="Override config data.folder_path (input_mode=excel_folder).", default=None)
    parser.add_argument("--config", default=str(CONFIG_PATH), help="Path to config.yaml.")
    return parser.parse_args()


# ---------- suppression decision ----------

def apply_suppression(
    anomalies: list,
    summaries: list,
    state: StateStore,
    enabled: bool,
    observed_metrics: set,
    escalation_z_delta: float = 2.0,
) -> list:
    """
    Decide which summaries survive suppression and return ONLY those.

    Behaviour:
      - For each metric that had an anomaly this run, look up current state
        and apply the rules below.
      - For each metric that was observed this run BUT had no anomaly, reset
        its state to normal (it has returned to baseline).
      - If suppression disabled: every anomaly is included, state is still
        updated so re-enabling suppression later doesn't double-fire.

    Returns the filtered list of summaries.
    """
    # 1. Reset metrics that were observed in-range this run.
    anomalous_metrics = {a.metric for a in anomalies}
    for metric in observed_metrics:
        if metric in anomalous_metrics:
            continue
        ms = state.get(metric)
        if ms.state == STATE_ACTIVE:
            state.reset(metric)

    kept = []
    for a, s in zip(anomalies, summaries):
        metric = a.metric
        ms = state.get(metric)

        if not enabled:
            # Always alert, but still record so suppression can take over later.
            state.upsert(_StateStore.make_state(metric, STATE_ACTIVE, abs(a.z_score), s["timestamp"], a.actual))
            s["is_escalation"] = False
            kept.append(s)
            continue

        if ms.state == STATE_NORMAL:
            # First anomaly for this metric - alert and arm.
            state.upsert(_StateStore.make_state(metric, STATE_ACTIVE, abs(a.z_score), s["timestamp"], a.actual))
            s["is_escalation"] = False
            kept.append(s)
            logger = logging.getLogger("anomaly-agent")
            logger.info(f"[{metric}] NEW anomaly - alert sent, state -> active_anomaly.")
            continue

        # STATE_ACTIVE - check for escalation.
        prev_z = abs(ms.last_alert_z) if ms.last_alert_z is not None else 0.0
        curr_z = abs(a.z_score)
        if (curr_z - prev_z) >= escalation_z_delta:
            state.upsert(_StateStore.make_state(metric, STATE_ACTIVE, curr_z, s["timestamp"], a.actual))
            s["is_escalation"] = True
            kept.append(s)
            logger = logging.getLogger("anomaly-agent")
            logger.info(
                f"[{metric}] ESCALATION: |z| jumped {prev_z:.2f} -> {curr_z:.2f} "
                f"(delta {curr_z - prev_z:.2f} >= {escalation_z_delta})."
            )
        else:
            logger = logging.getLogger("anomaly-agent")
            logger.info(
                f"[{metric}] SUPPRESSED: anomaly still active, |z| {curr_z:.2f} "
                f"not {escalation_z_delta}+ above last alert |z| {prev_z:.2f}."
            )
            # Still update last_value so resets work correctly.
            state.upsert(_StateStore.make_state(metric, STATE_ACTIVE, prev_z, ms.last_alert_at, a.actual))

    return kept


class _StateStore:
    """Tiny adapter so we can use dataclass-like init for state_store."""
    @staticmethod
    def make_state(metric, state, last_alert_z, last_alert_at, last_value):
        from state_store import MetricState
        return MetricState(
            metric=metric,
            state=state,
            last_alert_z=last_alert_z,
            last_alert_at=last_alert_at,
            last_value=last_value,
        )


# ---------- main ----------

def main() -> int:
    args = parse_args()
    config = load_config(Path(args.config))

    setup_logging(
        config.get("logging", {}).get("level", "INFO"),
        config.get("logging", {}).get("format", "%(levelname)s | %(message)s"),
    )
    logger = logging.getLogger("anomaly-agent")

    # .env (best-effort)
    if load_dotenv is not None:
        env_path = Path(args.config).parent / ".env"
        if env_path.is_file():
            load_dotenv(env_path)
            logger.info(f"Loaded environment from {env_path}")
    else:
        logger.debug("python-dotenv not installed; skipping .env loading.")

    # CLI overrides for input paths.
    if args.file:
        config.setdefault("data", {})["file_path"] = args.file
    if args.folder:
        config["data"]["folder_path"] = args.folder

    # 1. Load.
    try:
        df, ts_col = load_data(config)
    except DataLoadError as exc:
        logger.error(f"Data load failed: {exc}")
        return 2

    # 2. Detect.
    detection_cfg = config.get("detection", {})
    try:
        anomalies = detect_anomalies(
            df,
            timestamp_column=ts_col,
            window_size=int(detection_cfg.get("window_size", 30)),
            z_threshold=float(detection_cfg.get("z_threshold", 3.0)),
            detection_mode=detection_cfg.get("mode", "zscore"),
            seasonal_period=int(detection_cfg.get("seasonal_period", 7)),
            min_cycles=int(detection_cfg.get("min_cycles", 2)),
        )
    except (ValueError, KeyError) as exc:
        logger.error(f"Detection config invalid: {exc}")
        return 2

    # 3. Summaries.
    summaries = generate_summaries(anomalies)

    # 4. Correlation notes.
    corr_window = int(config.get("correlation_window_days", 0))
    co_occ = find_co_occurrences(anomalies, window_days=corr_window)
    attach_correlation_notes(summaries, co_occ)

    for s in summaries:
        logger.info(
            f"  - {s['metric']} @ {s['timestamp']} "
            f"({s['direction'].upper()}, z={s['z_score']:+.2f}): "
            f"actual={s['actual']:.2f}, baseline={s['rolling_mean']:.2f}"
            + (f" [co-occur: {', '.join(s['co_occurrences'])}]" if s['co_occurrences'] else "")
        )

    # 5. Suppression.
    suppression_cfg = config.get("suppression", {})
    store = StateStore(
        db_path=suppression_cfg.get("state_db_path", "anomaly_state.db")
    )
    observed_metrics = {
        c for c in df.columns
        if c != ts_col and c != "__source_file"
        and pd.api.types.is_numeric_dtype(df[c])
    }
    kept = apply_suppression(
        anomalies=anomalies,
        summaries=summaries,
        state=store,
        enabled=bool(suppression_cfg.get("enabled", False)),
        observed_metrics=observed_metrics,
        escalation_z_delta=float(suppression_cfg.get("escalation_z_delta", 2.0)),
    )

    # 6. Email alert (only kept summaries).
    sent = send_alert(config["smtp"], kept)
    if not sent and kept:
        logger.warning("Alert email was not sent; check SMTP config / env vars.")

    # 7. HTML report.
    report_cfg = config.get("report", {})
    report_path = generate_report(
        df=df,
        timestamp_column=ts_col,
        summaries=kept,
        anomalies=[
            a for a, s in zip(anomalies, summaries) if s in kept
        ],
        output_dir=report_cfg.get("output_dir", "./reports"),
        detection_mode=detection_cfg.get("mode", "zscore"),
    )
    if report_path:
        logger.info(f"HTML report: {report_path}")

    logger.info("Run complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
