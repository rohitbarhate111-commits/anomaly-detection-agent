"""Command-line orchestration for the anomaly detection pipeline."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

import pandas as pd
import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - requirements.txt includes python-dotenv
    load_dotenv = None

from anomaly_detector import detect_anomalies
from correlation import attach_correlation_notes, find_co_occurrences
from data_loader import DataLoadError, load_data
from email_alerter import send_alert
from report_generator import generate_report
from state_store import MetricState, StateStore, STATE_ACTIVE, STATE_NORMAL
from summary_generator import generate_summaries

CONFIG_PATH = Path(__file__).parent / "config.yaml"
LOGGER = logging.getLogger("anomaly-agent")


def setup_logging(level: str, fmt: str) -> None:
    """Configure application logging once for a CLI run."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("openpyxl").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def load_config(path: Path) -> Dict[str, Any]:
    """Load and validate the top-level YAML configuration structure."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Unable to load config '{path}': {exc}") from exc

    if not isinstance(config, dict):
        raise ValueError(f"Config '{path}' must contain a YAML mapping.")
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan time-series metrics for anomalies and generate alerts/reports."
    )
    parser.add_argument(
        "--file",
        help="Override data.file_path and use Excel file input mode.",
    )
    parser.add_argument(
        "--folder",
        help="Override data.folder_path and use Excel folder input mode.",
    )
    parser.add_argument("--config", default=str(CONFIG_PATH), help="Path to config.yaml.")
    return parser.parse_args()


def apply_suppression(
    anomalies: List,
    summaries: List[dict],
    state: StateStore,
    enabled: bool,
    observed_metrics: Set[str],
    escalation_z_delta: float = 2.0,
) -> List[dict]:
    """Filter repeated anomaly alerts while preserving escalation events."""
    if escalation_z_delta < 0:
        raise ValueError("escalation_z_delta must be >= 0")

    anomalous_metrics = {anomaly.metric for anomaly in anomalies}
    for metric in observed_metrics - anomalous_metrics:
        if state.get(metric).state == STATE_ACTIVE:
            state.reset(metric)

    kept = []
    for anomaly, summary in zip(anomalies, summaries):
        metric = anomaly.metric
        current = state.get(metric)
        current_z = abs(anomaly.z_score)

        if not enabled or current.state == STATE_NORMAL:
            state.upsert(
                MetricState(
                    metric=metric,
                    state=STATE_ACTIVE,
                    last_alert_z=current_z,
                    last_alert_at=summary["timestamp"],
                    last_value=anomaly.actual,
                )
            )
            summary["is_escalation"] = False
            kept.append(summary)
            if enabled:
                LOGGER.info("[%s] new anomaly - alerting.", metric)
            continue

        previous_z = abs(current.last_alert_z) if current.last_alert_z is not None else 0.0
        if current_z - previous_z >= escalation_z_delta:
            state.upsert(
                MetricState(
                    metric=metric,
                    state=STATE_ACTIVE,
                    last_alert_z=current_z,
                    last_alert_at=summary["timestamp"],
                    last_value=anomaly.actual,
                )
            )
            summary["is_escalation"] = True
            kept.append(summary)
            LOGGER.info(
                "[%s] escalation: |z| %.2f -> %.2f (delta %.2f).",
                metric,
                previous_z,
                current_z,
                current_z - previous_z,
            )
            continue

        state.upsert(
            MetricState(
                metric=metric,
                state=STATE_ACTIVE,
                last_alert_z=current.last_alert_z,
                last_alert_at=current.last_alert_at,
                last_value=anomaly.actual,
            )
        )
        LOGGER.info(
            "[%s] repeated anomaly suppressed: |z| %.2f, previous alert |z| %.2f.",
            metric,
            current_z,
            previous_z,
        )

    return kept


def main() -> int:
    args = parse_args()

    try:
        config = load_config(Path(args.config))
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    setup_logging(
        config.get("logging", {}).get("level", "INFO"),
        config.get("logging", {}).get("format", "%(asctime)s | %(levelname)s | %(name)s | %(message)s"),
    )

    if load_dotenv is not None:
        env_path = Path(args.config).parent / ".env"
        if env_path.is_file():
            load_dotenv(env_path)
            LOGGER.info("Loaded environment from %s", env_path)

    if args.file:
        config["input_mode"] = "excel_file"
        config.setdefault("data", {})["file_path"] = args.file
    if args.folder:
        config["input_mode"] = "excel_folder"
        config.setdefault("data", {})["folder_path"] = args.folder

    try:
        df, timestamp_column = load_data(config)

        detection_cfg = config.get("detection", {})
        anomalies = detect_anomalies(
            df,
            timestamp_column=timestamp_column,
            window_size=int(detection_cfg.get("window_size", 30)),
            z_threshold=float(detection_cfg.get("z_threshold", 3.0)),
            detection_mode=detection_cfg.get("mode", "zscore"),
            seasonal_period=int(detection_cfg.get("seasonal_period", 7)),
            min_cycles=int(detection_cfg.get("min_cycles", 2)),
        )

        summaries = generate_summaries(anomalies)
        correlation_window = int(config.get("correlation_window_days", 0))
        attach_correlation_notes(
            summaries,
            find_co_occurrences(anomalies, window_days=correlation_window),
        )

        for summary in summaries:
            LOGGER.info(
                "%s @ %s (%s, z=%+.2f): actual=%.2f, baseline=%.2f",
                summary["metric"],
                summary["timestamp"],
                summary["direction"].upper(),
                summary["z_score"],
                summary["actual"],
                summary["rolling_mean"],
            )

        suppression_cfg = config.get("suppression", {})
        store = StateStore(suppression_cfg.get("state_db_path", "anomaly_state.db"))
        observed_metrics = {
            column
            for column in df.columns
            if column not in {timestamp_column, "__source_file"}
            and pd.api.types.is_numeric_dtype(df[column])
        }
        kept = apply_suppression(
            anomalies,
            summaries,
            store,
            bool(suppression_cfg.get("enabled", False)),
            observed_metrics,
            float(suppression_cfg.get("escalation_z_delta", 2.0)),
        )

        sent = send_alert(config.get("smtp", {}), kept)
        if not sent and kept:
            LOGGER.warning("Alert email was not sent; check SMTP configuration.")

        report_cfg = config.get("report", {})
        report_path = generate_report(
            df=df,
            timestamp_column=timestamp_column,
            summaries=kept,
            anomalies=[anomaly for anomaly, summary in zip(anomalies, summaries) if summary in kept],
            output_dir=report_cfg.get("output_dir", "./reports"),
            detection_mode=detection_cfg.get("mode", "zscore"),
        )
        if report_path:
            LOGGER.info("HTML report: %s", report_path)

    except (DataLoadError, ValueError, KeyError, TypeError) as exc:
        LOGGER.error("Run failed: %s", exc)
        return 2

    LOGGER.info("Run complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
