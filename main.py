"""
main.py

Orchestrator and CLI entry point for the AI Anomaly Detection Agent.
Supports execution via command line or programmatically via run_pipeline().
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from anomaly_detector import detect_anomalies
from correlation import attach_correlation_notes, find_co_occurrences
from data_loader import DataLoadError, load_data, resolve_path
from email_alerter import send_alert
from report_generator import generate_report
from state_store import STATE_ACTIVE, STATE_NORMAL, MetricState, StateStore
from summary_generator import generate_summaries

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
        return yaml.safe_load(fh) or {}


def apply_suppression(
    anomalies: list,
    summaries: list,
    state: StateStore,
    enabled: bool,
    observed_metrics: set,
    escalation_z_delta: float = 2.0,
) -> list:
    """
    Applies per-metric state transitions and alert suppression logic.

    Rules:
      1. Metric returns in-range: reset state to STATE_NORMAL.
      2. First anomaly for metric: alert emitted, state becomes STATE_ACTIVE.
      3. Active anomaly: suppressed unless |z_current| - |z_previous| >= escalation_z_delta.
      4. Suppression disabled: all alerts emitted.
    """
    anomalous_metrics = {a.metric for a in anomalies}
    for metric in observed_metrics:
        if metric not in anomalous_metrics:
            ms = state.get(metric)
            if ms.state == STATE_ACTIVE:
                state.reset(metric)

    kept = []
    logger = logging.getLogger("anomaly-agent")

    for a, s in zip(anomalies, summaries):
        metric = a.metric
        curr_z = abs(a.z_score)
        ms = state.get(metric)

        if not enabled:
            state.upsert(MetricState(metric, STATE_ACTIVE, curr_z, s["timestamp"], a.actual))
            s["is_escalation"] = False
            kept.append(s)
            continue

        if ms.state == STATE_NORMAL:
            state.upsert(MetricState(metric, STATE_ACTIVE, curr_z, s["timestamp"], a.actual))
            s["is_escalation"] = False
            kept.append(s)
            logger.info(f"[{metric}] NEW anomaly detected -> alert emitted (state: active_anomaly).")
            continue

        # STATE_ACTIVE -> Check escalation
        prev_z = abs(ms.last_alert_z) if ms.last_alert_z is not None else 0.0
        if (curr_z - prev_z) >= escalation_z_delta:
            state.upsert(MetricState(metric, STATE_ACTIVE, curr_z, s["timestamp"], a.actual))
            s["is_escalation"] = True
            kept.append(s)
            logger.info(
                f"[{metric}] ESCALATION: |z| jumped {prev_z:.2f} -> {curr_z:.2f} "
                f"(delta {curr_z - prev_z:.2f} >= {escalation_z_delta})."
            )
        else:
            logger.info(
                f"[{metric}] SUPPRESSED: anomaly ongoing, |z|={curr_z:.2f} "
                f"below escalation delta (+{escalation_z_delta}) vs previous |z|={prev_z:.2f}."
            )
            state.upsert(MetricState(metric, STATE_ACTIVE, prev_z, ms.last_alert_at, a.actual))

    return kept


def run_pipeline(
    config: dict,
    file_override: Optional[str] = None,
    folder_override: Optional[str] = None,
    export_json_path: Optional[str] = None,
    base_dir: Optional[Path] = None,
) -> dict:
    """
    Executes the end-to-end anomaly detection pipeline programmatically.
    Returns structured results dictionary.
    """
    logger = logging.getLogger("anomaly-agent")
    base_dir = base_dir or Path(__file__).parent

    # Apply path overrides
    if file_override:
        config.setdefault("data", {})["file_path"] = file_override
        config["input_mode"] = "excel_file"
    if folder_override:
        config.setdefault("data", {})["folder_path"] = folder_override
        config["input_mode"] = "excel_folder"

    # 1. Load Data
    df, ts_col = load_data(config, base_dir=base_dir)

    # 2. Detect Anomalies
    detection_cfg = config.get("detection", {})
    detection_mode = detection_cfg.get("mode", "seasonal")
    window_size = int(detection_cfg.get("window_size", 30))
    z_thresh = float(detection_cfg.get("z_threshold", 3.0))
    seasonal_period = int(detection_cfg.get("seasonal_period", 7))
    min_cycles = int(detection_cfg.get("min_cycles", 2))

    anomalies = detect_anomalies(
        df=df,
        timestamp_column=ts_col,
        window_size=window_size,
        z_threshold=z_thresh,
        detection_mode=detection_mode,
        seasonal_period=seasonal_period,
        min_cycles=min_cycles,
    )

    # 3. Summaries
    summaries = generate_summaries(anomalies)

    # 4. Correlation
    corr_window = int(config.get("correlation_window_days", 0))
    co_occurrences = find_co_occurrences(anomalies, window_days=corr_window)
    attach_correlation_notes(summaries, co_occurrences)

    # 5. Suppression
    supp_cfg = config.get("suppression", {})
    state_db_path = resolve_path(supp_cfg.get("state_db_path", "anomaly_state.db"), base_dir=base_dir)
    store = StateStore(db_path=str(state_db_path))

    observed_metrics = {
        c for c in df.columns
        if c != ts_col and c != "__source_file" and pd.api.types.is_numeric_dtype(df[c])
    }

    kept_summaries = apply_suppression(
        anomalies=anomalies,
        summaries=summaries,
        state=store,
        enabled=bool(supp_cfg.get("enabled", True)),
        observed_metrics=observed_metrics,
        escalation_z_delta=float(supp_cfg.get("escalation_z_delta", 2.0)),
    )

    kept_anomalies = [a for a, s in zip(anomalies, summaries) if s in kept_summaries]

    # 6. Email Alert
    smtp_cfg = config.get("smtp", {})
    email_sent = send_alert(smtp_cfg, kept_summaries)

    # 7. HTML Report
    report_cfg = config.get("report", {})
    output_dir = report_cfg.get("output_dir", "./reports")
    report_path = generate_report(
        df=df,
        timestamp_column=ts_col,
        summaries=kept_summaries,
        anomalies=kept_anomalies,
        output_dir=output_dir,
        detection_mode=detection_mode,
        base_dir=base_dir,
    )

    result = {
        "status": "success",
        "detection_mode": detection_mode,
        "total_metrics": len(observed_metrics),
        "total_anomalies": len(anomalies),
        "emitted_anomalies": len(kept_summaries),
        "suppressed_anomalies": len(anomalies) - len(kept_summaries),
        "escalations": sum(1 for s in kept_summaries if s.get("is_escalation")),
        "email_sent": email_sent,
        "report_path": str(report_path) if report_path else None,
        "summaries": kept_summaries,
        "all_anomalies": [a.to_dict() for a in anomalies],
    }

    if export_json_path:
        out_json = resolve_path(export_json_path, base_dir=base_dir)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        logger.info(f"Exported scan results to {out_json}")

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI Anomaly Detection Agent - scans time-series metrics for outliers.",
    )
    parser.add_argument("--file", help="Override file input path.", default=None)
    parser.add_argument("--folder", help="Override folder input path.", default=None)
    parser.add_argument("--config", default=str(CONFIG_PATH), help="Path to config.yaml.")
    parser.add_argument("--export-json", default=None, help="Save structured scan results to JSON file.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_file = Path(args.config)
    config = load_config(config_file)

    log_cfg = config.get("logging", {})
    setup_logging(
        level=log_cfg.get("level", "INFO"),
        fmt=log_cfg.get("format", "%(asctime)s | %(levelname)s | %(name)s | %(message)s"),
    )
    logger = logging.getLogger("anomaly-agent")

    if load_dotenv is not None:
        env_path = config_file.parent / ".env"
        if env_path.is_file():
            load_dotenv(env_path)
            logger.info(f"Loaded credentials from {env_path}")

    try:
        res = run_pipeline(
            config=config,
            file_override=args.file,
            folder_override=args.folder,
            export_json_path=args.export_json,
            base_dir=config_file.parent,
        )
        logger.info(
            f"Run complete: {res['total_anomalies']} anomalies found "
            f"({res['emitted_anomalies']} emitted, {res['suppressed_anomalies']} suppressed, "
            f"{res['escalations']} escalations)."
        )
        return 0
    except DataLoadError as exc:
        logger.error(f"Data loading failed: {exc}")
        return 2
    except Exception as exc:
        logger.exception(f"Unexpected pipeline failure: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
