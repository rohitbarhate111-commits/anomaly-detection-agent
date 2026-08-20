"""
main.py

Orchestrates the full anomaly-detection run:

  1. Load config (config.yaml) and optional .env.
  2. Load Excel file (CLI arg overrides config).
  3. Detect anomalies.
  4. Generate business-context summaries.
  5. Send a single alert email if any anomalies were found.

Run:
    python main.py                       # uses config.yaml data.file_path
    python main.py --file path/to.xlsx   # override file path
"""

import argparse
import logging
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # dotenv is optional but convenient
    load_dotenv = None

import yaml

from data_loader import DataLoadError, load_excel
from anomaly_detector import detect_anomalies
from summary_generator import generate_summaries
from email_alerter import send_alert


CONFIG_PATH = Path(__file__).parent / "config.yaml"


def setup_logging(level: str, fmt: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
    )
    # Quiet down noisy libraries.
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("openpyxl").setLevel(logging.WARNING)


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI Anomaly Detection Agent - scans an Excel file for outliers.",
    )
    parser.add_argument(
        "--file",
        help="Path to the Excel file. Overrides config.yaml data.file_path.",
        default=None,
    )
    parser.add_argument(
        "--config",
        help="Path to a custom config.yaml.",
        default=str(CONFIG_PATH),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(Path(args.config))

    setup_logging(
        config.get("logging", {}).get("level", "INFO"),
        config.get("logging", {}).get("format", "%(levelname)s | %(message)s"),
    )
    logger = logging.getLogger("anomaly-agent")

    # Load .env if present (best-effort - missing is fine).
    if load_dotenv is not None:
        env_path = Path(args.config).parent / ".env"
        if env_path.is_file():
            load_dotenv(env_path)
            logger.info(f"Loaded environment from {env_path}")
    else:
        logger.debug("python-dotenv not installed; skipping .env loading.")

    # Resolve input file path.
    file_path = args.file or config["data"]["file_path"]

    # 1. Load.
    try:
        df = load_excel(
            file_path,
            timestamp_column=config["data"].get("timestamp_column"),
        )
    except DataLoadError as exc:
        logger.error(f"Data load failed: {exc}")
        return 2

    # 2. Detect.
    try:
        anomalies = detect_anomalies(
            df,
            timestamp_column=config["data"]["timestamp_column"],
            window_size=int(config["detection"]["window_size"]),
            z_threshold=float(config["detection"]["z_threshold"]),
        )
    except (ValueError, KeyError) as exc:
        logger.error(f"Detection config invalid: {exc}")
        return 2

    # 3. Summarize.
    summaries = generate_summaries(anomalies)

    for s in summaries:
        logger.info(
            f"  - {s['metric']} @ {s['timestamp']} "
            f"({s['direction'].upper()}, z={s['z_score']:+.2f}): "
            f"actual={s['actual']:.2f}, baseline={s['rolling_mean']:.2f}"
        )

    # 4. Alert (best-effort).
    sent = send_alert(config["smtp"], summaries)
    if not sent and summaries:
        logger.warning("Alert email was not sent; check SMTP config and env vars.")

    logger.info("Run complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
