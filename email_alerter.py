"""
email_alerter.py

Composes and sends a single summary email when any anomalies are found.
SMTP credentials are pulled from environment variables (names from config),
never hardcoded. Send failures are logged but do not abort the run.
"""

import logging
import os
import smtplib
from datetime import datetime
from email.message import EmailMessage
from typing import List

logger = logging.getLogger(__name__)


class EmailConfig:
    """Resolved SMTP settings + recipient list. Built once from config."""
    def __init__(
        self,
        server: str,
        port: int,
        use_tls: bool,
        username: str,
        password: str,
        sender: str,
        recipients: List[str],
    ):
        self.server = server
        self.port = port
        self.use_tls = use_tls
        self.username = username
        self.password = password
        self.sender = sender
        self.recipients = recipients


def resolve_email_config(smtp_cfg: dict) -> EmailConfig:
    """
    Pull credentials and recipients from env vars based on the names in config.

    Raises:
        EnvironmentError: If required env vars are missing.
    """
    def _required(name: str) -> str:
        val = os.environ.get(name)
        if not val:
            raise EnvironmentError(
                f"Required environment variable '{name}' is not set."
            )
        return val

    recipients_raw = _required(smtp_cfg["recipients_env"])
    recipients = [r.strip() for r in recipients_raw.split(",") if r.strip()]

    if not recipients:
        raise EnvironmentError(
            f"Environment variable '{smtp_cfg['recipients_env']}' is empty."
        )

    return EmailConfig(
        server=smtp_cfg["server"],
        port=int(smtp_cfg["port"]),
        use_tls=bool(smtp_cfg.get("use_tls", True)),
        username=_required(smtp_cfg["username_env"]),
        password=_required(smtp_cfg["password_env"]),
        sender=_required(smtp_cfg["sender_env"]),
        recipients=recipients,
    )


def build_email_body(summaries: List[dict]) -> tuple[str, str]:
    """
    Build (subject, body) for the alert email.

    Subject: [ANOMALY ALERT] {N} anomalies detected — {timestamp}
    Body: one block per anomaly (metric, what changed, severity, summary).
    """
    n = len(summaries)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    subject = f"[ANOMALY ALERT] {n} anomalies detected - {now}"

    lines = [
        f"Anomaly Detection Agent - {n} anomalies flagged at {now}",
        "=" * 60,
        "",
    ]

    for i, s in enumerate(summaries, start=1):
        lines.extend([
            f"#{i}. {s['metric']}  ({s['timestamp']})  -  {s['direction'].upper()}",
            f"    Severity : {s['severity']}  (z = {s['z_score']:+.2f})",
            f"    What changed: {s['what_changed']}",
            f"    Significance: {s['significance']}",
            f"    Possible impact: {s['possible_impact']}",
            "",
        ])

    lines.append("This is an automated alert. Investigate each item above.")
    return subject, "\n".join(lines)


def send_alert(smtp_cfg: dict, summaries: List[dict]) -> bool:
    """
    Send a single email summarising all detected anomalies.

    Returns:
        True if the email was sent successfully, False otherwise.

    Email send failures are logged but do NOT raise - a failed email should
    not crash the whole detection run.
    """
    if not summaries:
        logger.info("No anomalies to report - skipping email.")
        return False

    try:
        cfg = resolve_email_config(smtp_cfg)
    except EnvironmentError as exc:
        logger.error(f"Email config not resolved: {exc}")
        return False

    subject, body = build_email_body(summaries)

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = cfg.sender
    msg["To"] = ", ".join(cfg.recipients)
    msg.set_content(body)

    try:
        logger.info(
            f"Connecting to SMTP {cfg.server}:{cfg.port} "
            f"as {cfg.username} to deliver alert to {len(cfg.recipients)} recipient(s)."
        )
        with smtplib.SMTP(cfg.server, cfg.port, timeout=30) as smtp:
            if cfg.use_tls:
                smtp.starttls()
            smtp.login(cfg.username, cfg.password)
            smtp.send_message(msg)
        logger.info("Alert email sent successfully.")
        return True
    except smtplib.SMTPException as exc:
        logger.error(f"SMTP error while sending alert: {exc}")
    except OSError as exc:
        logger.error(f"Network error while sending alert: {exc}")
    except Exception as exc:  # last-resort: keep detection run alive
        logger.error(f"Unexpected error while sending alert: {exc}")

    return False
