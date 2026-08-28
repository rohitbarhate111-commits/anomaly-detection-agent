"""
email_alerter.py

Composes and sends a single summary email when any anomalies are found.
SMTP credentials are pulled from environment variables (names from config),
never hardcoded. Send failures are logged but do not abort the run.

v2 additions:
    - Per-summary `is_escalation` flag. Escalations are listed before regular
      anomalies and clearly labeled in both the subject and body.
    - `correlation_note` is appended after the standard fields when present.
"""

import logging
import os
import smtplib
from datetime import datetime
from email.message import EmailMessage
from typing import List, Tuple

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
    def _required(name: str) -> str:
        val = os.environ.get(name)
        if not val:
            raise EnvironmentError(f"Required environment variable '{name}' is not set.")
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


def build_email_body(summaries: List[dict]) -> Tuple[str, str]:
    """
    Build (subject, body) for the alert email.

    Subject: [ANOMALY ALERT] {N} anomalies detected — {timestamp}
             [ANOMALY ESCALATION] {N} escalated anomaly/ies — {timestamp}
             if any item has is_escalation=True.

    Body: for each anomaly — metric, what changed, severity, summary,
          correlation note (if any).
    """
    n = len(summaries)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    n_esc = sum(1 for s in summaries if s.get("is_escalation"))
    n_new = n - n_esc

    if n_esc and not n_new:
        subject = f"[ANOMALY ESCALATION] {n_esc} escalated anomaly/ies - {now}"
    elif n_esc and n_new:
        subject = f"[ANOMALY ALERT] {n} anomalies detected ({n_esc} escalated) - {now}"
    else:
        subject = f"[ANOMALY ALERT] {n} anomalies detected - {now}"

    lines = [
        f"Anomaly Detection Agent - {n} alert item(s) at {now}",
        f"  New anomalies: {n_new}  |  Escalations: {n_esc}",
        "=" * 60,
        "",
    ]

    escalations = [s for s in summaries if s.get("is_escalation")]
    regulars    = [s for s in summaries if not s.get("is_escalation")]

    def _render_block(idx: int, s: dict) -> List[str]:
        if s.get("is_escalation"):
            header = (
                f"#{idx}. [ESCALATION] {s['metric']}  ({s['timestamp']})  -  "
                f"{s['direction'].upper()}"
            )
        else:
            header = f"#{idx}. {s['metric']}  ({s['timestamp']})  -  {s['direction'].upper()}"
        block = [
            header,
            f"    Severity : {s['severity']}  (z = {s['z_score']:+.2f})",
            f"    What changed: {s['what_changed']}",
            f"    Significance: {s['significance']}",
            f"    Possible impact: {s['possible_impact']}",
        ]
        if s.get("correlation_note"):
            block.append(f"    Correlation: {s['correlation_note']}")
        block.append("")
        return block

    idx = 1
    for s in escalations:
        lines.extend(_render_block(idx, s))
        idx += 1
    for s in regulars:
        lines.extend(_render_block(idx, s))
        idx += 1

    lines.append("This is an automated alert. Investigate each item above.")
    return subject, "\n".join(lines)


def send_alert(smtp_cfg: dict, summaries: List[dict]) -> bool:
    """
    Send a single email summarising all detected anomalies.
    Returns True on success. Failures are logged but never raised.
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
            f"Connecting to SMTP {cfg.server}:{cfg.port} as {cfg.username} "
            f"to deliver alert to {len(cfg.recipients)} recipient(s)."
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
    except Exception as exc:
        logger.error(f"Unexpected error while sending alert: {exc}")

    return False
