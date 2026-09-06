"""
email_alerter.py

SMTP email alerting system.
Transmits a single consolidated alert per run with escalations highlighted first.
Fails gracefully without interrupting the detection pipeline if SMTP is unreachable.
"""

from __future__ import annotations

import logging
import os
import smtplib
from dataclasses import dataclass
from datetime import datetime
from email.message import EmailMessage
from typing import List, Sequence, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmailConfig:
    server: str
    port: int
    use_tls: bool
    username: str
    password: str
    sender: str
    recipients: List[str]


def resolve_email_config(smtp_cfg: dict) -> EmailConfig:
    """Resolves email credentials from environment variables."""
    user_env = smtp_cfg.get("username_env", "SMTP_USERNAME")
    pass_env = smtp_cfg.get("password_env", "SMTP_PASSWORD")
    sender_env = smtp_cfg.get("sender_env", "ALERT_SENDER")
    recip_env = smtp_cfg.get("recipients_env", "ALERT_RECIPIENTS")

    username = os.environ.get(user_env, "")
    password = os.environ.get(pass_env, "")
    sender = os.environ.get(sender_env, "")
    recipients_raw = os.environ.get(recip_env, "")

    missing = []
    if not username: missing.append(user_env)
    if not password: missing.append(pass_env)
    if not sender: missing.append(sender_env)
    if not recipients_raw: missing.append(recip_env)

    if missing:
        raise EnvironmentError(f"Missing required email environment variable(s): {', '.join(missing)}")

    recipients = [r.strip() for r in recipients_raw.split(",") if r.strip()]

    return EmailConfig(
        server=smtp_cfg.get("server", "smtp.gmail.com"),
        port=int(smtp_cfg.get("port", 587)),
        use_tls=bool(smtp_cfg.get("use_tls", True)),
        username=username,
        password=password,
        sender=sender,
        recipients=recipients,
    )


def build_email_content(summaries: Sequence[dict]) -> Tuple[str, str, str]:
    """
    Builds subject, plain-text body, and HTML body for alert emails.
    """
    n = len(summaries)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    escalations = [s for s in summaries if s.get("is_escalation")]
    regulars = [s for s in summaries if not s.get("is_escalation")]

    n_esc = len(escalations)
    n_reg = len(regulars)

    if n_esc and not n_reg:
        subject = f"[ANOMALY ESCALATION] {n_esc} escalated anomaly event(s) - {now_str}"
    elif n_esc:
        subject = f"[ANOMALY ALERT] {n} anomalies ({n_esc} escalated) - {now_str}"
    else:
        subject = f"[ANOMALY ALERT] {n} anomalies detected - {now_str}"

    text_lines = [
        f"AI Anomaly Detection Agent - Run Summary",
        f"Timestamp: {now_str} | Total: {n} (New: {n_reg}, Escalations: {n_esc})",
        "=" * 64,
        "",
    ]

    for idx, s in enumerate(escalations + regulars, start=1):
        tag = "[ESCALATION] " if s.get("is_escalation") else ""
        text_lines.append(f"#{idx}. {tag}{s['metric']} ({s['timestamp']}) - {s['direction'].upper()}")
        text_lines.append(f"    Severity : {s['severity']} (z = {s['z_score']:+.2f})")
        text_lines.append(f"    Change   : {s['what_changed']}")
        text_lines.append(f"    Impact   : {s['possible_impact']}")
        if s.get("correlation_note"):
            text_lines.append(f"    Co-occur : {s['correlation_note']}")
        text_lines.append("")

    text_lines.append("Investigate via the Anomaly Detector dashboard or generated HTML report.")
    plain_text = "\n".join(text_lines)

    html_parts = [
        f"<div style='font-family: sans-serif; max-width: 680px; margin: 0 auto; color: #1e293b;'>",
        f"<h2 style='color: #0f172a; margin-bottom: 4px;'>AI Anomaly Detection Alert</h2>",
        f"<p style='color: #64748b; font-size: 13px;'>Run completed at {now_str} | Total: <b>{n}</b> | Escalations: <b>{n_esc}</b></p>",
        f"<hr style='border: none; border-top: 1px solid #e2e8f0; margin: 16px 0;'/>",
    ]

    for idx, s in enumerate(escalations + regulars, start=1):
        is_esc = s.get("is_escalation")
        bg = "#fef2f2" if is_esc else "#f8fafc"
        border = "#ef4444" if is_esc else "#cbd5e1"
        badge = "<span style='background:#ef4444;color:#fff;padding:2px 6px;border-radius:4px;font-size:11px;'>ESCALATION</span> " if is_esc else ""
        html_parts.append(
            f"<div style='background:{bg}; border-left: 4px solid {border}; padding: 12px 16px; margin-bottom: 12px; border-radius: 4px;'>"
            f"<div style='font-weight: 600; font-size: 15px;'>#{idx}. {badge}{s['metric']} &mdash; {s['timestamp']} ({s['direction'].upper()})</div>"
            f"<div style='font-size: 13px; color: #475569; margin-top: 4px;'><b>Z-Score:</b> {s['z_score']:+.2f} | <b>Severity:</b> {s['severity']}</div>"
            f"<div style='font-size: 13px; color: #334155; margin-top: 6px;'>{s['what_changed']}</div>"
            f"<div style='font-size: 12px; color: #64748b; margin-top: 4px;'>{s['possible_impact']}</div>"
            + (f"<div style='font-size: 12px; color: #b45309; margin-top: 6px;'>{s['correlation_note']}</div>" if s.get("correlation_note") else "")
            + f"</div>"
        )

    html_parts.append("</div>")
    html_body = "".join(html_parts)

    return subject, plain_text, html_body


def send_alert(smtp_cfg: dict, summaries: Sequence[dict]) -> bool:
    """
    Transmits alert email. Returns True on success, False on failure.
    Never throws unhandled exceptions.
    """
    if not summaries:
        logger.info("No anomalies to report - skipping email.")
        return False

    try:
        cfg = resolve_email_config(smtp_cfg)
    except EnvironmentError as exc:
        logger.warning(f"Email alerting skipped: {exc}")
        return False

    subject, plain_body, html_body = build_email_content(summaries)

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = cfg.sender
    msg["To"] = ", ".join(cfg.recipients)
    msg.set_content(plain_body)
    msg.add_alternative(html_body, subtype="html")

    try:
        logger.info(f"Delivering alert via SMTP {cfg.server}:{cfg.port} to {len(cfg.recipients)} recipient(s)...")
        with smtplib.SMTP(cfg.server, cfg.port, timeout=20) as smtp:
            if cfg.use_tls:
                smtp.starttls()
            smtp.login(cfg.username, cfg.password)
            smtp.send_message(msg)
        logger.info("Alert email successfully dispatched.")
        return True
    except Exception as exc:
        logger.warning(f"SMTP dispatch failed ({exc}); pipeline will continue.")
        return False
