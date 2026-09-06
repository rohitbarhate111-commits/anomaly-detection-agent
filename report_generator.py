"""
report_generator.py

Generates standalone, self-contained HTML reports with inline base64 charts.
Zero external asset requests required.
"""

from __future__ import annotations

import base64
import io
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Anomaly Detection Report &mdash; {run_id}</title>
  <style>
    :root {{
      --bg: #0f172a;
      --card: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --red: #f87171;
      --green: #4ade80;
      --amber: #fbbf24;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      margin: 0;
      padding: 32px 24px;
    }}
    .container {{ max-width: 1100px; margin: 0 auto; }}
    header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--border);
      padding-bottom: 20px;
      margin-bottom: 28px;
    }}
    h1 {{ margin: 0; font-size: 24px; font-weight: 700; }}
    .badge {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
    }}
    .badge.esc {{ background: rgba(239,68,68,0.2); color: var(--red); border: 1px solid var(--red); }}
    .badge.norm {{ background: rgba(56,189,248,0.2); color: var(--accent); border: 1px solid var(--accent); }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 28px; }}
    .kpi-card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 16px;
    }}
    .kpi-val {{ font-size: 28px; font-weight: 700; color: var(--accent); }}
    .kpi-lbl {{ font-size: 12px; color: var(--muted); text-transform: uppercase; margin-top: 4px; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-bottom: 32px;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
    }}
    th, td {{ padding: 12px 16px; text-align: left; font-size: 13px; border-bottom: 1px solid var(--border); }}
    th {{ background: #131d31; color: var(--muted); font-weight: 600; text-transform: uppercase; font-size: 11px; }}
    tr:last-child td {{ border-bottom: none; }}
    .summary-card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 18px;
      margin-bottom: 16px;
    }}
    .summary-title {{ font-size: 16px; font-weight: 600; margin-bottom: 8px; display: flex; align-items: center; gap: 8px; }}
    .chart-container {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 24px;
      text-align: center;
    }}
    .chart-container img {{ max-width: 100%; height: auto; border-radius: 6px; }}
    .corr-note {{
      background: rgba(251,191,36,0.1);
      border-left: 3px solid var(--amber);
      color: #fde68a;
      padding: 8px 12px;
      font-size: 12px;
      margin-top: 10px;
      border-radius: 4px;
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>AI Anomaly Detection Report</h1>
        <div style="color: var(--muted); font-size: 13px; margin-top: 4px;">Run ID: {run_id} &bull; Generated {generated_at}</div>
      </div>
      <span class="badge norm">Engine: {detection_mode}</span>
    </header>

    <div class="grid">
      <div class="kpi-card">
        <div class="kpi-val">{n_anomalies}</div>
        <div class="kpi-lbl">Anomalies Detected</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-val" style="color: var(--red);">{n_escalations}</div>
        <div class="kpi-lbl">Escalations</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-val" style="color: var(--green);">{detection_mode}</div>
        <div class="kpi-lbl">Detection Algorithm</div>
      </div>
    </div>

    <h2>Flagged Events</h2>
    {summary_table}

    <h2>Time-Series Charts</h2>
    {charts_html}

    <h2>Detailed Incident Summaries</h2>
    {summaries_html}
  </div>
</body>
</html>
"""


def generate_report(
    df: pd.DataFrame,
    timestamp_column: str,
    summaries: Sequence[dict],
    anomalies: Sequence[object],
    output_dir: str | Path = "./reports",
    detection_mode: str = "zscore",
    base_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Renders self-contained HTML report and saves to output_dir."""
    if not summaries:
        logger.info("No anomalies to report; skipping HTML report generation.")
        return None

    out_dir = Path(output_dir)
    if not out_dir.is_absolute() and base_dir:
        out_dir = (base_dir / out_dir).resolve()
    else:
        out_dir = out_dir.resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"report_{run_id}.html"

    try:
        summary_table = _render_summary_table(summaries)
        charts_html = _render_charts(df, timestamp_column, anomalies)
        summaries_html = _render_summaries(summaries)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        n_esc = sum(1 for s in summaries if s.get("is_escalation"))

        html = _HTML_TEMPLATE.format(
            run_id=run_id,
            generated_at=now_str,
            detection_mode=detection_mode,
            n_anomalies=len(summaries),
            n_escalations=n_esc,
            summary_table=summary_table,
            charts_html=charts_html,
            summaries_html=summaries_html,
        )

        out_path.write_text(html, encoding="utf-8")
        logger.info(f"HTML report successfully written: {out_path}")
        return out_path
    except Exception as exc:
        logger.error(f"Failed generating HTML report: {exc}")
        return None


def _render_summary_table(summaries: Sequence[dict]) -> str:
    rows = []
    for s in summaries:
        esc_badge = '<span class="badge esc">ESCALATION</span> ' if s.get("is_escalation") else ''
        dir_color = "#4ade80" if s.get("direction") == "up" else "#f87171"
        rows.append(
            f"<tr>"
            f"<td>{esc_badge}<b>{s['metric']}</b></td>"
            f"<td>{s['timestamp']}</td>"
            f"<td style='color:{dir_color}; font-weight:600;'>{s['direction'].upper()}</td>"
            f"<td>{s['severity']}</td>"
            f"<td><b>{s['z_score']:+.2f}</b></td>"
            f"</tr>"
        )
    return (
        "<table>"
        "<thead><tr><th>Metric</th><th>Date</th><th>Direction</th><th>Severity</th><th>Z-Score</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def _render_summaries(summaries: Sequence[dict]) -> str:
    cards = []
    for s in summaries:
        esc_badge = '<span class="badge esc">ESCALATION</span> ' if s.get("is_escalation") else ''
        corr_html = f'<div class="corr-note">{s["correlation_note"]}</div>' if s.get("correlation_note") else ""
        cards.append(
            f'<div class="summary-card">'
            f'<div class="summary-title">{esc_badge}{s["metric"]} &mdash; {s["timestamp"]}</div>'
            f'<p><strong>Observation:</strong> {s["what_changed"]}</p>'
            f'<p><strong>Significance:</strong> {s["significance"]}</p>'
            f'<p><strong>Contextual Impact:</strong> {s["possible_impact"]}</p>'
            f'{corr_html}'
            f'</div>'
        )
    return "\n".join(cards)


def _render_charts(df: pd.DataFrame, timestamp_column: str, anomalies: Sequence[object]) -> str:
    by_metric: dict[str, list] = {}
    for a in anomalies:
        metric = getattr(a, "metric", None) or (a.get("metric") if isinstance(a, dict) else None)
        if metric:
            by_metric.setdefault(metric, []).append(a)

    charts = []
    for metric, items in by_metric.items():
        try:
            b64 = _chart_png_base64(df, timestamp_column, metric, items)
            charts.append(
                f'<div class="chart-container">'
                f'<h3 style="margin-top:0;">{metric}</h3>'
                f'<img src="data:image/png;base64,{b64}" alt="{metric} chart"/>'
                f'</div>'
            )
        except Exception as exc:
            logger.warning(f"Skipping chart rendering for {metric}: {exc}")

    return "\n".join(charts) if charts else "<p style='color:var(--muted);'>No charts generated.</p>"


def _chart_png_base64(df: pd.DataFrame, timestamp_column: str, metric: str, anomalies: Sequence[object]) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = df[[timestamp_column, metric]].dropna().sort_values(timestamp_column)
    if series.empty:
        raise ValueError(f"No valid data points for metric '{metric}'.")

    anomaly_ts = {
        pd.Timestamp(getattr(a, "timestamp", None) or a.get("timestamp"))
        for a in anomalies
    }

    fig, ax = plt.subplots(figsize=(10, 3.2), dpi=100)
    fig.patch.set_facecolor("#1e293b")
    ax.set_facecolor("#0f172a")

    ax.plot(series[timestamp_column], series[metric], color="#38bdf8", linewidth=1.5, label="Observed Metric")

    mask = series[timestamp_column].isin(anomaly_ts)
    if mask.any():
        ax.scatter(
            series.loc[mask, timestamp_column],
            series.loc[mask, metric],
            color="#ef4444", s=50, zorder=5, label="Anomaly Event"
        )

    ax.tick_params(colors="#94a3b8")
    ax.spines["bottom"].set_color("#334155")
    ax.spines["top"].set_color("#334155")
    ax.spines["left"].set_color("#334155")
    ax.spines["right"].set_color("#334155")
    ax.grid(True, color="#334155", alpha=0.4, linestyle="--")

    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc", fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")
