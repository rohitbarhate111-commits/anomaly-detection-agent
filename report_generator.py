"""Generate portable HTML reports for detected anomalies."""

import base64
import html
import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Anomaly Report - {run_id}</title>
<style>
  :root {{ --bg:#fff; --fg:#1f2937; --muted:#6b7280; --accent:#2563eb; --warn:#b91c1c; --border:#e5e7eb; --row-alt:#f9fafb; }}
  @media (prefers-color-scheme: dark) {{ :root {{ --bg:#111827; --fg:#f3f4f6; --muted:#9ca3af; --accent:#60a5fa; --warn:#f87171; --border:#374151; --row-alt:#1f2937; }} }}
  body {{ background:var(--bg); color:var(--fg); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif; max-width:960px; margin:24px auto; padding:0 16px; line-height:1.45; }}
  h1 {{ margin-bottom:.2em; }} .meta {{ color:var(--muted); margin-bottom:1.5em; }}
  table {{ width:100%; border-collapse:collapse; margin-bottom:2em; }} th,td {{ border-bottom:1px solid var(--border); padding:8px 10px; text-align:left; font-size:14px; }} th {{ background:var(--row-alt); }} tr:nth-child(even) td {{ background:var(--row-alt); }}
  .badge {{ display:inline-block; padding:2px 8px; border-radius:10px; font-size:12px; font-weight:600; }} .up {{ color:var(--warn); }} .down {{ color:var(--accent); }} .esc {{ background:var(--warn); color:#fff; }}
  .chart {{ margin:1.5em 0 2.5em; }} .chart img {{ max-width:100%; height:auto; border:1px solid var(--border); border-radius:6px; }}
  .summary {{ background:var(--row-alt); border:1px solid var(--border); border-radius:6px; padding:12px 16px; margin-bottom:1em; }} .summary h3 {{ margin:0 0 .4em; }} .summary p {{ margin:.3em 0; }} .corr {{ color:var(--warn); font-style:italic; }}
</style>
</head>
<body>
<h1>Anomaly Detection Report</h1>
<div class="meta"><strong>Run:</strong> {run_id} &nbsp;|&nbsp; <strong>Detection mode:</strong> {detection_mode} &nbsp;|&nbsp; <strong>Anomalies:</strong> {n_anomalies} &nbsp;|&nbsp; <strong>Generated:</strong> {generated_at}</div>
<h2>Summary</h2>{summary_table}
<h2>Trends</h2>{charts_html}
<h2>Business-context summaries</h2>{summaries_html}
</body>
</html>
"""


def generate_report(
    df: pd.DataFrame,
    timestamp_column: str,
    summaries: List[dict],
    anomalies: List,
    output_dir: str = "./reports",
    detection_mode: str = "zscore",
) -> Optional[Path]:
    """Write a self-contained HTML report and return its path."""
    if not anomalies:
        logger.info("No anomalies - skipping report generation.")
        return None

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"report_{run_id}.html"

    try:
        html_report = _HTML_TEMPLATE.format(
            run_id=html.escape(run_id),
            detection_mode=html.escape(str(detection_mode)),
            n_anomalies=len(summaries),
            generated_at=html.escape(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            summary_table=_render_summary_table(summaries),
            charts_html=_render_charts(df, timestamp_column, anomalies),
            summaries_html=_render_summaries(summaries),
        )
        out_path.write_text(html_report, encoding="utf-8")
    except (OSError, ValueError) as exc:
        logger.error("Failed to write HTML report: %s", exc)
        return None

    logger.info("Report written: %s", out_path.resolve())
    return out_path


def _render_summary_table(summaries: List[dict]) -> str:
    rows = []
    for summary in summaries:
        metric = html.escape(str(summary["metric"]))
        date = html.escape(str(summary["timestamp"]))
        severity = html.escape(str(summary["severity"]))
        direction = summary["direction"]
        direction_html = (
            f'<span class="up">▲ {html.escape(direction.upper())}</span>'
            if direction == "up"
            else f'<span class="down">▼ {html.escape(direction.upper())}</span>'
        )
        escalation = '<span class="badge esc">ESC</span> ' if summary.get("is_escalation") else ""
        rows.append(
            f"<tr><td>{escalation}{metric}</td><td>{date}</td><td>{direction_html}</td>"
            f"<td>{severity}</td><td>{summary['z_score']:+.2f}</td></tr>"
        )
    return (
        '<table><thead><tr><th>Metric</th><th>Date</th><th>Direction</th>'
        f"<th>Severity</th><th>Z-score</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def _render_summaries(summaries: List[dict]) -> str:
    blocks = []
    for summary in summaries:
        metric = html.escape(str(summary["metric"]))
        timestamp = html.escape(str(summary["timestamp"]))
        direction = html.escape(str(summary["direction"]).upper())
        direction_class = "up" if summary["direction"] == "up" else "down"
        arrow = "▲" if summary["direction"] == "up" else "▼"
        escalation = '<span class="badge esc">ESCALATION</span> ' if summary.get("is_escalation") else ""
        correlation = (
            f'<p class="corr">{html.escape(str(summary["correlation_note"]))}</p>'
            if summary.get("correlation_note") else ""
        )
        blocks.append(
            f'<div class="summary"><h3>{escalation}{metric} - {timestamp} '
            f'<span class="{direction_class}">({arrow} {direction})</span></h3>'
            f'<p><strong>What changed:</strong> {html.escape(str(summary["what_changed"]))}</p>'
            f'<p><strong>Significance:</strong> {html.escape(str(summary["significance"]))}</p>'
            f'<p><strong>Possible impact:</strong> {html.escape(str(summary["possible_impact"]))}</p>'
            f'{correlation}</div>'
        )
    return "\n".join(blocks)


def _render_charts(df: pd.DataFrame, timestamp_column: str, anomalies: List) -> str:
    """Render one inline chart per anomalous metric."""
    by_metric: Dict[str, List] = {}
    for anomaly in anomalies:
        by_metric.setdefault(anomaly.metric, []).append(anomaly)

    panels = []
    for metric, items in by_metric.items():
        try:
            encoded = _chart_png_base64(df, timestamp_column, metric, items)
        except (KeyError, ValueError, OSError) as exc:
            logger.warning("Skipping chart for %s: %s", metric, exc)
            continue
        safe_metric = html.escape(str(metric))
        panels.append(
            f'<div class="chart"><h3>{safe_metric}</h3>'
            f'<img src="data:image/png;base64,{encoded}" alt="{safe_metric} chart"></div>'
        )
    return "\n".join(panels) if panels else "<p>No charts to render.</p>"


def _chart_png_base64(df: pd.DataFrame, timestamp_column: str, metric: str, anomalies: List) -> str:
    """Render a chart to an in-memory PNG and return base64 text."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if timestamp_column not in df.columns or metric not in df.columns:
        raise KeyError(f"Missing chart column for '{metric}'.")
    series = df[[timestamp_column, metric]].dropna()
    if series.empty:
        raise ValueError(f"No data for metric '{metric}'.")

    anomaly_ts = {pd.Timestamp(anomaly.timestamp) for anomaly in anomalies}
    fig, ax = plt.subplots(figsize=(10, 3.5), dpi=100)
    ax.plot(series[timestamp_column], series[metric], linewidth=1.4, color="#2563eb")
    if anomaly_ts:
        mask = series[timestamp_column].isin(anomaly_ts)
        ax.scatter(series.loc[mask, timestamp_column], series.loc[mask, metric], color="#b91c1c", s=42, zorder=5, label="anomaly")
        ax.legend(loc="best", fontsize=9)
    ax.set_title(str(metric))
    ax.set_xlabel("Date")
    ax.set_ylabel(str(metric))
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png")
    finally:
        plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")
