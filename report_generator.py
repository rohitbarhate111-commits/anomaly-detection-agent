"""
report_generator.py

Generates a single self-contained HTML report per run.

Contents:
  - Run metadata (timestamp, detection mode, anomaly count).
  - Summary table of all anomalies (metric, direction, severity, timestamp, z).
  - One matplotlib line chart per anomalous metric, embedded as base64 PNG so
    the report has zero external dependencies at view time.
  - Full business-context summaries, same content as the email.

Output path: configured `report_output_dir` (default ./reports).
File name: report_{YYYYMMDD_HHMMSS}.html

Charts use matplotlib (Agg backend) so no display is required. The PNGs are
inlined as <img src="data:image/png;base64,..."> so the HTML is portable.
"""

import base64
import io
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# Minimum CSS reset + layout, inlined so the file works offline.
_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Anomaly Report - {run_id}</title>
<style>
  :root {{
    --bg: #ffffff;
    --fg: #1f2937;
    --muted: #6b7280;
    --accent: #2563eb;
    --warn: #b91c1c;
    --border: #e5e7eb;
    --row-alt: #f9fafb;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #111827;
      --fg: #f3f4f6;
      --muted: #9ca3af;
      --accent: #60a5fa;
      --warn: #f87171;
      --border: #374151;
      --row-alt: #1f2937;
    }}
  }}
  body {{
    background: var(--bg);
    color: var(--fg);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    max-width: 960px;
    margin: 24px auto;
    padding: 0 16px;
    line-height: 1.45;
  }}
  h1 {{ margin-bottom: 0.2em; }}
  .meta {{ color: var(--muted); margin-bottom: 1.5em; }}
  table {{ width: 100%; border-collapse: collapse; margin-bottom: 2em; }}
  th, td {{
    border-bottom: 1px solid var(--border);
    padding: 8px 10px;
    text-align: left;
    font-size: 14px;
  }}
  th {{ background: var(--row-alt); }}
  tr:nth-child(even) td {{ background: var(--row-alt); }}
  .badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 12px;
    font-weight: 600;
  }}
  .up    {{ color: var(--warn); }}
  .down  {{ color: var(--accent); }}
  .esc   {{ background: var(--warn); color: #fff; }}
  .chart {{ margin: 1.5em 0 2.5em 0; }}
  .chart img {{ max-width: 100%; height: auto; border: 1px solid var(--border); border-radius: 6px; }}
  .summary {{ background: var(--row-alt); border: 1px solid var(--border); border-radius: 6px;
              padding: 12px 16px; margin-bottom: 1em; }}
  .summary h3 {{ margin: 0 0 0.4em 0; }}
  .summary p  {{ margin: 0.3em 0; }}
  .corr {{ color: var(--warn); font-style: italic; }}
</style>
</head>
<body>
<h1>Anomaly Detection Report</h1>
<div class="meta">
  <strong>Run:</strong> {run_id} &nbsp;|&nbsp;
  <strong>Detection mode:</strong> {detection_mode} &nbsp;|&nbsp;
  <strong>Anomalies:</strong> {n_anomalies} &nbsp;|&nbsp;
  <strong>Generated:</strong> {generated_at}
</div>

<h2>Summary</h2>
{summary_table}

<h2>Trends</h2>
{charts_html}

<h2>Business-context summaries</h2>
{summaries_html}

</body>
</html>
"""


def generate_report(
    df: pd.DataFrame,
    timestamp_column: str,
    summaries: List[dict],
    anomalies: list,                       # List[Anomaly]
    output_dir: str = "./reports",
    detection_mode: str = "zscore",
) -> Optional[Path]:
    """
    Render the HTML report and write it to disk.

    Returns the Path of the written file, or None on failure (logged).
    """
    if not anomalies:
        logger.info("No anomalies - skipping report generation.")
        return None

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"report_{run_id}.html"

    try:
        summary_table = _render_summary_table(summaries)
        charts_html = _render_charts(df, timestamp_column, anomalies)
        summaries_html = _render_summaries(summaries)
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html = _HTML_TEMPLATE.format(
            run_id=run_id,
            detection_mode=detection_mode,
            n_anomalies=len(summaries),
            generated_at=generated_at,
            summary_table=summary_table,
            charts_html=charts_html,
            summaries_html=summaries_html,
        )

        out_path.write_text(html, encoding="utf-8")
        logger.info(f"Report written: {out_path.resolve()}")
        return out_path
    except Exception as exc:
        logger.error(f"Failed to write HTML report: {exc}")
        return None


# ---------- section renderers ----------

def _render_summary_table(summaries: List[dict]) -> str:
    rows = []
    for s in summaries:
        esc = '<span class="badge esc">ESC</span> ' if s.get("is_escalation") else ""
        direction = (
            f'<span class="up">▲ {s["direction"].upper()}</span>'
            if s["direction"] == "up"
            else f'<span class="down">▼ {s["direction"].upper()}</span>'
        )
        rows.append(
            f"<tr>"
            f"<td>{esc}{s['metric']}</td>"
            f"<td>{s['timestamp']}</td>"
            f"<td>{direction}</td>"
            f"<td>{s['severity']}</td>"
            f"<td>{s['z_score']:+.2f}</td>"
            f"</tr>"
        )
    return (
        "<table>"
        "<thead><tr><th>Metric</th><th>Date</th><th>Direction</th>"
        "<th>Severity</th><th>Z-score</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def _render_summaries(summaries: List[dict]) -> str:
    blocks = []
    for s in summaries:
        esc = '<span class="badge esc">ESCALATION</span> ' if s.get("is_escalation") else ""
        corr_html = (
            f'<p class="corr">{s["correlation_note"]}</p>'
            if s.get("correlation_note") else ""
        )
        blocks.append(
            f'<div class="summary">'
            f'<h3>{esc}{s["metric"]} - {s["timestamp"]} '
            f'<span class="{"up" if s["direction"]=="up" else "down"}">'
            f'({"▲" if s["direction"]=="up" else "▼"} {s["direction"].upper()})</span></h3>'
            f'<p><strong>What changed:</strong> {s["what_changed"]}</p>'
            f'<p><strong>Significance:</strong> {s["significance"]}</p>'
            f'<p><strong>Possible impact:</strong> {s["possible_impact"]}</p>'
            f'{corr_html}'
            f'</div>'
        )
    return "\n".join(blocks)


def _render_charts(
    df: pd.DataFrame,
    timestamp_column: str,
    anomalies: list,
) -> str:
    """One inline base64 PNG chart per metric that had anomalies."""
    # Group anomalies by metric.
    by_metric: dict[str, list] = {}
    for a in anomalies:
        by_metric.setdefault(a.metric, []).append(a)

    panels = []
    for metric, items in by_metric.items():
        try:
            b64 = _chart_png_base64(df, timestamp_column, metric, items)
            panels.append(
                f'<div class="chart">'
                f'<h3>{metric}</h3>'
                f'<img src="data:image/png;base64,{b64}" alt="{metric} chart">'
                f'</div>'
            )
        except Exception as exc:
            logger.warning(f"Skipping chart for {metric}: {exc}")

    return "\n".join(panels) if panels else "<p>No charts to render.</p>"


def _chart_png_base64(
    df: pd.DataFrame,
    timestamp_column: str,
    metric: str,
    anomalies: list,
) -> str:
    """Render a single line chart to an in-memory PNG and return base64 string."""
    import matplotlib
    matplotlib.use("Agg")  # no display
    import matplotlib.pyplot as plt

    series = df[[timestamp_column, metric]].dropna()
    if series.empty:
        raise ValueError(f"No data for metric '{metric}'.")

    # Build a set of anomaly timestamps for marker overlay.
    anomaly_ts = {pd.Timestamp(a.timestamp) for a in anomalies}

    fig, ax = plt.subplots(figsize=(10, 3.5), dpi=100)
    ax.plot(series[timestamp_column], series[metric], linewidth=1.4, color="#2563eb")
    if anomaly_ts:
        mask = series[timestamp_column].isin(anomaly_ts)
        ax.scatter(
            series.loc[mask, timestamp_column],
            series.loc[mask, metric],
            color="#b91c1c", s=42, zorder=5, label="anomaly",
        )
        ax.legend(loc="best", fontsize=9)

    ax.set_title(metric)
    ax.set_xlabel("Date")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")
