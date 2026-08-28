# AI Anomaly Detection Agent

A modular Python pipeline for detecting unusual time-series metric behaviour and turning it into actionable, non-causal operational signals.

It supports:

- Excel files, Excel folders, and SQLAlchemy-compatible databases as inputs
- rolling z-score detection and seasonal STL detection
- per-metric alert suppression with escalation handling
- co-occurrence hints without claiming causation
- one summary email per run
- self-contained HTML reports with inline charts

## Architecture

```text
Input source
    │
    ▼
 data_loader ──────► normalized DataFrame
    │
    ▼
anomaly_detector ──► Anomaly records
    │
    ├──► summary_generator ──► business-context summaries
    ├──► correlation ────────► co-occurrence notes
    │
    ▼
 state_store ──────► suppression / escalation decisions
    │
    ├──► email_alerter ──────► single alert email
    └──► report_generator ───► portable HTML report
```

The modules communicate through small data structures rather than sharing transport, storage, or presentation concerns.

## Project layout

```text
anomaly-detection-agent/
├── main.py
├── data_loader.py
├── db_loader.py
├── anomaly_detector.py
├── summary_generator.py
├── correlation.py
├── state_store.py
├── email_alerter.py
├── report_generator.py
├── generate_sample.py
├── tests/
│   └── test_detection_and_correlation.py
├── config.yaml
├── requirements.txt
├── .env.example
└── .gitignore
```

## Requirements and setup

Python 3.9+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

SMTP credentials are optional for local detection/report testing. If email delivery is enabled, copy `.env.example` to `.env` and provide the required environment variables. `.env` is ignored by Git.

## Quick start

Generate deterministic sample data and run the default seasonal detector:

```bash
python generate_sample.py --mode seasonal --out data/seasonal.xlsx
python main.py --file data/seasonal.xlsx
```

The run writes a self-contained HTML report under `reports/`. Email delivery is attempted only when the configured SMTP environment variables are available.

### Sample scenarios

```bash
python generate_sample.py --mode simple
python generate_sample.py --mode seasonal --out data/seasonal.xlsx
python generate_sample.py --mode ongoing --out data/ongoing.xlsx
python generate_sample.py --mode correlated --out data/correlated.xlsx
```

- `simple`: basic upward/downward anomalies
- `seasonal`: weekly seasonality for STL detection
- `ongoing`: persistent anomaly for suppression behaviour
- `correlated`: same-day anomalies for co-occurrence hints

## Configuration

`config.yaml` contains runtime settings for input, detection, suppression, correlation, reporting, SMTP, and logging.

Database credentials and queries can be referenced through environment variables:

```yaml
db:
  connection_string: "ENV:DB_CONNECTION_STRING"
  query: "ENV:DB_QUERY"
```

The database loader accepts SQLAlchemy-compatible URLs, so local SQLite can also be used for testing.

Important detection settings:

| Setting | Default | Purpose |
|---|---:|---|
| `detection.mode` | `seasonal` | `zscore` or `seasonal` |
| `detection.window_size` | `30` | Rolling baseline length |
| `detection.z_threshold` | `3.0` | Absolute z-score threshold |
| `detection.seasonal_period` | `7` | Seasonal period for STL |
| `detection.min_cycles` | `2` | Minimum history before seasonal detection |
| `suppression.enabled` | `true` | Suppress repeated active anomalies |
| `suppression.escalation_z_delta` | `2.0` | Extra z-score magnitude required for escalation |
| `correlation_window_days` | `0` | Same date when zero; pairwise ±N-day window otherwise |
| `report.output_dir` | `./reports` | HTML report destination |

When seasonal mode does not have enough history, the detector falls back to rolling z-score detection for that metric.

## Testing

Run the regression suite with the Python standard library test runner:

```bash
python -m unittest discover -s tests -v
```

The tests cover detector configuration validation, non-default DataFrame indexes, and the pairwise correlation-window behaviour.

## Design notes

- The detector does not make causal claims. Correlation output only identifies temporal co-occurrence.
- Suppression state is persisted in SQLite so repeated scheduled runs can share alert state.
- Email failures do not discard the generated detection/report results.
- Reports inline their chart images and escape user/data-derived HTML content before rendering.
- Generated Excel files, reports, and local SQLite state are intentionally excluded from version control.
