# AI Anomaly Detection Agent

A small, modular Python agent that scans time-series metrics from Excel files,
Excel folders, or a Postgres database, flags points that fall outside a
rolling baseline (z-score or seasonal STL), and:

- emails a single summary alert per run (with suppression + escalation),
- writes a self-contained HTML report with charts,
- groups co-occurring anomalies for root-cause hints (no causal claims).

## Layout

```
anomaly-detection-agent/
├── main.py                  # Orchestrator / CLI entry point
├── data_loader.py           # excel_file / excel_folder / database -> DataFrame
├── db_loader.py             # Postgres (SQLAlchemy) loader (optional)
├── anomaly_detector.py      # zscore or seasonal STL detection
├── summary_generator.py     # Plain-language anomaly summaries
├── correlation.py           # Same-timestamp grouping (non-causal)
├── state_store.py           # SQLite suppression state (per metric)
├── email_alerter.py         # Single email per run; escalation tagging
├── report_generator.py      # Self-contained HTML report w/ charts
├── generate_sample.py       # Synthetic .xlsx for the four test scenarios
├── config.yaml              # All tunables
├── requirements.txt
├── .env.example             # SMTP + (optional) DB credentials
└── data/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then edit .env
```

## Quick start (v1 flow still works)

```bash
python generate_sample.py                      # simple scenario
python main.py                                 # zscore mode, no suppression
```

## v2 scenarios

```bash
python generate_sample.py --mode seasonal   --out data/seasonal.xlsx
python generate_sample.py --mode ongoing    --out data/ongoing.xlsx
python generate_sample.py --mode correlated --out data/correlated.xlsx
```

Override config defaults per-run via CLI:

```bash
python main.py --file data/seasonal.xlsx      # v1 path, seasonal detection on
python main.py --folder data/folder_in        # excel_folder input mode
```

To use the database input mode, set `input_mode: database` in `config.yaml`
and provide a connection string + query under the `db:` section. You can also
reference env vars:

```yaml
db:
  connection_string: "ENV:DB_CONNECTION_STRING"
  query: "ENV:DB_QUERY"
```

## Configuration (config.yaml)

| Key                                 | Default                  | Meaning                                                  |
|-------------------------------------|--------------------------|----------------------------------------------------------|
| `input_mode`                        | `excel_file`             | `excel_file` / `excel_folder` / `database`               |
| `data.file_path` / `data.folder_path` | per mode              | Source of the data                                       |
| `detection.mode`                    | `seasonal`               | `zscore` (v1) or `seasonal` (v2 STL)                     |
| `detection.window_size`             | `30`                     | Trailing window for rolling baseline                     |
| `detection.z_threshold`             | `3.0`                    | Anomaly if `|z| > threshold`                             |
| `detection.seasonal_period`         | `7`                      | Period for STL decomposition                             |
| `detection.min_cycles`              | `2`                      | Need `>= period * min_cycles` of history for seasonal    |
| `suppression.enabled`               | `true`                   | Suppress repeat alerts while a metric stays anomalous    |
| `suppression.escalation_z_delta`    | `2.0`                    | `|z|` jump vs last alert that triggers an escalation      |
| `suppression.state_db_path`         | `anomaly_state.db`       | SQLite file for per-metric state                         |
| `correlation_window_days`           | `0`                      | `0` = same day only; `N>0` = ±N days                     |
| `report.output_dir`                 | `./reports`              | Where HTML reports are written                           |
| `smtp.*`                            | Gmail defaults           | SMTP transport; credentials from env vars                |

## Behaviour notes

- The first `window_size` rows per metric are skipped (insufficient baseline).
- Both upward and downward anomalies are tracked; `direction` is explicit.
- **One email per run**, summarising all kept anomalies; escalations listed first.
- Email send failures are logged but do not crash the detection run.
- Suppression is per metric, stored in SQLite; deleting `anomaly_state.db`
  resets everything.
- Co-occurrence hints are correlation only — the wording always says
  "may indicate a related cause, or may be coincidental — not confirmed causation."
- The HTML report is self-contained: charts are inlined as base64 PNG, no
  external requests when viewed.

## Testing without real SMTP / DB

```bash
python main.py --file data/sample_metrics.xlsx    # logs summaries, no email
```

For email plumbing, point `smtp.server` / `smtp.port` at a local debug SMTP
server (e.g. `python -m aiosmtpd -n`) or a service like Mailtrap.

For database mode, point `db.connection_string` at any SQLAlchemy-compatible
URL (SQLite works too: `sqlite:///./test.db`).
