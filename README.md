# AI Anomaly Detection Agent

A small, modular Python agent that scans an Excel file of time-series metrics,
flags points that fall outside ±N standard deviations of a rolling baseline,
and emails a single summary alert.

## Layout

```
anomaly-detection-agent/
├── main.py                 # Orchestrator / CLI entry point
├── data_loader.py          # Excel -> DataFrame, with loud failures
├── anomaly_detector.py     # Rolling z-score detection
├── summary_generator.py    # Plain-language anomaly summaries
├── email_alerter.py        # SMTP send (single email per run)
├── generate_sample.py      # Synthetic .xlsx generator for testing
├── config.yaml             # All tunable values, no magic numbers in code
├── requirements.txt        # Pinned dependencies
├── .env.example            # Template for SMTP credentials
└── data/
    └── (sample_metrics.xlsx goes here)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then edit .env with real SMTP creds
```

## Run end-to-end (no email needed)

```bash
python generate_sample.py                  # writes data/sample_metrics.xlsx
python main.py                             # reads it, detects anomalies, no email if no .env
python main.py --file data/other.xlsx      # override file path
```

## Run with email

Set these env vars (or use the `.env` file):

```
SMTP_USERNAME=...
SMTP_PASSWORD=...
ALERT_SENDER=...
ALERT_RECIPIENTS=alerts@example.com,oncall@example.com
```

Then `python main.py` — if any anomalies are found, one summary email is sent.

## Configuration

Everything tunable lives in `config.yaml`:

| Section         | Key                       | Default               | Meaning                                            |
|-----------------|---------------------------|-----------------------|----------------------------------------------------|
| `data`          | `file_path`               | `data/sample_metrics.xlsx` | Path to the input Excel file                  |
| `data`          | `timestamp_column`        | `Date`                | Name of the date column (auto-detected if null)    |
| `detection`     | `window_size`             | `30`                  | Trailing window for rolling mean/std               |
| `detection`     | `z_threshold`             | `3.0`                 | Flag points where `|z| > threshold`                |
| `smtp`          | `server` / `port` / `use_tls` | gmail / 587 / true  | SMTP transport                                    |
| `smtp`          | `username_env`, etc.      | `SMTP_USERNAME`       | Names of env vars holding credentials              |
| `logging`       | `level`                   | `INFO`                | Log level                                          |

## Behaviour notes

- The first `window_size` rows per metric are skipped (insufficient baseline).
- Both upward and downward anomalies are tracked; `direction` is explicit.
- One email per run, summarising **all** anomalies, never one-per-anomaly.
- Email send failures are logged but do not crash the detection run.
- No business glossary is provided; "Possible Impact" is inferred from the
  metric name using generic keyword hints in `summary_generator.py` — see the
  `_IMPACT_HINTS` block to extend it.

## Testing locally without real SMTP

Run `python main.py` without `.env` set — detection runs, summaries print to
the log, and the email step logs an error and continues. To verify email
plumbing end-to-end, use a service like Mailtrap or a local SMTP debug server
(such as `python -m smtpd`) and point `smtp.server` / `smtp.port` at it.
