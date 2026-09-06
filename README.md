# AI Anomaly Detection Agent

An autonomous, modular Python pipeline for detecting, correlating, and alerting on time-series anomalies across business and infrastructure metrics.

Supports Excel workbooks, directory batch aggregation, and PostgreSQL / SQLAlchemy-compatible relational databases. Features robust statistical rolling baselines, Seasonal STL decomposition, alert suppression with escalation tracking via SQLite, non-causal co-occurrence hints, and self-contained HTML reports with embedded charts.

---

## Architecture & Pipeline Flow

```
+-------------------------------------------------------------------------+
|                              DATA SOURCES                               |
|   Excel File (.xlsx)  |  Excel Directory Batch  |  PostgreSQL / SQL DB  |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                        DATA LOADER & VALIDATION                         |
|   - Auto-detect timestamp column  - Enforce chronological ordering      |
|   - Strip non-numeric telemetry   - Align multi-file schema columns     |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                        DETECTION ENGINE                                 |
|   - Mode A: Rolling Z-Score (local moving window)                       |
|   - Mode B: Seasonal STL (LOESS decomposition + residual z-scoring)     |
|   - Automatic fallback on insufficient cycle history (< 2 full cycles)  |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                        INTELLIGENCE & CORRELATION                       |
|   - Severity scoring & directional classification (UP / DOWN)           |
|   - Co-occurrence clustering (sliding temporal window, non-causal)      |
|   - Domain-aware plain language summary generation                      |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                    STATE MACHINE & ALERT SUPPRESSION                    |
|   - SQLite persistence (anomaly_state.db)                               |
|   - Suppress repeat alerts while metric remains anomalous               |
|   - Trigger ESCALATION alert if |z| jumps >= escalation_z_delta         |
|   - Reset state to normal once metric returns within baseline           |
+-------------------------------------------------------------------------+
                                    |
         +--------------------------+--------------------------+
         |                                                     |
         v                                                     v
+------------------+                                 +--------------------+
|   EMAIL ALERTER  |                                 |   REPORT BUILDER   |
| Single digest    |                                 | Self-contained     |
| Escalations top  |                                 | HTML + inline b64  |
| Graceful failure |                                 | matplotlib charts  |
+------------------+                                 +--------------------+
```

---

## Detection Algorithms

### 1. Rolling Z-Score (`detection.mode: "zscore"`)
Computes local mean $\mu_W$ and sample standard deviation $\sigma_W$ over a trailing window $W$:

$$z_t = \frac{y_t - \mu_W(y)}{\sigma_W(y)}$$

A data point is flagged as an anomaly whenever $|z_t| > z_{\text{threshold}}$.

### 2. Seasonal STL Decomposition (`detection.mode: "seasonal"`)
For metrics exhibiting recurrent cycles (e.g. daily, weekly seasonality):

$$y_t = \text{Trend}_t + \text{Seasonal}_t + \text{Residual}_t$$

- **Expected Baseline**: $\hat{y}_t = \text{Trend}_t + \text{Seasonal}_t$
- **Residual Deviation**: $r_t = y_t - \hat{y}_t$
- **Residual Z-Score**: $z_t = \frac{r_t - \mu_W(r)}{\sigma_W(r)}$

If a metric lacks sufficient history (fewer than $\text{seasonal\_period} \times \text{min\_cycles}$ observations), the agent automatically falls back to rolling z-score.

---

## Intelligence & State Rules

- **Directional Classification**: `UP` ($z > 0$) or `DOWN` ($z < 0$).
- **Severity Tiers**:
  - `moderate deviation`: $3.0 \le |z| < 4.5$
  - `critical outlier`: $4.5 \le |z| < 6.0$
  - `extreme outlier`: $|z| \ge 6.0$
- **Co-Occurrence Correlation**: Flags metrics anomalous within the same temporal window (default $0$ days = exact date). Adheres strictly to non-causal language:
  > *"Note: this anomaly coincided with unusual movement in [Other Metrics] on the same date. This may indicate a related cause, or may be coincidental - not confirmed causation."*
- **Suppression State Machine**:
  - `normal -> active_anomaly`: First breach generates an alert and records initial $|z|$.
  - `active_anomaly -> active_anomaly`: Repeated alerts are suppressed unless $|z_{\text{current}}| - |z_{\text{last}}| \ge \text{escalation\_z\_delta}$.
  - `active_anomaly -> normal`: Automatically resets when metric returns in-range.

---

## Project Structure

```
anomaly-detection-agent/
├── anomaly_detector.py      # Core z-score & seasonal STL detection logic
├── correlation.py           # Sliding-window co-occurrence clustering
├── data_loader.py           # Multi-source data ingestion & validation
├── db_loader.py             # PostgreSQL / SQLAlchemy database connector
├── email_alerter.py         # Consolidated SMTP alert dispatcher
├── generate_sample.py       # Synthetic time-series benchmark generator
├── main.py                  # CLI entry point & programmatic run_pipeline()
├── report_generator.py      # Standalone HTML report generator with charts
├── state_store.py           # SQLite state store for alert suppression
├── summary_generator.py     # Domain impact heuristics & plain-text summaries
├── config.yaml              # Pipeline configuration
├── requirements.txt         # Project dependencies
├── .env.example             # SMTP & database credentials template
├── data/                    # Benchmark Excel datasets & JSON fixtures
├── reports/                 # Output directory for HTML reports
└── tests/                   # Automated pytest test suite
    ├── test_anomaly_detector.py
    ├── test_correlation.py
    ├── test_data_loader.py
    ├── test_pipeline.py
    └── test_suppression.py
```

---

## Setup & Installation

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment variables (optional for SMTP/DB)
cp .env.example .env
```

---

## Configuration Reference (`config.yaml`)

| Setting | Default | Description |
| :--- | :--- | :--- |
| `input_mode` | `excel_file` | Source: `excel_file`, `excel_folder`, or `database` |
| `data.file_path` | `data/sample_metrics.xlsx` | Path to single workbook |
| `data.folder_path` | `data/folder_in` | Path to directory for batch aggregation |
| `data.timestamp_column` | `Date` | Date column name (auto-detected if null) |
| `detection.mode` | `seasonal` | Algorithm: `seasonal` (STL) or `zscore` |
| `detection.window_size` | `30` | Trailing window observations for baseline |
| `detection.z_threshold` | `3.0` | Cutoff multiplier for outlier classification |
| `detection.seasonal_period` | `7` | Expected seasonal cycle length (e.g. 7 = weekly) |
| `detection.min_cycles` | `2` | Minimum full cycles required for STL |
| `suppression.enabled` | `true` | Suppress repeat alerts for active anomalies |
| `suppression.escalation_z_delta` | `2.0` | Jump in \|z\| required to trigger escalation |
| `suppression.state_db_path` | `anomaly_state.db`| SQLite database file for state tracking |
| `correlation_window_days` | `0` | Temporal window for co-occurrence (0 = same day) |
| `report.output_dir` | `./reports` | Target directory for generated HTML reports |

---

## CLI & Programmatic Usage

### Command Line
```bash
# Run pipeline with default config
python main.py

# Override input file or folder
python main.py --file data/seasonal.xlsx
python main.py --folder data/folder_in

# Export structured JSON results (for APIs / Portfolio OS)
python main.py --file data/seasonal.xlsx --export-json data/scan_results.json
```

### Programmatic Python API
```python
from pathlib import Path
from main import load_config, run_pipeline

config = load_config(Path("config.yaml"))
results = run_pipeline(
    config=config,
    file_override="data/seasonal.xlsx",
    export_json_path="reports/output.json"
)

print(f"Detected {results['total_anomalies']} anomalies across {results['total_metrics']} metrics.")
for s in results['summaries']:
    print(f"[{s['metric']}] {s['what_changed']}")
```

---

## Running Automated Tests

Run the complete test suite across detection, seasonal STL, suppression, correlation, and data loaders:

```bash
python -m pytest tests/ -v
```

---

## Portfolio OS Integration

This repository is integrated into **Rohit Barhate — Portfolio OS** as a native developer monitoring application:
- **Project Card**: Featured under `Projects` with complete architectural details and direct application launch trigger.
- **Native OS App**: Interactive dashboard with real-time KPI metrics, interactive time-series timeline (normal values, model baseline, confidence thresholds, and anomaly markers), filterable anomaly feed, detailed inspection drawer with co-occurrence notes, and dynamic parameter tuning.

---

## Limitations & Roadmap

- **Current Limitations**:
  - Operates in batch mode over fixed datasets rather than real-time event streams.
  - Seasonality period must be configured explicitly (e.g. 7 for weekly, 24 for hourly) rather than auto-discovered via FFT/autocorrelation.
  - Co-occurrence detects temporal overlap; it does not model directed DAG causal structures.
- **Future Enhancements**:
  - Automated periodicity detection using periodograms / spectral analysis.
  - Streaming ingestion via Apache Kafka / AWS Kinesis.
  - Multivariate anomaly scoring via Isolation Forests or Autoencoders.
