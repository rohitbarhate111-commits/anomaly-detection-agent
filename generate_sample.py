"""
generate_sample.py

Tiny synthetic data generator. Produces a wide-format .xlsx with a few metrics
and at least one obvious injected anomaly per metric, so the agent can be
exercised end-to-end without a real business dataset.

Usage:
    python generate_sample.py                # writes data/sample_metrics.xlsx
    python generate_sample.py --out foo.xlsx --days 120
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate(days: int = 120, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days, freq="D")

    # --- Metric 1: Revenue (random walk, one spike injected) ---
    rev = np.cumsum(rng.normal(loc=10.0, scale=2.0, size=days)) + 500
    rev[60] += 250  # upward anomaly
    rev = np.maximum(rev, 0)

    # --- Metric 2: Error_Rate (small noise, one downward anomaly) ---
    err = np.clip(rng.normal(loc=1.5, scale=0.3, size=days), 0, None)
    err[80] = 0.0  # downward anomaly (suspiciously zero)

    # --- Metric 3: Latency_ms (random walk, one upward anomaly) ---
    lat = np.cumsum(rng.normal(loc=0.0, scale=1.5, size=days)) + 120
    lat[40] += 180  # upward anomaly

    # --- Metric 4: Active_Users (mild trend + noise) ---
    users = (
        1000
        + np.linspace(0, 200, days)
        + rng.normal(0, 25, days)
    ).astype(int)
    users[100] -= 600  # downward anomaly

    df = pd.DataFrame({
        "Date": dates,
        "Revenue": rev,
        "Error_Rate": err,
        "Latency_ms": lat,
        "Active_Users": users,
    })
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a sample Excel file for testing.")
    parser.add_argument("--out", default="data/sample_metrics.xlsx", help="Output .xlsx path.")
    parser.add_argument("--days", type=int, default=120, help="Number of daily rows.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate(days=args.days, seed=args.seed)
    df.to_excel(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path.resolve()}")
    print("Injected anomalies:")
    print("  - Revenue @ idx 60 (upward)")
    print("  - Error_Rate @ idx 80 (downward)")
    print("  - Latency_ms @ idx 40 (upward)")
    print("  - Active_Users @ idx 100 (downward)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
