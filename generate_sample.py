"""
generate_sample.py

Tiny synthetic data generator for the anomaly detection agent.

v1: --mode simple - basic metrics with one anomaly each.
v2 scenarios:
    seasonal  - weekly seasonality baked into every metric. Use with
                detection_mode=seasonal.
    ongoing   - one metric has a persistent multi-day anomaly so you can see
                suppression (first run alerts, subsequent runs do not).
    correlated- two metrics spike on the same day so correlation grouping
                fires. Use with correlation_window_days=0.

Usage:
    python generate_sample.py
    python generate_sample.py --mode seasonal
    python generate_sample.py --mode ongoing
    python generate_sample.py --mode correlated
    python generate_sample.py --out foo.xlsx --days 120
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _seasonal(days: int, period: int = 7, amplitude: float = 5.0) -> np.ndarray:
    """Return a periodic pattern (sine) of length `days` with the given period."""
    t = np.arange(days)
    return amplitude * np.sin(2 * np.pi * t / period)


def generate_simple(days: int = 120, seed: int = 7) -> pd.DataFrame:
    """v1 baseline scenario."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days, freq="D")

    rev = np.cumsum(rng.normal(loc=10.0, scale=2.0, size=days)) + 500
    rev[60] += 250
    rev = np.maximum(rev, 0)

    err = np.clip(rng.normal(loc=1.5, scale=0.3, size=days), 0, None)
    err[80] = 0.0

    lat = np.cumsum(rng.normal(loc=0.0, scale=1.5, size=days)) + 120
    lat[40] += 180

    users = (1000 + np.linspace(0, 200, days) + rng.normal(0, 25, days)).astype(int)
    users[100] -= 600

    return pd.DataFrame({
        "Date": dates,
        "Revenue": rev,
        "Error_Rate": err,
        "Latency_ms": lat,
        "Active_Users": users,
    })


def generate_seasonal(days: int = 180, seed: int = 11) -> pd.DataFrame:
    """Strong weekly seasonality on every metric - tests seasonal mode."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days, freq="D")
    season = _seasonal(days, period=7, amplitude=1.0)

    rev = 500 + 50 * season + rng.normal(0, 8, days)
    rev[90] += 220     # upward anomaly vs seasonal baseline

    err = 1.5 + 0.3 * season + rng.normal(0, 0.1, days)
    err[120] = 0.0     # downward anomaly

    lat = 120 + 8 * season + rng.normal(0, 1.2, days)
    lat[60] += 90      # upward

    return pd.DataFrame({
        "Date": dates,
        "Revenue": rev,
        "Error_Rate": err,
        "Latency_ms": lat,
    })


def generate_ongoing(days: int = 180, seed: int = 13) -> pd.DataFrame:
    """One metric is anomalous for ~5 consecutive days to test suppression."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days, freq="D")

    rev = 500 + rng.normal(0, 5, days).cumsum() * 0.5
    # A spike that persists for ~5 days, starting at index 70.
    rev[70:75] += 180

    err = np.clip(rng.normal(1.5, 0.3, days), 0, None)
    lat = 120 + rng.normal(0, 1.5, days).cumsum()
    users = (1000 + np.linspace(0, 50, days) + rng.normal(0, 25, days)).astype(int)

    return pd.DataFrame({
        "Date": dates,
        "Revenue": rev,
        "Error_Rate": err,
        "Latency_ms": lat,
        "Active_Users": users,
    })


def generate_correlated(days: int = 180, seed: int = 17) -> pd.DataFrame:
    """Two metrics spike on the SAME day to exercise correlation grouping."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days, freq="D")

    rev = np.cumsum(rng.normal(2, 1, days)) + 500
    err = np.clip(rng.normal(1.5, 0.2, days), 0, None)
    lat = 120 + rng.normal(0, 1.0, days).cumsum()

    # SAME-day spike pair: revenue up, error rate up.
    rev[80] += 220
    err[80] += 3.5

    # A second SAME-day spike pair elsewhere.
    lat[110] += 90
    rev[110] -= 150

    return pd.DataFrame({
        "Date": dates,
        "Revenue": rev,
        "Error_Rate": err,
        "Latency_ms": lat,
    })


_GENERATORS = {
    "simple":     generate_simple,
    "seasonal":   generate_seasonal,
    "ongoing":    generate_ongoing,
    "correlated": generate_correlated,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a sample Excel file for testing.")
    parser.add_argument("--out", default="data/sample_metrics.xlsx", help="Output .xlsx path.")
    parser.add_argument("--days", type=int, default=120, help="Number of daily rows.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--mode",
        default="simple",
        choices=sorted(_GENERATORS.keys()),
        help="Which scenario to generate.",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = _GENERATORS[args.mode](days=args.days, seed=args.seed)
    df.to_excel(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path.resolve()} (mode={args.mode})")
    print("Injected anomalies:")
    if args.mode == "simple":
        print("  - Revenue @ idx 60 (upward)")
        print("  - Error_Rate @ idx 80 (downward)")
        print("  - Latency_ms @ idx 40 (upward)")
        print("  - Active_Users @ idx 100 (downward)")
    elif args.mode == "seasonal":
        print("  - Revenue @ idx 90 (upward, weekly pattern)")
        print("  - Error_Rate @ idx 120 (downward, weekly pattern)")
        print("  - Latency_ms @ idx 60 (upward, weekly pattern)")
    elif args.mode == "ongoing":
        print("  - Revenue @ idx 70-74 (persistent upward, ~5 days)")
    elif args.mode == "correlated":
        print("  - Revenue + Error_Rate @ idx 80 (correlated)")
        print("  - Latency_ms + Revenue @ idx 110 (correlated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
