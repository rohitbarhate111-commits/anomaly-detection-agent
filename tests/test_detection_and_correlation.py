import unittest

import pandas as pd

from anomaly_detector import Anomaly, detect_anomalies
from correlation import find_co_occurrences


class DetectionTests(unittest.TestCase):
    def test_detection_works_with_non_default_dataframe_index(self):
        frame = pd.DataFrame(
            {
                "Date": pd.date_range("2026-01-01", periods=8, freq="D"),
                "metric": [10, 10, 10, 10, 10, 10, 10, 100],
            },
            index=list(range(10, 18)),
        )
        anomalies = detect_anomalies(frame, "Date", window_size=5, z_threshold=1.9)
        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies[0].metric, "metric")
        self.assertEqual(anomalies[0].direction, "up")

    def test_invalid_detection_configuration_is_rejected(self):
        frame = pd.DataFrame(
            {"Date": pd.date_range("2026-01-01", periods=3), "metric": [1, 2, 3]}
        )
        with self.assertRaises(ValueError):
            detect_anomalies(frame, "Date", window_size=1)
        with self.assertRaises(ValueError):
            detect_anomalies(frame, "Date", seasonal_period=1)
        with self.assertRaises(ValueError):
            detect_anomalies(frame, "Date", min_cycles=0)


class CorrelationTests(unittest.TestCase):
    def test_positive_window_matches_across_neighbouring_dates(self):
        anomalies = [
            Anomaly("a", pd.Timestamp("2026-01-02 00:00"), 1, 0, 1, 4, "up"),
            Anomaly("b", pd.Timestamp("2026-01-02 23:00"), 1, 0, 1, 4, "up"),
            Anomaly("c", pd.Timestamp("2026-01-03 23:00"), 1, 0, 1, 4, "up"),
        ]
        result = find_co_occurrences(anomalies, window_days=1)
        self.assertEqual(result["a"], ["b"])
        self.assertEqual(result["b"], ["a", "c"])
        self.assertEqual(result["c"], ["b"])

    def test_negative_window_is_rejected(self):
        with self.assertRaises(ValueError):
            find_co_occurrences([], window_days=-1)


if __name__ == "__main__":
    unittest.main()
