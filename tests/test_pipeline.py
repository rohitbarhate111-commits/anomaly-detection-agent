import tempfile
from pathlib import Path
import pandas as pd

from main import run_pipeline


def test_pipeline_end_to_end():
    with tempfile.TemporaryDirectory() as tmpdir:
        td = Path(tmpdir)
        data_file = td / "metrics.xlsx"
        dates = pd.date_range("2026-01-01", periods=60, freq="D")
        sales = [100.0] * 60
        sales[45] = 500.0  # obvious outlier

        df = pd.DataFrame({"Date": dates, "Sales": sales})
        df.to_excel(data_file, index=False)

        config = {
            "input_mode": "excel_file",
            "data": {"file_path": str(data_file), "timestamp_column": "Date"},
            "detection": {"mode": "zscore", "window_size": 20, "z_threshold": 3.0},
            "suppression": {"enabled": True, "state_db_path": str(td / "state.db"), "escalation_z_delta": 2.0},
            "correlation_window_days": 0,
            "report": {"output_dir": str(td / "reports")},
            "smtp": {},
        }

        export_json = td / "scan_out.json"
        res = run_pipeline(config, export_json_path=str(export_json), base_dir=td)

        assert res["status"] == "success"
        assert res["total_anomalies"] >= 1
        assert res["emitted_anomalies"] >= 1
        assert export_json.is_file()
