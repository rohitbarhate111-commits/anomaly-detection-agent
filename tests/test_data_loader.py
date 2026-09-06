import tempfile
from pathlib import Path
import pandas as pd
import pytest

from data_loader import DataLoadError, load_excel, load_excel_folder


def test_load_excel_clean():
    with tempfile.TemporaryDirectory() as tmpdir:
        fp = Path(tmpdir) / "test.xlsx"
        df_in = pd.DataFrame({
            "Date": pd.date_range("2026-01-01", periods=10, freq="D"),
            "Sales": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        })
        df_in.to_excel(fp, index=False)

        df_out = load_excel(fp)
        assert len(df_out) == 10
        assert "Date" in df_out.columns
        assert "Sales" in df_out.columns


def test_load_excel_missing_file():
    with pytest.raises(DataLoadError, match="not found"):
        load_excel("non_existent_file.xlsx")


def test_load_excel_no_numeric():
    with tempfile.TemporaryDirectory() as tmpdir:
        fp = Path(tmpdir) / "words.xlsx"
        df_in = pd.DataFrame({
            "Date": ["2026-01-01", "2026-01-02"],
            "Notes": ["hello", "world"],
        })
        df_in.to_excel(fp, index=False)
        with pytest.raises(DataLoadError, match="No numeric metric columns"):
            load_excel(fp)
