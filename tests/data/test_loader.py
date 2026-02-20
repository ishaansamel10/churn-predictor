"""Tests for ChurnDataLoader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from churn_predictor.data.loader import ChurnDataLoader, DataValidationError


def test_load_valid_csv(raw_churn_csv: Path) -> None:
    loader = ChurnDataLoader(raw_churn_csv)
    df = loader.load()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    assert "churn" in df.columns


def test_load_missing_file(tmp_path: Path) -> None:
    loader = ChurnDataLoader(tmp_path / "nonexistent.csv")
    with pytest.raises(FileNotFoundError):
        loader.load()


def test_load_missing_columns(tmp_path: Path) -> None:
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"customer_id": ["X"], "churn": [0]}).to_csv(bad_csv, index=False)
    loader = ChurnDataLoader(bad_csv)
    with pytest.raises(DataValidationError, match="Missing required columns"):
        loader.load()


def test_load_invalid_target(tmp_path: Path, raw_churn_df: pd.DataFrame) -> None:
    bad_df = raw_churn_df.copy()
    bad_df.loc[0, "churn"] = 99
    bad_csv = tmp_path / "bad_target.csv"
    bad_df.to_csv(bad_csv, index=False)
    loader = ChurnDataLoader(bad_csv)
    with pytest.raises(DataValidationError, match="invalid values"):
        loader.load()
