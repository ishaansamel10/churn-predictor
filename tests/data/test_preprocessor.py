"""Tests for ChurnPreprocessor."""

from __future__ import annotations

import pandas as pd
import pytest

from churn_predictor.data.preprocessor import ChurnPreprocessor
from churn_predictor.features.engineer import FeatureEngineer


def test_fit_transform_returns_xy(raw_churn_df: pd.DataFrame) -> None:
    engineer = FeatureEngineer()
    df_eng = engineer.transform(raw_churn_df)
    preprocessor = ChurnPreprocessor()
    X, y = preprocessor.fit_transform(df_eng)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == len(y) == 5


def test_transform_without_target(raw_churn_df: pd.DataFrame) -> None:
    engineer = FeatureEngineer()
    df_eng = engineer.transform(raw_churn_df)
    preprocessor = ChurnPreprocessor()
    preprocessor.fit_transform(df_eng)

    df_no_target = df_eng.drop(columns=["churn"])
    X = preprocessor.transform(df_no_target)
    assert isinstance(X, pd.DataFrame)
    assert len(X) == 5


def test_transform_before_fit_raises(raw_churn_df: pd.DataFrame) -> None:
    engineer = FeatureEngineer()
    df_eng = engineer.transform(raw_churn_df)
    preprocessor = ChurnPreprocessor()
    with pytest.raises(RuntimeError, match="not fitted"):
        preprocessor.transform(df_eng)
