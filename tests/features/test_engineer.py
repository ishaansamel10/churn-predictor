"""Tests for FeatureEngineer."""

from __future__ import annotations

import pandas as pd
import pytest

from churn_predictor.features.engineer import FeatureEngineer


def test_adds_charges_per_month(raw_churn_df: pd.DataFrame) -> None:
    df = FeatureEngineer().transform(raw_churn_df)
    assert "charges_per_month" in df.columns
    val = df.loc[df["tenure_months"] == 1, "charges_per_month"].iloc[0]
    assert val == pytest.approx(29.99, abs=0.01)


def test_adds_tenure_bucket(raw_churn_df: pd.DataFrame) -> None:
    df = FeatureEngineer().transform(raw_churn_df)
    assert "tenure_bucket" in df.columns
    assert df["tenure_bucket"].between(0, 3).all()


def test_adds_service_count(raw_churn_df: pd.DataFrame) -> None:
    df = FeatureEngineer().transform(raw_churn_df)
    assert "service_count" in df.columns
    assert df["service_count"].between(0, 2).all()


def test_transform_does_not_mutate_input(raw_churn_df: pd.DataFrame) -> None:
    original_cols = set(raw_churn_df.columns)
    FeatureEngineer().transform(raw_churn_df)
    assert set(raw_churn_df.columns) == original_cols
