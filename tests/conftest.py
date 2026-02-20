"""Shared pytest fixtures for the entire test suite."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from xgboost import XGBClassifier

from churn_predictor.api.main import create_app
from churn_predictor.data.preprocessor import ChurnPreprocessor
from churn_predictor.features.engineer import FeatureEngineer


@pytest.fixture()
def raw_churn_df() -> pd.DataFrame:
    """Minimal valid raw churn DataFrame for unit tests."""
    return pd.DataFrame(
        {
            "customer_id": ["C001", "C002", "C003", "C004", "C005"],
            "tenure_months": [1, 12, 24, 36, 60],
            "monthly_charges": [29.99, 49.99, 79.99, 99.99, 59.99],
            "total_charges": [29.99, 599.88, 1919.76, 3599.64, 3599.40],
            "num_products": [1, 2, 3, 2, 1],
            "has_internet_service": [True, True, False, True, True],
            "has_phone_service": [False, True, True, True, False],
            "contract_type": [
                "month-to-month",
                "one_year",
                "two_year",
                "month-to-month",
                "one_year",
            ],
            "churn": [1, 0, 0, 1, 0],
        }
    )


@pytest.fixture()
def raw_churn_csv(raw_churn_df: pd.DataFrame, tmp_path: Path) -> Path:
    """Write raw_churn_df to a temporary CSV file and return its path."""
    csv_path = tmp_path / "churn.csv"
    raw_churn_df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture()
def trained_artifact(
    raw_churn_df: pd.DataFrame,
) -> tuple[XGBClassifier, ChurnPreprocessor]:
    """Train a tiny model on the fixture data and return (model, preprocessor)."""
    engineer = FeatureEngineer()
    df_engineered = engineer.transform(raw_churn_df)

    preprocessor = ChurnPreprocessor()
    X, y = preprocessor.fit_transform(df_engineered)  # noqa: N806

    model = XGBClassifier(n_estimators=5, max_depth=2, random_state=42)
    model.fit(X, y)
    return model, preprocessor


@pytest.fixture()
def saved_model_path(
    trained_artifact: tuple[XGBClassifier, ChurnPreprocessor],
    tmp_path: Path,
) -> Path:
    """Persist the trained artifact and return the file path."""
    model, preprocessor = trained_artifact
    path = tmp_path / "test_model.joblib"
    joblib.dump({"model": model, "preprocessor": preprocessor}, path)
    return path


@pytest.fixture()
def api_client(saved_model_path: Path):
    """Return a FastAPI TestClient backed by the trained test model."""
    test_app = create_app(model_path=saved_model_path)
    with TestClient(test_app) as client:
        yield client
