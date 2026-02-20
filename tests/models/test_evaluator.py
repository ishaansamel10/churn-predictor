"""Tests for ModelEvaluator."""

from __future__ import annotations

import pandas as pd

from churn_predictor.features.engineer import FeatureEngineer
from churn_predictor.models.evaluator import EvaluationReport, ModelEvaluator


def test_evaluate_returns_report(trained_artifact, raw_churn_df) -> None:
    model, preprocessor = trained_artifact
    engineer = FeatureEngineer()
    df_eng = engineer.transform(raw_churn_df)
    X, y = preprocessor.fit_transform(df_eng)

    evaluator = ModelEvaluator(model)
    report = evaluator.evaluate(X, y)

    assert isinstance(report, EvaluationReport)
    assert 0.0 <= report.accuracy <= 1.0
    assert 0.0 <= report.roc_auc <= 1.0


def test_feature_importances_sorted(trained_artifact, raw_churn_df) -> None:
    model, preprocessor = trained_artifact
    engineer = FeatureEngineer()
    df_eng = engineer.transform(raw_churn_df)
    X, _ = preprocessor.fit_transform(df_eng)

    evaluator = ModelEvaluator(model)
    df_imp = evaluator.feature_importances(list(X.columns))

    assert isinstance(df_imp, pd.DataFrame)
    assert "feature" in df_imp.columns
    assert "importance" in df_imp.columns
    assert df_imp["importance"].is_monotonic_decreasing or len(df_imp) <= 1
