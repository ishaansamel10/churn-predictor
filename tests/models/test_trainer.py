"""Tests for ChurnModelTrainer."""

from __future__ import annotations

from pathlib import Path

import pytest
from xgboost import XGBClassifier

from churn_predictor.data.preprocessor import ChurnPreprocessor
from churn_predictor.models.trainer import ChurnModelTrainer


def test_train_returns_xgb_classifier(raw_churn_df) -> None:
    params = {"n_estimators": 3, "max_depth": 2, "random_state": 42}
    trainer = ChurnModelTrainer(params=params)
    model = trainer.train(raw_churn_df, eval_fraction=0.2)
    assert isinstance(model, XGBClassifier)


def test_save_and_load(raw_churn_df, tmp_path: Path) -> None:
    params = {"n_estimators": 3, "max_depth": 2, "random_state": 42}
    trainer = ChurnModelTrainer(params=params)
    trainer.train(raw_churn_df, eval_fraction=0.2)
    save_path = tmp_path / "model.joblib"
    trainer.save(save_path)
    assert save_path.exists()

    model, preprocessor = ChurnModelTrainer.load(save_path)
    assert isinstance(model, XGBClassifier)
    assert isinstance(preprocessor, ChurnPreprocessor)


def test_save_before_train_raises() -> None:
    trainer = ChurnModelTrainer()
    with pytest.raises(RuntimeError, match=r"Call train\(\)"):
        trainer.save("/tmp/should_not_exist.joblib")
