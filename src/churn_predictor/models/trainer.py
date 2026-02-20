"""XGBoost model training and persistence."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from xgboost import XGBClassifier

from churn_predictor.data.loader import ChurnDataLoader
from churn_predictor.data.preprocessor import ChurnPreprocessor
from churn_predictor.features.engineer import FeatureEngineer

logger = logging.getLogger(__name__)

DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
}


class ChurnModelTrainer:
    """Orchestrates the full train pipeline.

    Parameters
    ----------
    params:
        XGBoost hyperparameters. Defaults to ``DEFAULT_PARAMS``.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or DEFAULT_PARAMS
        self.model: XGBClassifier | None = None
        self.preprocessor: ChurnPreprocessor | None = None

    def train(
        self,
        df: pd.DataFrame,
        *,
        eval_fraction: float = 0.2,
        random_state: int = 42,
    ) -> XGBClassifier:
        """Fit the XGBoost classifier on the supplied DataFrame.

        Parameters
        ----------
        df:
            Raw churn DataFrame (output of ``ChurnDataLoader.load``).
        eval_fraction:
            Fraction of data to hold out for early-stopping eval set.
        random_state:
            Seed for the train/eval split.

        Returns
        -------
        XGBClassifier
            The fitted model.
        """
        from sklearn.model_selection import train_test_split

        engineer = FeatureEngineer()
        df_engineered = engineer.transform(df)

        self.preprocessor = ChurnPreprocessor()
        X, y = self.preprocessor.fit_transform(df_engineered)

        # Stratified split requires ≥2 samples per class in the eval set.
        # Fall back to non-stratified when the dataset is too small.
        n_eval = max(1, round(len(X) * eval_fraction))
        use_stratify = n_eval >= y.nunique() and y.value_counts().min() >= 2
        X_train, X_eval, y_train, y_eval = train_test_split(
            X,
            y,
            test_size=eval_fraction,
            stratify=y if use_stratify else None,
            random_state=random_state,
        )

        self.model = XGBClassifier(**self.params)
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_eval, y_eval)],
            verbose=False,
        )

        logger.info(
            "Model trained. Best iteration: %s",
            getattr(self.model, "best_iteration", "n/a"),
        )
        return self.model

    def save(self, path: str | Path) -> None:
        """Persist model + preprocessor to a single joblib file.

        Parameters
        ----------
        path:
            Destination file path (e.g., ``models/xgb_churn.joblib``).
        """
        if self.model is None or self.preprocessor is None:
            raise RuntimeError("Call train() before save().")
        artifact = {
            "model": self.model,
            "preprocessor": self.preprocessor,
        }
        joblib.dump(artifact, path)
        logger.info("Artifact saved to %s", path)

    @staticmethod
    def load(path: str | Path) -> tuple[XGBClassifier, ChurnPreprocessor]:
        """Load a persisted artifact.

        Returns
        -------
        tuple[XGBClassifier, ChurnPreprocessor]
            The model and its fitted preprocessor.
        """
        artifact: dict[str, Any] = joblib.load(path)
        return artifact["model"], artifact["preprocessor"]


def main() -> None:
    """Entry point for ``churn-train`` CLI command."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Train the churn XGBoost model.")
    parser.add_argument("--data-path", required=True, help="Path to churn CSV.")
    parser.add_argument(
        "--model-out",
        default="models/xgb_churn.joblib",
        help="Output path for the trained model artifact.",
    )
    args = parser.parse_args()

    loader = ChurnDataLoader(args.data_path)
    df = loader.load()

    trainer = ChurnModelTrainer()
    trainer.train(df)
    trainer.save(args.model_out)
    print(f"Model saved to {args.model_out}")


if __name__ == "__main__":
    main()
