"""Model evaluation metrics and reporting."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


@dataclass
class EvaluationReport:
    """Container for classification metrics."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: list[list[int]] = field(default_factory=list)
    classification_report: str = ""

    def log(self) -> None:
        """Emit all metrics to the logger."""
        logger.info("--- Evaluation Report ---")
        logger.info("Accuracy : %.4f", self.accuracy)
        logger.info("Precision: %.4f", self.precision)
        logger.info("Recall   : %.4f", self.recall)
        logger.info("F1 Score : %.4f", self.f1)
        logger.info("ROC-AUC  : %.4f", self.roc_auc)
        logger.info("\n%s", self.classification_report)


class ModelEvaluator:
    """Evaluates a fitted XGBClassifier on held-out data.

    Parameters
    ----------
    model:
        A fitted ``XGBClassifier``.
    threshold:
        Decision threshold for the positive class. Defaults to 0.5.
    """

    def __init__(self, model: XGBClassifier, threshold: float = 0.5) -> None:
        self.model = model
        self.threshold = threshold

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> EvaluationReport:
        """Compute classification metrics.

        Parameters
        ----------
        X:
            Feature matrix (preprocessed).
        y:
            Ground-truth binary labels.

        Returns
        -------
        EvaluationReport
            Populated metrics dataclass.
        """
        proba: np.ndarray = self.model.predict_proba(X)[:, 1]
        preds = (proba >= self.threshold).astype(int)

        report = EvaluationReport(
            accuracy=float(accuracy_score(y, preds)),
            precision=float(precision_score(y, preds, zero_division=0)),
            recall=float(recall_score(y, preds, zero_division=0)),
            f1=float(f1_score(y, preds, zero_division=0)),
            roc_auc=float(roc_auc_score(y, proba)),
            confusion_matrix=confusion_matrix(y, preds).tolist(),
            classification_report=classification_report(y, preds, zero_division=0),
        )
        report.log()
        return report

    def feature_importances(self, feature_names: list[str]) -> pd.DataFrame:
        """Return a sorted DataFrame of feature importances.

        Parameters
        ----------
        feature_names:
            Column names matching the training feature matrix.
        """
        importances = self.model.feature_importances_
        return (
            pd.DataFrame(
                {"feature": feature_names, "importance": importances}
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
