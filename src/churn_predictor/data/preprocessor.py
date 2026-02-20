"""Preprocessing pipeline: imputation, encoding, scaling."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

NUMERIC_FEATURES: list[str] = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "num_products",
]

CATEGORICAL_FEATURES: list[str] = [
    "contract_type",
]

BOOLEAN_FEATURES: list[str] = [
    "has_internet_service",
    "has_phone_service",
]

TARGET_COLUMN = "churn"
DROP_COLUMNS = ["customer_id"]


class ChurnPreprocessor:
    """Fits a scikit-learn ColumnTransformer and transforms DataFrames.

    The transformer is stateful: call ``fit_transform`` on training data,
    then ``transform`` on validation/test/inference data.
    """

    def __init__(self) -> None:
        self._transformer = self._build_transformer()
        self._is_fitted = False

    def fit_transform(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Fit on training data and return (X, y).

        Parameters
        ----------
        df:
            Raw DataFrame including the target column.

        Returns
        -------
        X:
            Transformed feature matrix as a dense DataFrame.
        y:
            Binary churn target Series (int).
        """
        X_raw, y = self._split_xy(df)
        X_transformed = self._transformer.fit_transform(X_raw)
        self._is_fitted = True
        logger.info("Preprocessor fitted on %d rows.", len(df))
        return self._to_dataframe(X_transformed), y

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted transformer to new data (no target column needed).

        Parameters
        ----------
        df:
            Raw feature DataFrame (target column optional; ignored if present).
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Preprocessor is not fitted yet. Call fit_transform first."
            )
        df_clean = df.drop(columns=[TARGET_COLUMN], errors="ignore")
        df_clean = df_clean.drop(columns=DROP_COLUMNS, errors="ignore")
        X_transformed = self._transformer.transform(df_clean)
        return self._to_dataframe(X_transformed)

    def _split_xy(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        y = df[TARGET_COLUMN].astype(int)
        X = df.drop(columns=[TARGET_COLUMN, *DROP_COLUMNS], errors="ignore")
        return X, y

    def _build_transformer(self) -> ColumnTransformer:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, NUMERIC_FEATURES),
                ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
                ("bool", "passthrough", BOOLEAN_FEATURES),
            ],
            remainder="drop",
        )

    def _to_dataframe(self, X: object) -> pd.DataFrame:
        """Convert transformer output array back to a named DataFrame."""
        feature_names = self._transformer.get_feature_names_out()
        return pd.DataFrame(
            np.array(X),
            columns=feature_names,
        )
