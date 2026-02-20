"""Derives new features from the raw data."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Adds domain-specific features to the raw DataFrame.

    Call ``transform`` before passing data to ``ChurnPreprocessor``.
    All new features are added on a copy of the input frame.
    """

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps.

        Parameters
        ----------
        df:
            Raw churn DataFrame as returned by ``ChurnDataLoader.load``.

        Returns
        -------
        pd.DataFrame
            DataFrame with additional engineered columns.
        """
        df = df.copy()
        df = self._add_charges_per_month(df)
        df = self._add_tenure_buckets(df)
        df = self._add_service_count(df)
        logger.info("Feature engineering complete. Shape: %s", df.shape)
        return df

    def _add_charges_per_month(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ratio of total charges to tenure; proxy for customer lifetime value."""
        df["charges_per_month"] = np.where(
            df["tenure_months"] > 0,
            df["total_charges"] / df["tenure_months"],
            df["monthly_charges"],
        )
        return df

    def _add_tenure_buckets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ordinal bucketing of tenure into loyalty tiers.

        0 - new        : 0-6 months
        1 - growing    : 7-24 months
        2 - established: 25-48 months
        3 — loyal      : 49+ months
        """
        df["tenure_bucket"] = pd.cut(
            df["tenure_months"],
            bins=[-1, 6, 24, 48, float("inf")],
            labels=[0, 1, 2, 3],
        ).astype(int)
        return df

    def _add_service_count(self, df: pd.DataFrame) -> pd.DataFrame:
        """Total number of active services (internet + phone)."""
        df["service_count"] = (
            df["has_internet_service"].astype(int)
            + df["has_phone_service"].astype(int)
        )
        return df
