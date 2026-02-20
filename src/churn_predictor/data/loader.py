"""CSV data loading with schema validation."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: list[str] = [
    "customer_id",
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "num_products",
    "has_internet_service",
    "has_phone_service",
    "contract_type",
    "churn",
]


class DataValidationError(ValueError):
    """Raised when the loaded DataFrame fails schema validation."""


class ChurnDataLoader:
    """Loads and validates churn CSV data.

    Parameters
    ----------
    data_path:
        Path to the CSV file containing raw churn data.
    """

    def __init__(self, data_path: str | Path) -> None:
        self.data_path = Path(data_path)

    def load(self) -> pd.DataFrame:
        """Load CSV, validate schema, and return a clean DataFrame.

        Returns
        -------
        pd.DataFrame
            Validated dataframe with correct dtypes.

        Raises
        ------
        FileNotFoundError
            If ``data_path`` does not exist.
        DataValidationError
            If required columns are missing or target contains invalid values.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        logger.info("Loading data from %s", self.data_path)
        df = pd.read_csv(self.data_path)

        self._validate(df)
        df = self._cast_dtypes(df)

        logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
        return df

    def _validate(self, df: pd.DataFrame) -> None:
        missing = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise DataValidationError(
                f"Missing required columns: {sorted(missing)}"
            )

        invalid_targets = df["churn"][~df["churn"].isin([0, 1])].unique()
        if len(invalid_targets) > 0:
            raise DataValidationError(
                f"'churn' column contains invalid values: {invalid_targets}. "
                "Expected 0 or 1 only."
            )

    def _cast_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
        df["churn"] = df["churn"].astype(int)
        df["has_internet_service"] = df["has_internet_service"].astype(bool)
        df["has_phone_service"] = df["has_phone_service"].astype(bool)
        return df
