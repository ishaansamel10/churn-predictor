"""Pydantic v2 request/response schemas."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class ContractType(StrEnum):
    MONTH_TO_MONTH = "month-to-month"
    ONE_YEAR = "one_year"
    TWO_YEAR = "two_year"


class CustomerFeatures(BaseModel):
    """Input schema for a single customer churn prediction request."""

    customer_id: str = Field(..., description="Unique customer identifier.")
    tenure_months: int = Field(..., ge=0, le=600, description="Months as a customer.")
    monthly_charges: float = Field(..., gt=0, description="Monthly bill amount (USD).")
    total_charges: float = Field(..., ge=0, description="Cumulative charges (USD).")
    num_products: int = Field(
        ..., ge=1, le=20, description="Number of subscribed products."
    )
    has_internet_service: bool = Field(
        ..., description="Whether customer has internet service."
    )
    has_phone_service: bool = Field(
        ..., description="Whether customer has phone service."
    )
    contract_type: ContractType = Field(..., description="Billing contract type.")

    @model_validator(mode="after")
    def validate_total_vs_monthly(self) -> CustomerFeatures:
        if self.tenure_months > 1 and self.total_charges < self.monthly_charges:
            raise ValueError(
                "total_charges cannot be less than monthly_charges "
                "when tenure_months > 1."
            )
        return self

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "customer_id": "CUST-0001",
                    "tenure_months": 24,
                    "monthly_charges": 65.50,
                    "total_charges": 1572.0,
                    "num_products": 3,
                    "has_internet_service": True,
                    "has_phone_service": True,
                    "contract_type": "month-to-month",
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Output schema for a churn prediction."""

    customer_id: str
    churn_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Probability of churn in [0, 1]."
    )
    churn_predicted: bool = Field(
        ..., description="True if churn_probability >= threshold."
    )
    threshold: float = Field(default=0.5, description="Decision threshold used.")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request body."""

    customers: list[CustomerFeatures] = Field(..., min_length=1, max_length=1000)


class BatchPredictionResponse(BaseModel):
    """Batch prediction response body."""

    predictions: list[PredictionResponse]
    count: int

    @model_validator(mode="after")
    def set_count(self) -> BatchPredictionResponse:
        self.count = len(self.predictions)
        return self


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    model_loaded: bool
    version: str
