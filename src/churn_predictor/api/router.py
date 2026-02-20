"""FastAPI route definitions for churn prediction endpoints."""

from __future__ import annotations

import logging
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status

from churn_predictor import __version__
from churn_predictor.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerFeatures,
    HealthResponse,
    PredictionResponse,
)
from churn_predictor.features.engineer import FeatureEngineer

logger = logging.getLogger(__name__)

router = APIRouter()


def get_model_artifact() -> tuple[object, object]:
    """Dependency placeholder — overridden in main.py via dependency_overrides."""
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Model not loaded. Start the server with a valid --model-path.",
    )


ModelArtifact = Annotated[tuple[object, object], Depends(get_model_artifact)]


@router.get("/health", response_model=HealthResponse, tags=["Ops"])
def health(artifact: ModelArtifact) -> HealthResponse:
    """Return service health and model load status."""
    model, _ = artifact
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        version=__version__,
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Prediction"],
)
def predict_single(
    customer: CustomerFeatures,
    artifact: ModelArtifact,
) -> PredictionResponse:
    """Predict churn probability for a single customer."""
    model, preprocessor = artifact
    return _run_prediction(customer, model, preprocessor)


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Prediction"],
)
def predict_batch(
    request: BatchPredictionRequest,
    artifact: ModelArtifact,
) -> BatchPredictionResponse:
    """Predict churn probability for up to 1 000 customers."""
    model, preprocessor = artifact
    predictions = [
        _run_prediction(customer, model, preprocessor)
        for customer in request.customers
    ]
    return BatchPredictionResponse(predictions=predictions, count=len(predictions))


def _run_prediction(
    customer: CustomerFeatures,
    model: object,
    preprocessor: object,
) -> PredictionResponse:
    from xgboost import XGBClassifier

    from churn_predictor.data.preprocessor import ChurnPreprocessor

    assert isinstance(model, XGBClassifier)
    assert isinstance(preprocessor, ChurnPreprocessor)

    engineer = FeatureEngineer()
    row = pd.DataFrame(
        [
            {
                "customer_id": customer.customer_id,
                "tenure_months": customer.tenure_months,
                "monthly_charges": customer.monthly_charges,
                "total_charges": customer.total_charges,
                "num_products": customer.num_products,
                "has_internet_service": customer.has_internet_service,
                "has_phone_service": customer.has_phone_service,
                "contract_type": customer.contract_type.value,
            }
        ]
    )

    row_engineered = engineer.transform(row)
    X = preprocessor.transform(row_engineered)

    proba = float(model.predict_proba(X)[0, 1])
    threshold = 0.5

    return PredictionResponse(
        customer_id=customer.customer_id,
        churn_probability=round(proba, 4),
        churn_predicted=proba >= threshold,
        threshold=threshold,
    )
