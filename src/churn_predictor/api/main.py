"""FastAPI application factory."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from churn_predictor import __version__
from churn_predictor.api.router import get_model_artifact, router
from churn_predictor.models.trainer import ChurnModelTrainer

logger = logging.getLogger(__name__)

_model_cache: dict[str, object] = {}


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    """Load the model artifact on startup; release on shutdown."""
    model_path = Path(
        application.state.model_path
        if hasattr(application.state, "model_path")
        else "models/xgb_churn.joblib"
    )

    if model_path.exists():
        logger.info("Loading model from %s", model_path)
        model, preprocessor = ChurnModelTrainer.load(model_path)
        _model_cache["model"] = model
        _model_cache["preprocessor"] = preprocessor
        logger.info("Model loaded successfully.")
    else:
        logger.warning(
            "Model artifact not found at %s. "
            "/predict endpoints will return 503 until a model is trained.",
            model_path,
        )

    yield

    _model_cache.clear()
    logger.info("Model cache cleared on shutdown.")


def create_app(model_path: str | Path | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Parameters
    ----------
    model_path:
        Optional override for the model artifact path.
        Useful in tests to inject a temporary model file.
    """
    application = FastAPI(
        title="Churn Predictor API",
        description="Customer churn prediction powered by XGBoost.",
        version=__version__,
        lifespan=lifespan,
    )

    if model_path is not None:
        application.state.model_path = str(model_path)

    def _get_artifact() -> tuple[object, object]:
        if "model" not in _model_cache:
            from fastapi import HTTPException, status

            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded.",
            )
        return _model_cache["model"], _model_cache["preprocessor"]

    application.dependency_overrides[get_model_artifact] = _get_artifact
    application.include_router(router, prefix="/api/v1")

    return application


app = create_app()


def main() -> None:
    """Entry point for ``churn-serve`` CLI command."""
    import argparse

    import uvicorn

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Run the churn prediction API.")
    parser.add_argument("--host", default="0.0.0.0")  # noqa: S104
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-path", default="models/xgb_churn.joblib")
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    app.state.model_path = args.model_path
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
