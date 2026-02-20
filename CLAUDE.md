# CLAUDE.md — Churn Predictor

## Project Overview

Customer churn prediction service. XGBoost classifier trained on customer features, served via a FastAPI REST API. Python 3.12, `src/` layout, single `pyproject.toml` config.

---

## Quick Commands

```bash
make dev        # install all deps (runtime + dev) into .venv
make lint       # ruff check — must be 0 violations
make format     # ruff format + auto-fix
make typecheck  # mypy strict — must be 0 errors
make test       # pytest + coverage (≥80% enforced, currently ~96%)
make test-fast  # pytest -x --no-cov (fast feedback loop)
make train      # train model: requires data/churn.csv → models/xgb_churn.joblib
make serve      # uvicorn on :8000 (returns 503 on /predict until trained)
make clean      # remove __pycache__, .mypy_cache, htmlcov, etc.
```

All commands use `.venv/bin/` directly — no need to activate the venv.

---

## Project Structure

```
churn-predictor/
├── pyproject.toml                  # single config: packaging, ruff, mypy, pytest
├── Makefile
├── src/
│   └── churn_predictor/
│       ├── __init__.py             # version = "0.1.0"
│       ├── data/
│       │   ├── loader.py           # ChurnDataLoader — CSV load + schema validation
│       │   └── preprocessor.py    # ChurnPreprocessor — stateful ColumnTransformer
│       ├── features/
│       │   └── engineer.py        # FeatureEngineer — charges_per_month, tenure_bucket, service_count
│       ├── models/
│       │   ├── trainer.py         # ChurnModelTrainer — train + joblib save/load
│       │   └── evaluator.py       # ModelEvaluator — metrics, EvaluationReport dataclass
│       └── api/
│           ├── schemas.py         # Pydantic v2 request/response models
│           ├── router.py          # GET /health, POST /predict, POST /predict/batch
│           └── main.py            # create_app() factory + lifespan model loading
└── tests/                         # mirrors src/churn_predictor/ exactly
    ├── conftest.py                 # shared fixtures: raw_churn_df, trained_artifact, api_client
    ├── data/
    ├── features/
    ├── models/
    └── api/
```

---

## Data Contract

The raw CSV (default: `data/churn.csv`) must contain these columns:

| Column | Type | Notes |
|---|---|---|
| `customer_id` | str | dropped before training |
| `tenure_months` | int | months as customer |
| `monthly_charges` | float | |
| `total_charges` | float | coerced via `pd.to_numeric` |
| `num_products` | int | |
| `has_internet_service` | bool | |
| `has_phone_service` | bool | |
| `contract_type` | str | `"month-to-month"`, `"one_year"`, `"two_year"` |
| `churn` | int | binary: 0 or 1 only |

`ChurnDataLoader` raises `DataValidationError` (subclass of `ValueError`) for missing columns or invalid target values.

---

## ML Pipeline Order

```
ChurnDataLoader.load()
    → FeatureEngineer.transform()       # adds charges_per_month, tenure_bucket, service_count
    → ChurnPreprocessor.fit_transform() # impute + scale numerics, OHE categoricals, passthrough booleans
    → train_test_split (stratified if dataset large enough, else non-stratified)
    → XGBClassifier.fit(eval_set=...)
    → joblib.dump({"model": ..., "preprocessor": ...})
```

**Inference path** (API):
```
CustomerFeatures (Pydantic) → FeatureEngineer.transform() → ChurnPreprocessor.transform() → model.predict_proba()
```

---

## Key Architectural Decisions

### Preprocessor stored alongside model
`ChurnPreprocessor` is serialized together with `XGBClassifier` in a single joblib artifact (`{"model": ..., "preprocessor": ...}`). This guarantees that inference-time transformations are byte-for-byte identical to training-time ones. Do not change this to a sklearn `Pipeline` — XGBoost early stopping requires a separate eval set, which breaks `Pipeline.fit`.

### FeatureEngineer is stateless, Preprocessor is stateful
`FeatureEngineer.transform()` is a pure function (no fit step). `ChurnPreprocessor` is stateful and must be fitted before calling `.transform()`. Always run `FeatureEngineer` first, then `ChurnPreprocessor` — the engineer operates on raw semantic columns; the preprocessor operates on the full feature set including engineered ones.

### FastAPI lifespan for model loading
The model is loaded once at startup via `@asynccontextmanager lifespan` into a module-level `_model_cache` dict in `main.py`. The `get_model_artifact` dependency in `router.py` is a placeholder — the real implementation is registered via `app.dependency_overrides[get_model_artifact]` inside `create_app()`. This pattern allows tests to inject a small model without touching app state.

### API returns 503 until a model is trained
If `models/xgb_churn.joblib` doesn't exist at startup, all `/predict` endpoints return `503 Service Unavailable`. This is intentional — run `make train` first.

---

## Coding Standards

### Toolchain
- **Formatter/linter**: `ruff` (replaces black + isort + flake8). Config in `pyproject.toml` under `[tool.ruff]`. Run `make format` to auto-fix, `make lint` to check.
- **Type checker**: `mypy` with `strict = true`. All source files must pass with zero errors.
- **Tests**: `pytest` with `pytest-cov`. Coverage threshold is 80% (currently ~96%). `filterwarnings = ["error"]` — all warnings are treated as errors.

### Typing
- All functions require type annotations (enforced by mypy `strict` + ruff `ANN` rules).
- Use `from __future__ import annotations` at the top of every source file for deferred evaluation.
- Uppercase `X` for feature matrices is suppressed with `# noqa: N806` — this is established ML convention.
- xgboost, sklearn, and joblib are in `[[tool.mypy.overrides]] ignore_missing_imports = true`.

### Imports
- isort profile enforced by ruff. Order: stdlib → third-party → first-party (`churn_predictor`).
- Lazy imports inside functions are allowed only in `router.py::_run_prediction` and `main.py::main` to avoid circular imports or heavy startup costs.

### Pydantic schemas
- All schemas use Pydantic v2 (`model_validator`, `field_validator`, `model_config`).
- `ContractType` inherits from `StrEnum` (not `str, Enum`) — ruff `UP042` enforced.
- Cross-field validation lives in `@model_validator(mode="after")`.

---

## Testing Patterns

### Fixtures (tests/conftest.py)
| Fixture | Scope | Description |
|---|---|---|
| `raw_churn_df` | function | 5-row in-memory DataFrame |
| `raw_churn_csv` | function | writes `raw_churn_df` to `tmp_path/churn.csv` |
| `trained_artifact` | function | 5-estimator XGBClassifier + fitted preprocessor |
| `saved_model_path` | function | joblib artifact in `tmp_path` |
| `api_client` | function | FastAPI `TestClient` with injected test model |

### API testing
`api_client` must be used as a pytest fixture (not instantiated manually). It uses `with TestClient(app) as client: yield client` — the `with` block is required to trigger the FastAPI lifespan and populate `_model_cache`. Skipping the context manager will result in 503 on all endpoints.

### Small dataset edge cases
The 5-row fixture is intentionally minimal. `ChurnModelTrainer.train()` gracefully falls back to non-stratified `train_test_split` when the eval set is too small for stratification (< 2 samples per class). Tests that call `train()` directly should pass small `n_estimators` (e.g. 3–5) and not rely on `best_iteration`.

---

## API Endpoints

Base prefix: `/api/v1`

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Returns `{"status": "ok", "model_loaded": bool, "version": str}` |
| `POST` | `/predict` | Single customer → `PredictionResponse` |
| `POST` | `/predict/batch` | Up to 1000 customers → `BatchPredictionResponse` |

Docs available at `/docs` (Swagger) and `/redoc` when the server is running.

---

## Guardrails

- **Do not commit `data/` or `models/`** — both are gitignored. CSV data and trained model artifacts belong in external stores (S3, MLflow, DVC).
- **Do not commit `.env` files** — gitignored. Use `.env.example` for documentation.
- **Do not use `sklearn.Pipeline` to wrap XGBoost** — early stopping requires a direct `eval_set`, which `Pipeline.fit` cannot pass through.
- **Do not call `ChurnPreprocessor.transform()` before `fit_transform()`** — it raises `RuntimeError`. The fitted preprocessor must come from the same training run that produced the model.
- **Do not change `filterwarnings = ["error"]` in `pyproject.toml`** — all sklearn/xgboost warnings must be suppressed at the source (e.g. `zero_division=0`), not silenced globally.
- **Keep `make lint`, `make typecheck`, and `make test` all passing at zero errors** before committing.
