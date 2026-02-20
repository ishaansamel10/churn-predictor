# Churn Predictor

Customer churn prediction service using XGBoost, served via a FastAPI REST API.

## Stack

- **Model** — XGBoost classifier with scikit-learn preprocessing pipeline
- **API** — FastAPI + Pydantic v2, `/predict` and `/predict/batch` endpoints
- **Tooling** — ruff (lint/format), mypy strict, pytest-cov (95%+ coverage)
- **Python** — 3.11+, `src/` layout, `pyproject.toml` single config

---

## Quickstart

```bash
# 1. Create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install all dependencies
make dev

# 3. Train the model (requires data/churn.csv — see Data Format below)
make train

# 4. Start the API server
make serve
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

---

## Data Format

Place your training data at `data/churn.csv`. Required columns:

| Column | Type | Description |
|---|---|---|
| `customer_id` | str | Unique identifier (dropped before training) |
| `tenure_months` | int | Months as a customer |
| `monthly_charges` | float | Monthly bill amount (USD) |
| `total_charges` | float | Cumulative charges (USD) |
| `num_products` | int | Number of subscribed products |
| `has_internet_service` | bool | Internet service active |
| `has_phone_service` | bool | Phone service active |
| `contract_type` | str | `month-to-month`, `one_year`, or `two_year` |
| `churn` | int | Target: `0` (retained) or `1` (churned) |

---

## API

### `GET /api/v1/health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "version": "0.1.0"
}
```

### `POST /api/v1/predict`

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST-001",
    "tenure_months": 3,
    "monthly_charges": 85.50,
    "total_charges": 256.50,
    "num_products": 1,
    "has_internet_service": true,
    "has_phone_service": false,
    "contract_type": "month-to-month"
  }'
```

```json
{
  "customer_id": "CUST-001",
  "churn_probability": 0.7842,
  "churn_predicted": true,
  "threshold": 0.5
}
```

### `POST /api/v1/predict/batch`

Same schema, wrapped in `{"customers": [...]}`. Accepts up to 1000 customers per request.

---

## Project Structure

```
src/churn_predictor/
├── data/
│   ├── loader.py        # CSV ingestion + schema validation
│   └── preprocessor.py  # Stateful ColumnTransformer (impute/scale/encode)
├── features/
│   └── engineer.py      # Feature engineering (charges_per_month, tenure_bucket, service_count)
├── models/
│   ├── trainer.py       # XGBClassifier training + joblib artifact persistence
│   └── evaluator.py     # Metrics: accuracy, precision, recall, F1, ROC-AUC
└── api/
    ├── schemas.py        # Pydantic v2 request/response models
    ├── router.py         # Route definitions
    └── main.py           # App factory + lifespan model loading
```

---

## Development

```bash
make lint       # ruff check
make format     # ruff format + auto-fix
make typecheck  # mypy strict
make test       # pytest + coverage report
make test-fast  # pytest -x, no coverage (fast feedback)
make clean      # remove caches and build artifacts
```

All three gates must pass before committing:

```bash
make lint && make typecheck && make test
```

---

## ML Pipeline

```
CSV → ChurnDataLoader → FeatureEngineer → ChurnPreprocessor → XGBClassifier
                                                    ↓
                              joblib artifact: {model, preprocessor}
```

The fitted `ChurnPreprocessor` is serialized alongside the model to guarantee identical transformations at training and inference time.
