PYTHON  := .venv/bin/python
PIP     := .venv/bin/pip
RUFF    := .venv/bin/ruff
MYPY    := .venv/bin/mypy
PYTEST  := .venv/bin/pytest
UVICORN := .venv/bin/uvicorn

MODEL_DIR := models
DATA_DIR  := data

.PHONY: install dev lint format typecheck test test-fast train serve clean help

install:
	$(PIP) install --upgrade pip
	$(PIP) install -e .

dev:
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	@echo "Dev environment ready. Activate with: source .venv/bin/activate"

lint:
	$(RUFF) check src/ tests/

format:
	$(RUFF) format src/ tests/
	$(RUFF) check --fix src/ tests/

typecheck:
	$(MYPY) src/churn_predictor/

test:
	$(PYTEST) tests/

test-fast:
	$(PYTEST) tests/ -x --no-cov

train:
	@mkdir -p $(MODEL_DIR)
	$(PYTHON) -m churn_predictor.models.trainer \
		--data-path $(DATA_DIR)/churn.csv \
		--model-out $(MODEL_DIR)/xgb_churn.joblib

serve:
	$(UVICORN) churn_predictor.api.main:app \
		--host 0.0.0.0 \
		--port 8000 \
		--reload

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov"     -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".coverage" -delete 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info/

help:
	@echo "Available targets:"
	@echo "  install    - Install runtime dependencies into .venv"
	@echo "  dev        - Install all dependencies including dev tools"
	@echo "  lint       - Run ruff linter"
	@echo "  format     - Run ruff formatter + auto-fix"
	@echo "  typecheck  - Run mypy type checker"
	@echo "  test       - Run full test suite with coverage"
	@echo "  test-fast  - Run tests, stop on first failure, no coverage"
	@echo "  train      - Train XGBoost model (requires DATA_DIR/churn.csv)"
	@echo "  serve      - Start FastAPI dev server on :8000"
	@echo "  clean      - Remove all generated cache/build artifacts"
