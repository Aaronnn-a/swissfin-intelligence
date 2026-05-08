# SwissFin Intelligence — developer task runner
# Usage: `make <target>`. On Windows, run via WSL/Git-Bash or use the equivalent
# commands listed in README.md.

PYTHON      ?= python
VENV        ?= .venv
VENV_BIN    := $(VENV)/bin
PIP         := $(VENV_BIN)/pip
PY          := $(VENV_BIN)/python
STREAMLIT   := $(VENV_BIN)/streamlit

# Example earnings-call URL used by the `download` and `pipeline` targets.
# Override on the CLI: `make pipeline URL="https://..." NAME="ubs_q3_2025"`.
URL  ?= https://www.youtube.com/watch?v=dQw4w9WgXcQ
NAME ?= example_call

.DEFAULT_GOAL := help

.PHONY: help setup install test lint format check clean run-app \
        download transcribe sentiment pipeline

help:  ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "} {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: $(VENV)/.installed  ## Create venv and install all dependencies.

$(VENV)/.installed: requirements.txt requirements-dev.txt
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip wheel
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e .
	@touch $(VENV)/.installed

install: setup  ## Alias for `setup`.

test: setup  ## Run the test suite.
	$(PY) -m pytest --cov=swissfin --cov-report=term-missing

lint: setup  ## Static analysis (ruff).
	$(VENV_BIN)/ruff check src scripts app tests

format: setup  ## Auto-format with black + ruff --fix.
	$(VENV_BIN)/ruff check --fix src scripts app tests
	$(VENV_BIN)/black src scripts app tests

check: lint test  ## Lint + test (CI gate).

run-app: setup  ## Launch the Streamlit demo UI.
	$(STREAMLIT) run app/streamlit_app.py

download: setup  ## Download a single earnings call. Vars: URL, NAME.
	$(PY) scripts/download_call.py --url "$(URL)" --name "$(NAME)"

transcribe: setup  ## Transcribe a saved audio file. Var: NAME.
	$(PY) scripts/transcribe.py --name "$(NAME)"

sentiment: setup  ## Score sentiment on an existing transcript. Var: NAME.
	$(PY) scripts/analyze_sentiment.py --name "$(NAME)"

pipeline: setup  ## Run download → transcribe → sentiment end-to-end.
	$(PY) scripts/run_pipeline.py --url "$(URL)" --name "$(NAME)"

clean:  ## Remove caches and build artefacts.
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
