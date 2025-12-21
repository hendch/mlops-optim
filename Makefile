# Variable declaration
PYTHON=python3
ENV_NAME=.make-env
REQUIREMENTS=requirements.txt
SRC_DIR=src
TEST_DIR=tests
IMAGE_NAME=ml-docker-app
# --- MLflow configuration ---
MLFLOW_HOST ?= 127.0.0.1
MLFLOW_PORT ?= 5000

# If you're using SQLite backend store (recommended for consistency)
MLFLOW_BACKEND_URI ?= sqlite:///mlflow.db

# If you want artifacts in a dedicated folder (optional)
MLFLOW_ARTIFACT_ROOT ?= ./mlrunsCONTAINER_NAME=ml-app

# Environment configuration
setup:
	@echo "Creating the virtual environment and installing dependencies..."
	@$(PYTHON) -m venv $(ENV_NAME)
requirements:
	@$(ENV_NAME)/bin/pip install -r $(REQUIREMENTS)

# Code quality

.PHONY: test quality pylint flake8 bandit

quality: test pylint flake8 bandit
	@echo "✅ Code quality checks passed."

test:
	@echo "🌸Running unit tests with pytest.."
	@$(ENV_NAME)/bin/pytest $(TEST_DIR)

pylint:
	@echo "👀Running pylint (code quality)..."
	@$(ENV_NAME)/bin/pylint --exit-zero $(SRC_DIR)

flake8:
	@echo "😑Running flake8 (style/format checks)..."
	@$(ENV_NAME)/bin/flake8 $(SRC_DIR)

bandit:
	@echo "🫠Running bandit (security checks)..."
	@$(ENV_NAME)/bin/bandit -r $(SRC_DIR)

ruff:
	@echo "🦝 Running Ruff (fast linter)..."
	@$(ENV_NAME)/bin/ruff check src tests
ruff-fix:
	@echo "🪄 Auto-formatting code with Ruff..."
	@$(ENV_NAME)/bin/ruff check --fix src tests
	@$(ENV_NAME)/bin/ruff format src tests

#Data preparation
prepare:
	@echo "Data preparation.."
	@$(ENV_NAME)/bin/python -m src.prepare_data


#Train Model
train:
	@echo "Training..."
	@$(ENV_NAME)/bin/python -m src.train

#Evaluate model
evaluate:
	@echo "Evaluation of the model.."
	@$(ENV_NAME)/bin/python -m src.test_model

#DVC pipeline
.PHONY: dvc-repro

dvc-repro:
	@echo "📦 Running DVC pipeline (prepare → train → evaluate)..."
	@$(ENV_NAME)/bin/dvc repro

.PHONY: ci

#local CI
ci:
	@echo "🚀 Running local CI pipeline (tests + quality + DVC pipeline)..."
	@$(MAKE) test
	@$(MAKE) pylint
	@$(MAKE) flake8
	@$(MAKE) bandit
	@$(MAKE) dvc-repro
	@$(MAKE) api-test
	@echo "✅ CI pipeline finished successfully."
#webhook using ngrok
.PHONY: webhook

webhook:
	@echo "🌐 Starting webhook server on http://localhost:8000 ..."
	@$(ENV_NAME)/bin/uvicorn src.webhook:app --host 0.0.0.0 --port 8000

#fastAPI
api:
	@echo "🚀 Starting FastAPI server..."
	$(ENV_NAME)/bin/uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
api-test:
	@echo "🌐 Running API tests..."
	$(ENV_NAME)/bin/pytest tests/test_api.py
#docker
build:
	docker build -t $(IMAGE_NAME)
run:
	docker run -d -p 5000:5000 --name $(CONTAINER_NAME) $(IMAGE_NAME)
clean:
	docker stop $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME) || true

#MLFlow
.PHONY: mlflow-ui mlflow-ui-bg mlflow-clean mlflow-open train-mlflow

mlflow-ui:
	@echo "Starting MLflow UI at http://$(MLFLOW_HOST):$(MLFLOW_PORT)"
	mlflow ui \
		--backend-store-uri $(MLFLOW_BACKEND_URI) \
		--host $(MLFLOW_HOST) \
		--port $(MLFLOW_PORT)

# Background start (useful if you want the terminal back)
mlflow-ui-bg:
	@echo "Starting MLflow UI in background at http://$(MLFLOW_HOST):$(MLFLOW_PORT)"
	nohup mlflow ui \
		--backend-store-uri $(MLFLOW_BACKEND_URI) \
		--host $(MLFLOW_HOST) \
		--port $(MLFLOW_PORT) \
		> mlflow-ui.log 2>&1 & echo $$! > mlflow-ui.pid
	@echo "MLflow PID: $$(cat mlflow-ui.pid) (logs: mlflow-ui.log)"

mlflow-clean:
	rm -f mlflow.db mlflow.db-shm mlflow.db-wal
	rm -rf mlruns
	rm -f mlflow-ui.log mlflow-ui.pid

# Train and log to MLflow (uses your existing target)
train-mlflow: train
	@echo "Training completed. View MLflow at http://$(MLFLOW_HOST):$(MLFLOW_PORT)"

# Optional: one command to start UI then train (UI blocks; good in 2 terminals)
run-mlflow: mlflow-ui

mlflow-stop:
	@if [ -f mlflow-ui.pid ]; then kill $$(cat mlflow-ui.pid); fi

#elastic search
monitoring-up:
	docker compose -f docker-compose.monitoring.yml up -d

monitoring-down:
	docker compose -f docker-compose.monitoring.yml down
