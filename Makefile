# Variable declaration
PYTHON=python3
ENV_NAME=.make-env
REQUIREMENTS=requirements.txt
SRC_DIR=src
TEST_DIR=tests

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
