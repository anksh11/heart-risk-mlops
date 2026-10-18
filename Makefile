PY      ?= python
IMAGE   ?= heart-risk-api:1.0.0
export PYTHONPATH := src:.

.PHONY: install data eda train train-quick test lint format docker-build docker-run smoke k8s-deploy k8s-delete load mlflow-ui serve

install:        ## create env from requirements.txt
	$(PY) -m pip install -r requirements.txt
data:           ## download + clean UCI data
	$(PY) -m heart_risk.data
eda:            ## regenerate EDA figures
	$(PY) -m heart_risk.eda
train:          ## full grid search, MLflow tracking, write models/
	$(PY) -m heart_risk.train
train-quick:
	$(PY) -m heart_risk.train --quick
test:
	$(PY) -m pytest --cov=src --cov=api --cov-report=term-missing
lint:
	ruff check . && ruff format --check .
format:
	ruff format . && ruff check --fix .
serve:          ## run the API without Docker
	uvicorn api.main:app --reload --port 8000
mlflow-ui:
	mlflow ui --backend-store-uri ./mlruns --port 5000
docker-build:
	docker build -t $(IMAGE) .
docker-run:
	docker run --rm -p 8000:8000 --name heart-risk-api $(IMAGE)
smoke:          ## hit a running API (BASE_URL=http://localhost:8000)
	./scripts/smoke_test.sh
k8s-deploy:
	./scripts/deploy_k8s.sh
k8s-delete:
	kubectl delete -k k8s --ignore-not-found
load:           ## generate traffic for the Grafana dashboard
	$(PY) scripts/load_test.py --url $${BASE_URL:-http://localhost:8080} --seconds 120
