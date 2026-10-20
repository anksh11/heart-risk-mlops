# Heart-Disease Risk Classifier — End-to-End MLOps

BITS Pilani · AIMLCZG523 MLOps · Assignment 1

The project trains a classifier on the UCI Heart Disease (Cleveland) data. It tracks every experiment in MLflow, packages the full preprocessing and model pipeline as one artefact, and serves it with FastAPI in a hardened container. The service runs on Kubernetes, with Prometheus and Grafana for monitoring. GitHub Actions lints, tests, trains, quality-gates and smoke-tests the container on every push.

| | |
|---|---|
| **Repository** | [github.com/anksh11/heart-risk-mlops](https://github.com/anksh11/heart-risk-mlops) |
| **Report** | [`reports/MLOps_Assignment1_Report.docx`](reports/MLOps_Assignment1_Report.docx) (source: [`reports/REPORT.md`](reports/REPORT.md)) |
| **Architecture** | [`reports/figures/arch/architecture.png`](reports/figures/arch/architecture.png) |
| **Selected model** | Logistic Regression (L2, C=1, balanced) — CV ROC-AUC **0.906 ± 0.020**, hold-out ROC-AUC **0.957** |

## Repository layout

```
├── src/heart_risk/        # library code: config, data, features, train, eda, predict
├── api/                   # FastAPI app (main.py) + pydantic schemas
├── tests/                 # pytest unit tests: data, features, model, API (36 tests)
├── notebooks/             # 01_eda, 02_training, 03_inference (executed, with outputs)
├── data/raw/              # UCI files (fetched by the download script)
├── data/processed/        # heart_clean.csv – cleaned dataset
├── models/                # heart_risk_model.joblib, model_meta.json, mlflow_model/
├── reports/               # report, model_comparison.csv, figures/{eda,training,arch}
├── k8s/                   # Kustomize: namespace, deployment, service(LB), ingress, HPA, monitoring/
├── monitoring/            # Prometheus/Grafana config for the docker-compose stack
├── scripts/               # download_data.sh, smoke_test.sh, deploy_k8s.sh, load_test.py
├── screenshots/           # evidence for the report
├── .github/workflows/ci.yml
├── Dockerfile, docker-compose.yml, Makefile
├── requirements.txt       # full dev/training environment (pinned)
├── requirements-serve.txt # minimal runtime deps baked into the image
└── params.yaml            # data split, CV, MLflow and hyper-parameter grids
```

## 1. Setup

You need Python 3.11, Docker, and optionally a local Kubernetes cluster (Colima `--kubernetes`, Docker Desktop, minikube or kind).

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src:.          # or use the Makefile, which sets it
```

## 2. Run the pipeline

```bash
make data         # download UCI zip -> data/raw, clean -> data/processed/heart_clean.csv
make eda          # 7 figures -> reports/figures/eda/
make train        # grid-search LR / RF / GBM, log to ./mlruns, write models/
make test         # 36 unit tests + coverage
make lint         # ruff
make mlflow-ui    # http://localhost:5000
```

You can also run the notebooks in `notebooks/` in order. They call the same library functions.

**Manual data download:** get <https://archive.ics.uci.edu/dataset/45/heart+disease>, unzip the `processed.*.data` files into `data/raw/`, and run `python -m heart_risk.data`.

## 3. Serve locally

```bash
make serve                                   # uvicorn on :8000, docs at /docs
python -m heart_risk.predict samples/patient_high_risk.json   # CLI inference
```

### Docker

```bash
make docker-build                            # docker build -t heart-risk-api:1.0.0 .
make docker-run                              # -> http://localhost:8000
BASE_URL=http://localhost:8000 make smoke    # health, predict x2, batch, 422 check, metrics
```

**Behind a TLS-intercepting proxy** (Zscaler, etc.): put the proxy's root CA as `docker/certs/<name>.crt` before you build. The file is git-ignored. The proxy must also allow `files.pythonhosted.org`.

Example request:

```bash
curl -X POST localhost:8000/predict -H 'content-type: application/json' -d '{
  "age": 58, "sex": 1, "cp": 4, "trestbps": 140, "chol": 289, "fbs": 0, "restecg": 2,
  "thalach": 128, "exang": 1, "oldpeak": 2.4, "slope": 2, "ca": 2, "thal": 7}'
# {"prediction":1,"label":"heart disease","probability_disease":0.9972,"confidence":0.9972,
#  "risk_band":"high","model_name":"logistic_regression","model_version":"1.0.0","request_id":"…"}
```

| Endpoint | Purpose |
|---|---|
| `POST /predict` | One patient. Returns prediction, confidence, probability and risk band |
| `POST /predict/batch` | Up to 500 patients |
| `GET /health` | Liveness and readiness; reports `degraded` if no model is loaded |
| `GET /model-info` | Training metadata: chosen model, params, CV and hold-out metrics |
| `GET /metrics` | Prometheus metrics |
| `GET /docs` | Swagger UI |

## 4. Deploy to Kubernetes

```bash
kubectl config use-context <your local cluster>   # e.g. colima / docker-desktop / minikube
make k8s-deploy          # builds the image, kubectl apply -k k8s, waits for rollouts
BASE_URL=http://localhost:8080 make smoke
make load                # 2 minutes of traffic so the dashboards have data
```

| URL | Service |
|---|---|
| <http://localhost:8080/docs> | API, through the LoadBalancer Service |
| <http://heart-risk.local/docs> | API, through the Ingress (add `127.0.0.1 heart-risk.local` to `/etc/hosts`) |
| <http://localhost:9090> | Prometheus (*Status → Targets*, *Alerts*) |
| <http://localhost:3000> | Grafana, with the **Heart Risk API** dashboard as home (anonymous viewer; admin/admin) |

On minikube, run `minikube tunnel` so the LoadBalancer services get an IP.

### Linux host (one interactive script)

On a Linux server (Rocky/RHEL/Amazon Linux or Ubuntu/Debian), `scripts/run_linux.sh` walks through every step: tool checks and installs, venv, data/EDA/train/test/lint, Docker build and `/predict` check, minikube with the `ingress` and `metrics-server` addons, deployment, smoke and load tests.

```bash
chmod +x scripts/run_linux.sh
./scripts/run_linux.sh          # menu
./scripts/run_linux.sh all      # everything in order
./scripts/run_linux.sh access   # one step: prereqs python pipeline docker k8s access smoke load status stop
```

It exposes the API, Prometheus and Grafana on `localhost:8080/9090/3000` with background `kubectl port-forward`s. On Linux, `minikube tunnel` serves each LoadBalancer on its cluster IP, not on localhost. To view the UIs from your own PC, keep `ssh -L 8080:localhost:8080 -L 9090:localhost:9090 -L 3000:localhost:3000 user@server` open.

Without Kubernetes, run `docker compose up --build`. It starts the same API, Prometheus and Grafana stack on ports 8000, 9090 and 3000.

To tear down, run `make k8s-delete`.

## 5. CI/CD

`.github/workflows/ci.yml` runs on every push and PR:

`lint` (ruff) → `test` (pytest + coverage, JUnit) → `train` (download → EDA → train → quality gate → job summary) → `docker` (build → run → smoke test → save image), plus `k8s-manifests` (kustomize + kubeconform).

Any failure stops the run: a lint error, a failing test, a model below CV ROC-AUC 0.80, or a container that doesn't answer. Each job uploads its artefacts: test reports, model, `mlruns`, figures and the Docker image tarball.
