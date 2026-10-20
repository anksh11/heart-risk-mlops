# Screenshots

| File | Shows | Status |
|---|---|---|
| `mlflow_experiment_runs.png` | All candidate and final runs in the `heart-disease-risk` experiment | ✅ captured |
| `mlflow_best_run_metrics.png` | Params and metrics of the selected run | ✅ captured |
| `mlflow_run_artifacts.png` | Logged artefacts (plots, tuning CSV, model and signature) | ✅ captured |
| `mlflow_model_registry.png` | `heart-risk-classifier` registered versions | ✅ captured |
| `api_swagger_docs.png` | FastAPI Swagger UI | ✅ captured |
| `local_smoke_test.txt` | `scripts/smoke_test.sh` output against the local API | ✅ captured |
| `pytest_output.txt`, `ruff_output.txt` | Test and lint results | ✅ captured |
| `docker_build_run.png` | `docker build` + `docker run` + `curl /predict` | ⬜ to capture (terminal proof: `docker_build_run.txt` ✅) |
| `k8s_pods_services.png` | `kubectl -n heart-risk get pods,svc,ingress,hpa` | ⬜ to capture (terminal proof: `minikube_deploy_and_smoke.txt` ✅) |
| `k8s_smoke_test.png` | `BASE_URL=http://localhost:8080 make smoke` | ⬜ to capture (terminal proof: `minikube_deploy_and_smoke.txt` ✅) |
| `prometheus_targets.png` | Prometheus *Status → Targets*, both API pods UP | ⬜ to capture |
| `grafana_dashboard.png` | Grafana **Heart Risk API** dashboard after `make load` | ⬜ to capture |
| `github_actions_run.png` | Green workflow run with all jobs | ⬜ to capture |
| `github_actions_artifacts.png` | Artefacts list and job summary of that run | ⬜ to capture |
