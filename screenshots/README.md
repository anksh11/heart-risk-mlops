# Screenshots

| File | Shows | Status |
|---|---|---|
| `mlflow_experiment_runs.png` | All candidate and final runs in the `heart-disease-risk` experiment | ✅ captured |
| `mlflow_best_run_metrics.png` | Params and metrics of the selected run | ✅ captured |
| `mlflow_run_artifacts.png` | Logged artefacts (plots, tuning CSV, model and signature) | ✅ captured |
| `mlflow_model_registry.png` | `heart-risk-classifier` registered versions | ✅ captured |
| `api_swagger_docs.png` | FastAPI Swagger UI (local run) | ✅ captured |
| `github_actions_run.png` | Green workflow run with all jobs and the training job summary | ✅ captured |
| `github_actions_artifacts.png` | Artefacts list of that run | ✅ captured |
| `docker_build_run.png` | `make docker-build` + `docker run` + `curl /predict` on the lab server | ✅ captured |
| `k8s_pods_services.png` | `make k8s-deploy` on minikube: resources, rollouts, pods, services, ingress | ✅ captured |
| `k8s_smoke_load_test.png` | Smoke test against the LoadBalancer, then repeated load tests | ✅ captured |
| `api_server.png` | Swagger UI served from the cluster through the LoadBalancer | ✅ captured |
| `pods_autoscaler.png` | `kubectl get hpa -w`: CPU 9 % → 113 %, replicas 2 → 4 | ✅ captured |
| `prometheus_targets.png` | Prometheus scraping both API pods (`heart_api_predictions_total` per pod) | ✅ captured |
| `grafana_dashboard.png` | Grafana **Heart Risk API** dashboard after the load tests | ✅ captured |
| `local_smoke_test.txt`, `docker_build_run.txt`, `minikube_deploy_and_smoke.txt` | Terminal transcripts of the same checks | ✅ captured |
| `pytest_output.txt`, `ruff_output.txt` | Test and lint results | ✅ captured |
