#!/usr/bin/env bash
# Build the image and deploy API + Prometheus + Grafana to the current kube-context.
# Works with Colima (k3s), Docker Desktop, minikube and kind.
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE=${IMAGE:-heart-risk-api:1.0.0}
CTX=$(kubectl config current-context)
echo "Deploying to context: $CTX"

docker build -t "$IMAGE" .
case "$CTX" in
  minikube) minikube image load "$IMAGE" ;;
  kind-*)   kind load docker-image "$IMAGE" --name "${CTX#kind-}" ;;
  *)        : ;;  # colima / docker-desktop share the docker daemon with the cluster
esac

kubectl apply -k k8s
kubectl -n heart-risk rollout status deploy/heart-risk-api --timeout=180s
kubectl -n heart-risk rollout status deploy/prometheus --timeout=180s
kubectl -n heart-risk rollout status deploy/grafana --timeout=180s
kubectl -n heart-risk get pods,svc,ingress,hpa -o wide
echo
echo "API:        http://localhost:8080/docs"
echo "Prometheus: http://localhost:9090"
echo "Grafana:    http://localhost:3000   (dashboard: Heart Risk API)"
