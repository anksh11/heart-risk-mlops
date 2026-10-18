#!/usr/bin/env bash
# End-to-end check of a running API instance. Exits non-zero on any failure.
set -euo pipefail
BASE_URL=${BASE_URL:-http://localhost:8000}
cd "$(dirname "$0")/.."

echo "== GET $BASE_URL/health";       curl -fsS "$BASE_URL/health"; echo
echo "== GET $BASE_URL/model-info";   curl -fsS "$BASE_URL/model-info" | head -c 400; echo
for f in samples/patient_high_risk.json samples/patient_low_risk.json; do
  echo "== POST /predict ($f)"
  curl -fsS -X POST "$BASE_URL/predict" -H 'content-type: application/json' -d @"$f"; echo
done
echo "== POST /predict/batch";  curl -fsS -X POST "$BASE_URL/predict/batch" -H 'content-type: application/json' -d @samples/batch.json; echo
echo "== POST /predict (invalid, expect 422)"
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE_URL/predict" -H 'content-type: application/json' -d '{"age": -1}')
[ "$code" = "422" ] && echo "422 OK" || { echo "expected 422, got $code"; exit 1; }
echo "== GET /metrics (excerpt)"; curl -fsS "$BASE_URL/metrics" | grep -E '^heart_api_(predictions_total|model_loaded)' | head
