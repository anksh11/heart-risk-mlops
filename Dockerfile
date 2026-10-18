# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Stage 1: install serving dependencies into an isolated venv.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder
WORKDIR /build
# Optional extra CA bundle for building behind a TLS-intercepting proxy.
# docker/certs/ is empty in the repo (and in CI); drop a *.crt there locally if needed.
COPY docker/certs/ /tmp/certs/
COPY requirements-serve.txt .
RUN if ls /tmp/certs/*.crt >/dev/null 2>&1; then cat /tmp/certs/*.crt > /tmp/ca.pem; export PIP_CERT=/tmp/ca.pem; fi; \
    python -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir -r requirements-serve.txt

# ---------------------------------------------------------------------------
# Stage 2: slim runtime, non-root, model baked in.
# ---------------------------------------------------------------------------
FROM python:3.11-slim
LABEL org.opencontainers.image.title="heart-risk-api" \
      org.opencontainers.image.description="Heart disease risk classifier (FastAPI)"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app \
    HEART_RISK_ROOT=/app \
    PATH=/opt/venv/bin:$PATH \
    PORT=8000

RUN useradd --create-home --uid 10001 app
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY src/ src/
COPY api/ api/
COPY models/heart_risk_model.joblib models/model_meta.json models/

USER 10001
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --workers ${WORKERS:-1} --no-access-log"]
