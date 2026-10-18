import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(trained_model, tmp_path, monkeypatch):
    import joblib

    model_path = tmp_path / "model.joblib"
    joblib.dump(trained_model, model_path)
    (tmp_path / "meta.json").write_text('{"model_name": "test_lr", "model_version": "0.0.1"}')

    import api.main as main

    monkeypatch.setattr(main, "MODEL_PATH", model_path)
    monkeypatch.setattr(main, "META_PATH", tmp_path / "meta.json")
    with TestClient(main.app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True
    assert r.json()["model_name"] == "test_lr"


def test_predict_returns_prediction_and_confidence(client, sample_patient):
    r = client.post("/predict", json=sample_patient)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1
    assert 0 <= body["probability_disease"] <= 1
    assert body["request_id"] == r.headers["x-request-id"]


def test_predict_propagates_request_id(client, sample_patient):
    r = client.post("/predict", json=sample_patient, headers={"x-request-id": "abc123"})
    assert r.json()["request_id"] == "abc123"


@pytest.mark.parametrize("field,value", [("age", -5), ("cp", 9), ("chol", "high"), ("unknown", 1)])
def test_predict_rejects_invalid_input(client, sample_patient, field, value):
    r = client.post("/predict", json={**sample_patient, field: value})
    assert r.status_code == 422


def test_predict_rejects_missing_required(client, sample_patient):
    sample_patient.pop("age")
    assert client.post("/predict", json=sample_patient).status_code == 422


def test_batch(client, sample_patient):
    r = client.post("/predict/batch", json={"patients": [sample_patient, sample_patient]})
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 2


def test_metrics_exposed_after_prediction(client, sample_patient):
    client.post("/predict", json=sample_patient)
    text = client.get("/metrics").text
    assert "heart_api_requests_total" in text
    assert "heart_api_predictions_total" in text
    assert "heart_api_model_loaded 1.0" in text


def test_service_degraded_without_model(tmp_path, monkeypatch, sample_patient):
    import api.main as main

    monkeypatch.setattr(main, "MODEL_PATH", tmp_path / "missing.joblib")
    main.load_model.cache_clear()
    with TestClient(main.app) as c:
        assert c.get("/health").json()["status"] == "degraded"
        assert c.post("/predict", json=sample_patient).status_code == 503
