import json

import joblib
import numpy as np
import pytest

from heart_risk import config
from heart_risk.data import split_xy
from heart_risk.predict import predict_records, risk_band
from heart_risk.train import evaluate, feature_importance, make_estimator


def test_model_learns_signal(trained_model, clean_df):
    X, y = split_xy(clean_df)
    metrics = evaluate(trained_model, X, y)
    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert metrics["roc_auc"] > 0.8  # synthetic data has a strong cp/thalach signal


def test_probabilities_are_valid(trained_model, clean_df):
    X, _ = split_xy(clean_df)
    p = trained_model.predict_proba(X)
    assert p.shape == (len(X), 2)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)


def test_pipeline_roundtrips_through_joblib(trained_model, clean_df, tmp_path):
    X, _ = split_xy(clean_df)
    path = tmp_path / "m.joblib"
    joblib.dump(trained_model, path)
    np.testing.assert_allclose(joblib.load(path).predict_proba(X), trained_model.predict_proba(X))


@pytest.mark.parametrize("name", ["logistic_regression", "random_forest", "gradient_boosting"])
def test_make_estimator(name):
    est = make_estimator(name, 0)
    assert hasattr(est, "predict_proba")


def test_make_estimator_rejects_unknown():
    with pytest.raises(ValueError):
        make_estimator("svm", 0)


def test_feature_importance_covers_all_features(trained_model):
    imp = feature_importance(trained_model)
    assert len(imp) == len(trained_model.named_steps["preprocess"].get_feature_names_out())
    assert (imp >= 0).all()


def test_predict_records_handles_missing_optional(trained_model, sample_patient):
    patient = {k: v for k, v in sample_patient.items() if k not in ("ca", "thal")}
    (res,) = predict_records(trained_model, [patient])
    assert res["prediction"] in (0, 1)
    assert 0.5 <= res["confidence"] <= 1.0
    assert res["risk_band"] in {"low", "moderate", "high"}


@pytest.mark.parametrize("p,band", [(0.05, "low"), (0.29, "low"), (0.3, "moderate"), (0.59, "moderate"), (0.9, "high")])
def test_risk_band(p, band):
    assert risk_band(p) == band


@pytest.mark.skipif(not config.MODEL_FILE.exists(), reason="trained artefact not present")
def test_shipped_model_meets_quality_bar():
    """Gate: the artefact produced by the training step must stay above a minimum bar."""
    meta = json.loads(config.MODEL_META_FILE.read_text())
    assert meta["cv_metrics"]["cv_roc_auc_mean"] >= 0.80
    assert meta["features"] == config.INPUT_FEATURES
