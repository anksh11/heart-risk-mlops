"""Shared fixtures.

Tests never touch the network or the real data folder: a synthetic but
realistically-distributed frame stands in for the UCI file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from heart_risk import config


def _synthetic_raw(n: int = 160, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    disease = rng.integers(0, 2, n)
    df = pd.DataFrame(
        {
            "age": rng.integers(29, 78, n).astype(float),
            "sex": rng.integers(0, 2, n).astype(float),
            "cp": np.where(disease, 4, rng.choice([1, 2, 3], n)).astype(float),
            "trestbps": rng.normal(132, 17, n).round(),
            "chol": rng.normal(246, 50, n).round(),
            "fbs": rng.integers(0, 2, n).astype(float),
            "restecg": rng.choice([0, 1, 2], n).astype(float),
            "thalach": (rng.normal(160, 18, n) - 20 * disease).round(),
            "exang": (disease & rng.integers(0, 2, n)).astype(float),
            "oldpeak": np.abs(rng.normal(0.6 + disease, 0.8, n)).round(1),
            "slope": rng.choice([1, 2, 3], n).astype(float),
            "ca": rng.choice([0, 1, 2, 3], n).astype(float),
            "thal": rng.choice([3, 6, 7], n).astype(float),
            "num": disease * rng.integers(1, 5, n),
        }
    )
    return df


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return _synthetic_raw()


@pytest.fixture
def clean_df(raw_df) -> pd.DataFrame:
    from heart_risk.data import clean

    return clean(raw_df)


@pytest.fixture
def sample_patient() -> dict:
    return {
        "age": 58,
        "sex": 1,
        "cp": 4,
        "trestbps": 140,
        "chol": 289,
        "fbs": 0,
        "restecg": 2,
        "thalach": 128,
        "exang": 1,
        "oldpeak": 2.4,
        "slope": 2,
        "ca": 2,
        "thal": 7,
    }


@pytest.fixture
def raw_dir(tmp_path, raw_df):
    """A temp dir containing a UCI-formatted file (with '?' for missing)."""
    out = raw_df.copy().astype(object)
    out.loc[0, "ca"] = "?"
    out.loc[1, "thal"] = "?"
    out.to_csv(tmp_path / config.SOURCE_FILES["cleveland"], header=False, index=False)
    return tmp_path


@pytest.fixture
def trained_model(clean_df):
    from sklearn.linear_model import LogisticRegression

    from heart_risk.data import split_xy
    from heart_risk.features import build_pipeline

    X, y = split_xy(clean_df)
    return build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
