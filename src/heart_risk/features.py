"""Feature engineering + preprocessing pipeline.

The full preprocessing lives inside a single sklearn Pipeline so that the exact
same transformations are applied at training time, in CI and inside the API
container – the serialized model is self-contained.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heart_risk import config


class ClinicalFeatureBuilder(BaseEstimator, TransformerMixin):
    """Adds two domain-motivated features.

    * ``hr_reserve_ratio`` – achieved max heart rate divided by the age-predicted
      maximum (220 - age). Chronotropic incompetence (< ~0.85) is a known marker
      of coronary artery disease, and the ratio removes the strong age confound
      from ``thalach``.
    * ``age_chol_index`` – age x cholesterol / 1000, a crude cumulative-exposure
      interaction term that linear models cannot learn on their own.
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X, columns=config.INPUT_FEATURES).copy()
        predicted_max = (220 - X["age"]).replace(0, np.nan)
        X["hr_reserve_ratio"] = X["thalach"] / predicted_max
        X["age_chol_index"] = X["age"] * X["chol"] / 1000.0
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(config.INPUT_FEATURES + config.ENGINEERED_FEATURES, dtype=object)


def build_preprocessor() -> Pipeline:
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    binary = SimpleImputer(strategy="most_frequent")
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    categories=[config.CATEGORY_LEVELS[c] for c in config.CATEGORICAL_FEATURES],
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )
    columns = ColumnTransformer(
        [
            ("num", numeric, config.NUMERIC_FEATURES + config.ENGINEERED_FEATURES),
            ("bin", binary, config.BINARY_FEATURES),
            ("cat", categorical, config.CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline(
        [
            ("clinical", ClinicalFeatureBuilder()),
            ("columns", columns),
        ]
    )


def build_pipeline(estimator) -> Pipeline:
    """Preprocessing + estimator as one fit/predict-able object."""
    return Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])
