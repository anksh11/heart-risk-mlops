import numpy as np
import pandas as pd
import pytest

from heart_risk import config
from heart_risk.data import split_xy
from heart_risk.features import ClinicalFeatureBuilder, build_preprocessor


def test_clinical_features_values():
    X = pd.DataFrame([{**dict.fromkeys(config.INPUT_FEATURES, 1.0), "age": 60, "thalach": 120, "chol": 250}])
    out = ClinicalFeatureBuilder().fit_transform(X)
    assert out.loc[0, "hr_reserve_ratio"] == pytest.approx(120 / 160)
    assert out.loc[0, "age_chol_index"] == pytest.approx(15.0)


def test_preprocessor_output_shape_and_no_nans(clean_df):
    X, _ = split_xy(clean_df)
    X.iloc[0, X.columns.get_loc("chol")] = np.nan
    X.iloc[1, X.columns.get_loc("thal")] = np.nan
    Z = build_preprocessor().fit_transform(X)
    n_onehot = sum(len(config.CATEGORY_LEVELS[c]) for c in config.CATEGORICAL_FEATURES)
    expected = len(config.NUMERIC_FEATURES) + len(config.ENGINEERED_FEATURES) + len(config.BINARY_FEATURES) + n_onehot
    assert Z.shape == (len(X), expected)
    assert not np.isnan(Z).any()


def test_numeric_features_are_standardised(clean_df):
    X, _ = split_xy(clean_df)
    pre = build_preprocessor().fit(X)
    Z = pd.DataFrame(pre.transform(X), columns=pre.get_feature_names_out())
    for col in config.NUMERIC_FEATURES:
        assert abs(Z[col].mean()) < 1e-6
        assert Z[col].std(ddof=0) == pytest.approx(1.0, abs=1e-6)


def test_unknown_category_is_ignored_not_error(clean_df):
    X, _ = split_xy(clean_df)
    pre = build_preprocessor().fit(X)
    row = X.iloc[[0]].copy()
    row["cp"] = 99
    assert pre.transform(row).shape[0] == 1
