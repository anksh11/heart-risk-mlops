import numpy as np
import pandas as pd

from heart_risk import config
from heart_risk.data import clean, load_raw, split_xy


def test_load_raw_parses_question_marks_as_nan(raw_dir):
    df = load_raw(["cleveland"], raw_dir=raw_dir)
    assert list(df.columns) == config.RAW_COLUMNS + ["source"]
    assert np.isnan(df.loc[0, "ca"])
    assert np.isnan(df.loc[1, "thal"])
    assert (df["source"] == "cleveland").all()


def test_clean_binarises_target(raw_df):
    out = clean(raw_df)
    assert "num" not in out.columns
    assert set(out[config.TARGET].unique()) <= {0, 1}
    assert (out[config.TARGET] == (raw_df["num"] > 0).astype(int)).all()


def test_clean_marks_impossible_values_missing(raw_df):
    raw_df.loc[0, "chol"] = 0
    raw_df.loc[1, "trestbps"] = 0
    raw_df.loc[2, "thal"] = 5  # not a valid code
    raw_df.loc[3, "ca"] = 9
    out = clean(raw_df)
    assert out.loc[[0, 1, 2, 3]].isna().sum().sum() == 4


def test_clean_drops_duplicates(raw_df):
    dup = pd.concat([raw_df, raw_df.iloc[:5]], ignore_index=True)
    assert len(clean(dup)) == len(clean(raw_df))


def test_clean_does_not_mutate_input(raw_df):
    snapshot = raw_df.copy()
    clean(raw_df)
    pd.testing.assert_frame_equal(raw_df, snapshot)


def test_split_xy_returns_model_features(clean_df):
    X, y = split_xy(clean_df)
    assert list(X.columns) == config.INPUT_FEATURES
    assert len(X) == len(y)
