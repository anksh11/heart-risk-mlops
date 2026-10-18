"""Central place for paths and the feature schema.

Everything else imports from here so the training code, the API and the tests
always agree on column names and types.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("HEART_RISK_ROOT", Path(__file__).resolve().parents[2]))

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_CSV = PROCESSED_DIR / "heart_clean.csv"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_FILE = MODELS_DIR / "heart_risk_model.joblib"
MODEL_META_FILE = MODELS_DIR / "model_meta.json"
MLFLOW_MODEL_DIR = MODELS_DIR / "mlflow_model"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
# The four "processed" files share the same 14-attribute layout. Cleveland is the
# canonical subset used in most of the literature and is what we train on by default.
SOURCE_FILES = {
    "cleveland": "processed.cleveland.data",
    "hungarian": "processed.hungarian.data",
    "switzerland": "processed.switzerland.data",
    "va": "processed.va.data",
}

RAW_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]
TARGET = "target"

# Continuous measurements -> imputed + scaled.
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
# Already 0/1 -> imputed, passed through.
BINARY_FEATURES = ["sex", "fbs", "exang"]
# Nominal codes -> imputed + one-hot encoded.
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "thal"]

INPUT_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

# Engineered features appended by heart_risk.features.ClinicalFeatureBuilder.
ENGINEERED_FEATURES = ["hr_reserve_ratio", "age_chol_index"]

# Valid ranges / codes, used both for cleaning and for API validation.
CATEGORY_LEVELS = {
    "cp": [1, 2, 3, 4],
    "restecg": [0, 1, 2],
    "slope": [1, 2, 3],
    "thal": [3, 6, 7],
}

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
