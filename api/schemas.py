"""Request / response models for the serving API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PatientFeatures(BaseModel):
    """One patient record, using the UCI attribute names and encodings."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
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
        },
    )

    age: float = Field(..., ge=1, le=120, description="Age in years")
    sex: Literal[0, 1] = Field(..., description="1 = male, 0 = female")
    cp: Literal[1, 2, 3, 4] = Field(..., description="Chest pain: 1 typical, 2 atypical, 3 non-anginal, 4 asymptomatic")
    trestbps: float = Field(..., ge=50, le=250, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=80, le=700, description="Serum cholesterol (mg/dl)")
    fbs: Literal[0, 1] = Field(..., description="Fasting blood sugar > 120 mg/dl")
    restecg: Literal[0, 1, 2] = Field(..., description="Resting ECG: 0 normal, 1 ST-T abnormality, 2 LV hypertrophy")
    thalach: float = Field(..., ge=50, le=250, description="Max heart rate achieved")
    exang: Literal[0, 1] = Field(..., description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=-3, le=10, description="ST depression induced by exercise")
    slope: Literal[1, 2, 3] = Field(..., description="Slope of peak exercise ST segment")
    ca: int | None = Field(None, ge=0, le=3, description="Major vessels coloured by fluoroscopy (0-3); optional")
    thal: Literal[3, 6, 7] | None = Field(None, description="3 normal, 6 fixed defect, 7 reversible defect; optional")


class BatchRequest(BaseModel):
    patients: list[PatientFeatures] = Field(..., min_length=1, max_length=500)


class Prediction(BaseModel):
    prediction: Literal[0, 1]
    label: str
    probability_disease: float
    confidence: float
    risk_band: Literal["low", "moderate", "high"]


class PredictionResponse(Prediction):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    model_version: str
    request_id: str


class BatchResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    predictions: list[Prediction]
    model_name: str
    model_version: str
    request_id: str
