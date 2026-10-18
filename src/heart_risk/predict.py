"""Model loading + inference helpers shared by the API and the CLI."""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from heart_risk import config

RISK_BANDS = [(0.3, "low"), (0.6, "moderate"), (1.01, "high")]


def risk_band(probability: float) -> str:
    for upper, label in RISK_BANDS:
        if probability < upper:
            return label
    return "high"


@lru_cache(maxsize=4)
def load_model(path: str | Path = config.MODEL_FILE):
    return joblib.load(path)


def load_meta(path: Path = config.MODEL_META_FILE) -> dict:
    return json.loads(Path(path).read_text()) if Path(path).exists() else {}


def predict_records(model, records: list[dict], threshold: float = 0.5) -> list[dict]:
    frame = pd.DataFrame(records, columns=config.INPUT_FEATURES)
    proba = model.predict_proba(frame)[:, 1]
    out = []
    for p in proba:
        label = int(p >= threshold)
        out.append(
            {
                "prediction": label,
                "label": "heart disease" if label else "no heart disease",
                "probability_disease": round(float(p), 4),
                # Confidence = probability of the class we actually predicted.
                "confidence": round(float(p if label else 1 - p), 4),
                "risk_band": risk_band(float(p)),
            }
        )
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score patients from a JSON file (object or list of objects)")
    parser.add_argument("input", type=Path)
    parser.add_argument("--model", type=Path, default=config.MODEL_FILE)
    args = parser.parse_args(argv)
    payload = json.loads(args.input.read_text())
    records = payload if isinstance(payload, list) else [payload]
    print(json.dumps(predict_records(load_model(args.model), records), indent=2))


if __name__ == "__main__":
    main()
