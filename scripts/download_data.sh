#!/usr/bin/env bash
# Download the UCI Heart Disease archive and write data/processed/heart_clean.csv.
# Manual alternative: download https://archive.ics.uci.edu/dataset/45/heart+disease,
# unzip processed.*.data into data/raw/ and run `python -m heart_risk.data`.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=src python -m heart_risk.data "$@"
