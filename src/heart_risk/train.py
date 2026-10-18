"""Train, tune, evaluate and track candidate models with MLflow.

Usage:
    python -m heart_risk.train                 # full grid from params.yaml
    python -m heart_risk.train --quick         # tiny grid, used by CI
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from heart_risk import __version__, config
from heart_risk.data import load_processed, split_xy
from heart_risk.features import build_pipeline

log = logging.getLogger(__name__)

SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]

QUICK_GRIDS = {
    "logistic_regression": {"model__C": [0.1, 1.0]},
    "random_forest": {"model__n_estimators": [100], "model__max_depth": [4]},
    "gradient_boosting": {"model__n_estimators": [50], "model__max_depth": [2]},
}


def make_estimator(name: str, random_state: int):
    if name == "logistic_regression":
        return LogisticRegression(solver="liblinear", max_iter=2000, random_state=random_state)
    if name == "random_forest":
        return RandomForestClassifier(random_state=random_state, n_jobs=-1)
    if name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=random_state)
    raise ValueError(f"unknown model {name!r}")


def load_params(path: Path = config.PROJECT_ROOT / "params.yaml") -> dict[str, Any]:
    with open(path) as fh:
        return yaml.safe_load(fh)


def evaluate(model, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba),
    }


def feature_importance(model) -> pd.Series:
    names = model.named_steps["preprocess"].get_feature_names_out()
    est = model.named_steps["model"]
    if hasattr(est, "feature_importances_"):
        values = est.feature_importances_
    else:
        values = np.abs(est.coef_.ravel())
    return pd.Series(values, index=names).sort_values(ascending=False)


def _plots(model, X_test, y_test, name: str, out: Path) -> list[Path]:
    paths = []

    fig, ax = plt.subplots(figsize=(5, 4.5))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
    ax.set_title(f"ROC – {name} (hold-out)")
    p = out / f"roc_{name}.png"
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, ax=ax, cmap="Blues", display_labels=["no disease", "disease"], colorbar=False
    )
    ax.set_title(f"Confusion matrix – {name}")
    p = out / f"confusion_{name}.png"
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)

    imp = feature_importance(model).head(15)[::-1]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(imp.index, imp.values, color="#3b6ea5")
    ax.set_title(f"Top features – {name}")
    ax.set_xlabel("importance" if hasattr(model.named_steps["model"], "feature_importances_") else "|coefficient|")
    p = out / f"importance_{name}.png"
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)
    return paths


def train_one(name: str, grid: dict, X_train, y_train, X_test, y_test, params: dict, fig_dir: Path) -> dict:
    rs = params["data"]["random_state"]
    cv = StratifiedKFold(n_splits=params["cv"]["folds"], shuffle=True, random_state=rs)
    refit = params["cv"]["refit_metric"]

    search = GridSearchCV(
        build_pipeline(make_estimator(name, rs)),
        param_grid=grid,
        scoring=SCORING,
        refit=refit,
        cv=cv,
        n_jobs=-1,
        return_train_score=False,
    )

    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags({"model_family": name, "code_version": __version__, "stage": "candidate"})
        t0 = time.perf_counter()
        search.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - t0

        best = search.best_estimator_
        i = search.best_index_
        cv_metrics = {f"cv_{m}_mean": float(search.cv_results_[f"mean_test_{m}"][i]) for m in SCORING}
        cv_metrics |= {f"cv_{m}_std": float(search.cv_results_[f"std_test_{m}"][i]) for m in SCORING}
        test_metrics = {f"test_{k}": float(v) for k, v in evaluate(best, X_test, y_test).items()}

        mlflow.log_params({k.replace("model__", ""): v for k, v in search.best_params_.items()})
        mlflow.log_params(
            {
                "grid_size": len(search.cv_results_["params"]),
                "cv_folds": params["cv"]["folds"],
                "refit_metric": refit,
                "n_train": len(X_train),
                "n_test": len(X_test),
            }
        )
        mlflow.log_metrics(cv_metrics | test_metrics | {"fit_seconds": fit_seconds})

        # Every grid point as a CSV so the tuning process is fully inspectable.
        with tempfile.TemporaryDirectory() as tmp:
            cvr = pd.DataFrame(search.cv_results_)
            keep = ["params"] + [c for c in cvr.columns if c.startswith(("mean_test_", "std_test_", "rank_test_"))]
            grid_csv = Path(tmp) / f"cv_results_{name}.csv"
            cvr[keep].sort_values(f"rank_test_{refit}").to_csv(grid_csv, index=False)
            mlflow.log_artifact(str(grid_csv), "tuning")

            report = Path(tmp) / "classification_report.txt"
            report.write_text(classification_report(y_test, best.predict(X_test), target_names=["absent", "present"]))
            mlflow.log_artifact(str(report), "evaluation")

        for p in _plots(best, X_test, y_test, name, fig_dir):
            mlflow.log_artifact(str(p), "plots")

        mlflow.sklearn.log_model(
            best,
            artifact_path="model",
            signature=infer_signature(X_train, best.predict_proba(X_train)[:, 1]),
            input_example=X_train.head(3),
        )

        log.info(
            "%-20s cv_roc_auc=%.3f±%.3f test_roc_auc=%.3f test_acc=%.3f (%.1fs)",
            name,
            cv_metrics["cv_roc_auc_mean"],
            cv_metrics["cv_roc_auc_std"],
            test_metrics["test_roc_auc"],
            test_metrics["test_accuracy"],
            fit_seconds,
        )
        return {
            "name": name,
            "run_id": run.info.run_id,
            "best_params": search.best_params_,
            "model": best,
            **cv_metrics,
            **test_metrics,
        }


def comparison_plot(results: list[dict], out: Path) -> Path:
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    df = pd.DataFrame({r["name"]: [r[f"cv_{m}_mean"] for m in metrics] for r in results}, index=metrics)
    err = pd.DataFrame({r["name"]: [r[f"cv_{m}_std"] for m in metrics] for r in results}, index=metrics)
    ax = df.plot.bar(yerr=err, figsize=(8, 4.5), capsize=3, rot=0, color=["#3b6ea5", "#e28743", "#5a9e6f"])
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("CV score (mean ± std)")
    ax.set_title("Cross-validated comparison of tuned models")
    ax.legend(loc="lower right")
    p = out / "model_comparison.png"
    plt.tight_layout()
    plt.savefig(p, dpi=120)
    plt.close()
    return p


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="tiny grids for CI smoke-training")
    parser.add_argument("--params", type=Path, default=config.PROJECT_ROOT / "params.yaml")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    params = load_params(args.params)

    df = load_processed()
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["data"]["test_size"], stratify=y, random_state=params["data"]["random_state"]
    )
    log.info("Train %s / test %s, positive rate %.2f", X_train.shape, X_test.shape, y.mean())

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", params["mlflow"]["tracking_uri"])
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(params["mlflow"]["experiment"])

    fig_dir = config.FIGURES_DIR / "training"
    fig_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for name, spec in params["models"].items():
        grid = QUICK_GRIDS[name] if args.quick else spec["grid"]
        results.append(train_one(name, grid, X_train, y_train, X_test, y_test, params, fig_dir))

    refit = params["cv"]["refit_metric"]
    # Select on CV score (not the hold-out) so the test set stays an unbiased estimate.
    winner = max(results, key=lambda r: r[f"cv_{refit}_mean"])
    log.info("Selected %s (cv_%s=%.3f)", winner["name"], refit, winner[f"cv_{refit}_mean"])

    summary = pd.DataFrame([{k: v for k, v in r.items() if k not in ("model",)} for r in results])
    summary_csv = config.REPORTS_DIR / "model_comparison.csv"
    summary.drop(columns=["best_params"]).to_csv(summary_csv, index=False)
    cmp_png = comparison_plot(results, fig_dir)

    # Refit the winning configuration on *all* data for the shipped artefact.
    final = build_pipeline(make_estimator(winner["name"], params["data"]["random_state"]))
    final.set_params(**winner["best_params"])
    final.fit(X, y)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, config.MODEL_FILE)
    meta = {
        "model_name": winner["name"],
        "model_version": __version__,
        "trained_at": pd.Timestamp.utcnow().isoformat(),
        "selection_metric": f"cv_{refit}_mean",
        "best_params": winner["best_params"],
        "cv_metrics": {k: round(v, 4) for k, v in winner.items() if k.startswith("cv_")},
        "holdout_metrics": {k: round(v, 4) for k, v in winner.items() if k.startswith("test_")},
        "mlflow_run_id": winner["run_id"],
        "n_training_rows": int(len(X)),
        "features": config.INPUT_FEATURES,
        "quick": bool(args.quick),
    }
    config.MODEL_META_FILE.write_text(json.dumps(meta, indent=2, default=str))

    with mlflow.start_run(run_name=f"final_{winner['name']}"):
        mlflow.set_tags({"stage": "production-candidate", "selected_from": winner["run_id"]})
        mlflow.log_params({k.replace("model__", ""): v for k, v in winner["best_params"].items()})
        mlflow.log_metrics({k: v for k, v in winner.items() if k.startswith(("cv_", "test_"))})
        mlflow.log_artifact(str(summary_csv), "comparison")
        mlflow.log_artifact(str(cmp_png), "comparison")
        mlflow.log_artifact(str(config.MODEL_META_FILE))
        mlflow.sklearn.log_model(
            final,
            artifact_path="model",
            signature=infer_signature(X, final.predict_proba(X)[:, 1]),
            input_example=X.head(3),
            registered_model_name="heart-risk-classifier",
        )

    # Also store a standalone MLflow-format copy next to the joblib file.
    import shutil

    shutil.rmtree(config.MLFLOW_MODEL_DIR, ignore_errors=True)
    mlflow.sklearn.save_model(final, str(config.MLFLOW_MODEL_DIR), input_example=X.head(3))

    log.info("Saved model -> %s", config.MODEL_FILE)
    print(summary.drop(columns=["best_params", "run_id"]).round(3).to_string(index=False))
    return meta


if __name__ == "__main__":
    main()
