"""Exploratory data analysis – produces the figures used in the report.

python -m heart_risk.eda          # writes PNGs to reports/figures/eda/
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from heart_risk import config
from heart_risk.data import clean, load_processed, load_raw

log = logging.getLogger(__name__)

PALETTE = {0: "#3b6ea5", 1: "#d1495b"}
LABELS = {0: "no disease", 1: "disease"}
CATEGORY_NAMES = {
    "cp": {1: "typical", 2: "atypical", 3: "non-anginal", 4: "asymptomatic"},
    "restecg": {0: "normal", 1: "ST-T abn.", 2: "LV hypertrophy"},
    "slope": {1: "upsloping", 2: "flat", 3: "downsloping"},
    "thal": {3: "normal", 6: "fixed", 7: "reversible"},
    "sex": {0: "female", 1: "male"},
    "exang": {0: "no", 1: "yes"},
    "fbs": {0: "<=120", 1: ">120"},
}


def _save(fig, name: str):
    out = config.FIGURES_DIR / "eda"
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / name, dpi=130)
    plt.close(fig)
    return out / name


def class_balance(df: pd.DataFrame):
    counts = df[config.TARGET].map(LABELS).value_counts()
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    bars = ax.bar(counts.index, counts.values, color=[PALETTE[0], PALETTE[1]])
    ax.bar_label(bars, labels=[f"{v} ({v / counts.sum():.0%})" for v in counts.values])
    ax.set_title("Class balance (Cleveland, n=%d)" % len(df))
    ax.set_ylabel("patients")
    return _save(fig, "01_class_balance.png")


def numeric_histograms(df: pd.DataFrame):
    cols = config.NUMERIC_FEATURES
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for ax, col in zip(axes.ravel(), cols, strict=True):
        for t in (0, 1):
            ax.hist(
                df.loc[df[config.TARGET] == t, col].dropna(),
                bins=20,
                alpha=0.6,
                color=PALETTE[t],
                label=LABELS[t],
                density=True,
            )
        ax.set_title(col)
    axes[0, 0].legend()
    fig.suptitle("Numeric feature distributions by diagnosis (density)")
    return _save(fig, "02_numeric_histograms.png")


def categorical_rates(df: pd.DataFrame):
    cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5))
    for ax, col in zip(axes.ravel(), cols, strict=False):
        rate = df.groupby(col)[config.TARGET].agg(["mean", "size"])
        names = [CATEGORY_NAMES.get(col, {}).get(int(i), str(i)) for i in rate.index]
        bars = ax.bar(names, rate["mean"], color="#d1495b", alpha=0.85)
        ax.bar_label(bars, labels=[f"n={n}" for n in rate["size"]], fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(f"{col}: disease rate")
        ax.tick_params(axis="x", rotation=25, labelsize=8)
    axes.ravel()[-1].axis("off")
    fig.suptitle("Share of patients with heart disease per category")
    return _save(fig, "03_categorical_disease_rate.png")


def correlation_heatmap(df: pd.DataFrame):
    corr = df[config.INPUT_FEATURES + [config.TARGET]].corr(method="spearman")
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={"shrink": 0.7},
        annot_kws={"size": 7},
        ax=ax,
    )
    ax.set_title("Spearman correlation (lower triangle)")
    return _save(fig, "04_correlation_heatmap.png")


def target_correlation(df: pd.DataFrame):
    corr = df[config.INPUT_FEATURES + [config.TARGET]].corr(method="spearman")[config.TARGET].drop(config.TARGET)
    corr = corr.reindex(corr.abs().sort_values().index)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(corr.index, corr.values, color=[PALETTE[int(v > 0)] for v in corr.values])
    ax.axvline(0, c="k", lw=0.8)
    ax.set_title("Spearman correlation with target")
    return _save(fig, "05_target_correlation.png")


def age_vs_thalach(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for t in (0, 1):
        d = df[df[config.TARGET] == t]
        ax.scatter(d["age"], d["thalach"], s=18, alpha=0.7, color=PALETTE[t], label=LABELS[t])
    x = np.arange(28, 80)
    ax.plot(x, 220 - x, "k--", lw=1, label="220 - age")
    ax.plot(x, 0.85 * (220 - x), "k:", lw=1, label="85% of predicted max")
    ax.set_xlabel("age")
    ax.set_ylabel("max heart rate (thalach)")
    ax.set_title("Heart-rate reserve motivates the engineered ratio feature")
    ax.legend(fontsize=8)
    return _save(fig, "06_age_vs_thalach.png")


def missingness_by_source():
    """Why we restrict to Cleveland: the other sites are mostly missing."""
    raw = clean(load_raw(list(config.SOURCE_FILES)))
    miss = raw.groupby("source")[config.INPUT_FEATURES].apply(lambda g: g.isna().mean())
    fig, ax = plt.subplots(figsize=(10, 3.2))
    sns.heatmap(miss, annot=True, fmt=".0%", cmap="Reds", vmin=0, vmax=1, cbar=False, ax=ax, annot_kws={"size": 8})
    ax.set_title("Fraction missing per feature and collection site (after cleaning)")
    ax.set_ylabel("")
    return _save(fig, "07_missingness_by_source.png"), miss


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = load_processed()
    paths = [
        class_balance(df),
        numeric_histograms(df),
        categorical_rates(df),
        correlation_heatmap(df),
        target_correlation(df),
        age_vs_thalach(df),
        missingness_by_source()[0],
    ]
    for p in paths:
        log.info("wrote %s", p)
    print(df.describe().T.round(2).to_string())


if __name__ == "__main__":
    main()
