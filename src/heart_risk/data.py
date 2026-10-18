"""Data acquisition and cleaning for the UCI Heart Disease dataset."""

from __future__ import annotations

import argparse
import io
import logging
import urllib.request
import zipfile
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from heart_risk import config

log = logging.getLogger(__name__)


def download_raw(dest: Path = config.RAW_DIR, url: str = config.UCI_ZIP_URL, force: bool = False) -> Path:
    """Download the UCI archive and extract the four `processed.*.data` files."""
    dest.mkdir(parents=True, exist_ok=True)
    wanted = set(config.SOURCE_FILES.values())
    if not force and all((dest / f).exists() for f in wanted):
        log.info("Raw files already present in %s – skipping download", dest)
        return dest

    log.info("Downloading %s", url)
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 - fixed trusted URL
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        for name in zf.namelist():
            if Path(name).name in wanted | {"heart-disease.names"}:
                (dest / Path(name).name).write_bytes(zf.read(name))
    log.info("Extracted raw files to %s", dest)
    return dest


def load_raw(sources: Iterable[str] = ("cleveland",), raw_dir: Path = config.RAW_DIR) -> pd.DataFrame:
    """Read one or more of the processed UCI files into a single frame.

    Missing values are encoded as '?' in the source files.
    """
    frames = []
    for src in sources:
        path = raw_dir / config.SOURCE_FILES[src]
        df = pd.read_csv(path, header=None, names=config.RAW_COLUMNS, na_values="?")
        df["source"] = src
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of a raw frame.

    * `num` (0-4) is collapsed into a binary `target` (0 = absent, >0 = present).
    * Physiologically impossible zeros (cholesterol / resting BP == 0, which occur
      in the non-Cleveland files) are treated as missing.
    * Out-of-vocabulary categorical codes are treated as missing.
    * Exact duplicate rows are dropped.

    Remaining missing values are *kept* on purpose: imputation happens inside the
    sklearn pipeline so it is fitted on training folds only and is shipped with
    the model.
    """
    out = df.copy()
    out[config.TARGET] = (out["num"] > 0).astype(int)
    out = out.drop(columns=["num"])

    for col in ("chol", "trestbps"):
        out.loc[out[col] <= 0, col] = np.nan

    for col, levels in config.CATEGORY_LEVELS.items():
        out.loc[~out[col].isin(levels), col] = np.nan

    for col in config.BINARY_FEATURES:
        out.loc[~out[col].isin([0, 1]), col] = np.nan

    if "ca" in out:
        out.loc[(out["ca"] < 0) | (out["ca"] > 3), "ca"] = np.nan

    before = len(out)
    out = out.drop_duplicates().reset_index(drop=True)
    if len(out) != before:
        log.info("Dropped %d duplicate rows", before - len(out))
    return out


def load_processed(path: Path = config.PROCESSED_CSV) -> pd.DataFrame:
    return pd.read_csv(path)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[config.INPUT_FEATURES].copy(), df[config.TARGET].copy()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download and clean the UCI Heart Disease data")
    parser.add_argument("--sources", nargs="+", default=["cleveland"], choices=list(config.SOURCE_FILES))
    parser.add_argument("--force", action="store_true", help="re-download even if files exist")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    download_raw(force=args.force)
    df = clean(load_raw(args.sources))
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.PROCESSED_CSV, index=False)
    log.info("Wrote %d rows x %d cols to %s", *df.shape, config.PROCESSED_CSV)
    log.info("Missing values per column:\n%s", df.isna().sum()[lambda s: s > 0].to_string() or "none")


if __name__ == "__main__":
    main()
