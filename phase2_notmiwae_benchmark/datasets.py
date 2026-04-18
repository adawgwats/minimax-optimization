"""Loaders for the 6 UCI datasets used in Ipsen et al. 2021 (not-MIWAE, ICLR).

Each loader returns a standardized (zero-mean unit-variance column-wise)
feature matrix with the target/class column dropped. Standardization is
applied BEFORE any MNAR injection — this matches `docs/notmiwae/task01.py`
lines 72-73 verbatim.

Sources:
  banknote: UCI direct HTTP (no header, 5 cols with last = class).
  concrete: OpenML mirror of UCI Concrete Compressive Strength (no xlrd dep).
  red:      UCI direct HTTP winequality-red.csv, sep=';', drop 'quality'.
  white:    UCI direct HTTP winequality-white.csv, sep=';', drop 'quality'.
  yeast:    UCI direct HTTP yeast.data, whitespace-delimited, drop name + class.
  breast:   sklearn.datasets.load_breast_cancer (Wisconsin Diagnostic).

All network fetches are cached under `phase2_notmiwae_benchmark/data_cache/`,
which is `.gitignore`d.
"""

from __future__ import annotations

import io
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


DATA_CACHE = Path(__file__).parent / "data_cache"


@dataclass(frozen=True)
class NotMIWAEDataset:
    """A loaded, standardized feature matrix ready for MNAR injection.

    X is zero-mean unit-variance column-wise; the target/class column is
    already dropped. `n_instances` and `n_features` reflect X's shape.
    """
    name: str
    X: np.ndarray  # float64, shape (n_instances, n_features)
    n_instances: int
    n_features: int
    source: str
    notes: str = ""


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _cache_path(filename: str) -> Path:
    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    return DATA_CACHE / filename


def _download_to_cache(url: str, filename: str) -> Path:
    """Download URL to the data_cache folder if not already present; return path."""
    path = _cache_path(filename)
    if not path.exists():
        with urllib.request.urlopen(url) as resp:  # nosec B310 — UCI HTTPS only
            data = resp.read()
        path.write_bytes(data)
    return path


def _standardize(X: np.ndarray) -> np.ndarray:
    """Column-wise zero-mean unit-variance, matching task01.py lines 72-73.

    Uses sample std = population std (numpy's default ddof=0), matching
    np.std behavior in the paper's code.
    """
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True)
    # Guard against exactly-constant columns (shouldn't happen in these
    # datasets, but defensively avoid 0/0 → NaN).
    sigma = np.where(sigma == 0, 1.0, sigma)
    return (X - mu) / sigma


# ---------------------------------------------------------------------------
# Per-dataset loaders
# ---------------------------------------------------------------------------


def load_banknote() -> NotMIWAEDataset:
    """UCI Banknote Authentication. 1372 × 4 features (last col = class).

    Source: https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt
    """
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "00267/data_banknote_authentication.txt"
    )
    path = _download_to_cache(url, "banknote.txt")
    df = pd.read_csv(path, header=None)
    # 5 columns: 4 features + class. Drop class (last).
    X = df.iloc[:, :-1].to_numpy(dtype=np.float64)
    X = _standardize(X)
    return NotMIWAEDataset(
        name="banknote",
        X=X,
        n_instances=X.shape[0],
        n_features=X.shape[1],
        source="UCI banknote authentication (direct HTTP)",
        notes="Dropped class (last column). 4 continuous features.",
    )


def load_concrete() -> NotMIWAEDataset:
    """UCI Concrete Compressive Strength. 1030 × 8 features (drop strength target).

    Sourced via sklearn OpenML to avoid xlrd dependency on the UCI .xls mirror.
    Tries multiple OpenML names in order of stability.
    """
    from sklearn.datasets import fetch_openml

    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    # Try a few known OpenML identifiers for the UCI Concrete dataset.
    candidates = [
        # (name, version)
        ("Concrete_Data", "active"),
        ("concrete_data", "active"),
        ("concrete-compressive-strength", "active"),
    ]
    last_err: Exception | None = None
    data = None
    for name, version in candidates:
        try:
            data = fetch_openml(
                name=name,
                version=version,
                as_frame=True,
                parser="auto",
                data_home=str(DATA_CACHE),
            )
            break
        except Exception as exc:  # pragma: no cover - network path
            last_err = exc
            continue
    if data is None:
        # Final fallback: direct HTTP to UCI .xls mirror. Requires openpyxl or xlrd.
        raise RuntimeError(
            f"Could not load Concrete dataset from any OpenML mirror. Last error: {last_err}. "
            f"Consider installing 'openpyxl' or 'xlrd' and falling back to the UCI .xls."
        )
    frame: pd.DataFrame = data.frame
    # Drop the target column. Some OpenML mirrors set `data.target` to None and
    # leave the compressive-strength column inside `data.frame`; we drop by name
    # pattern in that case. The 8 input features are:
    #   Cement / Blast Furnace Slag / Fly Ash / Water / Superplasticizer /
    #   Coarse Aggregate / Fine Aggregate / Age.
    target_col = None
    if getattr(data, "target", None) is not None and hasattr(data.target, "name"):
        target_col = data.target.name
    if target_col is None:
        # Fall back to pattern match: the strength column contains 'strength' or 'MPa'.
        for col in frame.columns:
            lc = str(col).lower()
            if "strength" in lc or "mpa" in lc or "compressive" in lc:
                target_col = col
                break
    feature_cols = [c for c in frame.columns if c != target_col]
    X_df = frame[feature_cols].apply(pd.to_numeric, errors="coerce")
    X_df = X_df.dropna(axis=0, how="any")
    X = X_df.to_numpy(dtype=np.float64)
    X = _standardize(X)
    return NotMIWAEDataset(
        name="concrete",
        X=X,
        n_instances=X.shape[0],
        n_features=X.shape[1],
        source="OpenML mirror of UCI Concrete Compressive Strength",
        notes="Dropped compressive-strength target. 8 continuous features.",
    )


def load_red_wine() -> NotMIWAEDataset:
    """UCI Red Wine Quality. 1599 × 11 features (drop quality target).

    Source: https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
    """
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "wine-quality/winequality-red.csv"
    )
    path = _download_to_cache(url, "winequality-red.csv")
    df = pd.read_csv(path, sep=";")
    if "quality" in df.columns:
        df = df.drop(columns=["quality"])
    X = df.to_numpy(dtype=np.float64)
    X = _standardize(X)
    return NotMIWAEDataset(
        name="red",
        X=X,
        n_instances=X.shape[0],
        n_features=X.shape[1],
        source="UCI winequality-red.csv (direct HTTP, sep=';')",
        notes="Dropped 'quality' target. 11 continuous features.",
    )


def load_white_wine() -> NotMIWAEDataset:
    """UCI White Wine Quality. 4898 × 11 features (drop quality target).

    Source: https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv
    """
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "wine-quality/winequality-white.csv"
    )
    path = _download_to_cache(url, "winequality-white.csv")
    df = pd.read_csv(path, sep=";")
    if "quality" in df.columns:
        df = df.drop(columns=["quality"])
    X = df.to_numpy(dtype=np.float64)
    X = _standardize(X)
    return NotMIWAEDataset(
        name="white",
        X=X,
        n_instances=X.shape[0],
        n_features=X.shape[1],
        source="UCI winequality-white.csv (direct HTTP, sep=';')",
        notes="Dropped 'quality' target. 11 continuous features.",
    )


def load_yeast() -> NotMIWAEDataset:
    """UCI Yeast. 1484 × 8 features (drop sequence-name and class label).

    Source: https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data
    Whitespace-delimited, no header. 10 columns: [name, 8 features, class].
    """
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "yeast/yeast.data"
    )
    path = _download_to_cache(url, "yeast.data")
    # yeast.data uses multiple-whitespace delimiter.
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    # Columns: [seq_name, mcg, gvh, alm, mit, erl, pox, vac, nuc, class]
    if df.shape[1] != 10:
        raise ValueError(f"Yeast expected 10 columns; got {df.shape[1]}")
    # Drop first (name) and last (class) columns.
    X_df = df.iloc[:, 1:-1]
    X = X_df.to_numpy(dtype=np.float64)
    X = _standardize(X)
    return NotMIWAEDataset(
        name="yeast",
        X=X,
        n_instances=X.shape[0],
        n_features=X.shape[1],
        source="UCI yeast.data (direct HTTP, whitespace-delimited)",
        notes="Dropped sequence-name and class-label columns. 8 continuous features.",
    )


def load_breast() -> NotMIWAEDataset:
    """Breast Cancer Wisconsin Diagnostic via sklearn. 569 × 30 features.

    Source: `sklearn.datasets.load_breast_cancer()` — the canonical sklearn
    mirror of the UCI WDBC dataset.
    """
    from sklearn.datasets import load_breast_cancer

    bunch = load_breast_cancer()
    X = np.asarray(bunch.data, dtype=np.float64)
    X = _standardize(X)
    return NotMIWAEDataset(
        name="breast",
        X=X,
        n_instances=X.shape[0],
        n_features=X.shape[1],
        source="sklearn.datasets.load_breast_cancer (UCI WDBC mirror)",
        notes="Dropped malignant/benign class. 30 continuous features.",
    )


# ---------------------------------------------------------------------------
# Registry + dispatcher
# ---------------------------------------------------------------------------


REGISTRY: dict[str, Callable[[], NotMIWAEDataset]] = {
    "banknote": load_banknote,
    "concrete": load_concrete,
    "red": load_red_wine,
    "white": load_white_wine,
    "yeast": load_yeast,
    "breast": load_breast,
}


def load(name: str) -> NotMIWAEDataset:
    """Load a dataset by short name. See REGISTRY for supported names."""
    if name not in REGISTRY:
        raise ValueError(f"Unknown dataset {name!r}. Known: {sorted(REGISTRY)}")
    return REGISTRY[name]()


def load_all() -> dict[str, NotMIWAEDataset]:
    """Load all six datasets; useful for smoke tests and the harness."""
    return {name: loader() for name, loader in REGISTRY.items()}
