"""Smoke tests for the Phase 2 not-MIWAE benchmark scaffold.

Covers:
  - All six dataset loaders produce standardized X with the expected shape.
  - self_masking_above_mean masks roughly half the entries in the first D/2
    columns (≥30%) and zero entries in the remaining columns.
  - imputation_rmse matches a synthetic hand-computed value.
  - Each baseline imputer runs without error on one dataset and produces a
    full (NaN-free) reconstruction.
"""

from __future__ import annotations

import numpy as np
import pytest

from phase2_notmiwae_benchmark import datasets as ds_mod
from phase2_notmiwae_benchmark import mnar_injection as mnar_mod
from phase2_notmiwae_benchmark import baselines as bl_mod


EXPECTED_SHAPES = {
    # name: (n_instances_expected, n_features_expected)
    "banknote": (1372, 4),
    "concrete": (1030, 8),
    "red": (1599, 11),
    "white": (4898, 11),
    "yeast": (1484, 8),
    "breast": (569, 30),
}


def _tol_equal(actual: int, expected: int, tol_frac: float = 0.01) -> bool:
    """Allow ±1% slack on instance counts for mirror drift."""
    slack = max(1, int(round(tol_frac * expected)))
    return abs(actual - expected) <= slack


@pytest.mark.parametrize("name", list(EXPECTED_SHAPES.keys()))
def test_dataset_loads_and_is_standardized(name: str) -> None:
    """Each loader returns the expected shape and standardized features."""
    try:
        dset = ds_mod.load(name)
    except Exception as exc:  # pragma: no cover — network failures logged
        pytest.skip(f"{name} loader failed (likely network / mirror issue): {exc}")

    exp_n, exp_d = EXPECTED_SHAPES[name]
    assert dset.X.ndim == 2, f"{name}: expected 2-D X"
    assert dset.X.shape[1] == exp_d, (
        f"{name}: expected {exp_d} features, got {dset.X.shape[1]}"
    )
    assert _tol_equal(dset.X.shape[0], exp_n), (
        f"{name}: expected ~{exp_n} rows, got {dset.X.shape[0]}"
    )
    # No NaNs in the raw (pre-masked) data.
    assert not np.isnan(dset.X).any(), f"{name}: raw X contains NaN"
    # Standardized: column means ≈ 0, column stds ≈ 1.
    np.testing.assert_allclose(dset.X.mean(axis=0), 0.0, atol=1e-6)
    np.testing.assert_allclose(dset.X.std(axis=0), 1.0, atol=1e-6)


def test_self_masking_above_mean_masks_first_half_only() -> None:
    """The first D/2 columns should be ~50% masked, rest fully observed."""
    rng = np.random.default_rng(0)
    N, D = 400, 6
    # Draw from a symmetric distribution so >mean hits ~50% per column.
    X = rng.standard_normal(size=(N, D))
    # Re-standardize for determinism (matches paper's pipeline).
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    result = mnar_mod.self_masking_above_mean(X, seed=0)

    observed = result.observed_mask
    half = D // 2
    # First D/2 columns: at least 30% missing, at most 70%.
    for j in range(half):
        frac_missing = 1.0 - observed[:, j].mean()
        assert frac_missing >= 0.30, (
            f"col {j}: only {frac_missing:.2%} missing (expected ≥30%)"
        )
        assert frac_missing <= 0.70, (
            f"col {j}: {frac_missing:.2%} missing (expected ≤70%)"
        )
    # Remaining columns: fully observed.
    for j in range(half, D):
        assert observed[:, j].all(), f"col {j}: expected fully observed"
    # X_nan should have NaN exactly where observed_mask is False.
    assert np.array_equal(np.isnan(result.X_nan), ~observed)
    # X_zero should have 0 where masked, original X elsewhere.
    assert np.all(result.X_zero[~observed] == 0.0)
    # Observed positions preserved.
    np.testing.assert_array_equal(result.X_zero[observed], X[observed])


def test_self_masking_matches_paper_formula() -> None:
    """Regenerate the paper's introduce_mising semantics on a fixed array."""
    X = np.array(
        [
            [1.0, 0.5, -0.2, 2.0],
            [0.5, 1.5, 0.3, -1.0],
            [-0.5, -1.0, 0.4, 0.0],
            [0.0, 2.0, -0.1, 0.5],
        ],
        dtype=np.float64,
    )
    # Expected behavior: first D/2 = 2 cols masked above mean; last 2 cols fully observed.
    means = X[:, :2].mean(axis=0)
    expected_mask_large = X[:, :2] > means
    result = mnar_mod.self_masking_above_mean(X)
    # Masked positions in the first 2 cols should match > mean.
    assert np.array_equal(~result.observed_mask[:, :2], expected_mask_large)
    # Last 2 cols fully observed.
    assert result.observed_mask[:, 2:].all()


def test_imputation_rmse_on_synthetic() -> None:
    """Hand-computed RMSE over masked entries only."""
    X_true = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    X_rec = np.array([[1.0, 0.0], [3.0, 5.0], [4.0, 6.0]])
    # Mask: observed everywhere except (0,1) and (2,0). Errors at those cells:
    # (0,1): (2 - 0)^2 = 4; (2,0): (5 - 4)^2 = 1. RMSE = sqrt((4 + 1) / 2) = sqrt(2.5).
    observed = np.array(
        [[True, False], [True, True], [False, True]]
    )
    rmse = mnar_mod.imputation_rmse(X_true, X_rec, observed)
    assert rmse == pytest.approx(np.sqrt(2.5))


def test_imputation_rmse_all_observed_returns_zero() -> None:
    """If no entries are masked, RMSE formula's denominator is 0 → 0.0."""
    X = np.zeros((3, 3))
    observed = np.ones((3, 3), dtype=bool)
    assert mnar_mod.imputation_rmse(X, X + 1.0, observed) == 0.0


@pytest.mark.parametrize("imputer_cls", [
    bl_mod.MeanImputer,
    bl_mod.MICEImputer,
    bl_mod.MissForestImputer,
])
def test_baseline_imputers_run_on_banknote(imputer_cls) -> None:
    """Each baseline runs end-to-end on the Banknote dataset and returns full X."""
    try:
        dset = ds_mod.load("banknote")
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"banknote loader failed: {exc}")

    result = mnar_mod.self_masking_above_mean(dset.X, seed=0)
    imputer = imputer_cls(random_state=0)
    X_rec = imputer.fit_impute(result.X_nan, observed_mask=result.observed_mask)

    assert X_rec.shape == dset.X.shape
    assert not np.isnan(X_rec).any(), (
        f"{imputer_cls.__name__}: reconstruction contains NaN"
    )
    # Observed positions should be preserved by the imputer (sklearn's imputers do this).
    np.testing.assert_allclose(
        X_rec[result.observed_mask],
        dset.X[result.observed_mask],
        atol=1e-10,
    )
    # Sanity-check: RMSE over masked entries is a finite non-negative number.
    rmse = mnar_mod.imputation_rmse(dset.X, X_rec, result.observed_mask)
    assert np.isfinite(rmse)
    assert rmse >= 0.0


def test_registry_and_load_all_cover_six_datasets() -> None:
    """REGISTRY has exactly the six expected datasets."""
    assert set(ds_mod.REGISTRY.keys()) == set(EXPECTED_SHAPES.keys())


def test_load_unknown_raises() -> None:
    with pytest.raises(ValueError):
        ds_mod.load("no-such-dataset")
