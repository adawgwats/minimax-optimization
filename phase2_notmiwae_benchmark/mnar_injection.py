"""Self-masking-above-mean MNAR injection per Ipsen et al. 2021 (not-MIWAE).

Verbatim reference: `docs/notmiwae/task01.py` lines 21-33. For the first
`D // 2` columns, values strictly greater than that column's mean are set to
NaN. The remaining `D - D // 2` columns are always fully observed. Because
standardized columns are roughly symmetric around zero, this yields
approximately 50% missing in each of the first `D // 2` columns and 0%
missing in the rest → overall ≈25% missing on the full matrix.

The `seed` argument is accepted for API symmetry with Phase 1 injectors but
is UNUSED — the mask is fully deterministic given X (per the paper's code).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MNARResult:
    """Result of applying the self-masking-above-mean MNAR mechanism.

    Attributes
    ----------
    X_nan : np.ndarray
        Float matrix with NaN at masked entries; observed entries hold the
        original X values.
    X_zero : np.ndarray
        Float matrix with 0.0 at masked entries; observed entries hold the
        original X values. Convenience for baselines that consume a
        zero-imputed matrix (the paper's `Xz`).
    observed_mask : np.ndarray
        Boolean matrix; True where observed, False where masked. This is the
        `S` in the paper's RMSE formula (with `1 - S` selecting masked
        entries).
    """
    X_nan: np.ndarray
    X_zero: np.ndarray
    observed_mask: np.ndarray


def self_masking_above_mean(X: np.ndarray, seed: int = 0) -> MNARResult:
    """Inject MNAR per Ipsen et al. 2021 `introduce_mising` (task01.py:21-33).

    For the first `D // 2` features, set any entry strictly greater than that
    column's mean to NaN. The remaining features are fully observed.

    Parameters
    ----------
    X : np.ndarray
        Shape (N, D) float matrix. Usually already standardized per the
        paper's protocol, but this function does not re-standardize — it
        operates on whatever X is passed in, matching the paper's code.
    seed : int
        Unused. Accepted for API symmetry with other MNAR injectors (the
        mask is deterministic given X).

    Returns
    -------
    MNARResult
        Named-tuple-style dataclass with `X_nan`, `X_zero`, `observed_mask`.
    """
    # seed is intentionally unused — the paper's mechanism is deterministic.
    del seed

    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D; got shape {X.shape}")

    N, D = X.shape
    X_nan = X.copy()
    half = D // 2
    if half > 0:
        # Compute column means on the first D/2 columns (exactly as in task01.py).
        col_means = np.mean(X_nan[:, :half], axis=0)
        # Boolean mask: True where the entry exceeds its column mean.
        mask_large = X_nan[:, :half] > col_means
        # Broadcast assign NaN. The paper's code writes via boolean indexing
        # on a view; we reproduce that behavior.
        left = X_nan[:, :half]
        left[mask_large] = np.nan
        X_nan[:, :half] = left

    X_zero = np.where(np.isnan(X_nan), 0.0, X_nan)
    observed_mask = ~np.isnan(X_nan)
    return MNARResult(X_nan=X_nan, X_zero=X_zero, observed_mask=observed_mask)


def imputation_rmse(
    X_true: np.ndarray,
    X_reconstructed: np.ndarray,
    observed_mask: np.ndarray,
) -> float:
    """Imputation RMSE evaluated only on masked entries.

    Verbatim formula from `docs/notmiwae/task01.py` line 115:

        RMSE = sqrt( sum( (Xtrain - Xrec)**2 * (1 - S) ) / sum(1 - S) )

    where `S` is the observed mask (True/1 where observed). Contributions
    from observed positions are weighted by zero and drop out.

    Parameters
    ----------
    X_true : np.ndarray
        Ground-truth matrix (e.g., the standardized pre-mask X).
    X_reconstructed : np.ndarray
        Imputer output, same shape as X_true. Values at observed positions
        do not matter for the metric (they are weighted out).
    observed_mask : np.ndarray
        Boolean matrix; True = observed, False = masked. Same shape as
        X_true.

    Returns
    -------
    float
        RMSE over masked entries. Returns 0.0 if no entries are masked.
    """
    X_true = np.asarray(X_true, dtype=np.float64)
    X_rec = np.asarray(X_reconstructed, dtype=np.float64)
    S = np.asarray(observed_mask, dtype=np.float64)  # 1 where observed
    missing_weight = 1.0 - S  # 1 where masked
    denom = missing_weight.sum()
    if denom == 0:
        return 0.0
    sq_err = (X_true - X_rec) ** 2
    return float(np.sqrt(np.sum(sq_err * missing_weight) / denom))
