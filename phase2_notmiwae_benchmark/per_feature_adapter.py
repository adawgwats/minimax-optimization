"""Per-feature Christensen imputer for not-MIWAE's element-wise MNAR protocol.

not-MIWAE masks half of each dataset's features (the first D/2 columns) above
the feature mean. Christensen's framework handles ROW-LEVEL MNAR on one
regression target at a time, not element-wise multi-feature missingness. To
bridge, we run Christensen's estimator ONCE PER MASKED FEATURE:

    for i in first_half_features:
        predictors = always-observed features (second D/2)
        target = feature i
        response_mask = observed_mask[:, i]
        fit ChristensenEstimator(MonotoneInY("decreasing"))
        predict feature i values for the masked rows

Then stack the predictions back into the feature matrix. RMSE is computed on
exactly the positions that were masked (matching not-MIWAE's metric).

The QClass is `MonotoneInY(direction="decreasing")` because self-masking-above-mean
is q(y) = 1{y <= mean}, which is a monotone decreasing step function in y.
`SelfMaskingAboveMean` is registered in christensen_core.reference_based_q with
delta=0.30 (see MECHANISM_DELTA).

Interface mirrors `baselines.py`:
    imp = ChristensenFeatureImputer()
    X_rec = imp.fit_impute(X_masked, observed_mask)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from christensen_core.estimator import ChristensenEstimator
from christensen_core.reference_based_q import adaptive_centered_q_for


@dataclass
class ChristensenFeatureImputer:
    """Per-feature Christensen imputer matching not-MIWAE's self-masking protocol.

    Assumes the first D/2 features may have missingness (per not-MIWAE's mask_first_half
    pattern) and the second D/2 are always observed. For each of the first D/2,
    runs a Christensen regression predicting that feature from the always-observed
    half, then overwrites the masked positions with the predictions.
    """
    mechanism_name: str = "SelfMaskingAboveMean"

    def fit_impute(self, X_masked: np.ndarray, observed_mask: np.ndarray) -> np.ndarray:
        """Impute a masked feature matrix via per-feature Christensen regression.

        Args:
            X_masked: (N, D) with np.nan at masked entries
            observed_mask: (N, D) bool, True where observed

        Returns:
            X_rec: (N, D) with the same observed values and Christensen-predicted
                values at masked positions.
        """
        X_masked = np.asarray(X_masked, dtype=float)
        observed_mask = np.asarray(observed_mask, dtype=bool)
        if X_masked.shape != observed_mask.shape:
            raise ValueError(
                f"shape mismatch: X_masked {X_masked.shape} vs observed_mask {observed_mask.shape}"
            )

        N, D = X_masked.shape
        half = D // 2
        # Always-observed half is the predictor block
        predictor_cols = list(range(half, D))
        # Masked half is what we impute
        target_cols = list(range(half))

        # Zero-fill NaN to comply with Christensen's Y_tilde convention
        X_work = np.where(np.isnan(X_masked), 0.0, X_masked)

        X_rec = X_work.copy()
        X_predictors = X_work[:, predictor_cols]

        for i in target_cols:
            y_i = X_work[:, i].copy()
            resp_i = observed_mask[:, i]

            # If fewer than 2 observed rows for this feature, fall back to zero
            # (should essentially never happen on not-MIWAE datasets but defensive)
            if resp_i.sum() < max(3, X_predictors.shape[1] + 1):
                X_rec[~resp_i, i] = 0.0
                continue

            # Build Q class via the reference-based adaptive dispatch
            # (handles q_hat from resp_i and mechanism-specific delta)
            q_cls = adaptive_centered_q_for(self.mechanism_name, resp_i)
            est = ChristensenEstimator(q_class=q_cls, fit_intercept=True)
            est.fit(X_predictors, y_i, resp_i)

            # Predict for ALL rows and overwrite the masked positions
            y_i_pred = est.predict(X_predictors)
            X_rec[~resp_i, i] = y_i_pred[~resp_i]

        return X_rec
