"""sklearn-style fit/predict wrapper around christensen_core.ChristensenEstimator.

Mirrors the shape of phase1_pereira_benchmark.minimax_adapter.ScoreMinimaxRegressor
so the benchmark harness can treat both methods uniformly.

Usage pattern in the harness:
    model = ChristensenRegressor(mechanism_name=mech)
    model.fit(X_train_arr, y_train_float, response_mask=mask)
    y_pred = model.predict(X_test_arr)

The mechanism name is needed because the right Q class depends on the
Pereira mechanism (see christensen_core.pereira_q). Passing it via the
constructor keeps the harness signature unchanged while giving this adapter
what it needs.

Uncertainty set Q follows Christensen's reference-based pattern (Christensen &
Connault 2023; Adjaho & Christensen 2022): a neighborhood of user-specified
radius `delta` around the empirical observation rate q_hat computed from the
training response_mask. See christensen_core.reference_based_q.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ChristensenRegressor:
    """Adapter: Pereira mechanism name + data q_hat -> reference-based QClass -> ChristensenEstimator.

    Q-specification policy (post-audit 2026-04-17):
      - Default behavior: use fixed `delta=0.30` regardless of mechanism_name. The
        mechanism_name is used ONLY for Q-class dispatch (e.g., MBOV_Lower -> monotone
        increasing; MBUV -> constant). This avoids the "oracle leak" where the
        benchmark passes pre-calibrated deltas per mechanism, giving the estimator
        unfair advantage over baselines.
      - Opt-in: if `use_mechanism_prior=True`, the adapter looks up mechanism-calibrated
        delta from MECHANISM_DELTA. This is legitimate ONLY when the user has genuine
        domain knowledge of the selection mechanism in deployment (not a benchmark
        protocol leak). For the Pereira benchmark, this flag is DISABLED by default.
      - Explicit `delta=<float>` always wins. For ablations.

    Historical note: pre-audit, the default was `delta=None` + `adaptive_centered_q_for`,
    which constituted an oracle leak because the benchmark supplied the true mechanism.
    Removed 2026-04-17 per AUDIT_v3_SYNTHESIS.md Finding #2.
    """
    mechanism_name: str
    fit_intercept: bool = True
    delta: float | None = 0.30  # Default fixed to avoid oracle leak
    use_mechanism_prior: bool = False  # Opt-in for legitimate domain-knowledge use

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray,
        response_mask: np.ndarray,
    ) -> "ChristensenRegressor":
        """Fit the Christensen estimator with a reference-based Q.

        Default uses `delta=0.30` centered on q_hat. Mechanism-prior lookup is
        opt-in via `use_mechanism_prior=True` to prevent benchmark oracle leaks.
        """
        from christensen_core.reference_based_q import (
            adaptive_centered_q_for,
            centered_q_for,
        )
        from christensen_core.estimator import ChristensenEstimator

        mask = np.asarray(response_mask, dtype=bool)
        if self.use_mechanism_prior and self.delta is None:
            q_cls = adaptive_centered_q_for(self.mechanism_name, mask)
        else:
            effective_delta = self.delta if self.delta is not None else 0.30
            q_cls = centered_q_for(self.mechanism_name, mask, delta=effective_delta)
        self._inner = ChristensenEstimator(q_class=q_cls, fit_intercept=self.fit_intercept)

        X_arr = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        Y_tilde = np.where(mask, np.asarray(y, dtype=float), 0.0)
        self._inner.fit(X_arr, Y_tilde, mask)
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Delegate to the inner estimator's predict."""
        X_arr = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        return self._inner.predict(X_arr)
