"""Classical imputation baselines for the Ipsen et al. 2021 benchmark.

All three baselines expose a common interface:

    imputer = <Baseline>(random_state=seed)
    X_reconstructed = imputer.fit_impute(X_masked, observed_mask)

`X_masked` may be a numpy array or DataFrame with NaN at masked entries
(sklearn's imputers take the mask implicitly via NaN; `observed_mask` is
accepted for API clarity but not required). `X_reconstructed` is a
fully-reconstructed numpy array with imputed values in the masked positions
and original values in the observed positions.

Baselines (matching `docs/notmiwae/task01.py` lines 112-132):

    MeanImputer        sklearn.impute.SimpleImputer(strategy='mean')
    MICEImputer        sklearn.impute.IterativeImputer(max_iter=10, random_state=seed)
    MissForestImputer  sklearn.impute.IterativeImputer(
                           estimator=RandomForestRegressor(n_estimators=100, random_state=seed))
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd


def _to_numpy(X) -> np.ndarray:
    """Convert DataFrame or array to float64 ndarray with NaN preserved."""
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=np.float64)
    return np.asarray(X, dtype=np.float64)


class MeanImputer:
    """Impute each masked entry with its column's observed mean.

    Mirrors `task01.py` lines 112-115:
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp.fit(Xnan); Xrec = imp.transform(Xnan)
    """

    def __init__(self, random_state: int = 0):
        # Mean imputation is deterministic; seed accepted for API symmetry.
        self.random_state = random_state

    def fit_impute(
        self,
        X_masked,
        observed_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        del observed_mask  # taken implicitly via NaN
        from sklearn.impute import SimpleImputer

        X = _to_numpy(X_masked)
        imp = SimpleImputer(missing_values=np.nan, strategy="mean")
        imp.fit(X)
        return imp.transform(X)


class MICEImputer:
    """Multivariate Imputation by Chained Equations, sklearn's IterativeImputer.

    Mirrors `task01.py` lines 119-123:
        imp = IterativeImputer(max_iter=10, random_state=0)
        imp.fit(Xnan); Xrec = imp.transform(Xnan)
    """

    def __init__(self, max_iter: int = 10, random_state: int = 0):
        self.max_iter = max_iter
        self.random_state = random_state

    def fit_impute(
        self,
        X_masked,
        observed_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        del observed_mask
        # `enable_iterative_imputer` is a required sklearn experimental import.
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer

        X = _to_numpy(X_masked)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imp = IterativeImputer(
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
            imp.fit(X)
            return imp.transform(X)


class MissForestImputer:
    """missForest-style iterative imputation using a RandomForest per feature.

    Mirrors `task01.py` lines 127-131:
        estimator = RandomForestRegressor(n_estimators=100)
        imp = IterativeImputer(estimator=estimator)
        imp.fit(Xnan); Xrec = imp.transform(Xnan)

    We additionally pass `random_state` through to both the imputer and the
    underlying RandomForest so seeded runs are reproducible.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        random_state: int = 0,
        max_iter: int = 10,
    ):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_iter = max_iter

    def fit_impute(
        self,
        X_masked,
        observed_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        del observed_mask
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.impute import IterativeImputer

        X = _to_numpy(X_masked)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimator = RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
            )
            imp = IterativeImputer(
                estimator=estimator,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
            imp.fit(X)
            return imp.transform(X)


REGISTRY = {
    "mean": MeanImputer,
    "mice": MICEImputer,
    "missforest": MissForestImputer,
}
