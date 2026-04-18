"""Tests for christensen_core.q_classes.MonotoneInY and its outer solver.

Scope:
    - Basic class contract (dim_theta, theta_bounds, q_values semantics)
    - Outer solver respects the monotone inequality constraint
    - Outer solver beats naive OLS-on-respondents under strong self-masking MNAR
"""
from __future__ import annotations

import numpy as np
import pytest

from christensen_core.q_classes import (
    ConstantQ,
    MonotoneInY,
    QClassConfig,
)
from christensen_core.outer_solver import solve_outer
from christensen_core.reference_based_q import centered_config


# ---------------------------------------------------------------------------
# Class-level tests
# ---------------------------------------------------------------------------
def test_dim_theta() -> None:
    """MonotoneInY(n_knots=5).dim_theta() == 5."""
    q = MonotoneInY(n_knots=5)
    assert q.dim_theta() == 5
    # Also check a different K.
    q2 = MonotoneInY(n_knots=7)
    assert q2.dim_theta() == 7


def test_theta_bounds_shape() -> None:
    """theta_bounds returns (K-vector, K-vector), both inside [q_min, q_max]."""
    config = QClassConfig(q_min=0.1, q_max=0.9)
    q = MonotoneInY(n_knots=5, config=config)
    low, high = q.theta_bounds()
    assert low.shape == (5,)
    assert high.shape == (5,)
    assert np.all(low >= config.q_min - 1e-12)
    assert np.all(high <= config.q_max + 1e-12)
    assert np.all(low <= high)
    # Specifically, bounds should be q_min and q_max replicated — the monotone
    # constraint is enforced by the outer solver, not baked into bounds.
    np.testing.assert_allclose(low, np.full(5, config.q_min))
    np.testing.assert_allclose(high, np.full(5, config.q_max))


def test_q_values_constant_theta_equals_constant() -> None:
    """With theta = [0.5, 0.5, 0.5, 0.5, 0.5], q_values is constant 0.5 everywhere
    (matches ConstantQ(0.5) behavior)."""
    q_mono = MonotoneInY(n_knots=5)
    theta = np.full(5, 0.5)
    Y_tilde = np.array([0.1, 0.25, 0.5, 0.75, 1.0])
    X = np.ones((5, 2))

    out_mono = q_mono.q_values(theta, X, Y_tilde)
    np.testing.assert_allclose(out_mono, np.full(5, 0.5))

    # Cross-check: ConstantQ at 0.5 on the same data gives the same vector.
    q_const = ConstantQ()
    out_const = q_const.q_values(np.array([0.5]), X, Y_tilde)
    np.testing.assert_allclose(out_mono, out_const)


def test_q_values_monotone_decreasing() -> None:
    """With decreasing theta and increasing Y_tilde, q_values is monotone
    non-increasing."""
    q = MonotoneInY(direction="decreasing", n_knots=5)
    theta = np.array([0.9, 0.8, 0.6, 0.4, 0.2])
    Y_tilde = np.array([0.1, 0.25, 0.5, 0.75, 1.0])  # all nonzero -> all observed
    X = np.ones((5, 2))

    out = q.q_values(theta, X, Y_tilde)
    diffs = np.diff(out)
    assert np.all(diffs <= 1e-12), f"not monotone decreasing: out={out}, diffs={diffs}"
    # Sanity: first entry should equal theta[0] (at y_lo = 0.1) and last
    # should equal theta[-1] (at y_hi = 1.0).
    assert out[0] == pytest.approx(theta[0])
    assert out[-1] == pytest.approx(theta[-1])


def test_q_values_all_zero_y_tilde_falls_back_to_mean() -> None:
    """Edge case: if Y_tilde is all zeros (no observed labels), q_values returns
    mean(theta) everywhere."""
    q = MonotoneInY(n_knots=5)
    theta = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
    Y_tilde = np.zeros(5)
    X = np.ones((5, 2))
    out = q.q_values(theta, X, Y_tilde)
    np.testing.assert_allclose(out, np.full(5, theta.mean()))


def test_q_values_clipped_to_box() -> None:
    """Output of q_values is clipped to [q_min, q_max]."""
    config = QClassConfig(q_min=0.2, q_max=0.8)
    q = MonotoneInY(direction="decreasing", n_knots=5, config=config)
    # Push theta outside the box; class must still return values inside.
    theta = np.array([1.5, 1.2, 0.9, 0.1, -0.5])
    Y_tilde = np.array([0.1, 0.25, 0.5, 0.75, 1.0])
    X = np.ones((5, 2))
    out = q.q_values(theta, X, Y_tilde)
    assert np.all(out >= config.q_min - 1e-12)
    assert np.all(out <= config.q_max + 1e-12)


# ---------------------------------------------------------------------------
# Outer solver tests
# ---------------------------------------------------------------------------
def _make_self_masking_above_mean_dataset(
    n: int = 1000,
    d: int = 4,
    noise: float = 1.0,
    steepness: float = 3.0,
    beta_true: np.ndarray | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Continuous-Y MNAR: q(y) = sigmoid(-steepness * (y - median(y))).

    Defaults picked so that OLS-on-respondents is clearly biased:
    larger noise relative to signal (noise=1.0 vs beta~0.3-0.5) plus strong
    self-masking (steepness=3) gives OLS a biased intercept and attenuated
    slopes, leaving room for a structured-Q adversary to improve.

    Returns (X_train, Y_tilde_train, response_mask_train, X_test, Y_test).
    """
    rng = np.random.default_rng(seed)
    if beta_true is None:
        beta_true = np.array([0.0, 0.5, -0.3, 0.2])[:d]
    # Train
    X_raw = rng.standard_normal((n, d - 1))
    X = np.concatenate([np.ones((n, 1)), X_raw], axis=1)
    y = X @ beta_true + noise * rng.standard_normal(n)

    # q(y) decreasing in y: self-masking above mean
    y_med = float(np.median(y))
    q_true = 1.0 / (1.0 + np.exp(steepness * (y - y_med)))
    # Floor above 0.05 to avoid extreme IPW weights
    q_true = np.clip(q_true, 0.10, 0.99)
    response = rng.random(n) < q_true
    Y_tilde = np.where(response, y, 0.0)

    # Test set (fully observed, from same distribution)
    n_test = max(500, n // 2)
    X_test_raw = rng.standard_normal((n_test, d - 1))
    X_test = np.concatenate([np.ones((n_test, 1)), X_test_raw], axis=1)
    Y_test = X_test @ beta_true + noise * rng.standard_normal(n_test)

    return X, Y_tilde, response.astype(bool), X_test, Y_test


def test_outer_solver_respects_monotone() -> None:
    """On synthetic continuous-Y MNAR data with known decreasing q(y) mechanism,
    MonotoneInY(direction='decreasing') should return a θ* that is monotone
    non-increasing up to numerical tolerance."""
    X, Y_tilde, mask, _X_test, _Y_test = _make_self_masking_above_mean_dataset(
        n=500, d=4, steepness=5.0, seed=0
    )
    q_class = MonotoneInY(
        direction="decreasing", n_knots=5, config=QClassConfig(q_min=0.05, q_max=1.0)
    )
    res = solve_outer(q_class, X, Y_tilde, mask)

    assert res.theta_star.shape == (5,)
    diffs = np.diff(res.theta_star)
    # Allow tiny numerical slack from SLSQP.
    assert np.all(diffs <= 1e-6), (
        f"theta* is not monotone decreasing: theta_star={res.theta_star}, diffs={diffs}"
    )


def test_outer_solver_increasing_direction_also_respects_monotone() -> None:
    """Symmetry check: MonotoneInY(direction='increasing') also returns a
    non-decreasing θ*."""
    X, Y_tilde, mask, _X_test, _Y_test = _make_self_masking_above_mean_dataset(
        n=400, d=4, steepness=3.0, seed=1
    )
    q_class = MonotoneInY(
        direction="increasing", n_knots=5, config=QClassConfig(q_min=0.05, q_max=1.0)
    )
    res = solve_outer(q_class, X, Y_tilde, mask)
    diffs = np.diff(res.theta_star)
    assert np.all(diffs >= -1e-6), (
        f"theta* not monotone increasing: theta_star={res.theta_star}"
    )


def test_outer_solver_beats_ols_under_selection() -> None:
    """Under strong self-masking-above-mean, Christensen+MonotoneInY should have
    lower average test MSE than OLS-on-respondents across 30 seeds, with 95% CI
    separation.

    We use a centered-config (delta=0.30 around q_hat), matching the
    reference-based setup Christensen uses in deployment. A wide box Q (e.g.,
    [0.05, 1.0]) lets the adversary pick arbitrarily small q's at large y,
    blowing up IPW weights and biasing β̂ toward 0; the centered config
    calibrates Q to a realistic neighborhood around the empirical response
    rate, which is the way we use this class in the benchmark pipeline.

    If this test fails, report honest numbers — it's signal about whether
    MonotoneInY on continuous Y is actually viable at this sample size.
    """
    christ_mses = []
    ols_mses = []
    n_seeds = 30

    for seed in range(n_seeds):
        X, Y_tilde, mask, X_test, Y_test = _make_self_masking_above_mean_dataset(
            n=1000, d=4, noise=1.0, steepness=3.0, seed=seed
        )
        # OLS on respondents only
        X_resp = X[mask]
        Y_resp = Y_tilde[mask]
        beta_ols, *_ = np.linalg.lstsq(X_resp, Y_resp, rcond=None)
        pred_ols = X_test @ beta_ols
        ols_mses.append(float(np.mean((pred_ols - Y_test) ** 2)))

        # Christensen + MonotoneInY, centered around the empirical response rate
        # (delta=0.30 is the default for MBOV-family / SelfMaskingAboveMean).
        q_hat = float(mask.mean())
        config = centered_config(q_hat, delta=0.30)
        q_class = MonotoneInY(direction="decreasing", n_knots=5, config=config)
        res = solve_outer(q_class, X, Y_tilde, mask)
        pred_christ = X_test @ res.beta_hat
        christ_mses.append(float(np.mean((pred_christ - Y_test) ** 2)))

    christ_mses = np.array(christ_mses)
    ols_mses = np.array(ols_mses)

    c_mean = christ_mses.mean()
    o_mean = ols_mses.mean()
    c_se = christ_mses.std(ddof=1) / np.sqrt(n_seeds)
    o_se = ols_mses.std(ddof=1) / np.sqrt(n_seeds)

    # Report via the assertion message so pytest -v shows the numbers.
    improvement = o_mean - c_mean
    # 95% CI separation: Christensen upper CI < OLS lower CI.
    c_hi = c_mean + 1.96 * c_se
    o_lo = o_mean - 1.96 * o_se

    msg = (
        f"Christensen MSE = {c_mean:.4f} +/- {1.96*c_se:.4f}, "
        f"OLS MSE = {o_mean:.4f} +/- {1.96*o_se:.4f}, "
        f"improvement = {improvement:.4f}, "
        f"Christensen 95%-upper = {c_hi:.4f}, OLS 95%-lower = {o_lo:.4f}"
    )
    assert c_hi < o_lo, f"No 95% CI separation of Christensen below OLS. {msg}"
