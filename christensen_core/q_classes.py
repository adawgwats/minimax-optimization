"""Structured uncertainty-set classes for Christensen's minimax problem.

PDF page 5 last paragraph:

    "Different Q represent different plausible types of non-response. For
    instance, perhaps Ỹᵢ is blood alcohol content and people are less likely
    to respond as this variable increases, so Q might be a set of decreasing
    functions q(x, y) = g(y) with g decreasing in y."

This module defines an abstract `QClass` interface and concrete implementations
corresponding to specific plausible selection regimes. Each concrete class:

1. Parameterizes a family of response-probability functions q(x, ỹ) by a
   finite-dimensional parameter θ.
2. Given θ, returns per-example q_values for use by moments.compute_r_n.
3. Exposes its parameter space so the outer solver can maximize over θ.

## Why classes matter for faithfulness

Christensen's Q is NOT a box `[q_min, q_max]` on per-example q's — that is the
DRO variant. Christensen's Q is a STRUCTURED class over the FUNCTION q(x,y).
"Set of decreasing functions of y" is an infinite-dimensional class; we need
a finite-dimensional approximation that preserves the structural restriction.

For binary Y (which is what the Pereira benchmark currently uses after
LPM binarization), the monotone-in-y class degenerates dramatically: q only
takes two values, q(·, 0) and q(·, 1). Specific q_classes handle this cleanly.

## Classes to implement for v1

- `ConstantQ`: q(x, ỹ) = q_0, a scalar. Reduces the estimator to classical
  OLS with MAR correction. Used as a sanity-check baseline and as the
  "correct" Q class for MNAR mechanisms that are actually MAR-on-label
  (e.g., MBUV on label-only injection).

- `MonotoneInY(direction)`: q(x, ỹ) = g(ỹ) where g is monotone in y, either
  increasing or decreasing. The "blood alcohol" example from the PDF is
  `MonotoneInY('decreasing')`. Well-matched to Pereira's MBOV_Lower
  (MBOV_Lower removes Y=0 observations preferentially, which in Christensen
  terms is g(0) < g(1), i.e., increasing in y).

- `MonotoneInScore(score_fn)`: q depends monotonically on a scalar score
  derived from (x, ỹ). Used for MBIR (missingness depends on an unobserved
  x-variable that correlates with some observable features).

- `Parametric2ParamForBinary`: for binary y, q has only two possible values
  (q_0 = q(·,0) and q_1 = q(·,1)), giving a 2-dimensional Q class. This is
  the right parameterization for any monotone class when y is binary.

Additional classes (`LipschitzInY`, `DependentOnUnobservedX`, etc.) are out
of scope for v1 and explicitly deferred.

## Parameter space conventions

Each class defines:

- `dim_theta`: dimension of the parameter space
- `theta_bounds`: box bounds on θ (for the outer optimizer)
- `q_values(theta, X, Y_tilde)` → (n,) array of per-example response probs
- `clip_to_box(q)` → q clipped into [q_min, q_max] for numerical safety

The outer solver (`outer_solver.py`) treats a QClass as a black-box parameter
space and maximizes the inner objective over θ. The structure of the class
determines the outer algorithm (grid, golden section, convex, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QClassConfig:
    """Shared config: numerical floor and ceiling on q values."""
    q_min: float = 0.05
    q_max: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 < self.q_min <= self.q_max <= 1.0:
            raise ValueError("require 0 < q_min <= q_max <= 1")


class QClass(ABC):
    """Base class for structured uncertainty sets."""

    config: QClassConfig

    @abstractmethod
    def dim_theta(self) -> int:
        """Dimension of the parameter space for this class."""

    @abstractmethod
    def theta_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns (low, high) each of shape (dim_theta,) — box bounds on θ."""

    @abstractmethod
    def q_values(self, theta: np.ndarray, X: np.ndarray, Y_tilde: np.ndarray) -> np.ndarray:
        """Return (n,) per-example response probabilities for the given θ."""

    def clip(self, q: np.ndarray) -> np.ndarray:
        """Clip q to [q_min, q_max]. Called by subclasses before returning."""
        return np.clip(q, self.config.q_min, self.config.q_max)


# ---------------------------------------------------------------------------
# ConstantQ
# ---------------------------------------------------------------------------
class ConstantQ(QClass):
    """Q = {q : q(x,y) = q_0, q_0 ∈ [q_min, q_max]}.

    Reduces Christensen's estimator to MAR OLS correction: β̂ = (1/q₀) * β̂_OLS.
    Serves as the "Q is trivial" baseline and the correct Q for MNAR mechanisms
    that happen to behave as MAR on the observed label (e.g., MBUV on a label
    column where the unobserved feature is independent of y).

    Parameter space: θ = [q_0], a scalar in [q_min, q_max].
    """

    def __init__(self, config: QClassConfig | None = None):
        self.config = config or QClassConfig()

    def dim_theta(self) -> int:
        return 1

    def theta_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.array([self.config.q_min], dtype=float),
            np.array([self.config.q_max], dtype=float),
        )

    def q_values(self, theta: np.ndarray, X: np.ndarray, Y_tilde: np.ndarray) -> np.ndarray:
        # Broadcast the single scalar parameter to the full sample length, then
        # clip into [q_min, q_max] for numerical safety in case the outer
        # solver supplies a θ slightly outside the box (rare but harmless).
        n = len(Y_tilde)
        out = np.full(n, float(theta[0]), dtype=float)
        return self.clip(out)


# ---------------------------------------------------------------------------
# Parametric2ParamForBinary
# ---------------------------------------------------------------------------
class Parametric2ParamForBinary(QClass):
    """Q = {q : q(x, ỹ) depends only on ỹ ∈ {0, 1}}.

    For binary Y, any q(x, y) that depends only on y collapses to two values:
    q_0 = q(·, 0), q_1 = q(·, 1). Box constraints `q_0, q_1 ∈ [q_min, q_max]`.

    Specialization flags:
        monotone: None | 'increasing' | 'decreasing'
            - 'increasing' enforces q_0 <= q_1 (equivalent to Christensen's
              "g decreasing in y" when y is encoded with high = bad outcome).
            - 'decreasing' enforces q_0 >= q_1.
            - None gives the full 2D box.

    This is the right Q class for label-level MNAR on binary outcomes, which
    is what most of the Pereira datasets become after LPM binarization.

    Parameter space: θ = [q_0, q_1], 2-dimensional.
    """

    def __init__(
        self,
        monotone: str | None = None,
        config: QClassConfig | None = None,
    ):
        self.config = config or QClassConfig()
        if monotone not in (None, "increasing", "decreasing"):
            raise ValueError("monotone must be None, 'increasing', or 'decreasing'")
        self.monotone = monotone

    def dim_theta(self) -> int:
        return 2

    def theta_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        # Plain box bounds on (q_0, q_1). The `monotone` flag is metadata that
        # the outer solver reads (via self.monotone) and converts into a linear
        # inequality constraint (q_0 <= q_1 or q_0 >= q_1). We do NOT bake that
        # constraint into the bounds returned here.
        q_min = self.config.q_min
        q_max = self.config.q_max
        return (
            np.array([q_min, q_min], dtype=float),
            np.array([q_max, q_max], dtype=float),
        )

    def q_values(self, theta: np.ndarray, X: np.ndarray, Y_tilde: np.ndarray) -> np.ndarray:
        # Convention: Y_tilde[i] == 0 selects theta[0] (q_0 = q(·,0)); any
        # other value selects theta[1] (q_1 = q(·,1)). For binary {0,1} labels
        # this is exactly the two-point parameterization. If a caller mistakenly
        # passes continuous y, every nonzero entry will collapse to theta[1]
        # silently — that is by design (the right Q class for continuous y is
        # MonotoneInY, not this one).
        y = np.asarray(Y_tilde)
        out = np.where(y == 0, float(theta[0]), float(theta[1])).astype(float)
        return self.clip(out)


# ---------------------------------------------------------------------------
# MonotoneInY (for continuous Y — future work)
# ---------------------------------------------------------------------------
class MonotoneInY(QClass):
    """Q = {q(x, y) = g(y), g monotone in y, g(y) ∈ [q_min, q_max]}.

    Continuous-y version. Parameterize g(y) as piecewise-linear on a grid of
    knots, with monotonicity enforced via ordered θ.

    Parameter space: θ = [g(y_1), g(y_2), ..., g(y_K)] for K grid points on
    the observed y range, subject to θ_i ≤ θ_{i+1} (for increasing direction).

    NOTE: Pereira's binary-label benchmark does not exercise this class; it is
    scaffolded here for future continuous-outcome extensions.
    """

    def __init__(
        self,
        direction: str = "decreasing",
        n_knots: int = 5,
        config: QClassConfig | None = None,
    ):
        if direction not in ("increasing", "decreasing"):
            raise ValueError("direction must be 'increasing' or 'decreasing'")
        self.config = config or QClassConfig()
        self.direction = direction
        self.n_knots = n_knots

    def dim_theta(self) -> int:
        return int(self.n_knots)

    def theta_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        # Plain box bounds; the monotone ordering θ_i <= θ_{i+1} (or the reverse)
        # is enforced by the outer solver via SLSQP inequality constraints, not
        # here — same pattern as Parametric2ParamForBinary.
        K = int(self.n_knots)
        q_min = self.config.q_min
        q_max = self.config.q_max
        return (
            np.full(K, q_min, dtype=float),
            np.full(K, q_max, dtype=float),
        )

    def q_values(self, theta: np.ndarray, X: np.ndarray, Y_tilde: np.ndarray) -> np.ndarray:
        """Piecewise-linear interpolation of g(y) over a K-knot grid.

        Protocol:
            1. Compute the y-range from the OBSERVED (non-zero) Y_tilde entries.
            2. Place K knots evenly over that range.
            3. np.interp(Y_tilde, y_knots, theta) gives g(Y_tilde[i]) for each i
               (for non-respondents where Y_tilde = 0, this returns whatever θ
               at y_knot[0] extrapolates to — their q is unused by compute_r_n
               so this is safe).
            4. Clip to [q_min, q_max].

        Edge case: if Y_tilde is all zeros (no respondents), fall back to
        the mean of theta (ConstantQ-like behavior).
        """
        theta = np.asarray(theta, dtype=float).reshape(-1)
        Y = np.asarray(Y_tilde, dtype=float).reshape(-1)
        K = int(self.n_knots)
        if theta.shape[0] != K:
            raise ValueError(f"theta has shape {theta.shape}; expected ({K},)")

        observed = Y[Y != 0.0]
        if observed.size == 0:
            # No observed labels; fall back to mean(theta) everywhere.
            out = np.full(Y.shape[0], float(theta.mean()), dtype=float)
            return self.clip(out)

        y_lo = float(observed.min())
        y_hi = float(observed.max())
        if y_hi <= y_lo:
            # All observed labels are identical; collapse to a single knot value.
            # np.interp handles degenerate grids poorly, so return mean(theta).
            out = np.full(Y.shape[0], float(theta.mean()), dtype=float)
            return self.clip(out)

        y_knots = np.linspace(y_lo, y_hi, K)
        # np.interp clamps to the endpoint values for x < y_knots[0] and
        # x > y_knots[-1], which gives the right "outside-range" semantics.
        out = np.interp(Y, y_knots, theta)
        return self.clip(out)
