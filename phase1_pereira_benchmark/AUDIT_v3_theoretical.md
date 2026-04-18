# Phase 1 Theoretical Audit, v3 — Independent Review

Independent assessment of `christensen_core/` and the Phase 1 Pereira benchmark
against Christensen's primary source (the 7-page 2020 note "Regression example
for prediction") and Christensen's broader published work. The reviewer formed
their own reading of the primary material before reading the repository's own
audit notes. Findings below.

---

## 1. What Christensen actually specifies

Christensen's 2020 note is short, self-contained, and essentially pedagogical.
Its content is:

- **Setup (§1.2).** Training data `(Xᵢ, Yᵢ)` are drawn i.i.d. from a fixed
  population `P`. Selection corrupts the labels: `Ỹᵢ = εᵢ Yᵢ` with
  `εᵢ ∼ Bernoulli(q(Xᵢ, Yᵢ))`. Under MAR `q` is constant and OLS-on-responders
  divided by `q` recovers `β = E[XX']⁻¹ E[XY]`. Under MNAR `q(·)` depends on
  unobserved `Y`, so naive reweighting fails.

- **Min-max objective (§1.3).** Christensen frames robustness to MNAR as
  ```
  min_{β̂}  max_{q ∈ Q}  E[ (β̂ − β)' E[Xₙ₊₁Xₙ₊₁'] (β̂ − β) ]
  ```
  — the worst-case (in `q ∈ Q`) squared-prediction-error excess over the
  Bayes-optimal linear predictor, where the expectation is taken under a single
  fixed `P` with the `q`-governed selection layer sitting on top.

- **Linear-estimator restriction + IPW trick (§1.3 → Problem 2).** Christensen
  restricts to estimators of the form `β̂ = M·b_n + m` with
  `b_n = (1/n) Σ Xᵢ Ỹᵢ`. Using the IPW identity
  `E[XY] = E[(ε/q) X Y] = E[(1/q) X Ỹ]`, and dropping a `q`-free constant, the
  problem reduces exactly to
  ```
  min_{M,m} max_{q∈Q}  (M b_n + m)' W_n (M b_n + m) − 2 (M b_n + m)' r_n(q)
  ```
  with `W_n = (1/n) Σ Xᵢ Xᵢ'` and `r_n(q) = (1/n) Σ_{i∈Rₙ} (1/q(Xᵢ, Ỹᵢ)) Xᵢ Ỹᵢ`.
  Non-respondents contribute `0` to `b_n` (because `Ỹᵢ = 0`) and are excluded
  from `r_n` by definition of `Rₙ`.

- **Inner solve (§1.5).** For fixed `q`, the inner is a quadratic in
  `a = vec([M | m])`. The FOC is
  `(bb ⊗ I_d) W (bb' ⊗ I_d) a = (bb ⊗ I_d) r`, rank-deficient but any
  least-squares solution yields the same `β̂ = M b_n + m`.

- **Uncertainty set Q.** Christensen gives exactly one example: for blood-alcohol
  censoring, `Q = { q(x, y) = g(y) : g decreasing in y }`. He does NOT specify a
  box `[q_min, q_max]`, a ball around an empirical reference, or a Wasserstein
  neighborhood. The class is shape-restricted and structural.

- **Outer solve.** The PDF hand-waves ("use the algorithms for min-max
  problems") and leaves it to the implementer.

**In Christensen's broader published work** (Christensen & Connault 2023,
Econometrica; Adjaho & Christensen 2022, QE R&R) the uncertainty set is
specified very differently from both the 2020 note and from this repo:

- *Counterfactual Sensitivity and Robustness* (2023) uses **φ-divergence
  neighborhoods** around a parametric reference distribution of latent
  variables. The radius is an explicit tuning/reporting parameter; the paper's
  central move is to produce a **profile of counterfactuals over radius**
  rather than to pick one radius.
- *Externally Valid Policy Choice* (2022) uses **Wasserstein neighborhoods**
  around the empirical joint distribution, with radius interpreted in terms of
  observable ATE deviations between populations — but again the radius is
  user-chosen and reported as a sensitivity axis, not calibrated adaptively
  from the data.

So the "Christensen pattern" for uncertainty sets in his published work is
(a) divergence/metric ball around an identified reference, (b) radius as a
transparent tuning knob, (c) results reported as a **profile over radius**,
not a single number. Data-adaptive radius selection does not appear in either
paper I reviewed.

The mathematical rigor expected in this literature is high: primitives are
stated precisely, the reference distribution and divergence are identified,
and identification / consistency results are proved. The 2020 note is
explicitly *not* at that bar — it's a teaching note. Any faithful
reimplementation therefore has to decide whether it is reproducing the
teaching note (in which case the Q is whatever structural class the author
declares) or attempting to extend to published-Christensen territory (in which
case the burden of rigor goes up a lot).

---

## 2. What the implementation does

**`moments.py`** implements `b_n`, `W_n`, `r_n(q)` exactly as the PDF states,
with a clean guard that `q_values > 0` on respondents. Non-respondents are
correctly excluded from `r_n` via `response_mask`.

**`inner_solver.py`** builds the Kronecker matrices directly:
`K_left = kron(bb, I_d)`, `K_right = kron(bb', I_d)`, then
`L = K_left @ W @ K_right`, `y = K_left @ r`, and solves
`np.linalg.lstsq(L, y, rcond=None)` before unstacking with
`A = a.reshape((d, d+1), order="F")`. This is a literal, non-clever translation
of §1.5. The rank-deficiency handling matches the PDF.

**`q_classes.py`** defines `ConstantQ` (1-D, a scalar), `Parametric2ParamForBinary`
(2-D with optional monotone flag), and `MonotoneInY` (K-knot piecewise-linear
with ordering constraint). The binary-2-param class is the clean specialization
of Christensen's monotone-g-of-y example to binary `y`.

**`outer_solver.py`** dispatches by class. For `ConstantQ`, 1-D bounded
`scipy.optimize.minimize_scalar`. For `Parametric2ParamForBinary`, a 15×15 grid
filtered by the monotone constraint, top-5 polished with L-BFGS-B (no monotone)
or SLSQP (with monotone inequality). For `MonotoneInY`, 10 random seeds + 3
structured constant seeds, top-3 polished under monotone inequality constraints.
A notable fast-path: during outer evaluation the code uses
`β̂ = W_n_pinv @ r_n` and evaluates the quadratic with that short form, skipping
the O(d⁶) Kronecker system. This is correct under the PDF's own claim that
any lstsq solution gives the same `β̂`; `solve_inner` is only re-called at the
best θ to populate (M*, m*) for bookkeeping.

**`estimator.py`** is a thin sklearn wrapper. It defensively zeros `Y_tilde`
at non-respondents, optionally prepends an intercept column, calls
`solve_outer`, stores `β̂ = M* b_n + m*`.

**`reference_based_q.py`** (the interesting one). This module does *not*
appear in the 2020 PDF. It builds a `QClassConfig` with
`q_min = max(0.01, q_hat − δ)`, `q_max = min(1.0, q_hat + δ)` — a **ball of
radius δ around the empirical observation rate**. A table
`MECHANISM_DELTA` assigns δ per Pereira mechanism
(0.30 for MBOV_Lower/Higher, 0.25 for Stochastic, 0.05 for MBUV / MBOV_Centered).
The module's docstring cites Christensen & Connault 2023 and Adjaho &
Christensen 2022 as justification.

**`phase1_pereira_benchmark/christensen_adapter.py`** glues these together,
passing `mechanism_name` into `adaptive_centered_q_for` to pick both the
Q-class structure *and* the δ at fit time.

---

## 3. Fidelity assessment

### Faithful pieces

- **The inner solver is a literal transcription of §1.5.** The FOC, the vec
  identity, the lstsq for the rank-deficient system, the column-major unstack
  — all correct. `test_solve_inner_produces_valid_solution` verifies
  `W β̂ = r`, which is the reduced-form FOC one obtains by multiplying the
  full vec FOC by `(bb' ⊗ I_d)` and using `bb' bb = 1 + ||b||²`, divided out.
  That is a fine sanity check.

- **The moment functions are exact.** The IPW identity Christensen uses is
  implemented as stated.

- **The estimator family is correct.** `β̂ = M b_n + m` with `M ∈ ℝ^{d×d}`
  and `m ∈ ℝᵈ`, as Problem 2 requires.

- **Reduction to MAR-OLS holds.** Under `ConstantQ` and MAR data,
  `β̂ ≈ β_OLS_on_responders` (test_reduction_to_ols.py); under full response,
  `β̂ = β_OLS_on_all_data`. These are the right legitimacy checks.

### Divergences

**D1 — The Q specification is not Christensen's.**  The 2020 note gives a
shape-restricted functional class: `{ g : g decreasing }`. The implementation
instead uses a **box** `[q_min, q_max]` (via `QClassConfig`) further **centered
on q̂** with radius δ chosen from a hand-curated table keyed by the *true*
mechanism name. Christensen's monotone condition enters only as a sign
constraint (`q₀ ≤ q₁` or `q₀ ≥ q₁`) on the 2-param box. The structural Q
has been silently replaced by an adaptive box. This matters because:

1. **The outer adversary has completely different semantics.** In Christensen's
   framing, the adversary over `{g decreasing}` can pick any decreasing
   function within broad bounds; in the repo's framing, the adversary is
   boxed into `[q̂ − δ, q̂ + δ]`. For small δ (e.g. 0.05 for MBUV), the
   adversary barely does anything and the estimator collapses to
   near-OLS-at-q̂. That is an *engineering* choice, not a *Christensen* choice.

2. **Data-adaptive centering isn't in Christensen's published corpus either.**
   The Christensen & Connault 2023 paper centers on a parametric reference,
   not the empirical observation rate; Adjaho & Christensen 2022 centers on
   the empirical joint distribution but the radius is a user-reported
   primitive, not a function of the data (see their sensitivity profiles over
   ε). Calling the repo's approach "reference-based" is a stretch — it is
   reference-based in a narrow, invented sense specific to this project.

3. **The δ table uses oracle mechanism knowledge.** `MECHANISM_DELTA` and
   `adaptive_centered_q_for(mechanism, ...)` read the mechanism name that
   generated the data and pick δ from a table that was explicitly calibrated
   against the magnitude of that mechanism's q-spread. This is *not a
   forecasting estimator.* It is a post-hoc-tuned-per-mechanism estimator.
   In any deployment the mechanism is unknown, which is the setting for
   which `DEFAULT_DELTA = 0.30` exists, and that default substantially
   degrades performance on mechanisms for which 0.30 is miscalibrated
   (the diagnostic script confirms this for MBUV / MBOV_Centered). Christensen
   would not call this a "faithful" reimplementation of his framework: the
   adversary is being pre-tuned to the ground truth.

**D2 — `MonotoneInY`'s knot placement uses the adversary's own observed `Ỹ`
to pick knots over `[y_min(observed), y_max(observed)]`.** For selective
observation this is a leaky construction: the observed range is
*endogenous to q*, so the Q-class parameterization depends on the realized
selection. A cleaner specification would use the known/declared support of
`Y` (for the benchmark, `[0,1]` post-normalization on `{0,1}` labels) or a
data-range estimated from a pilot. This doesn't affect the binary experiments
(which don't use MonotoneInY) but is relevant if MonotoneInY is later used
on continuous-`y` cases.

**D3 — Binary-label reduction glosses over Christensen's distinction between
`P(Y)` and `P(Ỹ)`.** The `Parametric2ParamForBinary.q_values` function keys
off `Y_tilde[i] == 0`. For respondents (`ε_i = 1`) this gives `q(·,0) = θ₀`,
`q(·,1) = θ₁` — correct. For non-respondents (`ε_i = 0`, `Ỹ_i = 0`) the same
dispatch returns `θ₀`. In `compute_r_n` this is harmless (non-respondents are
filtered by `response_mask`) but in any future code path that uses
`q_values(...)` directly on the full sample it would silently treat
non-respondents as-if-`y=0` respondents. Defensive OK in the current call
graph, but a latent footgun.

**D4 — Outer solver soundness is empirical, not certified.** The inner
function `θ → max val` is NOT concave in general (the repo docstring
acknowledges this). The code uses a 15×15 grid + top-5 L-BFGS-B/SLSQP polish
for 2-D, and 10+3 random seeds for K-D. This is a reasonable heuristic
implementation of a non-convex max, but:

- No guarantee the polish converges to the *actual* saddle.
- No Lagrangian / KKT check on the returned θ* to confirm local optimality
  w.r.t. both the box and the monotone constraint.
- No "worst case among polished candidates vs grid" assertion — the code
  falls back to the grid point if polish doesn't improve, but never compares
  against a *dense* grid at the end.
- Convergence of SLSQP on a non-smooth (box-clamped) surface is not
  guaranteed; `minimize_scalar` for 1-D is fine; L-BFGS-B for 2-D nonconvex is
  not.

For a 2-D problem this is probably adequate. For the K=5 MonotoneInY it is
meaningfully weaker — 10 + 3 seeds into a 5-D monotone cone is quite sparse
coverage, and there is no global-max certificate.

**D5 — The `Parametric2ParamForBinary` Q has a fundamental identifiability
concern under the centered-box regime.** If δ is small enough that the box
`[q̂ − δ, q̂ + δ]²` doesn't actually contain the ground-truth `(q₀, q₁)`, the
worst-case adversary is artificially weakened and `β̂` can be worse than a
wider-box Christensen (or even than OLS-at-q̂). The diagnostic script confirms
this. The repo papers over this by tuning δ to the mechanism — but in a real
application, δ must be set without knowing the mechanism, which means either
(i) a wide δ, which makes the adversary overconfidently pessimistic for
near-MAR data, or (ii) reporting a **profile over δ** as Christensen does in
his published work. The repo does neither.

**D6 — `MICE near-zero MSE` in the `ctg` benchmark is a dataset artifact,
not a real baseline.** `ctg` is binarized to "non-normal vs normal" with ~78%
class imbalance. MICE imputes observed-mean for missing y, which is
approximately a constant predictor close to 0; on a test set where ~78%
of labels are 0, MSE of "predict 0.22 everywhere" is dominated by the 22%
positive class and ends up very small (the observed MICE MSE of ~0.0001
matches this). That this shows up in the "least favorable cells" table makes
the 31M-percent headline numbers meaningless — the comparison is numerical
noise / degeneracy, not a theoretical loss. The report should be flagging
this, not tabulating it as a Christensen failure mode.

**D7 — The report's primary "win rate" calculation is vulnerable to
degeneracy cells.** The win/loss/tie methodology uses 95% CI separation. On
`ctg` MICE's tiny variance makes CIs collapse, so any non-degenerate
Christensen prediction "loses." Restricted to well-conditioned mechanisms on
non-degenerate datasets, the real comparison shrinks dramatically. A fair
report would either exclude `ctg` or rescale to normalized MSE (e.g., divide
by Var(y_test)).

---

## 4. NEW issues I flag (not in AUDIT_v2.md)

AUDIT_v2 is strong on identifying that the SGD-based `minimax_score`
estimator is **not** Christensen. But it pre-dates the `christensen_core/`
implementation, so it does not address:

1. **Oracle-tuned δ.** `MECHANISM_DELTA` is oracle-tuned to the mechanism.
   This is the single most important caveat for the Phase 1 headline.
   AUDIT_v2 doesn't discuss it because the faithful estimator didn't exist yet.
   In a journal-review setting this would be flagged as a leakage-style
   issue: the method gets access to the data-generating mechanism's *name*,
   which is rarely available in practice. The report footnote on "fidelity
   adaptive" is insufficient.

2. **`ctg` degeneracy dominates the least-favorable table and contaminates
   the overall win/loss counts.** All 10 of the "MICE beats Christensen by
   biggest %" entries are `ctg`. If `ctg` is excluded, the Christensen-loss
   tail essentially disappears. REPORT_v2 notes class imbalance obliquely in
   footnote 3 but does not actually break out or exclude degenerate cells.

3. **No profile-over-δ in the report.** Christensen's own published methodology
   (2023 Econometrica, 2022 AC) reports a *profile* of outcomes over the
   neighborhood radius. The repo reports a single-point estimate with δ
   picked from the mechanism table. A faithful replication-style report
   should include a sensitivity curve over δ for at least one (dataset,
   mechanism) pair. Without this, the 60% win rate on MBOV_Lower is
   indistinguishable from "δ was tuned to win on MBOV_Lower."

4. **No saddle-point certificate.** The report shows `min_{M,m} max_q` was
   *approximated*, but provides no evidence that `(M*, m*, θ*)` is a saddle
   (i.e. that the primal and dual gap is small). For a 2-param monotone box,
   a simple check — evaluate `max_θ f(M*,m*; θ)` on a dense grid and compare
   to the reported value — would be trivially cheap. This is standard
   practice in min-max benchmarking literature; its absence is surprising.

5. **The `Parametric2ParamForBinary` monotone=None code path is latent dead
   code for this benchmark.** All mechanisms map to a monotone variant via
   `centered_q_for` / `adaptive_centered_q_for`. The "no monotone flag" path
   is only exercised by unit tests. Not a bug, but means the "Parametric2
   class" is really only ever used as its monotone specialization.

6. **Unit-test for reduction-to-OLS uses `atol=0.1` on predictions.** That's
   a very loose tolerance for a claimed foundational equivalence. If the
   implementation is truly equal to MAR-OLS under ConstantQ at q̂, the
   tolerance should be at least 1e-6. 0.1 is large enough to hide a
   non-trivial systematic bias. The second test
   (`test_constant_q_recovers_beta_true_on_large_sample`) uses `atol=0.1` on
   predictions of magnitude ~O(1), which is similar — "close to noise level"
   rather than "bit-exact." A tight bit-exact test against
   `np.linalg.lstsq(X[resp], Y[resp])` scaled by `1/q̂` would be the right
   foundational invariant.

7. **`MonotoneInY` knot range uses observed `Ỹ`, which is
   q-endogenous.** (See D2.) Flagged as latent for future continuous-`y`
   work.

8. **Problem 3 (unobserved X for non-respondents) is not implemented.** The
   PDF §1.4 extension is declared out of scope (per IMPLEMENTATION_PLAN.md).
   For a strict faithful reimplementation this should be wired even if
   exercised only in a synthetic test, because it is part of the primary
   source's scope. For Phase 1 specifically, where X is always observed, this
   is a defensible omission — but the repo should not claim "faithful
   reimplementation of the 2020 note" without noting this.

---

## 5. Agreements and disagreements with AUDIT_v2.md

I read AUDIT_v2 only *after* completing sections 1–4 above. My findings and
theirs overlap substantially on the scope of the pre-`christensen_core`
estimator (`minimax_score` / SGD variant) — AUDIT_v2 correctly identifies
that it implements a different objective (reweighted empirical risk, not
quadratic parameter error), a different estimator family (β directly, not
`M b_n + m`), a different inner solve (projected-gradient, not closed-form
pinv), and a different Q (box + mean-equality, not structured class). I agree
with all of that.

**Where we agree:**

- **Divergence A (wrong objective in the SGD path).** Agreed. My assessment
  of `christensen_core/` is that it *fixes* A — the quadratic objective is
  implemented correctly in `inner_solver.py`. AUDIT_v2's concern is about
  pre-`christensen_core` code and remains valid there.
- **Divergence B (wrong estimator family in the SGD path).** Agreed, fixed in
  `christensen_core/`.
- **Divergence C (wrong inner solver in the SGD path).** Agreed, fixed.
- **Divergence E (proxy labels).** Agreed that Christensen's framing requires
  `Ỹᵢ = 0` for non-respondents and that injecting a proxy violates the IPW
  identity. `christensen_core/` does not do this; it correctly zeros
  non-respondent entries. AUDIT_v2's concern was about the SGD path.

**Where I think AUDIT_v2 is too generous:**

- AUDIT_v2 treats Divergence D ("Q: box vs structured class") as fully
  resolvable by implementing shape-restricted Q classes. The repo's
  `christensen_core/q_classes.py` does implement shape-restricted classes
  (monotone binary, MonotoneInY). But the `reference_based_q.py` module
  **layers a centered ball on top**, parameterized by a mechanism-oracle δ,
  which substantially undermines the shape restriction. AUDIT_v2 doesn't
  foresee or critique this design choice.

- AUDIT_v2 is silent on outer-solver soundness. A faithful reimplementation
  is not just "closed-form inner + structured Q" — it also needs a defensible
  outer-max procedure, which the current grid + polish implementation is at
  best empirical.

- AUDIT_v2 accepts mapping Pereira mechanisms 1-to-1 to Q classes without
  questioning whether `ctg`-style dataset degeneracies invalidate the win/loss
  metric (my D6/D7).

**Where I think AUDIT_v2 is *more* rigorous than I was:**

- AUDIT_v2's Divergence F (observation-rate clipping) is a concrete procedural
  bug in the SGD code that I wouldn't have caught without reading it. That
  issue is specific to the SGD path and does not carry over to
  `christensen_core/`, but AUDIT_v2 is right to flag it.

- AUDIT_v2 does explicitly catch that Problem 3 (unobserved X) is missing; I
  agree.

---

## 6. Bottom line: would I accept this in journal review?

**The inner solver and moments code: yes.** They are faithful, well-tested
(modulo loose tolerances), and literal transcriptions of §1.5.

**The overall `christensen_core` + Phase 1 claim: no, not in current form.**
Three issues would keep me from signing off:

1. **`reference_based_q.adaptive_centered_q_for` uses oracle mechanism
   knowledge to set δ.** This cannot be called "faithful Christensen." It is
   a post-hoc-tuned variant. The headline 22% win vs MICE / 60% on
   MBOV_Lower needs either a non-adaptive δ (one fixed value, reported with
   a sensitivity profile over δ) or an explicit disclosure that the method
   is oracle-tuned-to-mechanism. Ideally both.

2. **Outer max is heuristic with no saddle certificate.** For publication I
   would want, at minimum, a dense-grid check on
   `max_θ f(M*, m*; θ) ≈ f(M*, m*; θ*)` to confirm the polished θ* is a
   global maximizer. Easy to add; unclear why it's not there.

3. **The win/loss methodology is dominated by `ctg` degeneracy and by
   `MICE near-zero MSE` cases that aren't real losses.** Reporting without
   excluding or normalizing those cells is misleading. Either drop degenerate
   datasets or report per-cell normalized MSE.

These are fixable in a week of work — none require rewriting the core. But
as written, the v2 report overclaims. The REPORT_v2 caveat sentence
("the minimax estimator run here is SGD-with-online-score-based-adversary,
not the closed-form") points at a real problem in the old SGD path, but the
new `christensen_faithful` results have their *own* fidelity problems
(oracle δ, no saddle certificate, degenerate cells) that REPORT_v2 does not
address.

**One positive surprise:** the `inner_solver.py` implementation is cleaner
and more directly traceable to the PDF than I expected. The `solve_inner` →
`test_solve_inner_produces_valid_solution` round-trip (check
`W β̂ = r`) is exactly the right legitimacy invariant to assert on the
closed-form path. If the rest of the package matched this tier of care, the
review verdict would be much more favorable.

**Recommendation:** before any paper submission, (a) run a δ-sensitivity
profile on one representative (dataset, mechanism) pair and include it in
the report; (b) add a dense-grid outer-max certificate; (c) exclude `ctg`
or switch to normalized MSE; (d) tighten the reduction-to-OLS test to bit-
exact tolerance; (e) explicitly label `adaptive_centered_q_for` as
"oracle-δ, for calibration studies only" and provide a non-oracle default
path with a fixed δ that is disclosed up front.
