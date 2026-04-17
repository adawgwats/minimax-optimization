# Phase 1 Methodology Audit, v2 — Against Christensen's Primary Source

**Supersedes [AUDIT.md](AUDIT.md)**. The v1 audit compared our implementation to context.md's "v0 objective." That comparison is insufficient. context.md is a **paraphrase**, and on close reading of Christensen's actual PDF ([Minimax optimization.pdf](../docs/christensen_minimax_pdf.txt)) I find that context.md's "v0 objective" is substantively different from what Christensen wrote. This v2 audit compares our implementation to the **primary source**.

**Primary source**: `C:\Users\adawg\Downloads\Minimax optimization.pdf` — 7 pages, titled "Regression example for prediction," extracted to [docs/christensen_minimax_pdf.txt](../docs/christensen_minimax_pdf.txt) for searchability.

**Verification**: searching the PDF text for `group`, `groupwise`, `q_g`, `pi_g`, `L_g`, `sum_g` returns **zero matches**. The notion of groups is nowhere in Christensen's note. It was introduced by whoever wrote context.md.

---

## What Christensen actually specifies

### Setup (PDF §1.2)
- True `(Xᵢ, Yᵢ)ⁿᵢ₌₁ ~ P` with P fixed. β = argmin E[(Y-X'b)²] = E[XX']⁻¹E[XY].
- Observed data: `(Xᵢ, Ỹᵢ)` where `Ỹᵢ = εᵢ·Yᵢ`, `εᵢ ~ Bernoulli(qᵢ)`, `qᵢ = q(Xᵢ, Yᵢ)`.
- Under MAR (q constant), `β̂_OLS / q` recovers β. Under MNAR (q depends on X,Y), it does not.

### Objective (PDF §1.3, Problem 1)
```
min_{β̂}  max_{q ∈ Q}  E[ (β̂ - β)' E[Xₙ₊₁ Xₙ₊₁'] (β̂ - β) ]
```
This is **worst-case parameter-error quadratic form = worst-case mean-square prediction error**. It is NOT worst-case IPW-weighted loss.

### Linear estimator reduction (PDF §1.3, Problem 2)
Restrict to `β̂ = M · b_n + m` where `b_n = (1/n) Σ Xᵢ Ỹᵢ`, M ∈ ℝ^{d×d}, m ∈ ℝ^d. Using the IPW identity `E[XY] = E[(1/q) X Ỹ]`, the inner problem becomes, for FIXED q:
```
min_{M, m}  (Mb_n + m)' W_n (Mb_n + m) - 2(Mb_n + m)' r_n(q)
```
where:
- `W_n = (1/n) Σ Xᵢ Xᵢ'`
- `r_n(q) = (1/n) Σ_{i ∈ Rₙ} (1/q(Xᵢ, Ỹᵢ)) Xᵢ Ỹᵢ`  (IPW moment, sum over **responders only**)

### Inner solution (PDF §1.5)
For fixed q, the inner minimization has a **closed form** via the vec trick:
- Vectorize: `a = vec([M | m])` (stack columns)
- FOC: `(b ⊗ I_d) W (b' ⊗ I_d) a = (b ⊗ I_d) r`
- Solve via `pinv` or backslash. Matrix is rank-deficient but any solution yields the same prediction MSE.

### Uncertainty set Q (PDF §1.3, last paragraph)
> *"Diﬀerent Q represent diﬀerent plausible types of non-response. For instance, perhaps Ỹᵢ is blood alcohol content and people are less likely to respond as this variable increases, so Q might be a set of decreasing functions q(x, y) = g(y) with g decreasing in y."*

Q is a **structured functional class** — monotone, parametric, shape-restricted, or otherwise encoding domain knowledge about the selection mechanism. It is NOT a box `[q_min, q_max]`.

### Outer problem
The PDF says "we can now solve this problem using the algorithms for min-max problems" (page 5) — it does NOT specify SGD, alternating updates, online adversary, or any specific solver. The outer `max_q∈Q` is abstract; the specific Q determines the algorithm.

---

## What our code actually does

[minimax_adapter.py:108](minimax_adapter.py) constructs `ScoreMinimaxRegressor` which calls [gradient_validation.py:529 `train_robust_score`](../minimax_core/gradient_validation.py) which uses [adversary.py:84 `ScoreBasedObservationAdversary`](../minimax_core/adversary.py):

### Objective implemented
```
min_{β ∈ ℝᵈ⁺¹}  max_{q ∈ [q_min, q_max]ⁿ, mean(q) = q_bar}  Σᵢ (1/(n·qᵢ)) · ℓᵢ(β)
```
where ℓᵢ(β) is a per-example loss: observed → `(Xᵢ'β - Yᵢ)²`, unobserved → `(Xᵢ'β - proxy)²` (see §Divergence 2 below).

This is **not** Christensen's objective. Differences:
- Christensen: `E[(β̂ - β)' E[XX'] (β̂ - β)]` (worst-case quadratic form in parameter error)
- Ours: `Σᵢ (1/(n·qᵢ)) · ℓᵢ(β)` (worst-case IPW-reweighted empirical risk)

These are **not equivalent**. The IPW-reweighted empirical risk is an unbiased estimator of `E[ℓ(β)]` under the true q, but the min-max-of-reweighted-risk is NOT the min-max-of-quadratic-parameter-error. Our objective might arrive at similar β for some Q's and data regimes but the theoretical correspondence is not established.

### Estimator family
- Christensen: `β̂ = M · b_n + m` — a specific TWO-parameter family (M, m) over a single linear transform of a fixed sample moment `b_n`.
- Ours: β̂ is arbitrary `ℝᵈ⁺¹` — full d+1 free parameters found by SGD.

These are different parameter spaces. Christensen's has dim = d(d+1)+d = d²+2d parameters (M has d² entries + m has d); ours has d+1 (β directly). In practice both give a linear predictor `x'β`, but the OPTIMIZATION SURFACE differs.

### Inner solve
- Christensen: **closed form** via pinv/backslash on the vec system (§1.5).
- Ours: **one gradient step on q per outer epoch** (adversary.py:112–114), projected into `[q_min, q_max]` with equality constraint `mean(q)=q_bar`. Never solves inner exactly.

### Q class
- Christensen: "Q might be a set of decreasing functions q(x,y) = g(y) with g decreasing in y" — structured, domain-informed.
- Ours: `Q = { q ∈ ℝⁿ : q_min ≤ qᵢ ≤ q_max for all i, (1/n)Σqᵢ = q_bar }` — box + mean-equality, no structural restriction.

Our Q is richer in one sense (any per-example q pattern is allowed) and poorer in another (no encoding of directional selection, monotonicity, or functional shape).

---

## Re-mapped divergence list

I'm re-numbering because the v1 list was against the wrong spec. The true divergences against Christensen's primary source:

### Divergence A — Wrong objective (foundational)
- **Christensen (§1.3 Problem 1)**: `min_β̂ max_q E[(β̂-β)' E[XX'] (β̂-β)]`
- **Ours**: `min_β max_q Σᵢ ℓᵢ(β) / qᵢ`
- **Severity**: Foundational. The IPW-reweighted-risk objective and the quadratic-parameter-error objective are not the same problem.
- **Recoverable?** Only by implementing Christensen's objective directly. See "Recovery plan" below.

### Divergence B — Wrong estimator family (foundational)
- **Christensen (§1.3 Problem 2)**: `β̂ = M·b_n + m`, parameter space is (M, m).
- **Ours**: `β̂ = β` directly in `ℝᵈ⁺¹`, found by SGD.
- **Severity**: Foundational. We never implemented the (M, m) parameterization.
- **Recoverable?** Yes with a real implementation (see plan).

### Divergence C — Wrong inner solver (procedural)
- **Christensen (§1.5)**: closed form via pinv on the vec system.
- **Ours**: one projected gradient step per epoch.
- **Severity**: Procedural — but affects convergence quality substantially.
- **Recoverable?** Trivially, if we adopt Christensen's estimator family (B).

### Divergence D — Wrong Q (conceptual)
- **Christensen**: structured functional class like "decreasing g(y)".
- **Ours**: symmetric box `[0.25, 1.0]ⁿ` with mean-equality.
- **Severity**: Conceptual. Our Q doesn't encode directional selection. Against Pereira's MBOV_Lower (which IS directional-in-y), a correct Q would be "q(y) decreasing in y" — and the adversary could then be much stronger.
- **Recoverable?** Yes but requires implementing shape-restricted Q classes. Moderate engineering effort.

### Divergence E — Proxy labels (our invention)
- **Christensen**: "Ỹᵢ = 0 for individuals who don't respond. Those terms drop out of the sum." (PDF page 4). ONLY observed examples contribute to `r_n(q)`; unobserved contribute to `b_n` but not to `r_n`.
- **Ours (adapter + gradient_validation.py:540–551)**: unobserved rows get `proxy_label = mean(observed y)`, and a squared error `(pred - proxy)²` feeds into the adversary as `effective_scores`.
- **Severity**: Major. Christensen's derivation explicitly relies on `Ỹᵢ = 0` for non-respondents making those terms vanish. Injecting a non-zero proxy violates the IPW trick's core identity `E[XY] = E[(1/q)XỸ]`.
- **Recoverable?** Trivially, by removing the proxy path.

### Divergence F — Observation rate clipping (documented in v1 audit as Div #6)
- **Christensen**: Q is defined by the chosen class; empirical observation rate is a consequence, not a constraint.
- **Ours (gradient_validation.py:214–215)**: silently clamps `obs_rate` into `[q_min, q_max]`, effectively lying about the observation rate when mean missingness exceeds 75%.
- **Severity**: Major in our code, but **Christensen's theory has no such constraint at all**. This divergence is actually an artifact of our v1 reformulation (which introduced the budget constraint that Christensen didn't have).
- **Recoverable?** If we move to Christensen's actual Q (structured class), this issue disappears entirely.

### Divergence G — Adaptive LR (my hotfix, documented in v1 as Div #5)
- **Christensen**: No SGD, so no LR.
- **Ours**: adaptive `lr = lr_base / sqrt(n/200)`.
- **Severity**: Concerns only our SGD implementation. If we adopt Christensen's closed-form, this is moot.

---

## Summary table (v2)

| # | Divergence | Against Christensen PDF | Recoverable? |
|---|---|---|---|
| A | Objective: reweighted risk vs quadratic parameter error | **Yes, foundational** | Requires reimplementation |
| B | Estimator family: β ∈ ℝᵈ⁺¹ vs β̂ = Mb+m | **Yes, foundational** | Requires reimplementation |
| C | Inner solve: gradient step vs closed-form pinv | **Yes, procedural** | Follows from B |
| D | Q: box + equality vs structured function class | **Yes, conceptual** | Moderate engineering |
| E | Proxy labels for unobserved examples | **Yes, breaks IPW identity** | Trivial removal |
| F | Observation-rate clipping | **Yes, moot under true Q** | Disappears with D |
| G | Adaptive LR (my hotfix) | Concerns SGD only | Moot under B |

---

## What we actually tested in Phase 1

Restating the claim conservatively:

We implemented a **DRO-inspired variant** that borrows the "adversarial q" idea from Christensen's note but differs in objective, estimator family, solver, and uncertainty set. Specifically:
- Objective: min-max IPW-reweighted per-example empirical risk
- Estimator: direct linear β with SGD
- Q: box-constrained per-example q with mean-equality
- With an imputation-based proxy for unobserved losses (our invention)

On 350 (dataset, mechanism, rate) cells across 10 UCI medical datasets, this variant:
- beat MICE+OLS in 12% of cells with 95% CI separation
- concentrated wins on MBOV_Lower at rates ≥ 40% (with the 80% wins being partially explained by silent observation-rate clipping)

**This is NOT a test of Christensen's estimator.** The claim "Christensen's minimax framework beats MICE on outcome-correlated MNAR" is not supported by what we ran. What IS supported is "a loose DRO relative of Christensen's framework occasionally beats MICE."

---

## Recovery plan: implement the actual Christensen estimator

The PDF is 7 pages and spells out the algorithm completely. A faithful implementation is bounded:

### Component 1: Closed-form (M, m) solver for fixed q (~3–4h)
Translate §1.5 directly. Given q, compute:
- `b_n = (1/n) Σ Xᵢ Ỹᵢ`  (vector, d+1 incl. intercept if we prepend 1)
- `W_n = (1/n) Σ Xᵢ Xᵢ'`  (d×d)
- `r_n(q) = (1/n) Σ_{i ∈ Rₙ} (1/q(Xᵢ, Ỹᵢ)) Xᵢ Ỹᵢ`  (d-vector, sum over respondents only)
- Build `b = [b_n; 1]`, `A_matrix = (b⊗I) W (b'⊗I)`, `rhs = (b⊗I) r_n`
- Solve `A_matrix · a = rhs` via `numpy.linalg.pinv` (or `lstsq`)
- Unstack `a` into `M` (d×d) and `m` (d×1)
- Predict: `β̂ = M · b_n + m`; ŷ_test = X_test · β̂

### Component 2: Outer maximization over Q (~4–8h, depending on Q)
For the blood-alcohol-style example Q = {q(x,y) = g(y), g decreasing}:
- Discretize y (or use the sorted values of observed Ỹ) and parameterize g as monotone on that grid
- For each candidate g, compute the inner objective (closed form from Component 1)
- Maximize over the monotone cone — this is a convex or quasi-convex problem depending on how we parameterize
- For Pereira's MBOV_Lower, this Q is approximately right

For a more general Q (e.g., Lipschitz in (x,y)):
- More work to define numerically
- Could start with the monotone case and expand

### Component 3: Q-to-benchmark mapping
Our Pereira mechanisms correspond to specific Q classes:
- MBOV_Lower / MBOV_Higher: monotone-in-y
- MBUV: q independent of y (this is close to MAR, so IPW with constant q ≈ classical OLS/q̂)
- MBIR: q monotone in some auxiliary x (deleted by the mechanism)
- MBOV_Centered: q high at extremes of y, low at median — non-monotone; requires a richer Q class

This means some Pereira mechanisms are well-matched to Christensen's Q examples, others are not. The audit should note which Q-mismatch-free comparisons are interpretable.

### Estimated effort
- Faithful single-Q-class implementation: **8–12 hours** including tests and comparison vs. OLS baseline on a toy problem with known analytical optimum.
- Extension to multiple Q classes: +4–8 hours per class.
- Re-running Phase 1 benchmark against the faithful estimator: ~2 hours (most of harness reusable).

Total: roughly **2–3 focused days** for a real comparison.

---

## Recommendation

1. **Stop treating Phase 1 results as evidence about Christensen's framework.** The current REPORT.md should be renamed or prefaced as "DRO-variant benchmark, not a Christensen test."
2. **Implement the faithful estimator** (Components 1 and 2 above). This is the work that should have been done in 2020.
3. **Re-run Phase 1** with the faithful estimator. This time the results actually speak to Christensen's framework.
4. **THEN** scale to 30 seeds, write the paper, pitch to Christensen.

Until step 2 is done, I strongly recommend not scaling to 30 seeds or writing any paper. The 10-seed numbers we have represent a different method than the one we were supposed to be testing.
