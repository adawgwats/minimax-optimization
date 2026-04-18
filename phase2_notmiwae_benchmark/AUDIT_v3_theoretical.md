# Phase 2 Theoretical Audit (v3) — Christensen-Faithful on not-MIWAE Benchmark

**Auditor**: independent theoretical review
**Date**: 2026-04-16
**Scope**: evaluate whether the three Phase 2 adaptations (`MonotoneInY`, per-feature adapter, `SelfMaskingAboveMean`) are principled extensions of Christensen's framework or ad-hoc engineering.

---

## 1. Independent understanding of the task mismatch

Christensen's 2020 note (`docs/christensen_minimax.pdf`) is unambiguous about what it solves. The setup (§1.2) is:

- A **regression** problem: target Y, regressors X, true parameter β defined as the OLS projection β = E[XX']⁻¹E[XY].
- A **labels-only corruption**: ỹᵢ = εᵢYᵢ for a Bernoulli response εᵢ; Xᵢ is always fully observed.
- The **quantity of interest** is β (used downstream for prediction ŷ = x'β), not imputation of anything.

The min-max problem (Problem 1) minimizes worst-case **mean-square prediction error** `E[(β̂−β)' E[XX'] (β̂−β)]` over a Q class of response functions q(x,y). Christensen's §1.4 "Extension" briefly permits X itself to be non-observed (joint row-dropping) but still treats it as a regression problem. Nowhere in the note is the loss an **imputation RMSE on X** — it is always a prediction-error bound on β (and hence on ŷ).

not-MIWAE (Ipsen, Mattei, Frellsen, ICLR 2021) solves a fundamentally different problem. From §2 of the paper and `task01.py`:

- The goal is **imputation of missing feature entries X_m** given observed features X_o and mask s.
- The loss is **imputation RMSE on the masked entries of X**, not prediction error on some external Y.
- There is no regression target; in `task01.py` the original class column is explicitly dropped (line 64: `data = data[:, :-1]`).
- The missingness is **element-wise, across D/2 columns simultaneously**, each independently self-masked above its mean.
- Imputation is formally Exm[L(xm,x̂m)|xo,s] — a conditional expectation of X_m given X_o.

These are genuinely different tasks. Christensen's framework produces a β̂ with minimax prediction guarantees; it has no native notion of "impute X_m from X_o." The Phase 2 authors acknowledge this in PROTOCOL.md ("Deviation from Ipsen et al. 2021 — important and declared upfront"), but the implications are deeper than a labeled deviation: every step of Christensen's derivation relies on the **linear regression model being correct for the true E[Y|X]** (or at least being the best linear predictor with IPW-corrected moments). When you swap Y→X_j and X→X_{-j}, none of those derivations automatically transport.

A particularly worrying structural detail: Christensen's `r_n(q) = (1/n) Σ_{i∈R_n} (1/q(X_i,Ỹ_i)) X_i Ỹ_i` uses an IPW identity `E[XY] = E[(1/q) X Ỹ]` that is **exact when q is correct**. Under the per-feature adapter, q is parameterized as a monotone function of "Y" (now a feature X_j). The IPW identity still holds formally — but the **target of estimation** is now β_j = E[X_{-j}X_{-j}']⁻¹E[X_{-j}X_j], which is just a conditional-mean predictor of X_j from X_{-j}, not a structural causal parameter. This is closer to MICE-style conditional regression than to Christensen's econometric minimax.

**In short**: Christensen solves *"how do I estimate β under adversarial non-response so that ŷ = x'β has best worst-case MSE?"*. Ipsen et al. solve *"how do I impute x_m so that ||x_m − x̂_m|| is small in expectation?"*. These only coincide if the regression model E[X_j|X_{-j}] = X_{-j}'β_j is correctly specified — a strong and unverified assumption in Phase 2.

---

## 2. Theoretical soundness of `MonotoneInY`

Christensen's "g decreasing in y" (PDF p.5) is **a structural class of response functions** — effectively the set `{q : q(x,y) = g(y), g non-increasing, g∈[q_min,q_max]}`. This is infinite-dimensional. Any implementable Q must be a finite-dimensional approximation, and the theoretical question is what is lost by that approximation.

The Phase 2 `MonotoneInY` uses **K=5 knots with piecewise-linear interpolation on the observed-y range**. This has three theoretical concerns:

1. **Approximation gap is unquantified.** Christensen's note discusses Q as an abstract class; it says nothing about the error from replacing a rich class with a 5-knot piecewise-linear approximation. The Phase 2 code has no Lipschitz bound, no K-selection justification, no note on convergence as K→∞. K=5 is a guess. There is a defensible intuition (5 knots ≈ 4 linear pieces can roughly capture any smooth monotone function on a standardized range), but **no theorem backs it**. For a NeurIPS audience, this is noise; for a Journal of Econometrics audience, this is a significant gap.

2. **The class does not contain a hard step function.** The real mechanism in the not-MIWAE benchmark is `q(y) = 1{y ≤ mean(y)}` — a step that jumps from `q_max` (≈1) to `q_min` (≈0.05 under the clipping) at y = mean. A piecewise-linear function with 5 knots on the observed-y range (i.e., on y ≤ mean, since above-mean values are censored!) **cannot represent this step at all**. Worse, the knot grid is placed on `observed` values (`Y_tilde[Y_tilde != 0]` in `q_values`), which under self-masking-above-mean means the grid spans only y ≤ mean — the region where the true q is essentially 1. The adversary cannot even see the relevant y-range. When `Q` does not contain q_true, Christensen's minimax guarantee degrades to "minimax over Q, where Q may be arbitrarily far from the DGP in total variation." There is **no recovery of consistency or bias reduction** under this failure mode — the theorem simply does not apply.

3. **Monotone cone over empirical knots is not the same as monotone functions.** Enforcing `θ_{i-1} ≥ θ_i` on knot values gives a monotone piecewise-linear function on the **empirical knot grid**, but `np.interp` outside the grid clamps to endpoint values. So the effective function class has an artifact: `g(y) = θ_1` for all y ≤ y_lo, and `g(y) = θ_K` for all y ≥ y_hi. This is a boundary-artifact approximation, not a genuine monotone class. For standardized data with long tails this can be meaningful.

**Verdict on `MonotoneInY`**: the name is honest (it is monotone in y), but the parameterization is ad-hoc, K is unjustified, and in the specific regime where it is deployed (Phase 2, step-function self-masking), **it demonstrably cannot contain the truth**. This is a serious theoretical issue under Christensen's own framing, and is not discussed in PROTOCOL.md or REPORT.md.

---

## 3. Principledness of the per-feature adapter

This is the core theoretical question, and the honest answer is: **the per-feature adapter is using Christensen's framework as a regression engine, not as a joint-distribution imputer. Whether that counts as "principled" depends on whether you are willing to treat each column-wise regression as a legitimate application of Christensen in its own right.**

Principled aspects:
- For each masked feature j, the per-feature problem *is* a regression-under-missing-labels problem with X = X_{-j} (always observed by construction, since masked features live in the first D/2 and predictors in the second D/2) and Y = X_j (missing MNAR). This matches Christensen's setup (§1.2) exactly.
- The missingness mechanism `1{X_j > mean(X_j)}` is decreasing in X_j, so `MonotoneInY("decreasing")` is a structurally valid Q (modulo the step-function issue in §2).
- The predictors X_{-j} are never missing (by the benchmark protocol), so the "extension" to missing-X of §1.4 is not required.

Ad-hoc / unprincipled aspects:
1. **Model misspecification.** Christensen's guarantees assume the best **linear** predictor of Y from X is a meaningful target. If the conditional mean E[X_j|X_{-j}] is genuinely nonlinear (which it usually is on UCI datasets like wine quality or breast cancer features), then *neither* Christensen's β nor any β̂ has a meaningful interpretation as an "imputer of X_j". The estimator fits a linear model that OLS+q-hat would have fit; the minimax correction changes *which* β̂ gets chosen but does not fix nonlinearity. Since the benchmark metric is imputation RMSE, nonlinearity directly penalizes the method. MICE and missForest iterate through conditional models that can be nonlinear and use *all* features (including partially-observed ones via refinement).

2. **Independent regressions ignore cross-target dependence.** The adapter runs D/2 independent regressions of X_j on X_{-j}. The predictor set is the **always-observed half**; the other masked features are never used as predictors. This is an explicit choice acknowledged in REPORT.md ("legitimate methodological limitation of the per-feature adaptation — it ignores cross-masked-feature dependencies"). It is not principled — it is a simplification forced by the non-row-wise nature of the missingness. A more principled adaptation would chain the regressions (MICE-style) or model the joint, at which point you have effectively abandoned Christensen.

3. **δ calibration is mechanism-peeking.** The `MECHANISM_DELTA` table (`reference_based_q.py` line 84) uses `SelfMaskingAboveMean: 0.30`, which is a **domain-specific prior** calibrated off-line. The Phase 2 REPORT.md frames this as "mechanism-adaptive delta", but it is functionally "we looked at the mechanism and chose δ". This is not cheating (the mechanism is known in the benchmark), but it is not principled either — in any deployment where the mechanism is unknown, this lever collapses to `DEFAULT_DELTA = 0.30`, which is itself a hand-picked default. Christensen's framework does not specify how to choose δ; it is purely user input, and the "rationale" comments in `reference_based_q.py` are retrospective calibration.

4. **The "Y_tilde = 0 for non-respondents" convention is being abused.** Christensen's formulation works because Y_tilde = εY and the non-respondent terms drop from b_n (X·0 = 0) and r_n (sum over respondents only). In the per-feature adapter, the target X_j is zero-filled (`X_work = np.where(np.isnan(X_masked), 0.0, X_masked)`), which is correct for b_n/r_n. But then the **predictors X_predictors also use the standardized, always-observed X_{half:D}**, so X_predictors rows for non-respondents carry nontrivial values. That is fine for the moments, but it means the "X" of Christensen's derivation mixes (respondent X's with respondent y's) and (non-respondent X's with zero y's). This is Christensen's standard setup, but when you apply it per-feature, the "respondent" pattern is different for each feature j, so the moment W_n = (1/n) Σ X_i X_i' is computed on **all** n rows (responders + non-responders for j) — which is correct per Christensen's §1.5 but only because his X is *always observed*. This is technically consistent per the framework extension in §1.4, but it papers over the fact that the per-feature "Christensen" is really a family of D/2 independent IPW-corrected OLS regressions. Nothing about this is "joint minimax over q functions across features" — every feature gets its own q, and the Q-classes are not tied together.

5. **"Minimax guarantees" are per-feature, not joint.** The REPORT.md claim *"theoretical worst-case guarantees that not-MIWAE lacks"* is inflated. What the per-feature adapter gives is a minimax MSE bound **on the linear regression β̂_j for each j separately** — not a minimax bound on the joint imputation RMSE. Under worst-case q across all features simultaneously, the product of per-feature guarantees does not give a joint guarantee of the form "worst-case total imputation RMSE ≤ something." In fact, the worst-case-across-features is only bounded if you take a **union bound over the per-feature adversaries**, which is loose and unstated.

**Verdict on the per-feature adapter**: it is a legitimate-but-narrow application of Christensen — one regression at a time — dressed up as an "imputer." It is **not** a principled extension of Christensen's framework to joint feature imputation. It is also not a ridiculous hack: each per-feature regression is a valid Christensen problem. The framing that inflates "per-feature minimax OLS" into "theoretically-grounded imputer with worst-case guarantees" is where the principledness breaks down.

---

## 4. Novel issues not covered by the listed questions

**a. Degenerate standard deviation in the reported results.** In `results/aggregated.csv`, every cell except `missforest` has `std_rmse = 0.0`. This is because (i) the MNAR mask is deterministic given X, (ii) Mean / MICE with BayesianRidge default / Christensen with fixed rng(0) seeding in the outer solver are all deterministic. The REPORT.md acknowledges this ("RMSE across seeds: the self-masking MNAR mask is deterministic"). But then **running 5 seeds is performative** — it reports std=0 which could mislead a reader into thinking the method has been 5-fold validated. The "matches paper's 5 runs" framing conflates "we ran 5 copies" with "we have 5-seed variance." For Journal-of-Econometrics review, this is misleading reporting.

**b. Outer solver determinism hides instability.** The outer solver seeds `rng = np.random.default_rng(0)` inside `_solve_outer_monotone_in_y` (line 338). This is **hard-coded** independent of the user seed. So the Christensen branch always runs the same 10 random starts regardless of the outer seed. The variance across outer seeds is not explored. If you seeded the rng with the outer seed, you might find that the 10-start SLSQP routinely finds different local saddles on different starts — the reported θ* would then be seed-dependent. **This is an instability hidden by construction**, not an absence of instability.

**c. Local-saddle risk in the outer problem.** SLSQP on a nonconvex outer objective with 10 random starts + top-3 polish is a heuristic. For a K=5 monotone cone with clipped box bounds, the outer objective is piecewise-smooth (due to the `np.interp` linear segments) and generically has multiple local maxima. 10 random starts in a 5-dimensional cube is not dense. The reported `inner_value_at_star` is a **local saddle**, not provably the global. Consequences: (i) the minimax β̂ is not provably the minimax; (ii) the downstream ŷ = X_{-j}β̂_j may not be the worst-case-guarding imputer; (iii) there is no way to detect failure. None of this is mentioned in PROTOCOL.md. For NeurIPS this may pass; for Journal of Econometrics this would get a desk reject on the optimization section.

**d. Intercept column contaminates W_n.** With `fit_intercept=True`, X_aug has an all-ones column prepended. `W_n = (1/n) X_aug' X_aug` then has (1, mean of each predictor) in its top row/column. Because predictors X_{-j} are z-scored to zero-mean before masking, those means are ≈ 0 and W_n is approximately block-diagonal. Not a correctness bug, but worth noting: the standardization is load-bearing for conditioning.

**e. Predictor set shrinkage on `breast` is structural, not a bug.** `breast` has D=30, so half=15 predictors and 15 masked features. As the REPORT.md notes, MICE beats Christensen here because MICE uses all 29 features iteratively; the per-feature adapter uses only 15. **What is not noted** is that 15 features imputing 15 others, when the features are highly correlated (breast cancer features are famously redundant), leaves the always-observed 15 as a near-rank-deficient predictor set. The pinv in the inner solver papers over this but the minimax bound degrades.

**f. No comparison to IPW + sorted-q_hat OLS.** The natural "dumb" baseline is: per feature j, run OLS of X_j on X_{-j} with IPW weights set to the empirical q_hat_j on each side of the median (a non-minimax version of the same machinery). Does Christensen beat this, or is the minimax layer doing nothing? The Phase 2 results do not isolate this. Without that ablation, we cannot say the minimax outer loop is contributing; the win over Mean/MICE/missForest may be entirely attributable to IPW + linearity.

**g. The MICE "loss" on breast is the most informative cell but is quietly explained away.** MICE beating Christensen 0.946 vs 1.289 on breast is a 36% relative gap. The REPORT.md explains this as "MICE uses all features"; but a principled imputer should not be beaten by 36% on any dataset where the competing method is fundamentally simpler (BayesianRidge linear regressor, iterative). The asymmetry suggests MICE's iterative use of refinements is capturing something the per-feature-Christensen fundamentally cannot — which is worth emphasizing, not minimizing.

**h. Baseline protocol fidelity check is circular.** REPORT.md says "our baseline RMSE numbers match Ipsen et al. Table 1 within 0.01-0.03 ... confirming protocol fidelity." Then it notes that on `breast`, `MICE` is 0.946 vs the paper's 1.17 — a 0.22 gap, well outside the 0.03 tolerance. The "minor deviation" framing undersells it: on the dataset where MICE beats Christensen, MICE's number is 20% lower than the paper reports. If we compared Christensen to the **paper's** MICE number (1.17), Christensen would win breast too. This matters for the "4/6" vs "5/6 vs MICE" headline — the result depends on which MICE number is used, and the Phase 2 authors quietly chose the worse one for their side.

---

## 5. Is the headline claim defensible?

The claim in REPORT.md: *"Christensen-faithful beats MICE on 4/6 datasets, missForest on 6/6, Mean on 6/6. Does not beat not-MIWAE."*

Factually, the numbers in `aggregated.csv` support those counts (against the **re-run** baselines). But the framing has several problems:

1. **The headline is a benchmark claim, but the method is not optimized for the benchmark metric.** Christensen optimizes prediction MSE on β̂; imputation RMSE is not the loss Christensen's framework minimizes. Beating imputation-optimized baselines (MICE, missForest) on their own metric is impressive only if you concede that the method you beat was not tuned for MNAR. The missForest loss across 6/6 is partly because missForest uses a MAR-assumption iterative refinement, which is biased under above-mean self-masking. The Christensen win over missForest is therefore "we correctly modeled the mechanism" — a win for *any* mechanism-aware method, not specifically for Christensen.

2. **The "fast, theoretically-grounded alternative to deep-generative" framing is misleading.** Christensen does not guarantee better imputation RMSE than not-MIWAE; it guarantees bounded prediction MSE under worst-case q within the per-feature Q class. These are different quantities. The reported 6/6 loss to not-MIWAE is not a "linear vs deep" gap — it is the right outcome under the benchmark metric, because not-MIWAE is the correct tool for the job (joint imputation under MNAR) and Christensen is a tool for a different job (regression under MNAR labels).

3. **δ = 0.30 for SelfMaskingAboveMean is peek-at-the-answer.** The entry in `MECHANISM_DELTA` was set knowing the benchmark mechanism. This is not a deployable recipe; it is a per-benchmark tuning. Without this tuning (δ falls back to 0.30 anyway, which is still the same number — but obtained by "default" rather than mechanism-specific calibration), the result should be replicated. Phase 2 does not run with un-tuned δ as an ablation.

**What should change:**

1. Reframe the claim narrowly: *"Per-feature IPW-corrected minimax linear regression, using a decreasing monotone Q class, beats sklearn defaults for Mean / MICE / missForest on a standardized 6-UCI benchmark under self-masking-above-mean MNAR."* Drop the "Christensen-faithful" label — it is per-feature Christensen applied D/2 times, not Christensen's framework extended to imputation.

2. Add an ablation table: (a) OLS + fixed q_hat (no minimax), (b) ConstantQ with centered-ball δ=0.30, (c) MonotoneInY with K=3/5/10, (d) paper's MICE number vs sklearn MICE number. This isolates what the minimax layer buys.

3. Acknowledge the approximation gap: `MonotoneInY` does not contain the true step function; the minimax guarantee is over the **implementable** class, not the truth.

4. Acknowledge that "5 seeds" gives σ=0 for Christensen; this is not a meaningful variance estimate. Either vary the outer-solver seed, vary the data subsample, or drop the "5-seed" framing.

5. Drop the implicit "theoretically-grounded" superiority over not-MIWAE; they solve different problems, and the benchmark metric is not the one Christensen minimizes.

---

## 6. Bottom line: venue-specific verdict

**NeurIPS (ML venue)**: **Marginal reject in current form; tractable rebuttal.** Reviewers will care about (a) benchmark results, (b) novelty over MICE/missForest, (c) empirical honesty. The 4/6 vs MICE win is real but narrow; the 6/6 vs missForest / Mean is a correct-mechanism-aware win that any mechanism-aware baseline would achieve. The "theoretical guarantees" framing will be pushed back on — NeurIPS reviewers will want either (i) a theorem stating the guarantee in terms of the benchmark metric, or (ii) the framing downgraded. The per-feature adapter is implementable and beats baselines, which would survive on empirical merit if the framing is honest. The `MonotoneInY`-cannot-contain-step-function issue will be raised in review but can be addressed by a comment ("this is minimax over the approximating class, not the true mechanism"). The σ=0 reporting and the quietly-different MICE-on-breast number would not kill the paper but would force a revision.

**Journal of Econometrics (econ venue)**: **Reject in current form; major revision required for resubmit.** An econometrics reviewer will read Christensen's 2020 note closely and immediately notice that Phase 2 has substituted features for labels and declared the problem the same. They will ask: "What is the target parameter? What is the identification argument? Under what assumptions is the per-feature β_j interpretable?" The honest answer (β_j is the best linear predictor of X_j from X_{-j}, and Christensen's minimax gives a worst-case bound over a specific approximating monotone class) is defensible but requires a formal theorem statement — absent from the current writeup. The "mechanism-adaptive δ" will be flagged as data-snooping without a formal prior-calibration procedure. The SLSQP + 10-start optimization without a global-convergence argument will be flagged on the numerical-methods side. The claim of "minimax guarantees" will be tested by asking: "guarantees with respect to what loss, over what class, with what approximation gap?" — and the current manuscript has no answer. A revision that (i) states and proves the per-feature minimax theorem, (ii) bounds the approximation error from K-knot discretization, (iii) replaces "5 seeds" with a proper variance analysis, and (iv) runs the ablations above would be publishable. In current form it is not.

---

## Summary (one paragraph)

Phase 2 takes Christensen's regression-under-missing-labels framework and applies it D/2 times, once per masked feature, on the not-MIWAE imputation benchmark. Each individual application is a valid Christensen problem, but the aggregation is not a principled extension to joint imputation — it is a bank of independent IPW-corrected OLS regressions with per-feature monotone Q classes. The `MonotoneInY` class with K=5 knots is a finite-dimensional approximation whose approximation gap is not analyzed and which provably does not contain the true step-function mechanism, so Christensen's minimax guarantee applies to the approximating class, not the data-generating process. The `SelfMaskingAboveMean` entry with δ=0.30 is benchmark-calibrated domain knowledge, not an uninformed prior. The headline "beats MICE on 4/6, missForest on 6/6" is numerically correct against the re-run baselines but depends on the sklearn-default MICE (which is worse than the paper's R-`mice` numbers on `breast`) and reports σ=0 across 5 seeds, making the "5-seed" framing performative. The work is a reasonable empirical probe of "can an IPW-corrected linear imputer, with a structurally-aware adversary, beat MICE defaults on self-masked UCI data?" — the answer is "yes, on 4 of 6 datasets" — but the "Christensen-faithful" and "theoretically-grounded" labels oversell the theoretical contribution.
