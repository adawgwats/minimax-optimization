# AUDIT v3 — Empirical Review of Phase 2 (not-MIWAE Head-to-Head)

Independent methodological and empirical audit of
`phase2_notmiwae_benchmark/` against Ipsen, Mattei & Frellsen (ICLR 2021).
Numbers below are recomputed from
`phase2_notmiwae_benchmark/results/raw_results.csv` (120 rows = 6 datasets
× 5 seeds × 4 methods).

---

## 1. Baseline reproducibility — actual numerical discrepancies

Paper Table 1 vs our measured aggregates:

| Dataset  | Mean (paper / ours / Δ) | MICE (paper / ours / Δ)   | missForest (paper / ours / Δ) |
|----------|-------------------------|----------------------------|-------------------------------|
| banknote | 1.73 / 1.726 / −0.004   | 1.41 / 1.411 / +0.001      | 1.28 / 1.279 / −0.001         |
| concrete | 1.85 / 1.847 / −0.003   | **1.70 / 1.868 / +0.168**  | 1.76 / 1.806 / +0.046         |
| red      | 1.83 / 1.838 / +0.008   | 1.68 / 1.681 / +0.001      | 1.64 / 1.633 / −0.007         |
| white    | 1.74 / 1.739 / −0.001   | 1.41 / 1.412 / +0.002      | 1.63 / 1.629 / −0.001         |
| yeast    | **1.69 / 1.729 / +0.039** | **1.72 / 1.756 / +0.036** | **1.66 / 1.712 / +0.052**      |
| breast   | 1.82 / 1.820 / 0.000    | **1.17 / 0.946 / −0.224**  | **1.57 / 1.486 / −0.084**      |

Banknote/red/white match the paper to 2 decimals on all three baselines —
reassuring. The other three datasets have meaningful deviations.

- **breast MICE: 0.946 vs paper 1.17 (Δ = −0.224).** REPORT.md attributes
  this to sklearn vs R `mice`. That's plausible but untested. The R `mice`
  default uses pmm (predictive mean matching), sklearn uses BayesianRidge
  regression — fundamentally different imputers despite shared branding.
  Ipsen et al. `task01.py` literally uses `IterativeImputer(max_iter=10,
  random_state=0)` — the same sklearn call we use — so the gap should be
  close to zero. A ~0.22 RMSE discrepancy on an RMSE-1 metric is 19% —
  that is not a "minor deviation." It suggests the paper's Table 1 MICE
  number may itself have been computed from a different pipeline than the
  `task01.py` posted on GitHub.
- **concrete MICE: 1.868 vs paper 1.70 (Δ = +0.17).** Our MICE is WORSE
  than the paper's. 10% relative error on a published SOTA baseline.
  This matters because Christensen beats MICE 1.424 vs 1.868 on concrete —
  if the "true" MICE is 1.70 (paper), the win margin shrinks from −0.44 to
  −0.28, still a win but materially smaller.
- **yeast: all three baselines are higher by 0.036–0.052.** Systematic,
  not random. Suggests a protocol difference (dataset version, feature
  order, target dropped or kept, one-hot on the class column, etc.).
  REPORT.md does not acknowledge this.

A reviewer will absolutely run `diff` on these numbers. The current framing
— "baselines match the paper" — is false on at least 3 of 6 datasets. The
honest framing is: banknote/red/white reproduce; breast/concrete/yeast
don't, and we don't know why.

---

## 2. Comparison fairness — is the head-to-head apples-to-apples?

**No, and the asymmetry is not disclosed in REPORT.md at the level it
deserves.** Three structural issues:

1. **Information sets are different.** not-MIWAE, MICE, and missForest use
   ALL features (including masked-half features, through iterative re-
   imputation or joint modeling) as predictors for every masked target.
   The Christensen adapter in `per_feature_adapter.py` (line 69) only uses
   the second-half (always-observed) D/2 features as predictors — it never
   sees the other masked features. On breast, that means predicting each
   of 15 masked features from 15 always-observed features, while MICE uses
   all 29 others. **This is not noise; it's a hard information bottleneck.**
   REPORT.md mentions this only in the "Where Christensen loses" paragraph
   on breast, framed as "a legitimate methodological limitation" — but it
   is present on ALL datasets, not just breast. The fact that Christensen
   still wins on 4/6 doesn't make the comparison fair; it makes the other
   methods' failure to exploit their information advantage suspicious.

2. **Model classes are different in ways that matter differently on
   different datasets.** not-MIWAE is a joint deep generative model that
   explicitly models the missingness mechanism. Christensen is a linear
   per-feature regression with a structured adversarial adjustment. The
   claim "Christensen dominates shallow imputers" sweeps over the fact
   that MICE, missForest, and Christensen all make different assumptions
   about structure: MICE assumes MAR per-feature conditional on others,
   missForest assumes tree-representable structure, Christensen assumes a
   selective-observation model with a known Q class. The MNAR mechanism
   here is exactly the one Christensen's Q class was designed for
   (`SelfMaskingAboveMean` via `MonotoneInY(decreasing)`). **Christensen
   is being tested on the adversarial scenario it was built for; MICE is
   being tested outside its stated assumptions.** That's not cheating, but
   a careful reviewer will ask: does Christensen generalize to MNAR
   mechanisms it was not designed for? Phase 2 doesn't answer this.

3. **Per-feature vs joint is a design choice, not a limitation.** REPORT.md
   frames per-feature as a necessary consequence of Christensen being
   row-level. That's only half-true: nothing prevents an iterative variant
   (Christensen-in-MICE-loop) that uses current imputations of the other
   masked features. That variant isn't implemented, and its absence means
   we don't know whether Christensen loses on breast because (a) linear is
   wrong for breast, or (b) it starved itself of 15 predictors. Until that
   ablation exists, the "fair comparison is to shallow imputers" claim is
   unsubstantiated.

---

## 3. Statistical methodology issues

**Seed variance collapsed to zero for three of four methods.** Recomputed
from the CSV:

| Method                | std across 5 seeds (max across datasets) |
|-----------------------|------------------------------------------|
| mean                  | 0.000                                    |
| mice                  | 0.000                                    |
| christensen_faithful  | 0.000                                    |
| missforest            | 0.018 (red)                              |

MICE returning identical RMSE across seeds is not a code bug — sklearn's
`IterativeImputer` with default `sample_posterior=False` and
`imputation_order='ascending'` is fully deterministic, and BayesianRidge
has no stochastic component. The `random_state` argument has no effect.
Mean is trivially deterministic. Christensen (closed-form inner + SLSQP
outer) is deterministic if SLSQP init is deterministic. The mask is
deterministic by construction (§3 below).

Consequence: **5-seed runs are compute waste for three of four methods.**
The "±std" columns in REPORT.md are literally 0.000 for those methods, so
any claim of "statistical significance" or "confidence interval" is
misleading — there is no variance to estimate. This is consistent with
what the paper reports (their Mean/MICE also have ±0.00), but it needs to
be called out, not papered over. The correct framing is: "these are point
estimates under deterministic pipelines; seeds only vary missForest."

**Deterministic mask compounds the problem.** `self_masking_above_mean` is
fully deterministic given X (PROTOCOL.md §"MNAR mechanism" and
mnar_injection.py:66 confirm — `seed` is `del`'d). So the seed budget
doesn't sample the mask distribution either. A proper variance estimate on
MNAR imputation would at minimum bootstrap X or perturb the mean
threshold; neither is done. The CIs in REPORT.md are not CIs; they are
estimation noise of a single stochastic method (missForest).

**Small-sample concern on breast.** Breast: N=569, D=30, half=15 features
imputed. Measured mask rate is 0.199 (not the nominal 0.25) because
breast-cancer features are right-skewed — the mean-split puts <50% above
the mean. Per masked feature, we have 569 − n_masked observed rows
(~456 rows) fitting a linear model in 15 predictors + intercept. n/p ≈ 30;
not catastrophic, but far tighter than on white wine (n/p ≈ 800). The
per-feature regression is likely overfitting on breast, which bears on
whether "linear is wrong" vs "we starved ourselves of predictors."

---

## 4. Interpretation — is the headline defensible?

**Partially.** Let me break down claim by claim:

- *"Beats MICE on 4/6."* Recomputed: banknote −0.235, concrete −0.444,
  red −0.188, yeast −0.269 are wins. White: Christensen 1.426 vs MICE
  1.412 = **+0.015, a loss, not a tie**, contra REPORT.md. Breast: +0.343
  loss. So the record is **4 wins, 2 losses, 0 ties**, not "4/6 ties on
  1 loses on 1." This is a direct factual error in the headline table.
- *"Beats missForest on 6/6, Mean on 6/6."* Verified from the CSV.
  Correct.
- *"Does NOT beat not-MIWAE."* Correct — Christensen loses on all 6 vs
  not-MIWAE's best variant. But this comparison is against numbers we did
  not re-run (see §5 below).
- *"Christensen is ~100× faster (13ms vs minutes)."* This compares
  apples to oranges. Mean is ~1ms, MICE is ~100ms, Christensen is
  20–220ms, missForest is 6–145s, not-MIWAE is "minutes." On 4 of 6
  datasets Christensen is slower than MICE, not faster — the 13ms figure
  is for banknote only, the smallest dataset. The "~100× faster than
  not-MIWAE" claim is defensible but not novel; mean imputation is
  ~10⁶× faster than not-MIWAE and nobody cites that.
- *"Theoretical worst-case guarantees that not-MIWAE lacks."* Not audited
  here, but in the non-asymptotic regime of N=569 breast, any asymptotic
  guarantee is hard to cash. The paper submission would need to cite the
  specific sample-complexity theorem and show it bites at N=569.

---

## 5. Reviewer concerns NOT mentioned in REPORT.md

1. **"not-MIWAE best" is best-of-three cherry-picking.** The paper reports
   three variants: agnostic, self-masking, self-masking-known. On
   banknote, self-masking is 1.88±0.85 (huge variance!) but
   self-masking-known is 0.74±0.05. Using the best of three across
   datasets without telling the reader you did is cherry-picking and
   overstates the gap we fail to close. PROTOCOL.md §"Baselines we take
   from Ipsen et al." does note this, but REPORT.md just says "not-MIWAE
   best" — a reviewer reading REPORT.md alone would miss it.

2. **The banknote not-MIWAE number (0.57) is the PPCA+self-masking-known
   variant, not the full deep not-MIWAE.** Paper Table 1 shows deep
   not-MIWAE self-masking-known on banknote is 0.74±0.05, not 0.57.
   The 0.57 figure is from "not-MIWAE - PPCA." That's a linear PPCA with
   an MNAR head, not a deep generative model. If we're comparing against
   that, the gap is smaller than REPORT.md implies, AND it is the same
   class of model (linear) as Christensen — so the "linear vs deep"
   framing is wrong. The correct comparison is Christensen vs
   not-MIWAE-PPCA-self-masking-known, both linear-ish methods with MNAR
   awareness. On concrete: paper not-MIWAE-PPCA-self-masking-known = 1.31,
   Christensen = 1.424. Gap = 0.11. On red: 1.13 vs 1.493, gap 0.36. On
   white: 0.99 vs 1.426, gap 0.44. On yeast: 0.77 vs 1.487, gap 0.72. On
   breast: 0.72 vs 1.289, gap 0.57. So even the linear-linear comparison
   Christensen loses, but the gap story is very different from "linear
   vs deep."

3. **Baseline reproducibility gap (§1) is waved off.** REPORT.md line 67:
   "Our MICE RMSE (0.946) is lower than their reported 1.17 — likely due
   to different sklearn MICE defaults." This is a priori unlikely given
   we literally call `IterativeImputer(max_iter=10, random_state=seed)`
   exactly like `task01.py`. More likely causes: different dataset
   version, different standardization, different NaN handling in the
   randomization path. This is unresolved; it shouldn't be dismissed as
   "minor."

4. **We're scored against numbers that are 4 years old on the wrong
   computer.** The paper reports 5 runs on an unspecified GPU setup with
   TF1.x. Not-MIWAE training is known to be seed-sensitive and
   early-stopping-sensitive; subsequent literature (e.g., Collier et al.
   2020, Ghalebikesabi et al. 2022) has re-run not-MIWAE and gotten
   different numbers on some datasets. Cite those re-runs OR re-run
   not-MIWAE ourselves. Citing Table 1 as a fixed target assumes
   reproducibility that is not established.

5. **Seed claim in PROTOCOL.md is inconsistent with the data.** PROTOCOL.md
   line 163 says "Reporting: mean ± std over 5 seeds." The std is
   literally zero for 18 of 24 (dataset, method) cells. PROTOCOL.md
   line 168-169 acknowledges this, but REPORT.md line 73 is more candid
   than the headline table. The headline should carry a ⁎ footnote.

6. **Dataset loader discrepancies not audited.** Yeast +0.039 / +0.036 /
   +0.052 consistent positive deviations strongly suggest a dataset
   version issue (e.g., different one-hot treatment of the class column
   or a slightly different row count). We haven't checked the loader
   against the one task01.py uses. This could swing results.

7. **No ablation on δ=0.30 or Q class choice.** PROTOCOL.md cites
   `MonotoneInY(direction="decreasing")` with δ=0.30, adaptively
   centered. No sensitivity analysis. A reviewer will ask what happens
   at δ=0.1, 0.5, 1.0, and whether these were tuned on test data
   (circular).

8. **Transductive evaluation = possible overfit.** PROTOCOL.md line 156
   notes the RMSE is evaluated on the same rows used to fit. Christensen
   here is fit on observed entries and evaluated on masked entries of
   the same dataset. That's standard for the benchmark, but for a
   closed-form linear method with 15 predictors on 569 rows (breast),
   the "train on observed, test on masked" split is not a clean
   generalization test — the observed and masked rows come from a
   non-random partition (above/below mean), i.e. exactly the selection
   bias Christensen's Q class is designed for. So Christensen is
   evaluated on a distribution shift that its Q class knows about by
   construction. This is not wrong; it's the point of the benchmark.
   But it does mean Phase 2 doesn't test generalization of the Q choice.

---

## 6. Bottom line — would this survive review at ICLR/NeurIPS?

**In its current form: no — not as a headline contribution. Maybe as an
appendix or workshop paper.** The core issues:

- **Reproducibility gap on 3 of 6 baselines.** Breast MICE is off by 19%,
  concrete MICE by 10%, yeast all three baselines by 2–3%. An ICLR
  reviewer will run `task01.py` and notice. The "likely due to sklearn
  defaults" excuse doesn't hold when we use the same sklearn call as
  task01.py.

- **Factual error in the headline table.** White is a Christensen loss
  vs MICE, not a tie. The 4-1-1 record is actually 4-2-0. Small, but
  the kind of thing a referee flags as sloppy.

- **Cherry-picking "not-MIWAE best."** Using best-of-three variants
  without running those variants ourselves is overclaiming. The
  banknote 0.57 number cites the wrong row of Table 1 (PPCA variant, not
  the deep not-MIWAE).

- **The comparison is not apples-to-apples.** Christensen uses strictly
  less information (D/2 predictors) than MICE, missForest, and not-MIWAE
  (all D−1 or joint). That Christensen still wins on 4/6 vs MICE is an
  interesting result, but the right framing is "a per-feature restricted
  imputer matches or beats joint imputers on MNAR when the Q class is
  known," not "Christensen dominates." This reframe is more defensible
  and actually strengthens the theoretical contribution.

- **The speed claim doesn't hold as stated.** Christensen is not 100×
  faster than MICE; on the CSV, it is often slower than MICE. The 100×
  claim is only vs not-MIWAE, which is expected because one is closed-
  form and the other trains a neural net for 100k iters.

- **Variance story is non-existent.** 5-seed runs with deterministic
  mask and deterministic imputers yield zero std for 3/4 methods. CIs
  are not meaningful. This needs honest acknowledgment, not a "we match
  the paper's 5 runs" hand-wave.

**What would fix it:**

1. Re-run not-MIWAE ourselves (or cite a published re-run) so the
   comparison isn't against a 4-year-old table.
2. Add a Christensen-in-MICE-loop variant to isolate the per-feature
   information bottleneck from the Q-class contribution.
3. Resolve or honestly document the baseline reproducibility gaps
   (breast MICE, concrete MICE, yeast all).
4. Replace "5 seeds" with bootstrapped or X-perturbed CIs, or drop the
   ±std column entirely and report point estimates.
5. Fix the white-is-a-tie claim and the banknote not-MIWAE 0.57 attribution.
6. Add a δ and Q-class ablation.
7. Reframe the headline from "Christensen dominates shallow imputers" to
   "a per-feature-restricted Christensen matches joint imputers on MNAR
   with a structured Q — promising even without joint modeling." Less
   grand, more defensible.

With those fixes the work is a solid empirical contribution. Without them
a competent reviewer hits it for (1) cherry-picking and (2) inconsistent
reproducibility and recommends reject.
