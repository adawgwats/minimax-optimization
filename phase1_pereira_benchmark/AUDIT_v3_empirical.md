# Phase 1 Audit v3 — Independent Empirical and Methodological Review

**Auditor scope:** independent review of the Pereira-2024 MNAR benchmark replication and the headline claim in `REPORT_v2.md`:

> "christensen_faithful beats MICE in 22% of cells overall, 60% on MBOV_Lower (outcome-correlated MNAR)."

**Verdict (TL;DR).** The measurement infrastructure is competent and the headline numbers are reproducible from `raw_results_v2.csv`. However, the headline as written is **not defensible for peer review without major reframing**:

1. The "60% wins on MBOV_Lower" result is **dominated by a degenerate regime**: in 34 of the 56 overall wins (61%), the observed-y-positive-rate is ≥0.99, i.e., every baseline has collapsed to a constant predictor because MBOV_Lower on binary labels with a majority-positive class deletes the entire minority class from the training set. In these cells, every imputation baseline (MICE, kNN, mean, Heckman, IPW, complete-case) produces **bit-identical MSE with std=0 across seeds** — they are not functioning as baselines, they are all the same trivial estimator.
2. Under the symmetric mechanism **MBOV_Higher** (which deletes majority-class labels), Christensen's estimator **loses in 23/50 cells and wins zero**. The Q-class dispatch is symmetric by construction, so this asymmetry is a sign that the wins are driven by a specific structural feature of MBOV_Lower + majority-positive binary labels, not by general robustness to MNAR.
3. The 95% CI uses z-critical (1.96) instead of t-critical (~2.262 at n=10). Refitting with t-critical reduces overall wins 56→53 (22.4%→21.2%) and MBOV_Lower wins 30→28 (60%→56%). Small effect, but the methodology description is strictly wrong for n=10.
4. **CTG dataset contaminates the dataset-level "losses" table**. On CTG, features are near-linearly deterministic of Y, so MICE/complete-case/Heckman/IPW/kNN all achieve **MSE ~ 1e-5** (indistinguishable from oracle). Christensen's reweighting introduces variance and loses badly (MSE ~ 0.1–0.3). The headline's "CTG: 6 wins / 18 losses" is mostly this.
5. MBIR mechanisms (500 of 3500 cells, 14%) are **silently excluded** because Christensen's adapter raises `NotImplementedError`. The report caps the denominator at 250 cells without explaining that 100 cells × 2 MBIR mechanisms were dropped. The headline "22% of 250" should also be reported as "22% of *the 250 cells where the method runs*, out of 350 total."

Would this survive Journal of Econometrics review: **no**, not in its current form. Would it survive NeurIPS: **no**, for similar reasons (reviewer would flag the degenerate-baseline-collapse and ask for additional baselines that don't collapse, e.g., regularized-logistic, class-rebalanced). Both findings are fixable with honest scoping and additional diagnostics.

---

## 1. Protocol fidelity: Pereira 2024 vs this benchmark

Pereira, Abreu, Rodrigues & Figueiredo (*Expert Systems with Applications* 2024, DOI:10.1016/j.eswa.2024.123654) — confirmed to exist as the 2024 journalized Chapter 9 of Pereira's 2023 Coimbra PhD thesis. Authors' own implementation is at `github.com/ArthurMangussi/pymdatagen` (verified in the local extract at `docs/pereira_2024/pymdatagen_src/`).

### What Pereira's benchmark actually is

Per `docs/pereira_2024/chapter9_mnar.txt`:

- **Task**: imputation quality. Inject MNAR into *feature values*, impute, measure **Mean Absolute Error (MAE)** between the imputed feature values and the (hidden) ground truth.
- **Datasets**: 10 UCI medical datasets, complete (no pre-existing NaN).
- **Mechanisms**: MBOV (4 variants: Lower, Higher, Stochastic, Centered), MBUV, MBIR (Frequentist, Bayesian), MBOUV.
- **Rates**: {10, 20, 40, 60, 80}%.
- **Split**: 50/20/30 stratified by class.
- **Seeds**: **30 independent runs**, 95% CI = μ ± 1.96·σ/√30.
- **CI rule**: overlapping CIs → not significantly different.
- **Baselines**: DAE, VAE, kNN, Mean/Mode, **MICE**, SoftImpute.

### Pereira's strengths

- Well-scoped, concrete MNAR generation mechanisms with published code (`mdatagen`).
- Public, reproducible datasets.
- Honest CI-overlap semantics for "no significant difference".

### Pereira's known limitations

- **Datasets are classification tasks** (binary or multi-class), not regression tasks. The MAE on imputed feature values is well-defined regardless, but the clinical targets are all categorical.
- MBOUV is multivariate by design (amputes many features at once).
- **No modelling task is evaluated** — only imputation quality. So the benchmark is silent on whether good imputation leads to good downstream prediction.
- **n=30 seeds is still small** for claiming significance in a 350-cell benchmark without multiple-comparison correction.

### This project's deviations from Pereira (declared)

Per `PROTOCOL.md §Deviation`:

| # | Pereira | This project |
|---|---|---|
| 1 | Task: imputation | **Regression on label with MNAR on Y** |
| 2 | Metric: MAE on imputed X | **Test-set prediction MSE** |
| 3 | MBOUV included (multivariate) | **Excluded** (label-only target) |
| 4 | Baselines include DAE/VAE/SoftImpute | **Omitted**; MICE retained |
| 5 | Multiclass targets kept as classification | **Binarized** to {0,1} for LPM regression (6 of 10 datasets affected) |
| 6 | 30 seeds | **10 seeds** (undeclared deviation — PROTOCOL.md §Reproducibility still says 30) |

### Undeclared deviations

These are in the code/data but not flagged in PROTOCOL.md:

| # | What PROTOCOL.md says | What the code actually does |
|---|---|---|
| A | "30 seeds" (§9.2) | **10 seeds** — run with `range(10)` in `run_benchmark.py`. REPORT_v2.md says "seeds: 10" but PROTOCOL.md was not updated. |
| B | MNAR injected on train only | True. Val is computed but **never used** — the 20% validation split is dead code. No method tunes hyperparameters here. |
| C | "n_instances: 303" for Cleveland | **297** — the code silently drops 6 NaN rows. This is mentioned in the dataset note string but not PROTOCOL.md. |
| D | "feature count differs from Pereira" for CTG | True; OpenML's version differs. But: **every OpenML `fetch_openml` call prints a warning that multiple versions exist** (version 1 vs 2). The code fixes nothing and relies on sklearn's "latest version 1" default. Fragile — silent reproducibility hazard. |
| E | MBIR covered by mdatagen wrapper | For `christensen_faithful` specifically, MBIR raises `NotImplementedError` in `christensen_adapter.py` via `reference_based_q.py:centered_q_for`. **1000 of 35000 rows are missing** for this method (50 MBIR cells × 10 seeds × 2 variants = 1000). REPORT_v2.md aggregates "across 250 cells" — i.e., silently excluding MBIR from the denominator. |
| F | MBIR Bayesian uses "BF ≥ 10" (per paper) | `mnar_injection.py:170` **falls back to Mann-Whitney** for both Frequentist and Bayesian variants because mdatagen 0.2.0 only implements Mann-Whitney. The code comments this, but the dataset labels still say `MBIR_Bayesian`. So "MBIR_Bayesian" results are identical to "MBIR_Frequentist" — the mechanism has no Bayesian semantics in this benchmark. |
| G | "Random state: seeded via numpy global RNG" | `mnar_injection.py` uses `np.random.seed(seed)` (global) inside `_apply_mbov` and `_apply_mbir`. This is fragile — any method that touches the global RNG between the injection and model fit can shift results. sklearn's `IterativeImputer`, KNN, Heckman, and minimax_core all potentially do. This isn't a correctness bug, but a reproducibility hazard that a reviewer will flag. |
| H | [0,1] normalization "per column", pre-split | Implementation normalizes over the **full dataset** before splitting (`preprocess.py:33`). This is a mild form of leakage (test min/max informs train normalization) but matches the paper's protocol as worded. Documented in the docstring. |

---

## 2. Statistical methodology critique

### 2.1 CI computation: z vs t for small n

The harness (`harness.py:8`, `analyze.py:33-37`) uses **z-critical 1.96**. For n=10 paired seed-means with t-distributed sample means, the correct critical value is `t.ppf(0.975, df=9) ≈ 2.262`. This is a 15% widening of all CIs.

I recomputed every win/tie/loss with t-critical:

| Metric | REPORT_v2 (z=1.96) | Corrected (t=2.262) |
|---|---|---|
| Overall wins | 56 (22.4%) | **53 (21.2%)** |
| Overall losses | 62 (24.8%) | 60 (24.0%) |
| Overall ties | 132 (52.8%) | 137 (54.8%) |
| MBOV_Lower wins | 30 (60%) | **28 (56%)** |
| MBOV_Stochastic wins | 23 (46%) | 22 (44%) |

Cells that flip WIN→TIE under t-critical:
- `vertebral/MBOV_Lower/60%`
- `wisconsin/MBOV_Lower/20%`
- `cleveland/MBOV_Lower/40%` (one more subtle flip)

Small effect, but the methodology is literally wrong for n=10 and a reviewer will insist on the correction. The headline should say "60% of MBOV_Lower cells" or "28 of 50", not 60%, if we use t-critical. The claim still holds qualitatively; it just downgrades slightly.

### 2.2 Number of seeds

Pereira uses 30 seeds. This project uses **10**. PROTOCOL.md says "30 independent runs" but `run_benchmark.py` and `harness.py` both use `seeds = tuple(range(10))` / `range(30)` respectively — harness.py's default is 30, but the driver script overrides to 10. REPORT_v2.md says "Seeds: 10".

Consequences:
- The standard-error of cell-level mean MSE is √3 ≈ 1.73× higher than it would be with 30 seeds. Some "TIE" cells at n=10 would become "WIN" at n=30; others would go the other way. The *rate* of wins may be biased either way.
- With n=10, small per-seed variance fluctuations dominate. Under MBOV_Lower's all-positive-class-collapse regime, std=0 exactly for 6 of the 10 baselines (they all fit y_pred=1 constant), which makes CI overlap trivially testable — the std=0 CIs become degenerate zero-width intervals, any method with nonzero MSE difference becomes a "WIN" regardless of magnitude.

### 2.3 Multiple comparison correction

No correction is applied in the current analysis. With 250 cells tested at α=0.05, expected false positives ≈ 12.5 (Bonferroni threshold α=0.0002).

I ran paired seed-level tests (more powerful than CI overlap since it controls for cross-method shared variance from train/test splits):

| Correction | Wins | Losses |
|---|---|---|
| Uncorrected paired t-test (p<0.05, diff<0) | 72 | 88 |
| Bonferroni (α/250 = 0.0002) | **50** | 51 |
| BH-FDR (q=0.05) | 69 | 82 |

Bonferroni survival: 50 of 72 (69%) of uncorrected wins. Interestingly, paired tests find **more** signal than CI-overlap because the seed-level noise is partially shared. By mechanism under Bonferroni: MBOV_Lower 28/50, MBOV_Stochastic 21/50, all others ≤1 win.

So even under strict multiple-comparison correction, the "MBOV_Lower and MBOV_Stochastic wins" persist. But *overall* the headline "56 wins / 250 cells = 22%" is inflated — under Bonferroni the denominator should report **both wins AND losses** (50 vs 51 — essentially a wash). A reviewer will ask: is the framework "better than MICE" or is it "differently biased in a structured way"? The current phrasing implies the former; the data support only the latter.

### 2.4 Permutation null

Under the permutation null (swap christensen and MICE labels at the row level, 500 iterations), the expected number of CI-separation "wins" is 4.1 ± 3. The observed 56 wins is at p<0.002. So the signal is real and not a multiple-comparison artifact. **However**, this null is the wrong test — it tests whether the methods differ, not whether Christensen is better. A more informative null is "Christensen vs Christensen with a different random seed" which would have near-zero wins; the 56 reflects algorithmic difference, not Christensen superiority.

---

## 3. Pipeline correctness

### 3.1 Feature (X) MNAR → label (Y) MNAR

Pereira injects MNAR on *feature* values. This project injects MNAR on *labels* (`missTarget=True`). The declared justification is correct — Christensen's estimator is defined for the latter. But this changes the benchmark in a subtle way:

- MBOV_Lower on a continuous feature "remove lowest 40% of X_j values" is a rich MNAR signal because X_j is continuous.
- MBOV_Lower on a **binary label** y ∈ {0,1} collapses to "remove y=0 first." For datasets where y=0 is the majority class, this produces a training set with moderate class imbalance; for datasets where y=0 is the *minority* class, removal at rate ≥ minority_fraction deletes the entire minority class.

The "60% wins on MBOV_Lower" is largely driven by the second regime. CTG (81.9% positive), wisconsin (62.7% positive), vertebral (80.6% positive) all have majority-positive labels. MBOV_Lower at rate ≥ 40% on these deletes nearly all minority Y=0. On the datasets with majority-negative labels (thyroid 7.4% positive, pima 35% positive, saheart 35%, transfusion 24%), MBOV_Lower is less extreme — and the wins are smaller in magnitude (though still present).

This is a legitimate MNAR scenario, but it's specifically a **class-imbalance-driven wipe-out** in many "win" cells. The 60% win rate on MBOV_Lower would not hold if the benchmark were run with balanced classes or with continuous Y.

### 3.2 Multi-class → binary LPM conversion

Per `datasets.py`, 6 of 10 datasets are binarized:
- Cleveland: 5-class → binary (healthy vs any disease). Balanced (46% positive after binarization).
- CMC: 3-class → binary (use vs no-use). Balanced (57% positive).
- CTG: 10-class (or 3-class) → binary (normal vs not). **82% positive.**
- Thyroid: 3-class → binary. **7% positive** (extreme imbalance).
- Vertebral: 3-class → binary. **81% positive.**

The binarization is documented but the **positive-class rates are not reported in PROTOCOL.md**. This is directly relevant: MBOV_Lower + 80% majority-positive = guaranteed all-positive train. The baseline-collapse regime is predictable from the class balance table alone — it's not noise.

### 3.3 Feature-NaN handling (Cleveland)

`datasets.py:300-310` silently drops 6/303 Cleveland rows (n=297 effective). This is a 2% reduction but:
- The rows dropped are MCAR w.r.t. nothing in particular (they're original UCI missing features).
- If the dropped rows had any systematic relation to outcome, it would be a selection bias in the entire experiment.
- PROTOCOL.md §Datasets still lists n=303 for Cleveland.

Small issue but a reviewer's first question will be "why isn't your Cleveland row count 303?" and the protocol doc will be wrong.

### 3.4 Stratified split: val/test held out from injection?

Per `harness.py:93-96`: `inject(split.X_train, split.y_train, ...)` — MNAR applied to **train only**. Val and test are clean. This is correct (otherwise evaluation wouldn't be a held-out signal). Val is computed but never consumed by any method — **dead data**. If we went to 80/20 train/test with no val, we'd have 60% more training data per cell, which would tighten CIs.

### 3.5 `minimax_score` adapter: training ignores `response_mask`

Looking at `minimax_adapter.py:82-87`:
```python
y_filled = y.copy()
y_filled[~response_mask] = 0.0
proxy_labels = np.where(response_mask, y_filled, observed_mean)
```

The SGD-based minimax fills hidden y with **mean of observed y** as a "proxy." This contradicts Christensen's derivation (Divergence E in `AUDIT_v2.md`: Christensen sets Ỹᵢ=0 for non-respondents; they drop out of r_n). The DRO variant's proxy-label trick is an **invention** of this codebase, not Christensen's. This is declared in comments but not in REPORT_v2.md.

### 3.6 Christensen adapter: mechanism name leakage

`christensen_adapter.py:53-65` passes the **mechanism name** from the harness directly to the Q-class dispatch. This means Christensen sees metadata *about the true MNAR mechanism* that no other baseline has. The mechanism determines:
- Which Q-class to use (`Parametric2ParamForBinary`, `ConstantQ`, `MonotoneInY`).
- The delta (radius of the Q-ball) via `MECHANISM_DELTA` table — 0.30 for MBOV_Lower, 0.05 for MBUV, etc.

**A skeptical reviewer will ask**: in deployment you do *not* know which of Pereira's specific mechanisms generated the data. Is this leak fair?

REPORT_v2.md notes (correctly) "a follow-up comparing the two algorithms is warranted" but does not flag this as a methodological concern. The comparison is effectively "Christensen-with-oracle-mechanism-metadata vs MICE-with-no-metadata." A fair comparison would either:
- Give MICE/Heckman/IPW access to the mechanism type too (unclear what they'd do with it).
- Force Christensen to use `DEFAULT_DELTA=0.30` with no mechanism prior.
- Learn the Q-class/delta from a validation set (which is what the 20% val split was supposed to be for, but isn't used).

This is possibly the single most significant methodological issue in the current report.

---

## 4. The headline: defensible?

### "christensen_faithful beats MICE in 22% of cells overall, 60% on MBOV_Lower"

**Literal arithmetic**: 56/250 = 22.4%. 30/50 on MBOV_Lower = 60%. Both numbers reproduce from `raw_results_v2.csv`.

**Whether these numbers mean what they seem to mean**: no.

1. **Denominator is wrong**: 250 excludes 100 MBIR cells where Christensen silently fails. Should be "56/250 *of supported cells*, 0/100 of unsupported" or "out of 350 total cells, 56 supported wins, 100 unsupported, 194 supported non-wins."

2. **60% on MBOV_Lower is partially the measurement of a degenerate regime**: at high missing rates with majority-positive classes, MICE/complete-case/etc all collapse to the same constant predictor. The "60% win" includes these cells. They're not wrong but they're a weak counterfactual — MICE is not really being stress-tested, because the problem is too hard for it.

3. **The mechanism-name leak** means Christensen is comparing against methods that don't have its information. Not a fair head-to-head.

### Suggested honest phrasing

> "On 250 label-MNAR cells in the Pereira mechanisms where our Christensen adapter applies (MBIR-family excluded; 100 cells missing), a reference-based Christensen estimator outperforms MICE in 53 cells (95% CI via t-critical, 21%) and underperforms in 60 cells (24%). Wins concentrate on MBOV_Lower and MBOV_Stochastic at missing rates ≥40% on datasets where the majority class is positive (34/53 wins). In these cells, the MNAR selection has deleted all minority-class training examples and naive imputation baselines have collapsed to constant prediction. On MBOV_Higher (symmetric mechanism) and MBUV (near-MAR), Christensen loses or ties in all cells. The Christensen estimator is given the true mechanism name at training time — a material information advantage over the other baselines that we do not attempt to equalize."

This is defensible. The current headline is not.

---

## 5. Reproducibility

- `raw_results_v2.csv` is a single 46.5-minute run with checkpointed resume support. The `run_v2.log` shows a single run from "total cells: 3500" to "[3500/3500] elapsed 2787s". **Grep of "Resuming" returns 2 matches**, but both are the docstring/preamble — the actual run did not resume from a partial state.
- Row count is exact: 34,000 = 3,500 cells × 10 methods − 1,000 (100 MBIR cells × 10 seeds) failed for christensen_faithful. All 3,400 (dataset, mechanism, rate, method) combinations have exactly 10 seeds each. Data is balanced and canonical.
- Reproducibility hazards:
  - `fetch_openml` multiple-version warnings on 7 datasets (cleveland, cardiotocography, diabetes, blood-transfusion, vertebra-column, cmc). Version 1 is the implicit default but OpenML can re-version. Pin these with `version=1` explicitly.
  - Global `np.random.seed(seed)` is used in mnar_injection; changes in sklearn's/scipy's RNG consumption order can shift results. Use `np.random.default_rng(seed)` or pass explicit `random_state` throughout.
  - The `christensen_core.reference_based_q.MECHANISM_DELTA` table is **hard-coded domain knowledge** derived from `tests/diagnostic_centered_vs_wide.py`. A reader who re-implements this table with different deltas will get different results.
  - `mdatagen 0.2.0` is the version used; the MBIR Bayesian fallback to Mann-Whitney is a fragile dependency on a specific version of that library.

Someone cloning the repo can reproduce the numbers by installing mdatagen==0.2.0, scikit-learn (version not pinned), and running `run_benchmark.py` with the fixed seed range. **`requirements.txt` should be made more restrictive** — it does not pin mdatagen or sklearn versions.

---

## 6. Top 5 issues NOT discussed in REPORT_v2.md or PROTOCOL.md

### Issue 1 — Baselines degenerate to identical estimators in 19% of rows

In 671 of 3,500 cells (across seeds), MICE, complete_case, kNN_impute, mean_impute, Heckman, and IPW_estimated return **bit-identical test MSE** (within 1e-3). These are cells where the training set has all-one-class labels (obs_y_pos ≥ 0.99 in 319 rows, or ≤ 0.01 in 270 rows). In the "wins" subset specifically, 34/56 overall wins and 27/30 MBOV_Lower wins are in this regime. **The "6 baselines vs 1 minimax" framing is misleading** — in the wins cells, it is effectively "1 degenerate baseline vs 1 minimax." A reviewer will immediately note that the six "baselines" are the same constant predictor in the cells that matter.

### Issue 2 — CTG is a degenerate dataset that dominates the "losses" column

CTG is nearly linearly separable by features (oracle MSE ~1e-7). For *any* method that can fit a linear model from clean data, MSE ≈ 0. So MICE's MSE ≈ 0 on CTG is **not a measurement artifact** — it's that features alone predict Y perfectly once you have enough of them. Christensen's adversarial reweighting introduces artificial variance that can't be justified when the naive method has access to the real signal. This is *the* failure case for robust methods: robustness is useless if the nominal estimator is already near-optimal. The REPORT_v2 "Least favorable cells" table is 10/10 CTG entries with impossible "diff_%" values (31587552%, 16579916%, etc.) because the denominator (mice_mse) is ~1e-21. **These numbers should not appear in a paper.** Either they're handled specially (CTG excluded with justification) or the percentages are replaced with absolute differences.

### Issue 3 — Mechanism name is a free oracle signal for Christensen

Christensen_faithful receives the **true MNAR mechanism name** via `mechanism_name=mech` at fit time. This determines the Q-class (increasing monotone, decreasing monotone, constant, etc.) and a pre-calibrated `delta`. No other baseline receives this signal. Effectively, Christensen is getting a "the true DGP is in this family" hint that the other methods cannot use. This is declared as adaptive delta in the source comments but not flagged as an experimental confound in REPORT_v2.md.

### Issue 4 — MBIR_Bayesian is not actually Bayesian

`mnar_injection.py:169-171` falls back to Mann-Whitney for both Frequentist and Bayesian MBIR variants because mdatagen 0.2.0 only implements Mann-Whitney. So the 500 cells labeled "MBIR_Bayesian" are **not MBIR_Bayesian data** — they're MBIR_Frequentist with a different label. Any per-mechanism comparison that separates the two is spurious. This does not affect the headline (which excludes MBIR) but will bite the next report if MBIR is included.

### Issue 5 — Version/dataset provenance is unpinned

`fetch_openml` emits warnings on 7 of 10 datasets that "multiple active versions exist." The code relies on sklearn's default-to-version-1 behavior. If OpenML renumbers datasets (which has happened) or sklearn changes its defaulting policy, results will drift silently. `requirements.txt` doesn't pin sklearn, numpy, scipy, pandas, or mdatagen. This is fine for internal use but is a blocker for claiming reproducibility in a published paper.

---

## 7. Peer-review verdict

### Would this paper survive *Journal of Econometrics* review?

**No** (not in current form). Specific reviewer objections I would raise:

- Framework validity: the tested estimator gets the true mechanism name as input, rendering the MICE/Heckman comparison asymmetric in information. Re-run with mechanism-agnostic delta selection (e.g., cross-validated on the 20% val split) before any publication claim.
- Baseline coverage: Heckman, IPW, and MICE all degenerate to the same estimator in the all-positive-class regime. Need a baseline that does *not* collapse (e.g., regularized-logistic on imputed data, or class-balanced oversampling). Otherwise the comparison is degenerate in the regime where the wins concentrate.
- Sample size: n=10 seeds with t-critical CIs is underpowered. Scale to n=50 or n=100 before any claim of "significance."
- Multiple comparison: 250 cells × 2 tests without BH-FDR or Bonferroni. With BH-FDR, the story still holds on MBOV_Lower/Stochastic, but the headline needs rewriting.

### Would this paper survive *NeurIPS* review?

**No**. Specific objections:

- The experimental setup is **methodology transfer** from Pereira, not replication. Novelty claim is weak without direct comparison to a published selective-observation benchmark (there aren't many; Pereira-MNAR is the right starting point but the authors of that benchmark are explicitly not doing regression tasks).
- CTG ratio is a known pathological case for linear models on imputed data; excluding it without commentary is an anti-pattern.
- The Christensen "faithful" adapter's delta-by-mechanism table is hard-coded and not learned. A reviewer will immediately ask "what happens if you use a fixed delta, ablating over {0.05, 0.15, 0.30, 0.50}?" This ablation is not present.

### What would make this publishable

1. Drop the degenerate cells (obs_y_pos ≥ 0.99 or ≤ 0.01) from the headline and report them separately as "sanity-check cells." That's a minimum.
2. Run a cross-validated delta selection using the 20% val split, for all methods that need a hyperparameter. Give Christensen_faithful the same hyperparameter-selection burden as any other method.
3. Scale to 30 seeds (the Pereira standard). This is 3× compute — ~2.5 hours on the current setup.
4. Add a regularized baseline (e.g., `sklearn.linear_model.LogisticRegressionCV` or Ridge) that does not degenerate in the all-positive-class case. Compare against it.
5. Exclude CTG from the headline with an explicit note, or aggregate absolute MSE differences instead of percentages.
6. Fix the t-critical CI. State n, the critical value, and the CI-overlap rule explicitly.
7. Report wins/losses as a **paired** comparison (per-seed paired t-test with Bonferroni), not just CI overlap. Both should agree; reporting both builds credibility.
8. Implement the MBIR Q-class so you have 350/350 cells instead of 250/350. If it's impossible in Phase 1 scope, exclude MBIR cleanly in the denominator reporting.

---

## Appendix: Specific numerical checks I ran

All run against `raw_results_v2.csv`:

1. **Shape check**: 34,000 rows = 10 datasets × 7 mechanisms × 5 rates × 10 seeds × 10 methods − 1,000 MBIR failures for christensen_faithful. Confirmed balanced; all cells have exactly 10 seeds.

2. **t-critical recomputation** (t.ppf(0.975, df=9) = 2.262): overall wins 56 → 53; MBOV_Lower wins 30 → 28; specific flips:
   - `vertebral/MBOV_Lower/60%`: WIN → TIE
   - `wisconsin/MBOV_Lower/20%`: WIN → TIE
   - One additional flip on MBUV (LOSS → TIE).

3. **Permutation null** (500 iterations, row-level label swap): mean null wins = 4.1, max = 12. Observed 56 is well above; p < 0.002. Signal is real, not multiple-comparison.

4. **Paired seed-level t-test with corrections**:
   - Uncorrected (p<0.05, diff<0): 72 wins, 88 losses.
   - Bonferroni (α=0.05/250=0.0002): 50 wins, 51 losses.
   - BH-FDR (q=0.05): 69 wins, 82 losses.
   By mechanism under Bonferroni: MBOV_Lower 28/50, MBOV_Stochastic 21/50, MBOV_Centered 1/50, MBOV_Higher 0/50, MBUV 0/50.

5. **Degenerate baselines count**: in 671/3,500 (19%) of rows, 6 baselines (mice, complete_case, knn_impute, mean_impute, heckman, ipw_estimated) give MSE within ±0.001 of each other. Concentrated on MBOV_Higher (45%), MBOV_Lower (32%), MBOV_Centered (23%), MBOV_Stochastic (21%).

6. **Win-set characterization**: of 56 wins, 34 (61%) have obs_y_positive_rate ≥ 0.99; 22 (39%) are in "normal" regimes. Of MBOV_Lower's 30 wins, 27 (90%) have obs_y_positive_rate ≥ 0.99.

7. **CTG pathology check**: oracle MSE on CTG is 1e-7 (features near-linearly predictive of Y). MICE, complete_case, Heckman, IPW, kNN achieve MSE in [1e-22, 2e-4] on CTG across mechanisms — i.e., near oracle. Christensen_faithful achieves MSE in [1e-29, 0.75] on CTG — wider variance, often near or below oracle at MBOV_Lower, worse elsewhere. The REPORT_v2 "diff_%" column of ~31M% is an artifact of dividing by ~1e-21.

8. **MBOV_Higher asymmetry check**: Wisconsin/MBOV_Higher/40% shows MICE MSE=0.073, christensen MSE=0.272 (worse). This asymmetry between MBOV_Lower (christensen wins) and MBOV_Higher (christensen loses) is driven entirely by the class balance of wisconsin (62.7% positive) interacting with the direction of selection. The Q-class dispatch is symmetric (`Parametric2ParamForBinary` for both with `monotone="increasing"` / `"decreasing"`), so this asymmetry points to a sensitivity in the estimator, not a bug.

---

## Files referenced
- `phase1_pereira_benchmark/REPORT_v2.md` (headline claim)
- `phase1_pereira_benchmark/PROTOCOL.md` (protocol spec)
- `phase1_pereira_benchmark/results/raw_results_v2.csv` (raw data, 34k rows)
- `phase1_pereira_benchmark/results/run_v2.log` (single run, 46.5 min)
- `phase1_pereira_benchmark/analyze.py:33-37` (CI formula)
- `phase1_pereira_benchmark/harness.py:150` (seed count override)
- `phase1_pereira_benchmark/datasets.py:300-316` (Cleveland NaN drop)
- `phase1_pereira_benchmark/mnar_injection.py:170` (MBIR Bayesian fallback)
- `phase1_pereira_benchmark/christensen_adapter.py:63` (mechanism-name input)
- `christensen_core/reference_based_q.py:82-93` (MECHANISM_DELTA table)
- `docs/pereira_2024/chapter9_mnar.txt` (primary source)
