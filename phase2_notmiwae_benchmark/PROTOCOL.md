# Phase 2 Protocol: Ipsen et al. 2021 (not-MIWAE) UCI MNAR Imputation Benchmark

## Source of truth

- **Paper**: Ipsen, N.B., Mattei, P.-A., & Frellsen, J. (2021). *not-MIWAE: Deep
  Generative Modelling with Missing not at Random Data*. ICLR 2021.
  [arXiv:2006.12871](https://arxiv.org/abs/2006.12871).
- **Primary spec**: `docs/notmiwae/paper_text.txt` (extract) and
  `docs/notmiwae/paper.pdf` (original).
- **Canonical code**: `docs/notmiwae/task01.py`, `notMIWAE.py`, `MIWAE.py`,
  `utils.py`, `trainer.py` — pulled from the authors' GitHub. The self-masking
  injection mechanism and the imputation-RMSE metric are implemented verbatim
  against `task01.py` (see below).

## Deviation from Ipsen et al. 2021 — important and declared upfront

**Ipsen et al. 2021 benchmarks deep generative imputation**: their not-MIWAE
fits a joint VAE-style model with an explicit missingness model and imputes X
by posterior sampling.

**This study benchmarks structured minimax regression as an imputer**: we
adopt Ipsen et al.'s datasets, MNAR mechanism, and imputation-RMSE metric, but
the imputation engine under test is the Christensen per-feature regression
head with the `MonotoneInY` QClass (shared implementation with Phase 1). Each
masked feature j is imputed by running Christensen regression of X_j on
X_{-j} with the selective-observation assumption baked into the QClass, then
writing the prediction back into the masked entries.

**Framing for a paper**: *"We evaluate a different class of imputer —
structured minimax regression — against Ipsen et al.'s not-MIWAE on their
exact benchmark: six UCI datasets, self-masking above the feature mean
applied to the first D/2 columns, imputation RMSE over masked entries only."*
The MNAR mechanism and the metric are unchanged; the imputer is different.
Our comparison uses the not-MIWAE numbers reported in their Table 1 (we do
not re-run not-MIWAE; that is out of scope for this phase).

## Datasets

All six datasets are sourced from the UCI Machine Learning Repository via
robust mirrors (OpenML / direct HTTP / sklearn). Every dataset is
standardized to zero-mean unit-variance column-wise BEFORE any masking. The
target/class column is dropped everywhere — the task is feature-only
imputation.

| Dataset      | # Instances (expected) | # Features (expected) | Source                                     |
|--------------|------------------------|------------------------|--------------------------------------------|
| banknote     |                   1372 |                      4 | UCI direct HTTP (`data_banknote_authentication.txt`) |
| concrete     |                   1030 |                      8 | OpenML (`concrete_data`)                    |
| red          |                   1599 |                     11 | UCI direct HTTP (`winequality-red.csv`)     |
| white        |                   4898 |                     11 | UCI direct HTTP (`winequality-white.csv`)   |
| yeast        |                   1484 |                      8 | UCI direct HTTP (`yeast.data`)              |
| breast       |                    569 |                     30 | sklearn `load_breast_cancer()`              |

Shape counts above match the UCI documentation. Phase 2 loaders assert the
instance count is within a ±1% tolerance of the expected value (defensive
against line-ending quirks and silent mirror drift).

## MNAR mechanism (verbatim from task01.py)

`docs/notmiwae/task01.py` lines 21-33:

```python
def introduce_mising(X):
    N, D = X.shape
    Xnan = X.copy()
    # ---- MNAR in D/2 dimensions
    mean = np.mean(Xnan[:, :int(D / 2)], axis=0)
    ix_larger_than_mean = Xnan[:, :int(D / 2)] > mean
    Xnan[:, :int(D / 2)][ix_larger_than_mean] = np.nan

    Xz = Xnan.copy()
    Xz[np.isnan(Xnan)] = 0

    return Xnan, Xz
```

Interpretation:

- Features are split on `D // 2`. The first `D // 2` columns are MNAR-masked;
  the remaining columns are always observed.
- For each of the first `D // 2` columns, compute that column's mean on the
  standardized data. (After standardization the expected mean is 0 up to
  floating-point residual, but the code calls `np.mean` on the actual array.)
- Any entry strictly greater than its column mean is set to NaN. Entries
  equal to or less than the mean stay observed.
- Because standardized columns are roughly symmetric around zero, this yields
  approximately 50% missing in each of the first `D // 2` columns and 0%
  missing in the rest → total realized rate ≈ 25% across the whole matrix.
- This mechanism is called **self-masking above the mean** and is described
  in the paper §4 / Table 1 as the "self-masking" setting `selfmasking`.

Our `mnar_injection.py::self_masking_above_mean` reproduces this
element-for-element. The `seed` argument is accepted for API symmetry with
Phase 1 injectors but is unused — the mask is deterministic.

## Metric (verbatim from task01.py)

`docs/notmiwae/task01.py` line 115 (Mean branch; identical for MICE / MISSFOREST):

```python
RMSE = np.sqrt(np.sum((Xtrain - Xrec) ** 2 * (1 - S)) / np.sum(1 - S))
```

Where:
- `Xtrain` is the standardized ground-truth matrix (no NaN).
- `Xrec` is the fully-reconstructed imputed matrix (ground-truth values kept
  in observed entries, imputed values substituted in masked entries).
- `S ∈ {0, 1}^{N×D}` is the observed-indicator matrix (`1` where observed,
  `0` where masked). `1 - S` is therefore `1` on masked entries.
- RMSE is evaluated ONLY on masked entries; observed entries contribute zero
  to both numerator and denominator.

## Baselines

### Baselines we implement (in `baselines.py`)

All three operate on the NaN-masked matrix directly — sklearn's imputers take
the mask implicitly via NaN. All three take an optional `random_state` seed.

| Baseline      | Implementation                                                                     |
|---------------|------------------------------------------------------------------------------------|
| Mean          | `sklearn.impute.SimpleImputer(strategy='mean')`                                    |
| MICE          | `sklearn.impute.IterativeImputer(max_iter=10, random_state=seed)`                  |
| missForest    | `sklearn.impute.IterativeImputer(estimator=RandomForestRegressor(n_estimators=100))` |

`missForest` uses sklearn's iterative-imputation loop with a RandomForest
regressor per feature. This is a close analogue of the original R `missForest`
package; both Ipsen et al. and subsequent work treat them interchangeably.

### Baselines we take from Ipsen et al. Table 1 (we do NOT re-run these)

MIWAE, not-MIWAE (best missingness-model variant), PPCA, and the low-rank
joint-model — we use the reported numbers as targets to beat. These are not
re-run in Phase 2; re-running would require a TensorFlow 1.x installation
plus significant compute. Their reported numbers:

| Dataset  | not-MIWAE best | MICE | missForest | Mean |
|----------|----------------|------|------------|------|
| Banknote |           0.57 | 1.41 |       1.28 | 1.73 |
| Concrete |           1.12 | 1.70 |       1.76 | 1.85 |
| Red      |           1.07 | 1.68 |       1.64 | 1.83 |
| White    |           0.99 | 1.41 |       1.63 | 1.74 |
| Yeast    |           0.77 | 1.72 |       1.66 | 1.69 |
| Breast   |           0.72 | 1.17 |       1.57 | 1.82 |

(Reproduced from their Table 1. "not-MIWAE best" = the best of their
reported `selfmasking` / `selfmasking_known` / `linear` variants on each
dataset.)

## Experimental protocol

- **Runs**: 5 independent seeds (matches Ipsen et al.'s 5-run mean±std).
- **Standardization**: zero-mean unit-variance column-wise applied BEFORE
  injection. Mean and std are computed on the full dataset (the task is
  transductive imputation — there is no held-out split).
- **Train/test split**: none. The entire dataset is the training set for the
  imputer, and RMSE is evaluated on the MNAR-masked entries of that same
  dataset (transductive imputation, matching Ipsen et al.).
- **Seeds**: used for (a) the baseline imputers that accept `random_state`
  and (b) any stochastic element of the Christensen adapter. The mask itself
  is deterministic per the paper's code.
- **Metric**: imputation RMSE on masked entries only (formula above).
- **Reporting**: mean ± std over 5 seeds per (dataset, method) cell.

## Seeds

Five seeds: `[0, 1, 2, 3, 4]` by convention. The MNAR mask is deterministic
so the seed only varies the stochastic imputer internals (MICE, missForest
random-forest bootstrap, and any Christensen adapter randomness).

## Expected runtime

N/A at scaffold stage (no harness yet). Baseline imputers are fast enough
(<30s per dataset per seed on CPU); the Christensen adapter will be
quantified when its harness lands.

## Declared deviations

1. **Imputer**: their not-MIWAE / MIWAE are replaced with a Christensen
   per-feature regression adapter. Declared and framed above. The MNAR
   mechanism and metric are unchanged.
2. **not-MIWAE numbers are copied from Table 1**, not re-run. Any comparison
   against them inherits the exact training setup described in Ipsen et al.
   §4 and should be cited as such.
3. **Concrete sourced via OpenML** rather than the UCI `.xls` mirror (which
   requires `xlrd` and has fragile Excel parsing). OpenML's
   `concrete_data` / equivalent mirrors the same 1030-row × 8-feature
   tabular data; the target (compressive strength) is dropped.
4. **Breast Cancer sourced via `sklearn.datasets.load_breast_cancer`** rather
   than raw UCI. This is the standard sklearn mirror of the Wisconsin
   Diagnostic Breast Cancer (WDBC) dataset and matches the paper's 30
   numeric features × 569 instances.
5. **Seed count = 5** matches the paper; Phase 1 used 30 seeds for its
   regression variance budget, but Ipsen et al. report 5 runs.

## Reproducibility

- Branch: `main` (Phase 2 scaffold)
- Python: ≥3.11 (matches repo root `pyproject.toml`)
- Dependencies: `numpy`, `pandas`, `scikit-learn` — already required by the
  repo. No new dependencies are introduced in this scaffold task.
- Random state: seeded via the baseline imputers' `random_state` argument and
  any downstream Christensen adapter's explicit seed.
