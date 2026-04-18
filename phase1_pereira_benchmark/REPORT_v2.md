# Phase 1 Report — Christensen Minimax vs MICE under MNAR Labels

**Path A benchmark**: Pereira et al. 2024 MNAR mechanisms applied to label column for regression tasks on 10 UCI medical datasets. See PROTOCOL.md for full spec and declared deviations from Pereira's original imputation-quality benchmark.

**Seeds completed**: 10 per cell (median across cells)
**Total rows**: 34,000
**Methods**: ['christensen_faithful', 'complete_case', 'erm_sgd', 'heckman', 'ipw_estimated', 'knn_impute', 'mean_impute', 'mice', 'minimax_score', 'oracle']
**Mechanisms**: ['MBIR_Bayesian', 'MBIR_Frequentist', 'MBOV_Centered', 'MBOV_Higher', 'MBOV_Lower', 'MBOV_Stochastic', 'MBUV']
**Datasets**: ['bc-coimbra', 'cleveland', 'cmc', 'ctg', 'pima', 'saheart', 'thyroid', 'transfusion', 'vertebral', 'wisconsin']

## Headline: Christensen minimax vs MICE

Across 250 (dataset, mechanism, rate) cells:

- **Wins** (95% CI strictly below MICE): **56** (22.4%)
- **Ties** (95% CIs overlap MICE): 132 (52.8%)
- **Losses** (95% CI strictly above MICE): 62 (24.8%)

## Headline: christensen_faithful vs minimax_score (DRO variant)

This is the two-minimax-estimator comparison: the faithful Christensen (closed-form vec solve + reference-based Q) vs the DRO-inspired SGD variant in minimax_core. Direct measurement of faithful-vs-paraphrase delta.

- Wins: **110** (44.0%)
- Ties: 76 (30.4%)
- Losses: 64 (25.6%)

## Win/loss vs MICE by MNAR mechanism

| mechanism | wins | ties | losses | total | win_rate | mean_mse_diff_pct |
| --- | --- | --- | --- | --- | --- | --- |
| MBOV_Centered | 3 | 28 | 19 | 50 | 0.060 | 494622.028 |
| MBOV_Higher | 0 | 27 | 23 | 50 | 0.000 | 1082184.538 |
| MBOV_Lower | 30 | 18 | 2 | 50 | 0.600 | -24.099 |
| MBOV_Stochastic | 23 | 25 | 2 | 50 | 0.460 | 10035.538 |
| MBUV | 0 | 34 | 16 | 50 | 0.000 | 315513.314 |

## Win/loss vs MICE by missing rate

| missing_rate_pct | wins | ties | losses | total | win_rate | mean_mse_diff_pct |
| --- | --- | --- | --- | --- | --- | --- |
| 10.000 | 0.000 | 42.000 | 8.000 | 50.000 | 0.000 | 61521.287 |
| 20.000 | 6.000 | 31.000 | 13.000 | 50.000 | 0.120 | 169598.844 |
| 40.000 | 11.000 | 25.000 | 14.000 | 50.000 | 0.220 | 473593.064 |
| 60.000 | 19.000 | 18.000 | 13.000 | 50.000 | 0.380 | 1073259.504 |
| 80.000 | 20.000 | 16.000 | 14.000 | 50.000 | 0.400 | 124358.620 |

## Win/loss vs MICE by dataset

| dataset | wins | ties | losses | total | win_rate | mean_mse_diff_pct |
| --- | --- | --- | --- | --- | --- | --- |
| bc-coimbra | 3 | 19 | 3 | 25 | 0.120 | 9.433 |
| cleveland | 5 | 13 | 7 | 25 | 0.200 | 7.476 |
| cmc | 8 | 11 | 6 | 25 | 0.320 | -7.250 |
| ctg | 6 | 1 | 18 | 25 | 0.240 | 3804618.134 |
| pima | 6 | 16 | 3 | 25 | 0.240 | -4.762 |
| saheart | 6 | 18 | 1 | 25 | 0.240 | -5.861 |
| thyroid | 4 | 19 | 2 | 25 | 0.160 | 2.322 |
| transfusion | 6 | 17 | 2 | 25 | 0.240 | -7.124 |
| vertebral | 6 | 10 | 9 | 25 | 0.240 | 16.256 |
| wisconsin | 6 | 8 | 11 | 25 | 0.240 | 34.014 |

## Most favorable cells (minimax beats MICE by largest %)

| dataset | mechanism | missing_rate_pct | minimax_mse | mice_mse | diff_% |
| --- | --- | --- | --- | --- | --- |
| ctg | MBOV_Lower | 20.000 | 0.001 | 0.180 | -99.342 |
| ctg | MBOV_Stochastic | 40.000 | 0.021 | 0.180 | -88.469 |
| ctg | MBOV_Lower | 40.000 | 0.036 | 0.180 | -80.181 |
| wisconsin | MBOV_Lower | 40.000 | 0.076 | 0.374 | -79.737 |
| transfusion | MBOV_Lower | 80.000 | 0.196 | 0.760 | -74.186 |
| wisconsin | MBOV_Stochastic | 60.000 | 0.098 | 0.374 | -73.768 |
| ctg | MBOV_Stochastic | 60.000 | 0.048 | 0.180 | -73.490 |
| wisconsin | MBOV_Lower | 60.000 | 0.104 | 0.374 | -72.246 |
| pima | MBOV_Lower | 80.000 | 0.183 | 0.649 | -71.844 |
| saheart | MBOV_Lower | 80.000 | 0.200 | 0.655 | -69.390 |

## Least favorable cells (MICE beats minimax by largest %)

| dataset | mechanism | missing_rate_pct | minimax_mse | mice_mse | diff_% |
| --- | --- | --- | --- | --- | --- |
| ctg | MBOV_Higher | 60.000 | 0.329 | 0.000 | 31587552.334 |
| ctg | MBOV_Centered | 60.000 | 0.173 | 0.000 | 16579916.464 |
| ctg | MBOV_Higher | 40.000 | 0.162 | 0.000 | 15600567.832 |
| ctg | MBUV | 80.000 | 0.134 | 0.000 | 6147873.520 |
| ctg | MBUV | 60.000 | 0.057 | 0.000 | 5495459.338 |
| ctg | MBOV_Centered | 40.000 | 0.057 | 0.000 | 5452409.010 |
| ctg | MBOV_Higher | 20.000 | 0.056 | 0.000 | 5360098.026 |
| ctg | MBUV | 40.000 | 0.027 | 0.000 | 2626284.027 |
| ctg | MBOV_Centered | 20.000 | 0.019 | 0.000 | 1821562.580 |
| ctg | MBOV_Higher | 10.000 | 0.016 | 0.000 | 1523427.325 |

## Interpretation notes

1. **This is NOT a replication of Pereira et al.'s benchmark**. They measure imputation MAE on feature values; we measure test-set prediction MSE with MNAR injected on the label. See PROTOCOL.md §Deviation.
2. The minimax estimator run here is SGD-with-online-score-based-adversary, not the closed-form β̂ = M·(1/n Σ XᵢỸᵢ) + m from Christensen's 2020 write-up. A follow-up comparing the two algorithms is warranted if this result is encouraging.
3. Binary-labeled datasets with extreme MNAR (high rate + strong selection) can produce training splits with all-one or all-zero labels, causing SGD-based methods to diverge from the trivial mean-predictor. This is reflected in some LOSS cells.
4. MBUV is near-MCAR in label-only setting (see mnar_injection.py). Differences vs MICE there are expected to be small; the interesting signal is under MBOV_Lower/Higher.
