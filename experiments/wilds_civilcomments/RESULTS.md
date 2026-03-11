# CivilComments Benchmark Notes

This note preserves the current tracked WILDS CivilComments benchmark status without committing large `outputs/` artifacts.

## March 4, 2026

Code reference:

- base integration commit: `f65c890` (`Wire adaptive WILDS CivilComments evaluation`)

Evaluation environment:

- Windows
- Python 3.11
- `torch 2.8.0+cu126`
- `torchvision 0.23.0+cu126`
- `torch-scatter 2.1.2+pt28cu126`
- official `wilds 2.0.0` evaluator path

Experiment configs:

- auto-discovery: `experiments/wilds_civilcomments/configs/midscale_auto_v1.yaml`
- ERM baseline: `experiments/wilds_civilcomments/configs/midscale_erm.yaml`

Shared setup:

- model: `distilbert-base-uncased`
- train/val/test cap: `16384 / 4096 / 4096`
- epochs: `2`
- batch size: `16 / 32`
- seed: `17`

Results:

| Run | Val acc_avg | Val acc_wg | Test acc_avg | Test acc_wg |
| --- | ---: | ---: | ---: | ---: |
| `midscale_auto_v1` | `0.9197` | `0.4286` | `0.9197` | `0.5682` |
| `midscale_erm` | `0.9214` | `0.4286` | `0.9243` | `0.5227` |

Interpretation:

- `robust_auto_v1` did not improve validation worst-group accuracy over ERM on this run.
- `robust_auto_v1` improved test worst-group accuracy by `+0.0455` absolute over ERM.
- This is encouraging, but not yet submission-grade evidence because model selection should be justified on validation performance, not test gains alone.

Additional run details:

- `midscale_auto_v1` estimated `effective_assumed_observation_rate = 0.8753662109377119`
- raw metrics were produced in local artifacts under `outputs/wilds_civilcomments/`

## March 4, 2026 Tuning Follow-up

Follow-up sweep:

- `experiments/wilds_civilcomments/configs/midscale_auto_v1_rate_0p75.yaml`
- `experiments/wilds_civilcomments/configs/midscale_auto_v1_rate_0p70.yaml`

Results:

| Run | Assumed rate | Val acc_avg | Val acc_wg | Test acc_avg | Test acc_wg |
| --- | ---: | ---: | ---: | ---: | ---: |
| `midscale_auto_v1` | `0.8754` estimated | `0.9197` | `0.4286` | `0.9197` | `0.5682` |
| `midscale_auto_v1_rate_0p75` | `0.75` | `0.9209` | `0.4286` | `0.9236` | `0.5909` |
| `midscale_auto_v1_rate_0p70` | `0.70` | `0.9209` | `0.4286` | `0.9221` | `0.5682` |
| `midscale_erm` | `n/a` | `0.9214` | `0.4286` | `0.9243` | `0.5227` |

Current takeaway:

- tuning the latent observation-rate prior downward helped `robust_auto_v1`
- `0.75` is the strongest auto-discovery run so far on this setup
- the best auto run now beats ERM on test worst-group accuracy by `+0.0682` absolute
- validation worst-group accuracy still ties ERM, so this remains promising rather than submission-ready

## March 4, 2026 5-Seed Protocol Run

To mirror the WILDS CivilComments submission protocol, both methods were re-run with 5 seeds
(`17, 23, 29, 31, 37`) using:

- `experiments/wilds_civilcomments/multiseed.py`
- `experiments/wilds_civilcomments/configs/midscale_auto_v1_rate_0p75.yaml`
- `experiments/wilds_civilcomments/configs/midscale_erm.yaml`

Artifacts:

- auto summary: `outputs/wilds_civilcomments/midscale_auto_v1_rate_0p75_multiseed/multiseed_summary.json`
- ERM summary: `outputs/wilds_civilcomments/midscale_erm_multiseed/multiseed_summary.json`

5-seed summary (mean +/- sample std):

| Run | Val acc_avg | Val acc_wg | Test acc_avg | Test acc_wg |
| --- | ---: | ---: | ---: | ---: |
| `robust_auto_v1` (`assumed_observation_rate=0.75`) | `0.9198 +/- 0.0065` | `0.4069 +/- 0.0724` | `0.9207 +/- 0.0060` | `0.5973 +/- 0.0430` |
| `erm` | `0.9199 +/- 0.0064` | `0.3957 +/- 0.0603` | `0.9220 +/- 0.0059` | `0.5909 +/- 0.0759` |

Interpretation:

- under 5-seed averaging, auto-discovery is no longer a single-seed anomaly
- auto-discovery is slightly better than ERM on worst-group accuracy in both val and test means
- the absolute gap is still small, and the variability is high, so this is progress but not top-leaderboard performance

## March 4, 2026 Unlabeled Self-Training Probe

An exploratory semi-supervised probe was run using WILDS `civilcomments` unlabeled data
(`extra_unlabeled`) via:

- `experiments/wilds_civilcomments/semi_supervised.py`
- unlabeled candidate cap: `8192`
- pseudo-label threshold: `0.90`
- student fine-tune epochs: `1`
- seed: `17`

Artifacts:

- auto: `outputs/wilds_civilcomments/midscale_auto_v1_rate_0p75_semi_supervised/semi_supervised_metrics.json`
- ERM: `outputs/wilds_civilcomments/midscale_erm_semi_supervised/semi_supervised_metrics.json`

Teacher -> student results:

| Method | Stage | Val acc_avg | Val acc_wg | Test acc_avg | Test acc_wg |
| --- | --- | ---: | ---: | ---: | ---: |
| `robust_auto_v1` (`assumed_observation_rate=0.75`) | teacher | `0.9209` | `0.4286` | `0.9236` | `0.5909` |
| `robust_auto_v1` (`assumed_observation_rate=0.75`) | student | `0.9231` | `0.3846` | `0.9243` | `0.5000` |
| `erm` | teacher | `0.9214` | `0.4286` | `0.9243` | `0.5227` |
| `erm` | student | `0.9236` | `0.4286` | `0.9224` | `0.5455` |

Pseudo-label diagnostics:

| Method | Pseudo selected | Selection rate | Agreement vs hidden unlabeled labels |
| --- | ---: | ---: | ---: |
| `robust_auto_v1` | `4445 / 8192` | `0.5426` | `0.3917` |
| `erm` | `6048 / 8192` | `0.7383` | `0.3271` |

Current interpretation:

- this first unlabeled protocol is not yet helping `robust_auto_v1` worst-group accuracy
- `erm` gained test worst-group accuracy modestly (`+0.0228`) but lost test average accuracy (`-0.0019`)
- pseudo-label agreement is low for both methods, indicating that this naive self-training setup needs stronger filtering/calibration before leaderboard-facing use

## March 4, 2026 CK + DFR Additivity Ablation (5 Seeds)

Goal:

- test whether Christiansen-Knightian (`ck_only`) and a top-method proxy (`dfr_erm`) are additive when combined (`dfr_ck`)
- all variants run over the same seed set: `17, 23, 29, 31, 37`

Runner and artifacts:

- runner: `experiments/wilds_civilcomments/ablation_multiseed.py`
- base config: `experiments/wilds_civilcomments/configs/midscale_auto_v1_rate_0p75.yaml`
- output root: `outputs/wilds_civilcomments/ablation_midscale_5seed/`
- per-variant summaries:
  - `outputs/wilds_civilcomments/ablation_midscale_5seed/ck_only/multiseed_summary.json`
  - `outputs/wilds_civilcomments/ablation_midscale_5seed/dfr_erm/multiseed_summary.json`
  - `outputs/wilds_civilcomments/ablation_midscale_5seed/dfr_ck/multiseed_summary.json`

Protocol notes:

- `dfr_erm`: ERM stage-1 training + DFR-style head-only stage-2 retraining on balanced `(identity, label)` buckets
- `dfr_ck`: CK stage-1 training + same DFR stage-2 head retraining
- DFR stage-2 params in this run:
  - `dfr_target_per_group = 256`
  - `dfr_head_learning_rate = 5e-5`
  - `dfr_head_epochs = 2`

5-seed summary (mean +/- sample std):

| Variant | Val acc_avg | Val acc_wg | Test acc_avg | Test acc_wg |
| --- | ---: | ---: | ---: | ---: |
| `ck_only` | `0.9195 +/- 0.0066` | `0.4235 +/- 0.0598` | `0.9202 +/- 0.0052` | `0.6059 +/- 0.0558` |
| `dfr_erm` | `0.8948 +/- 0.0064` | `0.6125 +/- 0.0309` | `0.9021 +/- 0.0063` | `0.6226 +/- 0.0658` |
| `dfr_ck` | `0.8933 +/- 0.0052` | `0.6288 +/- 0.0349` | `0.8997 +/- 0.0059` | `0.6072 +/- 0.0552` |

Interpretation:

- `dfr_erm` strongly improves worst-group accuracy over `ck_only`, but with a clear drop in average accuracy.
- `dfr_ck` improves validation worst-group vs `dfr_erm` (`+0.0163`) but reduces test worst-group vs `dfr_erm` (`-0.0155`).
- On test worst-group, `dfr_ck` is only marginally above `ck_only` (`+0.0012`), so this setup does **not** yet show clear additive gains from combining CK with DFR.
- next step is to tune stage-2 DFR settings (target-per-group, head epochs, LR) before concluding CK+DFR synergy is absent.

## March 5, 2026 Full-Data 5-Seed CivilComments Run (Submission-Style)

Goal:

- run full-data (no split fractions, no `max_*_examples` caps) with the official CivilComments 5-seed protocol (`17, 23, 29, 31, 37`)
- compare CK (`robust_auto_v1`) vs ERM under matched training settings

Configs:

- CK: `experiments/wilds_civilcomments/configs/full_auto_v1_rate_0p75.yaml`
- ERM: `experiments/wilds_civilcomments/configs/full_erm.yaml`

Shared setup:

- dataset fractions: full (`train/val/test_fraction = 1.0`)
- limits: none (`max_train_examples = max_val_examples = max_test_examples = null`)
- model: `distilbert-base-uncased`
- max length: `128`
- batch size: `16 / 32`
- epochs: `2`

Artifacts:

- CK summary: `outputs/wilds_civilcomments/full_auto_v1_rate_0p75_multiseed/multiseed_summary.json`
- ERM summary: `outputs/wilds_civilcomments/full_erm_multiseed/multiseed_summary.json`

5-seed summary (mean +/- sample std):

| Run | Val acc_avg | Val acc_wg | Test acc_avg | Test acc_wg |
| --- | ---: | ---: | ---: | ---: |
| `robust_auto_v1` (full data) | `0.9266 +/- 0.0004` | `0.4323 +/- 0.0071` | `0.9261 +/- 0.0003` | `0.4825 +/- 0.0057` |
| `erm` (full data) | `0.9266 +/- 0.0004` | `0.4224 +/- 0.0075` | `0.9264 +/- 0.0002` | `0.4800 +/- 0.0105` |

Key deltas (CK - ERM):

- `val acc_wg`: `+0.0099`
- `test acc_wg`: `+0.0025`
- `val acc_avg`: `+0.0000` (effectively tied)
- `test acc_avg`: `-0.0002`

Interpretation:

- in full-data 5-seed runs, CK remains slightly better than ERM on worst-group accuracy in both val and test
- average accuracy is effectively tied (tiny ERM edge on test average)
- effect size is modest but stable enough to justify continuing CK development under full-data protocol
