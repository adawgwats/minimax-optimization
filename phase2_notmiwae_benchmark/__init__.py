"""Phase 2 benchmark: Ipsen et al. 2021 (not-MIWAE) MNAR imputation benchmark.

See PROTOCOL.md for the full specification. This package scaffolds dataset
loaders, the self-masking-above-mean MNAR injector, and classical imputation
baselines (Mean / MICE / missForest). The Christensen per-feature regression
adapter and the benchmark harness itself are intentionally NOT in this package
yet — they are added in a follow-up task.
"""
