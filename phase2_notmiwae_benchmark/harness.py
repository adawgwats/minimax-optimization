"""Benchmark harness: run all imputers across all 6 datasets × 5 seeds, collect
imputation RMSE matching not-MIWAE Table 1 metric.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .datasets import REGISTRY as DATASET_REGISTRY, load as load_dataset
from .mnar_injection import self_masking_above_mean, imputation_rmse
from .baselines import MeanImputer, MICEImputer, MissForestImputer
from .per_feature_adapter import ChristensenFeatureImputer


METHOD_FACTORIES = {
    "mean": lambda seed: MeanImputer(),
    "mice": lambda seed: MICEImputer(random_state=seed),
    "missforest": lambda seed: MissForestImputer(random_state=seed),
    "christensen_faithful": lambda seed: ChristensenFeatureImputer(),
}

METHOD_ORDER = ("mean", "mice", "missforest", "christensen_faithful")


@dataclass
class CellResult:
    dataset: str
    seed: int
    method: str
    rmse: float
    fit_seconds: float
    n_masked: int
    n_total: int


def run_cell(dataset_name: str, seed: int, methods: tuple[str, ...] = METHOD_ORDER) -> list[CellResult]:
    """Run all methods on one (dataset, seed) cell."""
    ds = load_dataset(dataset_name)
    inj = self_masking_above_mean(ds.X, seed=seed)

    results: list[CellResult] = []
    for method_name in methods:
        factory = METHOD_FACTORIES[method_name]
        model = factory(seed)
        start = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X_rec = model.fit_impute(inj.X_nan, inj.observed_mask)
            rmse = imputation_rmse(X_true=ds.X, X_reconstructed=X_rec, observed_mask=inj.observed_mask)
        except Exception as exc:
            rmse = float("nan")
            elapsed = time.perf_counter() - start
            results.append(CellResult(
                dataset=dataset_name, seed=seed, method=method_name,
                rmse=rmse, fit_seconds=elapsed,
                n_masked=int((~inj.observed_mask).sum()), n_total=ds.X.size,
            ))
            print(f"  WARN {dataset_name}/seed{seed}/{method_name}: {type(exc).__name__}: {str(exc)[:120]}")
            continue
        elapsed = time.perf_counter() - start
        results.append(CellResult(
            dataset=dataset_name, seed=seed, method=method_name,
            rmse=float(rmse), fit_seconds=elapsed,
            n_masked=int((~inj.observed_mask).sum()), n_total=ds.X.size,
        ))
    return results


def run_benchmark(
    datasets: tuple[str, ...] = tuple(DATASET_REGISTRY.keys()),
    seeds: tuple[int, ...] = tuple(range(5)),
    methods: tuple[str, ...] = METHOD_ORDER,
    out_csv: Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the full benchmark grid. Returns per-cell DataFrame."""
    rows: list[CellResult] = []
    total = len(datasets) * len(seeds)
    count = 0
    t0 = time.perf_counter()
    for ds_name in datasets:
        for seed in seeds:
            count += 1
            cell_results = run_cell(ds_name, seed, methods=methods)
            rows.extend(cell_results)
            if verbose:
                elapsed = time.perf_counter() - t0
                eta = elapsed * (total / count - 1) if count > 0 else 0
                print(f"  [{count}/{total}] {ds_name:10} seed={seed}  elapsed {elapsed:.0f}s  ETA {eta:.0f}s")
            if out_csv is not None:
                pd.DataFrame([vars(r) for r in rows]).to_csv(out_csv, index=False)
    df = pd.DataFrame([vars(r) for r in rows])
    if out_csv is not None:
        df.to_csv(out_csv, index=False)
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Mean RMSE per (dataset, method) across seeds, with std."""
    grouped = df.groupby(["dataset", "method"])["rmse"]
    n = grouped.count()
    mean = grouped.mean()
    std = grouped.std(ddof=1)
    return pd.DataFrame({"mean_rmse": mean, "std_rmse": std, "n_seeds": n}).reset_index()
