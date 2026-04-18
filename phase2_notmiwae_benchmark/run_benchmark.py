"""Entry point: run the Phase 2 benchmark (6 datasets × 5 seeds × 4 methods)."""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

from .datasets import REGISTRY as DATASET_REGISTRY
from .harness import run_benchmark, aggregate, METHOD_ORDER


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5, help="Seeds per cell (not-MIWAE uses 5).")
    p.add_argument("--datasets", type=str, default="all")
    p.add_argument("--methods", type=str, default="all")
    p.add_argument("--output", type=Path,
                   default=Path(__file__).parent / "results" / "raw_results.csv")
    p.add_argument("--agg-output", type=Path,
                   default=Path(__file__).parent / "results" / "aggregated.csv")
    return p.parse_args(argv)


def main(argv=None):
    warnings.filterwarnings("ignore")
    args = parse_args(argv)
    datasets = tuple(DATASET_REGISTRY.keys()) if args.datasets == "all" else tuple(args.datasets.split(","))
    methods = METHOD_ORDER if args.methods == "all" else tuple(args.methods.split(","))
    seeds = tuple(range(args.seeds))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"[run_benchmark] datasets={len(datasets)} seeds={len(seeds)} methods={len(methods)}")
    print(f"[run_benchmark] output: {args.output}")

    t0 = time.perf_counter()
    df = run_benchmark(datasets=datasets, seeds=seeds, methods=methods, out_csv=args.output)
    elapsed = time.perf_counter() - t0
    print(f"[run_benchmark] done in {elapsed/60:.1f} minutes; {len(df)} rows")

    agg = aggregate(df)
    agg.to_csv(args.agg_output, index=False)
    print(f"[run_benchmark] aggregation saved to {args.agg_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
