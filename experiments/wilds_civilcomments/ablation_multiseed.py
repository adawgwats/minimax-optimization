from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Mapping, Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minimax_hf import MinimaxTrainer

from experiments.wilds_civilcomments.common import (
    IDENTITY_FIELDS,
    NON_IDENTITY_GROUP,
    config_to_dict,
    load_experiment_config,
    metadata_row_to_dict,
)
from experiments.wilds_civilcomments.train import (
    _build_minimax_config,
    _build_training_arguments,
    _require_transformers,
    evaluate_split,
)
from experiments.wilds_civilcomments.dataset import load_civilcomments_splits


DEFAULT_SEEDS = (17, 23, 29, 31, 37)
DEFAULT_VARIANTS = ("dfr_erm", "ck_only", "dfr_ck")
SUMMARY_METRICS = (
    ("overall_accuracy", ("overall_accuracy",)),
    ("worst_group_accuracy", ("worst_group_accuracy",)),
    ("overall_auroc", ("overall_auroc",)),
    ("worst_group_auroc", ("worst_group_auroc",)),
    ("wilds_acc_avg", ("wilds_eval", "acc_avg")),
    ("wilds_acc_wg", ("wilds_eval", "acc_wg")),
)


class ReindexedDataset:
    def __init__(self, base_dataset: Any, indices: Sequence[int]) -> None:
        self.base_dataset = base_dataset
        self.indices = [int(index) for index in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Any:
        return self.base_dataset[self.indices[index]]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CK/DFR ablations on WILDS CivilComments with multiseed summaries."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Base YAML/JSON config path (usually robust_auto_v1 config).",
    )
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=list(DEFAULT_VARIANTS),
        choices=list(DEFAULT_VARIANTS),
        help="Ablation variants to run.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Seeds for WILDS-style multiseed reporting.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help=(
            "Optional output root. Defaults to "
            "<base_output_dir>_ablation_multiseed/<variant>/seed_<seed>."
        ),
    )
    parser.add_argument(
        "--dfr-target-per-group",
        type=int,
        default=256,
        help="Target examples per (identity,label) bucket for DFR head training.",
    )
    parser.add_argument(
        "--dfr-head-learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate for DFR head-only retraining.",
    )
    parser.add_argument(
        "--dfr-head-epochs",
        type=int,
        default=3,
        help="Epochs for DFR head-only retraining.",
    )
    return parser.parse_args(argv)


def run_ablation_multiseed(
    *,
    config_path: str | Path,
    variants: Sequence[str],
    seeds: Sequence[int],
    output_root: str | Path | None,
    dfr_target_per_group: int,
    dfr_head_learning_rate: float,
    dfr_head_epochs: int,
) -> dict[str, Any]:
    base_config = load_experiment_config(config_path)
    if not seeds:
        raise ValueError("at least one seed is required.")

    root_dir = (
        Path(output_root)
        if output_root is not None
        else Path(f"{base_config.output_dir}_ablation_multiseed")
    )
    root_dir.mkdir(parents=True, exist_ok=True)

    combined_summary: dict[str, Any] = {
        "config_path": str(config_path),
        "base_config": config_to_dict(base_config),
        "variants": {},
    }

    for variant in variants:
        variant_root = root_dir / variant
        variant_root.mkdir(parents=True, exist_ok=True)
        seed_artifacts: list[dict[str, Any]] = []
        seed_runs: list[dict[str, Any]] = []

        for seed in seeds:
            run_output_dir = variant_root / f"seed_{int(seed)}"
            run_output_dir.mkdir(parents=True, exist_ok=True)
            run_config = _build_variant_config(
                base_config,
                variant=variant,
                seed=int(seed),
                output_dir=str(run_output_dir),
            )
            artifact = _train_and_evaluate_variant(
                config=run_config,
                variant=variant,
                dfr_target_per_group=dfr_target_per_group,
                dfr_head_learning_rate=dfr_head_learning_rate,
                dfr_head_epochs=dfr_head_epochs,
            )
            seed_artifacts.append(artifact)
            metrics_path = run_output_dir / "metrics.json"
            seed_runs.append(
                {
                    "seed": int(seed),
                    "output_dir": str(run_output_dir),
                    "metrics_path": str(metrics_path),
                }
            )
            _release_accelerator_memory()

        summary = _aggregate_multiseed_metrics(
            artifacts=seed_artifacts,
            base_config=config_to_dict(base_config),
            seeds=[int(seed) for seed in seeds],
            output_root=str(variant_root),
            config_path=str(config_path),
            seed_runs=seed_runs,
            variant=variant,
        )
        summary_path = variant_root / "multiseed_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        combined_summary["variants"][variant] = summary
        print(_render_multiseed_summary(summary))
        print("")

    combined_path = root_dir / "ablation_summary.json"
    combined_path.write_text(json.dumps(combined_summary, indent=2, sort_keys=True), encoding="utf-8")
    return combined_summary


def _build_variant_config(
    base_config: Any,
    *,
    variant: str,
    seed: int,
    output_dir: str,
) -> Any:
    if variant not in DEFAULT_VARIANTS:
        raise ValueError(f"unsupported variant: {variant}")
    method = "robust_auto_v1" if variant in {"ck_only", "dfr_ck"} else "erm"
    return replace(
        base_config,
        method=method,
        seed=seed,
        output_dir=output_dir,
    )


def _train_and_evaluate_variant(
    *,
    config: Any,
    variant: str,
    dfr_target_per_group: int,
    dfr_head_learning_rate: float,
    dfr_head_epochs: int,
) -> dict[str, Any]:
    deps = _require_transformers()
    Trainer = deps["Trainer"]
    TrainingArguments = deps["TrainingArguments"]
    AutoModelForSequenceClassification = deps["AutoModelForSequenceClassification"]
    set_seed = deps["set_seed"]

    set_seed(config.seed)
    random.seed(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wilds_dataset, splits, collator = load_civilcomments_splits(config)
    minimax_config: Any = None
    effective_assumed_observation_rate: float | None = None
    if config.method == "robust_auto_v1":
        minimax_config, effective_assumed_observation_rate = _build_minimax_config(
            config,
            train_split=splits["train"],
        )

    model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=2)
    stage1_args = _build_training_arguments(
        TrainingArguments,
        output_dir=str(output_dir / "stage1"),
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        num_train_epochs=config.num_train_epochs,
        seed=config.seed,
        remove_unused_columns=False,
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=50,
        report_to=[],
    )
    if config.method == "erm":
        trainer: Any = Trainer(
            model=model,
            args=stage1_args,
            train_dataset=splits["train"].dataset,
            eval_dataset=splits["val"].dataset,
            data_collator=collator,
        )
    else:
        trainer = MinimaxTrainer(
            model=model,
            args=stage1_args,
            train_dataset=splits["train"].dataset,
            eval_dataset=splits["val"].dataset,
            data_collator=collator,
            minimax_config=minimax_config,
        )
    stage1_result = trainer.train()

    dfr_info: dict[str, Any] | None = None
    if variant in {"dfr_erm", "dfr_ck"}:
        dfr_indices = _build_dfr_balanced_indices(
            metadata_rows=splits["train"].metadata_rows,
            metadata_fields=splits["train"].metadata_fields,
            labels=splits["train"].labels,
            seed=config.seed,
            target_per_group=dfr_target_per_group,
        )
        dfr_dataset = ReindexedDataset(splits["train"].dataset, dfr_indices)
        _freeze_model_for_linear_head(trainer.model)
        stage2_args = _build_training_arguments(
            TrainingArguments,
            output_dir=str(output_dir / "stage2_dfr"),
            learning_rate=dfr_head_learning_rate,
            weight_decay=config.weight_decay,
            per_device_train_batch_size=config.train_batch_size,
            per_device_eval_batch_size=config.eval_batch_size,
            num_train_epochs=dfr_head_epochs,
            seed=config.seed,
            remove_unused_columns=False,
            save_strategy="no",
            logging_strategy="steps",
            logging_steps=50,
            report_to=[],
        )
        stage2_trainer = Trainer(
            model=trainer.model,
            args=stage2_args,
            train_dataset=dfr_dataset,
            eval_dataset=splits["val"].dataset,
            data_collator=collator,
        )
        stage2_result = stage2_trainer.train()
        trainer = stage2_trainer
        dfr_info = {
            "target_per_group": dfr_target_per_group,
            "balanced_examples": len(dfr_indices),
            "head_learning_rate": dfr_head_learning_rate,
            "head_epochs": dfr_head_epochs,
            "stage2_runtime": float(stage2_result.metrics.get("train_runtime", 0.0)),
        }

    evaluated_splits = {
        split_name: evaluate_split(
            trainer=trainer,
            split=splits[split_name],
            wilds_dataset=wilds_dataset,
        )
        for split_name in ("val", "test")
    }
    metrics_payload = {
        "variant": variant,
        "config": config_to_dict(config),
        "train": {
            "stage1_runtime": float(stage1_result.metrics.get("train_runtime", 0.0)),
            "observed_examples": sum(1 for observed in splits["train"].observed_mask if observed),
            "total_examples": len(splits["train"].observed_mask),
            "effective_assumed_observation_rate": effective_assumed_observation_rate,
            "dfr": dfr_info,
        },
        "val": evaluated_splits["val"][0],
        "test": evaluated_splits["test"][0],
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True), encoding="utf-8")
    return metrics_payload


def _build_dfr_balanced_indices(
    *,
    metadata_rows: Sequence[Sequence[int]],
    metadata_fields: Sequence[str],
    labels: Sequence[int],
    seed: int,
    target_per_group: int,
) -> list[int]:
    if target_per_group <= 0:
        raise ValueError("target_per_group must be positive.")

    buckets: dict[str, list[int]] = {}
    for index, (metadata_row, label) in enumerate(zip(metadata_rows, labels)):
        metadata = metadata_row_to_dict(metadata_row, metadata_fields)
        active_groups = [
            identity
            for identity in IDENTITY_FIELDS
            if metadata.get(identity, 0) == 1
        ]
        if not active_groups:
            active_groups = [NON_IDENTITY_GROUP]
        for group_name in active_groups:
            key = f"{group_name}:y:{int(label)}"
            buckets.setdefault(key, []).append(index)

    if not buckets:
        raise ValueError("no DFR buckets were created from the training metadata.")

    rng = random.Random(seed + 7919)
    balanced_indices: list[int] = []
    for indices in buckets.values():
        if not indices:
            continue
        if len(indices) >= target_per_group:
            selected = rng.sample(indices, target_per_group)
        else:
            selected = list(indices)
            deficit = target_per_group - len(indices)
            selected.extend(rng.choice(indices) for _ in range(deficit))
        balanced_indices.extend(selected)
    rng.shuffle(balanced_indices)
    return balanced_indices


def _freeze_model_for_linear_head(model: Any) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False

    trainable_heads = 0
    for head_name in ("pre_classifier", "classifier", "score"):
        if not hasattr(model, head_name):
            continue
        module = getattr(model, head_name)
        for parameter in module.parameters():
            parameter.requires_grad = True
            trainable_heads += 1
    if trainable_heads <= 0:
        raise ValueError("no trainable classification head found for DFR stage.")


def _aggregate_multiseed_metrics(
    *,
    artifacts: Sequence[Mapping[str, Any]],
    base_config: Mapping[str, Any],
    seeds: Sequence[int],
    output_root: str,
    config_path: str,
    seed_runs: Sequence[Mapping[str, Any]],
    variant: str,
) -> dict[str, Any]:
    if len(artifacts) != len(seeds):
        raise ValueError("artifacts and seeds must have the same length.")
    return {
        "variant": variant,
        "config_path": config_path,
        "output_root": output_root,
        "base_config": dict(base_config),
        "seeds": [int(seed) for seed in seeds],
        "seed_runs": [dict(seed_run) for seed_run in seed_runs],
        "num_seeds": len(seeds),
        "val": _summarize_split(artifacts, "val"),
        "test": _summarize_split(artifacts, "test"),
    }


def _summarize_split(
    artifacts: Sequence[Mapping[str, Any]],
    split_name: str,
) -> dict[str, dict[str, float | int | None]]:
    split_summary: dict[str, dict[str, float | int | None]] = {}
    for metric_name, path in SUMMARY_METRICS:
        values = [
            float(value)
            for artifact in artifacts
            if (value := _get_nested_value(artifact.get(split_name, {}), path)) is not None
        ]
        split_summary[metric_name] = _summarize_numeric_values(values)
    return split_summary


def _summarize_numeric_values(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _get_nested_value(payload: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
        if value is None:
            return None
    return value


def _render_multiseed_summary(summary: Mapping[str, Any]) -> str:
    lines = [
        f"variant: {summary.get('variant', 'unknown')}",
        f"seeds: {', '.join(str(seed) for seed in summary.get('seeds', []))}",
        f"output_root: {summary.get('output_root', 'n/a')}",
        "",
        "split  metric                mean     std",
        "-----  ------------------  -------  ------",
    ]
    for split in ("val", "test"):
        split_summary = summary.get(split, {})
        for metric_name in (
            "overall_accuracy",
            "worst_group_accuracy",
            "overall_auroc",
            "worst_group_auroc",
            "wilds_acc_avg",
            "wilds_acc_wg",
        ):
            metric_summary = split_summary.get(metric_name, {})
            lines.append(
                f"{split:<5}"
                f"  {metric_name:<18}"
                f"  {_format_number(metric_summary.get('mean')):>7}"
                f"  {_format_number(metric_summary.get('std')):>6}"
            )
    return "\n".join(lines)


def _format_number(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _release_accelerator_memory() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_ablation_multiseed(
        config_path=args.config,
        variants=args.variants,
        seeds=args.seeds,
        output_root=args.output_root,
        dfr_target_per_group=args.dfr_target_per_group,
        dfr_head_learning_rate=args.dfr_head_learning_rate,
        dfr_head_epochs=args.dfr_head_epochs,
    )


if __name__ == "__main__":
    main()
