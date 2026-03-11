from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Any, Mapping, Sequence

from experiments.wilds_civilcomments.common import (
    CivilCommentsExperimentConfig,
    IDENTITY_FIELDS,
    metadata_row_to_dict,
    synthetic_observation_probability,
)


@dataclass(frozen=True)
class CivilCommentsSplitMetrics:
    overall_accuracy: float
    overall_auroc: float | None
    worst_group_accuracy: float | None
    worst_group_auroc: float | None
    group_accuracy: dict[str, float]
    group_accuracy_counts: dict[str, int]
    group_auroc: dict[str, float]
    group_auroc_counts: dict[str, int]


DEFAULT_TARGET_RECALL = 0.90
DEFAULT_STRESS_LEVELS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
DEFAULT_WORST_GROUP_ACCURACY_FLOOR = 0.45


def compute_civilcomments_wilds_eval(
    labels: Sequence[int],
    predicted_labels: Sequence[int],
    metadata_rows: Sequence[Sequence[Any] | Mapping[str, Any]],
    metadata_fields: Sequence[str],
) -> dict[str, float]:
    if not (len(labels) == len(predicted_labels) == len(metadata_rows)):
        raise ValueError("labels, predicted_labels, and metadata_rows must align.")
    if not labels:
        raise ValueError("at least one evaluation example is required.")

    label_ints = [int(label) for label in labels]
    pred_ints = [int(label) for label in predicted_labels]
    metadata_dicts = [
        metadata_row_to_dict(metadata_row, metadata_fields)
        for metadata_row in metadata_rows
    ]

    results: dict[str, float] = {
        "acc_avg": _accuracy(label_ints, pred_ints),
    }
    worst_group_accuracy: float | None = None
    for identity in IDENTITY_FIELDS:
        for label_value in (0, 1):
            indices = [
                index
                for index, metadata in enumerate(metadata_dicts)
                if metadata.get(identity, 0) == 1 and label_ints[index] == label_value
            ]
            group_name = f"{identity}:1,y:{label_value}"
            if indices:
                group_accuracy = _accuracy(
                    [label_ints[index] for index in indices],
                    [pred_ints[index] for index in indices],
                )
                results[f"acc_{group_name}"] = group_accuracy
                results[f"count_{group_name}"] = float(len(indices))
                if worst_group_accuracy is None or group_accuracy < worst_group_accuracy:
                    worst_group_accuracy = group_accuracy
            else:
                results[f"acc_{group_name}"] = 0.0
                results[f"count_{group_name}"] = 0.0
    results["acc_wg"] = 0.0 if worst_group_accuracy is None else worst_group_accuracy
    return results


def logits_to_predictions_and_scores(
    logits: Sequence[Sequence[float]] | Sequence[float],
) -> tuple[list[int], list[float]]:
    if not logits:
        return [], []

    first = logits[0]
    if isinstance(first, (list, tuple)):
        positive_scores: list[float] = []
        predicted_labels: list[int] = []
        for row in logits:
            if len(row) == 1:
                positive_score = _sigmoid(float(row[0]))
                predicted_label = int(positive_score >= 0.5)
            else:
                probabilities = _softmax([float(value) for value in row])
                positive_score = probabilities[-1]
                predicted_label = int(max(range(len(probabilities)), key=probabilities.__getitem__))
            positive_scores.append(positive_score)
            predicted_labels.append(predicted_label)
        return predicted_labels, positive_scores

    positive_scores = [_sigmoid(float(value)) for value in logits]
    predicted_labels = [int(score >= 0.5) for score in positive_scores]
    return predicted_labels, positive_scores


def compute_civilcomments_metrics(
    labels: Sequence[int],
    predicted_labels: Sequence[int],
    positive_scores: Sequence[float],
    metadata_rows: Sequence[Sequence[Any] | Mapping[str, Any]],
    metadata_fields: Sequence[str],
) -> CivilCommentsSplitMetrics:
    if not (len(labels) == len(predicted_labels) == len(positive_scores) == len(metadata_rows)):
        raise ValueError("labels, predicted_labels, positive_scores, and metadata_rows must align.")
    if not labels:
        raise ValueError("at least one evaluation example is required.")

    label_ints = [int(label) for label in labels]
    pred_ints = [int(label) for label in predicted_labels]
    scores = [float(score) for score in positive_scores]
    metadata_dicts = [
        metadata_row_to_dict(metadata_row, metadata_fields)
        for metadata_row in metadata_rows
    ]

    group_accuracy: dict[str, float] = {}
    group_accuracy_counts: dict[str, int] = {}
    worst_group_accuracy: float | None = None

    for identity in IDENTITY_FIELDS:
        for label_value in (0, 1):
            indices = [
                index
                for index, metadata in enumerate(metadata_dicts)
                if metadata.get(identity, 0) == 1 and label_ints[index] == label_value
            ]
            if not indices:
                continue
            group_name = f"{identity}:1,y:{label_value}"
            accuracy = _accuracy(
                [label_ints[index] for index in indices],
                [pred_ints[index] for index in indices],
            )
            group_accuracy[group_name] = accuracy
            group_accuracy_counts[group_name] = len(indices)
            if worst_group_accuracy is None or accuracy < worst_group_accuracy:
                worst_group_accuracy = accuracy

    group_auroc: dict[str, float] = {}
    group_auroc_counts: dict[str, int] = {}
    worst_group_auroc: float | None = None
    for identity in IDENTITY_FIELDS:
        indices = [
            index
            for index, metadata in enumerate(metadata_dicts)
            if metadata.get(identity, 0) == 1
        ]
        if not indices:
            continue
        group_labels = [label_ints[index] for index in indices]
        group_scores = [scores[index] for index in indices]
        group_auc = binary_auroc(group_labels, group_scores)
        if group_auc is None:
            continue
        group_auroc[identity] = group_auc
        group_auroc_counts[identity] = len(indices)
        if worst_group_auroc is None or group_auc < worst_group_auroc:
            worst_group_auroc = group_auc

    return CivilCommentsSplitMetrics(
        overall_accuracy=_accuracy(label_ints, pred_ints),
        overall_auroc=binary_auroc(label_ints, scores),
        worst_group_accuracy=worst_group_accuracy,
        worst_group_auroc=worst_group_auroc,
        group_accuracy=group_accuracy,
        group_accuracy_counts=group_accuracy_counts,
        group_auroc=group_auroc,
        group_auroc_counts=group_auroc_counts,
    )


def metrics_to_dict(metrics: CivilCommentsSplitMetrics) -> dict[str, Any]:
    return asdict(metrics)


def format_split_metrics(split_name: str, metrics: CivilCommentsSplitMetrics) -> str:
    worst_group_accuracy = (
        "n/a" if metrics.worst_group_accuracy is None else f"{metrics.worst_group_accuracy:.4f}"
    )
    overall_auroc = "n/a" if metrics.overall_auroc is None else f"{metrics.overall_auroc:.4f}"
    worst_group_auroc = (
        "n/a" if metrics.worst_group_auroc is None else f"{metrics.worst_group_auroc:.4f}"
    )
    return (
        f"{split_name}: "
        f"accuracy={metrics.overall_accuracy:.4f}, "
        f"worst_group_accuracy={worst_group_accuracy}, "
        f"auroc={overall_auroc}, "
        f"worst_group_auroc={worst_group_auroc}"
    )


def select_threshold_for_target_recall(
    labels: Sequence[int],
    positive_scores: Sequence[float],
    *,
    target_recall: float = DEFAULT_TARGET_RECALL,
) -> float:
    if len(labels) != len(positive_scores):
        raise ValueError("labels and positive_scores must have the same length.")
    if not labels:
        raise ValueError("at least one example is required.")
    if not 0.0 < target_recall <= 1.0:
        raise ValueError("target_recall must be in (0, 1].")

    positive_class_scores = sorted(
        (
            float(score)
            for label, score in zip(labels, positive_scores)
            if int(label) == 1
        )
    )
    if not positive_class_scores:
        return 1.0

    positive_count = len(positive_class_scores)
    # Choose the highest threshold that still achieves target recall.
    index = max(0, min(positive_count - 1, math.floor((1.0 - target_recall) * positive_count)))
    return positive_class_scores[index]


def compute_operating_point_metrics(
    *,
    labels: Sequence[int],
    positive_scores: Sequence[float],
    metadata_rows: Sequence[Sequence[Any] | Mapping[str, Any]],
    metadata_fields: Sequence[str],
    threshold: float,
) -> dict[str, float | str | int | None]:
    if not (len(labels) == len(positive_scores) == len(metadata_rows)):
        raise ValueError("labels, positive_scores, and metadata_rows must align.")
    if not labels:
        raise ValueError("at least one evaluation example is required.")

    predicted_labels = [1 if float(score) >= threshold else 0 for score in positive_scores]
    metadata_dicts = [
        metadata_row_to_dict(metadata_row, metadata_fields)
        for metadata_row in metadata_rows
    ]
    confusion = _confusion_counts(labels, predicted_labels)

    group_fpr: dict[str, float] = {}
    group_fnr: dict[str, float] = {}
    for identity in IDENTITY_FIELDS:
        indices = [
            index
            for index, metadata in enumerate(metadata_dicts)
            if metadata.get(identity, 0) == 1
        ]
        if not indices:
            continue

        group_labels = [int(labels[index]) for index in indices]
        group_predictions = [int(predicted_labels[index]) for index in indices]
        group_confusion = _confusion_counts(group_labels, group_predictions)
        group_fpr_value = _safe_rate(group_confusion.fp, group_confusion.fp + group_confusion.tn)
        group_fnr_value = _safe_rate(group_confusion.fn, group_confusion.fn + group_confusion.tp)
        if group_fpr_value is not None:
            group_fpr[identity] = group_fpr_value
        if group_fnr_value is not None:
            group_fnr[identity] = group_fnr_value

    worst_group_fpr_name = max(group_fpr, key=group_fpr.get) if group_fpr else None
    worst_group_fnr_name = max(group_fnr, key=group_fnr.get) if group_fnr else None

    return {
        "threshold": float(threshold),
        "count": len(labels),
        "predicted_positive_rate": _safe_rate(confusion.tp + confusion.fp, len(labels)),
        "precision": _safe_rate(confusion.tp, confusion.tp + confusion.fp),
        "recall": _safe_rate(confusion.tp, confusion.tp + confusion.fn),
        "fpr": _safe_rate(confusion.fp, confusion.fp + confusion.tn),
        "fnr": _safe_rate(confusion.fn, confusion.fn + confusion.tp),
        "worst_group_fpr": group_fpr.get(worst_group_fpr_name) if worst_group_fpr_name is not None else None,
        "worst_group_fpr_name": worst_group_fpr_name,
        "worst_group_fnr": group_fnr.get(worst_group_fnr_name) if worst_group_fnr_name is not None else None,
        "worst_group_fnr_name": worst_group_fnr_name,
    }


def compute_hidden_risk_stress_curve(
    *,
    labels: Sequence[int],
    positive_scores: Sequence[float],
    metadata_rows: Sequence[Sequence[Any] | Mapping[str, Any]],
    metadata_fields: Sequence[str],
    threshold: float,
    base_config: CivilCommentsExperimentConfig,
    stress_levels: Sequence[float] = DEFAULT_STRESS_LEVELS,
    worst_group_accuracy_floor: float = DEFAULT_WORST_GROUP_ACCURACY_FLOOR,
) -> dict[str, Any]:
    if not (len(labels) == len(positive_scores) == len(metadata_rows)):
        raise ValueError("labels, positive_scores, and metadata_rows must align.")
    if not labels:
        raise ValueError("at least one evaluation example is required.")
    if not stress_levels:
        raise ValueError("stress_levels must contain at least one value.")
    if not 0.0 <= worst_group_accuracy_floor <= 1.0:
        raise ValueError("worst_group_accuracy_floor must be in [0, 1].")

    metadata_dicts = [
        metadata_row_to_dict(metadata_row, metadata_fields)
        for metadata_row in metadata_rows
    ]
    predicted_labels = [1 if float(score) >= threshold else 0 for score in positive_scores]

    curve: list[dict[str, float | None]] = []
    tail_worst_group_points: list[tuple[float, float]] = []
    failure_count = 0
    normalized_stress_levels = [float(level) for level in stress_levels]
    for severity in normalized_stress_levels:
        if severity < 0.0:
            raise ValueError("stress level values must be non-negative.")
        stressed_config = _config_for_stress(base_config, severity)
        observed_weights = [
            synthetic_observation_probability(metadata, metadata_fields, stressed_config)
            for metadata in metadata_rows
        ]
        hidden_weights = [1.0 - weight for weight in observed_weights]

        observed_metrics = _weighted_accuracy_bundle(
            labels=labels,
            predicted_labels=predicted_labels,
            metadata_dicts=metadata_dicts,
            weights=observed_weights,
        )
        tail_metrics = _weighted_accuracy_bundle(
            labels=labels,
            predicted_labels=predicted_labels,
            metadata_dicts=metadata_dicts,
            weights=hidden_weights,
        )

        tail_worst_group_accuracy = tail_metrics["worst_group_accuracy"]
        if tail_worst_group_accuracy is not None:
            tail_worst_group_points.append((severity, tail_worst_group_accuracy))
            if tail_worst_group_accuracy < worst_group_accuracy_floor:
                failure_count += 1

        curve.append(
            {
                "severity": severity,
                "effective_observation_rate": sum(observed_weights) / len(observed_weights),
                "observed_weighted_accuracy": observed_metrics["overall_accuracy"],
                "observed_weighted_worst_group_accuracy": observed_metrics["worst_group_accuracy"],
                "tail_weighted_accuracy": tail_metrics["overall_accuracy"],
                "tail_weighted_worst_group_accuracy": tail_worst_group_accuracy,
            }
        )

    tail_worst_group_values = [point[1] for point in tail_worst_group_points]
    tail_aurc = _normalized_trapezoid_area(tail_worst_group_points)

    summary = {
        "worst_group_accuracy_floor": float(worst_group_accuracy_floor),
        "tail_worst_group_accuracy_aurc": tail_aurc,
        "tail_worst_group_accuracy_min": min(tail_worst_group_values) if tail_worst_group_values else None,
        "tail_worst_group_accuracy_max": max(tail_worst_group_values) if tail_worst_group_values else None,
        "tail_worst_group_failure_rate_below_floor": (
            failure_count / len(tail_worst_group_points) if tail_worst_group_points else None
        ),
    }

    return {
        "stress_levels": normalized_stress_levels,
        "curve": curve,
        "summary": summary,
    }


def binary_auroc(labels: Sequence[int], positive_scores: Sequence[float]) -> float | None:
    if len(labels) != len(positive_scores):
        raise ValueError("labels and positive_scores must have the same length.")
    if not labels:
        raise ValueError("at least one example is required.")

    positives = sum(1 for label in labels if int(label) == 1)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None

    ordered_pairs = sorted(
        ((float(score), int(label)) for label, score in zip(labels, positive_scores)),
        key=lambda item: item[0],
    )
    positive_rank_sum = 0.0
    rank = 1
    index = 0
    while index < len(ordered_pairs):
        tie_end = index + 1
        while tie_end < len(ordered_pairs) and ordered_pairs[tie_end][0] == ordered_pairs[index][0]:
            tie_end += 1
        tie_count = tie_end - index
        average_rank = (2 * rank + tie_count - 1) / 2.0
        positive_count = sum(label for _score, label in ordered_pairs[index:tie_end])
        positive_rank_sum += average_rank * positive_count
        rank += tie_count
        index = tie_end

    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


@dataclass(frozen=True)
class _ConfusionCounts:
    tp: float
    fp: float
    tn: float
    fn: float


def _confusion_counts(
    labels: Sequence[int],
    predicted_labels: Sequence[int],
) -> _ConfusionCounts:
    if len(labels) != len(predicted_labels):
        raise ValueError("labels and predicted_labels must have the same length.")

    tp = fp = tn = fn = 0.0
    for label, prediction in zip(labels, predicted_labels):
        label_int = int(label)
        prediction_int = int(prediction)
        if label_int == 1 and prediction_int == 1:
            tp += 1.0
        elif label_int == 0 and prediction_int == 1:
            fp += 1.0
        elif label_int == 0 and prediction_int == 0:
            tn += 1.0
        elif label_int == 1 and prediction_int == 0:
            fn += 1.0
    return _ConfusionCounts(tp=tp, fp=fp, tn=tn, fn=fn)


def _weighted_accuracy_bundle(
    *,
    labels: Sequence[int],
    predicted_labels: Sequence[int],
    metadata_dicts: Sequence[Mapping[str, int]],
    weights: Sequence[float],
) -> dict[str, float | None]:
    if not (len(labels) == len(predicted_labels) == len(metadata_dicts) == len(weights)):
        raise ValueError("labels, predicted_labels, metadata_dicts, and weights must align.")

    total_weight = sum(weights)
    overall_accuracy = (
        sum(
            float(weight) * float(int(int(label) == int(prediction)))
            for label, prediction, weight in zip(labels, predicted_labels, weights)
        )
        / total_weight
        if total_weight > 0.0
        else None
    )

    group_accuracies: list[float] = []
    for identity in IDENTITY_FIELDS:
        for label_value in (0, 1):
            numerator = 0.0
            denominator = 0.0
            for index, metadata in enumerate(metadata_dicts):
                if metadata.get(identity, 0) != 1:
                    continue
                if int(labels[index]) != label_value:
                    continue
                weight = float(weights[index])
                if weight <= 0.0:
                    continue
                denominator += weight
                numerator += weight * float(int(int(labels[index]) == int(predicted_labels[index])))
            if denominator > 0.0:
                group_accuracies.append(numerator / denominator)

    return {
        "overall_accuracy": overall_accuracy,
        "worst_group_accuracy": min(group_accuracies) if group_accuracies else None,
    }


def _config_for_stress(
    config: CivilCommentsExperimentConfig,
    severity: float,
) -> CivilCommentsExperimentConfig:
    penalty_scale = 1.0 + severity
    return replace(
        config,
        toxic_penalty=min(config.toxic_penalty * penalty_scale, 1.0),
        identity_penalty=min(config.identity_penalty * penalty_scale, 1.0),
        identity_toxic_interaction_penalty=min(
            config.identity_toxic_interaction_penalty * penalty_scale,
            1.0,
        ),
    )


def _normalized_trapezoid_area(points: Sequence[tuple[float, float]]) -> float | None:
    if not points:
        return None
    if len(points) == 1:
        return points[0][1]
    sorted_points = sorted(points, key=lambda item: item[0])
    area = 0.0
    for (x0, y0), (x1, y1) in zip(sorted_points[:-1], sorted_points[1:]):
        area += (x1 - x0) * (y0 + y1) / 2.0
    x_start = sorted_points[0][0]
    x_end = sorted_points[-1][0]
    if x_end == x_start:
        return sorted_points[-1][1]
    return area / (x_end - x_start)


def _safe_rate(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    return float(numerator) / float(denominator)


def _accuracy(labels: Sequence[int], predicted_labels: Sequence[int]) -> float:
    if len(labels) != len(predicted_labels):
        raise ValueError("labels and predicted_labels must have the same length.")
    return sum(int(label == prediction) for label, prediction in zip(labels, predicted_labels)) / len(labels)


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        denominator = 1.0 + math.exp(-value)
        return 1.0 / denominator
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _softmax(values: Sequence[float]) -> list[float]:
    max_value = max(values)
    exps = [math.exp(value - max_value) for value in values]
    total = sum(exps)
    return [value / total for value in exps]
