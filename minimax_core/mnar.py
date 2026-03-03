from __future__ import annotations

import random
import math
from dataclasses import dataclass
from statistics import mean
from typing import Mapping, Sequence


MNAR_VIEW_MODES = (
    "explicit_missing",
    "drop_unobserved",
    "truncate_after_unobserved",
)


@dataclass(frozen=True)
class SyntheticMNARConfig:
    seed: int = 0
    view_mode: str = "explicit_missing"
    base_observation_probability: float = 0.95
    distressed_penalty: float = 0.35
    drought_penalty: float = 0.10
    exit_penalty: float = 0.15
    min_observation_probability: float = 0.05
    max_observation_probability: float = 1.0

    def __post_init__(self) -> None:
        if self.view_mode not in MNAR_VIEW_MODES:
            raise ValueError(
                f"view_mode must be one of {MNAR_VIEW_MODES}, got {self.view_mode!r}."
            )
        if not 0.0 <= self.min_observation_probability <= self.max_observation_probability <= 1.0:
            raise ValueError("observation probability bounds must satisfy 0 <= min <= max <= 1.")
        if not 0.0 <= self.base_observation_probability <= 1.0:
            raise ValueError("base_observation_probability must be between 0 and 1.")


@dataclass(frozen=True)
class SyntheticMNARResult:
    observed_mask: tuple[bool, ...]
    keep_mask: tuple[bool, ...]
    observed_values: tuple[float | None, ...]
    observation_probabilities: tuple[float, ...]
    observation_rate: float
    stable_observation_rate: float
    distressed_observation_rate: float


def sample_group_mnar_mask(
    *,
    group_ids: Sequence[str],
    base_observed_mask: Sequence[bool],
    q_values: Mapping[str, float],
    seed: int,
    epoch_index: int,
    min_observed_per_group: int = 1,
) -> list[bool]:
    if len(group_ids) != len(base_observed_mask):
        raise ValueError("group_ids and base_observed_mask must have the same length.")

    sampled_mask: list[bool] = []
    eligible_by_group: dict[str, list[int]] = {}
    retained_by_group: dict[str, list[int]] = {}

    for index, (group_id, base_observed) in enumerate(zip(group_ids, base_observed_mask)):
        if not base_observed:
            sampled_mask.append(False)
            continue
        probability = float(q_values[group_id])
        rng = random.Random(hash((seed, epoch_index, index, group_id, round(probability, 6))))
        keep = rng.random() < probability
        sampled_mask.append(keep)
        eligible_by_group.setdefault(group_id, []).append(index)
        if keep:
            retained_by_group.setdefault(group_id, []).append(index)

    _rescue_group_mask(
        sampled_mask=sampled_mask,
        eligible_by_group=eligible_by_group,
        retained_by_group=retained_by_group,
        minimum=min_observed_per_group,
    )
    return sampled_mask


def build_group_adversarial_mask(
    *,
    group_ids: Sequence[str],
    base_observed_mask: Sequence[bool],
    q_values: Mapping[str, float],
    losses: Sequence[float],
    min_observed_per_group: int = 1,
) -> list[bool]:
    if not (len(group_ids) == len(base_observed_mask) == len(losses)):
        raise ValueError("group_ids, base_observed_mask, and losses must have the same length.")

    keep_mask = [False for _ in group_ids]
    eligible_by_group: dict[str, list[int]] = {}
    for index, (group_id, base_observed) in enumerate(zip(group_ids, base_observed_mask)):
        if base_observed:
            eligible_by_group.setdefault(group_id, []).append(index)

    for group_id, indices in eligible_by_group.items():
        if not indices:
            continue
        target_keep = math.ceil(float(q_values[group_id]) * len(indices))
        target_keep = max(min_observed_per_group, min(len(indices), target_keep))
        ranked = sorted(indices, key=lambda index: (float(losses[index]), index))
        for index in ranked[:target_keep]:
            keep_mask[index] = True
    return keep_mask


def sample_score_mnar_mask(
    *,
    base_observed_mask: Sequence[bool],
    q_values: Sequence[float],
    seed: int,
    epoch_index: int,
    min_observed: int = 1,
) -> list[bool]:
    if len(base_observed_mask) != len(q_values):
        raise ValueError("base_observed_mask and q_values must have the same length.")

    sampled_mask: list[bool] = []
    eligible_indices: list[int] = []
    retained_indices: list[int] = []
    for index, (base_observed, probability) in enumerate(zip(base_observed_mask, q_values)):
        if not base_observed:
            sampled_mask.append(False)
            continue
        rng = random.Random(hash((seed, epoch_index, index, round(float(probability), 6))))
        keep = rng.random() < float(probability)
        sampled_mask.append(keep)
        eligible_indices.append(index)
        if keep:
            retained_indices.append(index)

    if eligible_indices and len(retained_indices) < min_observed:
        ranked = sorted(eligible_indices, key=lambda idx: float(q_values[idx]), reverse=True)
        for index in ranked[:min_observed]:
            sampled_mask[index] = True
    return sampled_mask


def apply_synthetic_mnar(
    *,
    labels: Sequence[float],
    group_ids: Sequence[str],
    path_indices: Sequence[int],
    step_indices: Sequence[int],
    config: SyntheticMNARConfig,
    weather_regimes: Sequence[str] | None = None,
    farm_alive_next_year: Sequence[bool] | None = None,
) -> SyntheticMNARResult:
    if not (
        len(labels) == len(group_ids) == len(path_indices) == len(step_indices)
    ):
        raise ValueError("labels, group_ids, path_indices, and step_indices must have the same length.")
    if weather_regimes is not None and len(weather_regimes) != len(labels):
        raise ValueError("weather_regimes must have the same length as labels.")
    if farm_alive_next_year is not None and len(farm_alive_next_year) != len(labels):
        raise ValueError("farm_alive_next_year must have the same length as labels.")

    observed_mask: list[bool] = []
    observation_probabilities: list[float] = []
    for index, (label, group_id, path_index, step_index) in enumerate(
        zip(labels, group_ids, path_indices, step_indices)
    ):
        probability = config.base_observation_probability
        is_distressed = label < 0.0 or group_id == "distressed"
        if is_distressed:
            probability -= config.distressed_penalty
        if weather_regimes is not None and weather_regimes[index] == "drought":
            probability -= config.drought_penalty
        if farm_alive_next_year is not None and not farm_alive_next_year[index]:
            probability -= config.exit_penalty

        probability = min(
            max(probability, config.min_observation_probability),
            config.max_observation_probability,
        )
        rng = random.Random(hash((config.seed, path_index, step_index, group_id, round(label, 6))))
        observed_mask.append(rng.random() < probability)
        observation_probabilities.append(probability)

    keep_mask = _build_keep_mask(
        observed_mask=observed_mask,
        path_indices=path_indices,
        config=config,
    )
    observed_mask, keep_mask = _ensure_nonempty_training_view(
        observed_mask=observed_mask,
        keep_mask=keep_mask,
        observation_probabilities=observation_probabilities,
        config=config,
    )
    observed_values = [
        label if observed else None
        for label, observed in zip(labels, observed_mask)
    ]

    stable_observed = [observed for observed, group in zip(observed_mask, group_ids) if group == "stable"]
    distressed_observed = [
        observed for observed, group in zip(observed_mask, group_ids) if group == "distressed"
    ]
    return SyntheticMNARResult(
        observed_mask=tuple(observed_mask),
        keep_mask=tuple(keep_mask),
        observed_values=tuple(observed_values),
        observation_probabilities=tuple(observation_probabilities),
        observation_rate=(mean(1.0 if observed else 0.0 for observed in observed_mask) if observed_mask else 0.0),
        stable_observation_rate=(
            mean(1.0 if observed else 0.0 for observed in stable_observed)
            if stable_observed
            else 1.0
        ),
        distressed_observation_rate=(
            mean(1.0 if observed else 0.0 for observed in distressed_observed)
            if distressed_observed
            else 1.0
        ),
    )


def build_proxy_labels(
    *,
    observed_values: Sequence[float | None],
    group_ids: Sequence[str],
    observed_mask: Sequence[bool],
    label_scale: float,
) -> list[float]:
    if not (
        len(observed_values) == len(group_ids) == len(observed_mask)
    ):
        raise ValueError("observed_values, group_ids, and observed_mask must have the same length.")

    observed_by_group: dict[str, list[float]] = {}
    scaled_observed_values: list[float] = []
    for value, group_id, observed in zip(observed_values, group_ids, observed_mask):
        if not observed or value is None:
            continue
        scaled_value = float(value) / label_scale
        observed_by_group.setdefault(group_id, []).append(scaled_value)
        scaled_observed_values.append(scaled_value)

    global_proxy = mean(scaled_observed_values) if scaled_observed_values else 0.0
    proxies: list[float] = []
    for value, group_id, observed in zip(observed_values, group_ids, observed_mask):
        if observed and value is not None:
            proxies.append(float(value) / label_scale)
            continue
        group_values = observed_by_group.get(group_id, [])
        proxies.append(mean(group_values) if group_values else global_proxy)
    return proxies


def _build_keep_mask(
    *,
    observed_mask: Sequence[bool],
    path_indices: Sequence[int],
    config: SyntheticMNARConfig,
) -> list[bool]:
    if config.view_mode == "explicit_missing":
        return [True] * len(observed_mask)
    if config.view_mode == "drop_unobserved":
        return [bool(observed) for observed in observed_mask]

    dropped_paths: set[int] = set()
    keep_mask: list[bool] = []
    for observed, path_index in zip(observed_mask, path_indices):
        if path_index in dropped_paths:
            keep_mask.append(False)
            continue
        if not observed:
            keep_mask.append(False)
            dropped_paths.add(path_index)
            continue
        keep_mask.append(True)
    return keep_mask


def _rescue_group_mask(
    *,
    sampled_mask: list[bool],
    eligible_by_group: Mapping[str, list[int]],
    retained_by_group: Mapping[str, list[int]],
    minimum: int,
) -> None:
    if minimum <= 0:
        return
    for group_id, eligible_indices in eligible_by_group.items():
        currently_retained = len(retained_by_group.get(group_id, []))
        if currently_retained >= minimum:
            continue
        missing = minimum - currently_retained
        rescue_candidates = [index for index in eligible_indices if not sampled_mask[index]]
        for index in rescue_candidates[:missing]:
            sampled_mask[index] = True


def _ensure_nonempty_training_view(
    *,
    observed_mask: list[bool],
    keep_mask: list[bool],
    observation_probabilities: Sequence[float],
    config: SyntheticMNARConfig,
) -> tuple[list[bool], list[bool]]:
    if any(keep_mask):
        return observed_mask, keep_mask

    if not observation_probabilities:
        return observed_mask, keep_mask

    rescue_index = max(
        range(len(observation_probabilities)),
        key=lambda index: observation_probabilities[index],
    )
    observed_mask[rescue_index] = True
    if config.view_mode == "explicit_missing":
        keep_mask = [True] * len(observed_mask)
    else:
        keep_mask[rescue_index] = True
    return observed_mask, keep_mask
