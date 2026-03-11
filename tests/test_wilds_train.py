from types import SimpleNamespace

from experiments.wilds_civilcomments.common import CivilCommentsExperimentConfig
from experiments.wilds_civilcomments.metrics import CivilCommentsSplitMetrics
from experiments.wilds_civilcomments import train as train_module
from experiments.wilds_civilcomments.train import _build_minimax_config, _build_training_arguments


class _LegacyTrainingArguments:
    def __init__(self, *, output_dir: str, eval_strategy: str, save_strategy: str) -> None:
        self.output_dir = output_dir
        self.eval_strategy = eval_strategy
        self.save_strategy = save_strategy


class _ModernTrainingArguments:
    def __init__(self, *, output_dir: str, evaluation_strategy: str, save_strategy: str) -> None:
        self.output_dir = output_dir
        self.evaluation_strategy = evaluation_strategy
        self.save_strategy = save_strategy


def test_build_training_arguments_supports_eval_strategy_name() -> None:
    args = _build_training_arguments(
        _LegacyTrainingArguments,
        output_dir="outputs",
        save_strategy="epoch",
    )

    assert args.eval_strategy == "no"


def test_build_training_arguments_supports_evaluation_strategy_name() -> None:
    args = _build_training_arguments(
        _ModernTrainingArguments,
        output_dir="outputs",
        save_strategy="epoch",
    )

    assert args.evaluation_strategy == "no"


def test_build_training_arguments_preserves_save_strategy() -> None:
    args = _build_training_arguments(
        _ModernTrainingArguments,
        output_dir="outputs",
        save_strategy="no",
    )

    assert args.save_strategy == "no"


def test_build_minimax_config_maps_auto_discovery_mode() -> None:
    metadata_fields = [
        "male",
        "female",
        "LGBTQ",
        "christian",
        "muslim",
        "other_religions",
        "black",
        "white",
        "identity_any",
        "severe_toxicity",
        "obscene",
        "threat",
        "insult",
        "identity_attack",
        "sexual_explicit",
        "y",
    ]
    train_split = SimpleNamespace(
        metadata_rows=[
            [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        metadata_fields=metadata_fields,
    )

    minimax_config, assumed_rate = _build_minimax_config(
        CivilCommentsExperimentConfig(method="robust_auto_v1"),
        train_split=train_split,
    )

    assert minimax_config.uncertainty_mode == "adaptive_v1"
    assert assumed_rate is not None
    assert 0.0 < assumed_rate < 1.0


def test_train_from_config_skips_minimax_builder_for_erm(tmp_path, monkeypatch) -> None:
    class _FakeTrainingArguments:
        def __init__(
            self,
            *,
            output_dir: str,
            evaluation_strategy: str,
            save_strategy: str,
            **kwargs,
        ) -> None:
            self.output_dir = output_dir
            self.evaluation_strategy = evaluation_strategy
            self.save_strategy = save_strategy
            self.kwargs = kwargs

    class _FakeTrainer:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def train(self) -> SimpleNamespace:
            return SimpleNamespace(metrics={"train_runtime": 1.0})

        def save_model(self, _path: str) -> None:
            raise AssertionError("save_model should not be called when save_final_checkpoint is false")

    class _FakeAutoModel:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return object()

    split = SimpleNamespace(
        dataset=[],
        labels=[0, 1],
        metadata_rows=[[0], [1]],
        metadata_fields=["y"],
        observed_mask=[True, True],
    )

    metrics = CivilCommentsSplitMetrics(
        overall_accuracy=1.0,
        overall_auroc=1.0,
        worst_group_accuracy=1.0,
        worst_group_auroc=1.0,
        group_accuracy={},
        group_accuracy_counts={},
        group_auroc={},
        group_auroc_counts={},
    )

    monkeypatch.setattr(
        train_module,
        "_require_transformers",
        lambda: {
            "AutoModelForSequenceClassification": _FakeAutoModel,
            "Trainer": _FakeTrainer,
            "TrainingArguments": _FakeTrainingArguments,
            "set_seed": lambda _seed: None,
        },
    )
    monkeypatch.setattr(
        train_module,
        "load_civilcomments_splits",
        lambda _config: (object(), {"train": split, "val": split, "test": split}, object()),
    )
    monkeypatch.setattr(train_module, "build_training_group_summary", lambda _split: {})
    monkeypatch.setattr(train_module, "evaluate_split", lambda **_kwargs: ({"overall_accuracy": 1.0}, metrics))
    monkeypatch.setattr(
        train_module,
        "_predict_split",
        lambda **_kwargs: train_module.SplitPredictionOutput(
            predicted_labels=[0, 1],
            positive_scores=[0.1, 0.9],
        ),
    )
    monkeypatch.setattr(
        train_module,
        "_build_minimax_config",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("ERM should not call _build_minimax_config")),
    )
    monkeypatch.setattr(
        train_module,
        "MinimaxTrainer",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("ERM should not instantiate MinimaxTrainer")),
    )

    config = CivilCommentsExperimentConfig(
        method="erm",
        output_dir=str(tmp_path / "erm_run"),
        save_strategy="no",
        save_final_checkpoint=False,
        download=False,
    )
    payload = train_module.train_from_config(config)

    assert payload["train"]["effective_assumed_observation_rate"] is None
