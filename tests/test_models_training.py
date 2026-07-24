"""
===============================================================================
test_models_training.py
===============================================================================
Verify baseline models, checkpointing, freezing, and best-state restoration.

Responsibilities:
  - Smoke-test TinyCNN and EfficientNet input/output contracts.
  - Exercise atomic checkpoints and staged best-model handoff.

Design principles:
  - Tiny deterministic tensors isolate training-state behavior.
  - Monkeypatches replace only costly epoch internals while preserving
    orchestration.

Boundaries:
  - Full optimization quality and GPU throughput are outside unit-test scope.
  - Dataset construction is covered by dedicated data and dataset tests.
===============================================================================
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config.config_paths import ProjectPaths
from src.models.models_efficientnet import EfficientNetB0YCbCr
from src.models.models_freezing import FineTuneStage, configure_stage, efficientnet_stages, trainable_parameters
from src.models.models_tinycnn import TinyCNN
from src.training import training_loop, training_staged
from src.training.training_artifacts import (
    SingleRunArtifactPaths,
    StagedRunArtifactPaths,
    resolve_artifact_paths,
)
from src.training.training_checkpoint import load_checkpoint, save_checkpoint
from src.training.training_loop import RunSummary, TrainingResult


def test_training_artifact_paths_preserve_tinycnn_physical_destinations(tmp_path: Path) -> None:
    paths = ProjectPaths(root=tmp_path)
    artifacts = resolve_artifact_paths(
        paths,
        dataset_name="ALASKA2",
        model_name="TinyCNN_Y",
        run_id="alaska2_retrain_tiny10_effnet10_seed42_20260721",
    )

    assert isinstance(artifacts, SingleRunArtifactPaths)
    assert artifacts.run_name == "TinyCNN_Y_alaska2_retrain_tiny10_effnet10_seed42_20260721"
    assert artifacts.checkpoint_path == (
        tmp_path / "checkpoints/alaska2/TinyCNN_Y/TinyCNN_Y_alaska2_retrain_tiny10_effnet10_seed42_20260721_best.pt"
    )
    assert artifacts.history_path == (
        tmp_path / "reports/alaska2/TinyCNN_Y/TinyCNN_Y_alaska2_retrain_tiny10_effnet10_seed42_20260721_history.csv"
    )
    assert not artifacts.checkpoint_path.exists()


def test_training_artifact_paths_preserve_efficientnet_staged_directories(tmp_path: Path) -> None:
    paths = ProjectPaths(root=tmp_path)
    artifacts = resolve_artifact_paths(
        paths,
        dataset_name="ALASKA2",
        model_name="EfficientNetB0_YCbCr_FineTuned",
        run_id="alaska2_retrain_tiny10_effnet10_seed42_20260721",
        staged=True,
    )

    assert isinstance(artifacts, StagedRunArtifactPaths)
    assert artifacts.run_name == ("EfficientNetB0_YCbCr_FineTuned_alaska2_retrain_tiny10_effnet10_seed42_20260721")
    assert artifacts.checkpoint_directory == (
        tmp_path / "checkpoints/alaska2/EfficientNetB0_YCbCr_FineTuned/"
        "EfficientNetB0_YCbCr_FineTuned_alaska2_retrain_tiny10_effnet10_seed42_20260721"
    )
    assert artifacts.history_directory == (
        tmp_path / "reports/alaska2/EfficientNetB0_YCbCr_FineTuned/"
        "EfficientNetB0_YCbCr_FineTuned_alaska2_retrain_tiny10_effnet10_seed42_20260721"
    )
    assert not artifacts.checkpoint_directory.exists()


@pytest.mark.parametrize(
    ("dataset_name", "model_name", "run_id", "invalid_field"),
    [
        ("", "TinyCNN_Y", "run_1", "dataset_name"),
        ("ALASKA2", "bad/model", "run_1", "model_name"),
        ("ALASKA2", "TinyCNN_Y", " ../run", "run_id"),
    ],
)
def test_training_artifact_paths_reject_invalid_components(
    tmp_path: Path,
    dataset_name: str,
    model_name: str,
    run_id: str,
    invalid_field: str,
) -> None:
    with pytest.raises(ValueError, match=invalid_field):
        resolve_artifact_paths(
            ProjectPaths(root=tmp_path),
            dataset_name=dataset_name,
            model_name=model_name,
            run_id=run_id,
        )


def test_tinycnn_smoke_shape_and_input_error() -> None:
    model = TinyCNN().eval()
    with torch.inference_mode():
        assert model(torch.zeros(2, 1, 64, 64)).shape == (2, 1)
    with pytest.raises(ValueError, match=r"\[N, 1"):
        model(torch.zeros(2, 3, 64, 64))


def test_efficientnet_smoke_and_stage_uniqueness() -> None:
    model = EfficientNetB0YCbCr(weights=None).eval()
    with torch.inference_mode():
        assert model(torch.zeros(1, 3, 64, 64)).shape == (1, 1)
    stages = efficientnet_stages(
        model,
        head_learning_rate=1e-3,
        block_learning_rate=1e-4,
        head_epochs=2,
        block_epochs=1,
    )
    assert stages[0].name == "head_stem"
    assert len({stage.name for stage in stages}) == len(stages)
    feature_modules = [stage.modules[1] for stage in stages]
    assert len({id(module) for module in feature_modules}) == len(feature_modules)
    configure_stage(model, stages[1])
    selected = {id(parameter) for module in stages[1].modules for parameter in module.parameters()}
    assert {id(parameter) for parameter in trainable_parameters(model)} == selected


def test_checkpoint_roundtrip_is_device_safe(tmp_path: Path) -> None:
    model = nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
    path = save_checkpoint(
        tmp_path / "model.pt",
        model=model,
        optimizer=optimizer,
        epoch=3,
        validation_weighted_auc=0.7,
        validation_accuracy=0.6,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10)
    checkpoint = load_checkpoint(path, model=model, device="cpu", optimizer=optimizer)
    assert checkpoint["epoch"] == 3
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[name])


def test_run_experiment_creates_checkpoint_and_history_directories(tmp_path: Path) -> None:
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loader = DataLoader(
        TensorDataset(
            torch.tensor([[-2.0], [-1.0], [1.0], [2.0]]),
            torch.tensor([0.0, 0.0, 1.0, 1.0]),
        ),
        batch_size=4,
        shuffle=False,
    )
    checkpoint_path = tmp_path / "nested" / "checkpoints" / "best.pt"
    history_path = tmp_path / "nested" / "reports" / "history.csv"

    result = training_loop.run_experiment(
        model,
        loader,
        loader,
        nn.BCEWithLogitsLoss(),
        optimizer,
        num_epochs=1,
        patience=1,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        progress=False,
    )

    assert checkpoint_path.is_file()
    assert history_path.is_file()
    assert result.summary.best_checkpoint == checkpoint_path


def test_run_experiment_restores_in_memory_best_without_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    dataset = TensorDataset(torch.tensor([[0.0], [1.0]]), torch.tensor([0.0, 1.0]))
    loader = DataLoader(dataset, batch_size=2)
    epoch = 0
    validation_scores = iter([0.8, 0.5])

    def fake_train(*args, **kwargs):
        nonlocal epoch
        del args, kwargs
        epoch += 1
        with torch.no_grad():
            model.weight.fill_(float(epoch))
        return 0.1, 0.5

    def fake_validate(*args, **kwargs):
        del args, kwargs
        return training_loop._ValidationPass(0.1, 0.5, next(validation_scores))

    monkeypatch.setattr(training_loop, "_train_epoch", fake_train)
    monkeypatch.setattr(training_loop, "_validate_epoch", fake_validate)
    result = training_loop.run_experiment(
        model,
        loader,
        loader,
        nn.BCEWithLogitsLoss(),
        optimizer,
        num_epochs=3,
        patience=1,
        progress=False,
    )
    assert model.weight.item() == pytest.approx(1.0)
    assert result.summary.best_epoch == 1
    assert result.summary.final_epoch == 2
    assert result.summary.best_checkpoint is None


def test_staged_training_hands_global_best_state_to_next_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = nn.Linear(1, 1, bias=False)
    stages = (
        FineTuneStage("first", (model,), learning_rate=0.1, epochs=1),
        FineTuneStage("second", (model,), learning_rate=0.1, epochs=1),
    )
    dataset = TensorDataset(torch.tensor([[0.0], [1.0]]), torch.tensor([0.0, 1.0]))
    loader = DataLoader(dataset, batch_size=2)
    starting_weights: list[float] = []

    def fake_run_experiment(
        model_arg: nn.Linear,
        *args,
        run_name: str,
        **kwargs,
    ) -> TrainingResult:
        del args, kwargs
        starting_weights.append(model_arg.weight.item())
        stage_number = len(starting_weights)
        with torch.no_grad():
            model_arg.weight.fill_(float(stage_number))
        score = 0.8 if stage_number == 1 else 0.4
        history = pd.DataFrame(
            [{"epoch": 1, "train_loss": 0.1, "train_acc": 0.5, "val_loss": 0.1, "val_acc": 0.5, "val_wauc": score}]
        )
        return TrainingResult(
            history=history,
            summary=RunSummary(
                run_name=run_name,
                best_epoch=1,
                best_val_accuracy=0.5,
                best_val_weighted_auc=score,
                final_epoch=1,
                final_train_accuracy=0.5,
                final_val_accuracy=0.5,
                final_val_weighted_auc=score,
                early_stopped=False,
                best_checkpoint=None,
            ),
        )

    monkeypatch.setattr(training_staged, "run_experiment", fake_run_experiment)
    result = training_staged.run_staged_fine_tuning(
        model,
        stages,
        loader,
        loader,
        nn.BCEWithLogitsLoss(),
        device="cpu",
        patience=1,
        progress=False,
    )
    assert starting_weights[1] == pytest.approx(1.0)
    assert model.weight.item() == pytest.approx(1.0)
    assert result.best_stage == "first"
    assert result.best_validation_weighted_auc == pytest.approx(0.8)
