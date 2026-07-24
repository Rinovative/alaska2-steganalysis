"""
===============================================================================
training_staged.py
===============================================================================
Best-state handoff across EfficientNet stages.

Responsibilities:
  - Configure each declared stage from a fully frozen starting state.
  - Construct an optimizer from only trainable parameters.
  - Hand the best state seen across all completed stages to the next stage.
  - Keep per-stage histories and optional checkpoint files distinct.

Design principles:
  - A worse stage cannot overwrite the globally best validation state. Test
    data is not accepted by this API.

Boundaries:
  - Stage definitions come from the models package; test evaluation remains
    separate.

Notes:
  - Classifier parameters may be optimized in every stage, while each feature
    block is selected in one stage only.
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..models.models_freezing import FineTuneStage, configure_stage, trainable_parameters
from .training_checkpoint import capture_model_state, restore_model_state
from .training_loop import TrainingResult, run_experiment

__all__ = ["StagedTrainingResult", "run_staged_fine_tuning"]


@dataclass(frozen=True, slots=True)
class StagedTrainingResult:
    """Store per-stage results and the globally best EfficientNet state identity.

    Parameters
    ----------
    stages
        Completed stage results in execution order.
    best_stage
        Name of the stage with highest validation weighted AUC.
    best_validation_weighted_auc
        Best score observed across all stages.
    history
        Concatenated stage-labelled epoch histories.
    """

    stages: tuple[TrainingResult, ...]
    best_stage: str
    best_validation_weighted_auc: float
    history: pd.DataFrame


def run_staged_fine_tuning(
    model: torch.nn.Module,
    stages: tuple[FineTuneStage, ...],
    train_loader: DataLoader,
    validation_loader: DataLoader,
    criterion: torch.nn.Module,
    *,
    device: str | torch.device,
    patience: int,
    checkpoint_directory: str | Path | None = None,
    history_directory: str | Path | None = None,
    progress: bool = True,
) -> StagedTrainingResult:
    """Run fine-tuning stages with global best-state handoff and restoration.

    Parameters
    ----------
    model
        EfficientNet adapter optimized in place.
    stages
        Non-empty ordered fine-tuning stage definitions.
    train_loader
        Shuffled training DataLoader.
    validation_loader
        Independent loader used for stage and model selection.
    criterion
        Binary loss module.
    device
        Explicit training device.
    patience
        Per-stage early-stopping patience.
    checkpoint_directory
        Optional directory for distinct stage checkpoints.
    history_directory
        Optional directory for distinct stage histories.
    progress
        Whether stage training displays progress bars.

    Returns
    -------
    StagedTrainingResult
        Stage results, concatenated history, and global-best identity.

    Raises
    ------
    ValueError
        If no stages are supplied or a stage selects no parameters.
    GPUPreflightError
        If CUDA is requested but unavailable.
    RuntimeError
        If no global best state is produced.

    Notes
    -----
    A worse later stage cannot replace the global best state; test data is never accepted.
    """
    if not stages:
        raise ValueError("At least one fine-tuning stage is required.")
    checkpoint_root = Path(checkpoint_directory) if checkpoint_directory is not None else None
    history_root = Path(history_directory) if history_directory is not None else None
    global_best_score = float("-inf")
    global_best_state: dict[str, torch.Tensor] | None = None
    global_best_stage = ""
    results: list[TrainingResult] = []
    histories: list[pd.DataFrame] = []

    for stage_index, stage in enumerate(stages):
        if global_best_state is not None:
            restore_model_state(model, global_best_state)
        configure_stage(model, stage)
        optimizer = torch.optim.Adam(trainable_parameters(model), lr=stage.learning_rate)
        checkpoint_path = checkpoint_root / f"{stage.name}_best.pt" if checkpoint_root else None
        history_path = history_root / f"{stage.name}_history.csv" if history_root else None
        result = run_experiment(
            model,
            train_loader,
            validation_loader,
            criterion,
            optimizer,
            num_epochs=stage.epochs,
            device=device,
            run_name=stage.name,
            checkpoint_path=checkpoint_path,
            history_path=history_path,
            patience=patience,
            progress=progress,
        )
        results.append(result)
        stage_history = result.history.copy()
        stage_history.insert(0, "stage_index", stage_index)
        stage_history.insert(1, "stage", stage.name)
        histories.append(stage_history)
        if result.summary.best_val_weighted_auc > global_best_score:
            global_best_score = result.summary.best_val_weighted_auc
            global_best_state = capture_model_state(model)
            global_best_stage = stage.name

    if global_best_state is None:
        raise RuntimeError("Staged training produced no model state.")
    restore_model_state(model, global_best_state)
    model.eval()
    return StagedTrainingResult(
        stages=tuple(results),
        best_stage=global_best_stage,
        best_validation_weighted_auc=global_best_score,
        history=pd.concat(histories, ignore_index=True),
    )
