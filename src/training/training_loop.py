"""
===============================================================================
training_loop.py
===============================================================================
Correct binary training and validation.

Responsibilities:
  - Train binary-logit models with device-safe tensor transfers.
  - Compute loss, accuracy, predictions, and weighted AUC in one validation
    pass.
  - Retain and restore the best model in memory, with optional persistent
    checkpoint.
  - Report best-epoch and final-epoch metrics as distinct values.

Design principles:
  - Validation never drives gradients. The returned model is always restored
    to the best validation weighted-AUC state, even when no file is written.

Boundaries:
  - Test-set evaluation is deliberately absent and must happen after model
    selection.

Notes:
  - Frozen modules are kept in evaluation mode so their batch-normalization
    statistics do not change.
===============================================================================
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..config.config_device import resolve_device
from ..evaluation.evaluation_metrics import weighted_auc
from ..models.models_freezing import apply_frozen_eval
from .training_checkpoint import capture_model_state, restore_model_state, save_checkpoint

__all__ = ["RunSummary", "TrainingResult", "run_experiment"]


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Store unambiguous best-epoch and final-epoch training metrics.

    Parameters
    ----------
    run_name
        Unique artifact and display identifier.
    best_epoch
        Epoch with the highest validation weighted AUC.
    best_val_accuracy
        Validation accuracy at the best epoch.
    best_val_weighted_auc
        Highest validation weighted AUC.
    final_epoch
        Last epoch executed before completion or early stopping.
    final_train_accuracy
        Training accuracy in the final executed epoch.
    final_val_accuracy
        Validation accuracy in the final executed epoch.
    final_val_weighted_auc
        Validation weighted AUC in the final executed epoch.
    early_stopped
        Whether patience terminated the run.
    best_checkpoint
        Persisted best-checkpoint path, if requested.
    """

    run_name: str
    best_epoch: int
    best_val_accuracy: float
    best_val_weighted_auc: float
    final_epoch: int
    final_train_accuracy: float
    final_val_accuracy: float
    final_val_weighted_auc: float
    early_stopped: bool
    best_checkpoint: Path | None


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Store epoch history and summary for a model restored to its best state.

    Parameters
    ----------
    history
        Per-epoch train and validation metric dataframe.
    summary
        Best-versus-final metric and artifact summary.
    """

    history: pd.DataFrame
    summary: RunSummary


@dataclass(frozen=True, slots=True)
class _ValidationPass:
    loss: float
    accuracy: float
    weighted_auc: float


def _move_inputs(
    inputs: torch.Tensor | tuple[torch.Tensor, ...],
    device: torch.device,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    non_blocking = device.type == "cuda"
    if isinstance(inputs, tuple):
        return tuple(value.to(device, non_blocking=non_blocking) for value in inputs)
    return inputs.to(device, non_blocking=non_blocking)


def _train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    progress: bool,
    description: str,
) -> tuple[float, float]:
    model.train()
    apply_frozen_eval(model)
    total_loss = 0.0
    correct = 0
    count = 0
    iterator = tqdm(loader, desc=description, leave=False, disable=not progress)
    for inputs, labels in iterator:
        moved_inputs = _move_inputs(inputs, device)
        labels = labels.to(device, non_blocking=device.type == "cuda").float().view(-1, 1)
        optimizer.zero_grad(set_to_none=True)
        logits = model(moved_inputs).view(-1, 1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch_count = labels.shape[0]
        total_loss += float(loss.item()) * batch_count
        correct += int(((torch.sigmoid(logits) >= 0.5) == labels.bool()).sum().item())
        count += batch_count
        iterator.set_postfix(loss=float(loss.item()))
    if count == 0:
        raise ValueError("Training loader is empty.")
    return total_loss / count, correct / count


def _validate_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
) -> _ValidationPass:
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0
    targets: list[torch.Tensor] = []
    probabilities: list[torch.Tensor] = []
    with torch.inference_mode():
        for inputs, labels in loader:
            moved_inputs = _move_inputs(inputs, device)
            labels = labels.to(device, non_blocking=device.type == "cuda").float().view(-1, 1)
            logits = model(moved_inputs).view(-1, 1)
            loss = criterion(logits, labels)
            probability = torch.sigmoid(logits)
            batch_count = labels.shape[0]
            total_loss += float(loss.item()) * batch_count
            correct += int(((probability >= 0.5) == labels.bool()).sum().item())
            count += batch_count
            targets.append(labels.cpu().view(-1))
            probabilities.append(probability.cpu().view(-1))
    if count == 0:
        raise ValueError("Validation loader is empty.")
    y_true = torch.cat(targets).numpy()
    y_probability = torch.cat(probabilities).numpy()
    return _ValidationPass(
        loss=total_loss / count,
        accuracy=correct / count,
        weighted_auc=weighted_auc(y_true, y_probability),
    )


def run_experiment(
    model: torch.nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    *,
    num_epochs: int = 30,
    device: str | torch.device = "cpu",
    run_name: str | None = None,
    checkpoint_path: str | Path | None = None,
    history_path: str | Path | None = None,
    patience: int = 5,
    minimum_improvement: float = 1e-4,
    progress: bool = True,
) -> TrainingResult:
    """Train a binary-logit model and restore its best validation state.

    Parameters
    ----------
    model
        Model optimized and restored in place.
    train_loader
        Shuffled training DataLoader.
    validation_loader
        Independent loader used only for model selection.
    criterion
        Binary loss callable returning a scalar tensor.
    optimizer
        Optimizer bound to the intended trainable parameters.
    num_epochs
        Positive maximum epoch count.
    device
        Explicit device; unavailable CUDA raises instead of falling back.
    run_name
        Optional unique run identifier; the class name is the fallback.
    checkpoint_path
        Optional atomic best-checkpoint destination.
    history_path
        Optional CSV history destination.
    patience
        Positive epochs without sufficient improvement before stopping.
    minimum_improvement
        Non-negative weighted-AUC increase required as improvement.
    progress
        Whether to show training progress bars.

    Returns
    -------
    TrainingResult
        Per-epoch history and best/final summary after best-state restoration.

    Raises
    ------
    GPUPreflightError
        If CUDA is requested but unavailable.
    ValueError
        If arguments or a train/validation loader are invalid or empty.
    RuntimeError
        If no best model state is produced.

    Notes
    -----
    The function never accepts or evaluates a test loader.
    """
    if num_epochs <= 0 or patience <= 0:
        raise ValueError("num_epochs and patience must be positive.")
    if minimum_improvement < 0:
        raise ValueError("minimum_improvement must be non-negative.")
    resolved = resolve_device(device)
    model.to(resolved)
    name = run_name or model.__class__.__name__
    destination = Path(checkpoint_path) if checkpoint_path is not None else None

    history_rows: list[dict[str, float | int]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_accuracy = float("nan")
    best_score = -np.inf
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
        train_loss, train_accuracy = _train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            resolved,
            progress=progress,
            description=f"{name} {epoch}/{num_epochs}",
        )
        validation = _validate_epoch(model, validation_loader, criterion, resolved)
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_accuracy,
                "val_loss": validation.loss,
                "val_acc": validation.accuracy,
                "val_wauc": validation.weighted_auc,
            }
        )

        if validation.weighted_auc > best_score + minimum_improvement:
            best_score = validation.weighted_auc
            best_accuracy = validation.accuracy
            best_epoch = epoch
            best_state = capture_model_state(model)
            epochs_without_improvement = 0
            if destination is not None:
                save_checkpoint(
                    destination,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    validation_weighted_auc=validation.weighted_auc,
                    validation_accuracy=validation.accuracy,
                )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError("Training completed without a best model state.")
    final = history_rows[-1]
    restore_model_state(model, best_state)
    model.to(resolved)
    model.eval()

    history = pd.DataFrame.from_records(history_rows)
    if history_path is not None:
        history_destination = Path(history_path)
        history_destination.parent.mkdir(parents=True, exist_ok=True)
        history.to_csv(history_destination, index=False)

    summary = RunSummary(
        run_name=name,
        best_epoch=best_epoch,
        best_val_accuracy=best_accuracy,
        best_val_weighted_auc=float(best_score),
        final_epoch=int(final["epoch"]),
        final_train_accuracy=float(final["train_acc"]),
        final_val_accuracy=float(final["val_acc"]),
        final_val_weighted_auc=float(final["val_wauc"]),
        early_stopped=epochs_without_improvement >= patience,
        best_checkpoint=destination,
    )
    return TrainingResult(history=history, summary=summary)
