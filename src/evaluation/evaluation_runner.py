"""
===============================================================================
evaluation_runner.py
===============================================================================
Device-safe one-pass binary evaluation.

Responsibilities:
  - Collect loss, accuracy, labels, probabilities, and weighted AUC in one
    pass.
  - Support tensor and tuple-of-tensor model inputs.
  - Use inference mode and non-blocking accelerator transfers.

Design principles:
  - Evaluation is a pure result-producing operation. It never mutates result
    tables, renders widgets, or selects checkpoints.

Boundaries:
  - The caller is responsible for loading the intended best checkpoint before
    testing.

Notes:
  - Test loaders must remain separate from validation loaders by
    construction.
===============================================================================
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config.config_device import resolve_device
from .evaluation_metrics import classification_scores, predict_binary, weighted_auc

__all__ = ["BinaryEvaluation", "evaluate_binary_model"]


@dataclass(frozen=True, slots=True)
class BinaryEvaluation:
    """Store loss, accuracy, weighted AUC, targets, and probabilities from one pass.

    Parameters
    ----------
    loss
        Mean criterion loss or ``None`` when no criterion was supplied.
    accuracy
        Binary decision accuracy.
    weighted_auc
        Official ALASKA2 weighted AUC.
    y_true
        Collected binary targets.
    y_probability
        Collected Stego probabilities.
    """

    loss: float | None
    accuracy: float
    weighted_auc: float
    y_true: np.ndarray
    y_probability: np.ndarray


def _move_inputs(
    inputs: torch.Tensor | tuple[torch.Tensor, ...],
    device: torch.device,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    non_blocking = device.type == "cuda"
    if isinstance(inputs, tuple):
        return tuple(value.to(device, non_blocking=non_blocking) for value in inputs)
    return inputs.to(device, non_blocking=non_blocking)


def evaluate_binary_model(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: str | torch.device,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> BinaryEvaluation:
    """Evaluate a binary-logit model in exactly one DataLoader pass.

    Parameters
    ----------
    model
        Binary-logit model accepting a tensor or tuple of tensors.
    loader
        Evaluation loader reserved for the requested partition.
    device
        Explicit device; unavailable CUDA raises instead of falling back.
    criterion
        Optional per-batch loss callable.

    Returns
    -------
    BinaryEvaluation
        Collected loss, metrics, targets, and probabilities.

    Raises
    ------
    GPUPreflightError
        If CUDA is requested but unavailable.
    ValueError
        If logits mismatch labels or the loader is empty.
    """
    resolved = resolve_device(device)
    model.to(resolved)
    model.eval()
    labels_all: list[torch.Tensor] = []
    probabilities_all: list[torch.Tensor] = []
    total_loss = 0.0
    total_count = 0

    with torch.inference_mode():
        for inputs, labels in loader:
            moved_inputs = _move_inputs(inputs, resolved)
            labels = labels.to(resolved, non_blocking=resolved.type == "cuda").float().view(-1, 1)
            logits = model(moved_inputs).view(-1, 1)
            if logits.shape != labels.shape:
                raise ValueError(f"Logit shape {logits.shape} does not match label shape {labels.shape}.")
            if criterion is not None:
                total_loss += float(criterion(logits, labels).item()) * labels.shape[0]
            total_count += labels.shape[0]
            labels_all.append(labels.cpu().view(-1))
            probabilities_all.append(torch.sigmoid(logits).cpu().view(-1))

    if total_count == 0:
        raise ValueError("Cannot evaluate an empty loader.")
    y_true = torch.cat(labels_all).numpy()
    y_probability = torch.cat(probabilities_all).numpy()
    predictions = predict_binary(y_probability)
    scores = classification_scores(y_true, predictions)
    return BinaryEvaluation(
        loss=total_loss / total_count if criterion is not None else None,
        accuracy=scores["accuracy"],
        weighted_auc=weighted_auc(y_true, y_probability),
        y_true=y_true,
        y_probability=y_probability,
    )
