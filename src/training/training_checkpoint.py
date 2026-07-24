"""
===============================================================================
training_checkpoint.py
===============================================================================
Compact, device-safe checkpoint contracts.

Responsibilities:
  - Clone model states to CPU for reliable in-memory restoration.
  - Atomically save model and optional optimizer state.
  - Load trusted project checkpoints with an explicit map location.

Design principles:
  - Checkpoint paths are caller-owned. Writes use a sibling temporary file
    followed by an atomic replace.

Boundaries:
  - This module does not choose when a checkpoint is best or construct
    models.

Notes:
  - Only project-produced tensor/primitive checkpoint dictionaries are
    supported.
===============================================================================
"""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

__all__ = ["capture_model_state", "load_checkpoint", "restore_model_state", "save_checkpoint"]

ModelState = dict[str, torch.Tensor]


def capture_model_state(model: torch.nn.Module) -> ModelState:
    """Clone every model state tensor onto the CPU.

    Parameters
    ----------
    model
        Model whose current state is captured.

    Returns
    -------
    dict[str, torch.Tensor]
        Independent CPU clone suitable for later restoration.
    """
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def restore_model_state(model: torch.nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    """Restore a captured model state with strict key matching.

    Parameters
    ----------
    model
        Model receiving the captured tensors.
    state
        Complete state mapping produced by ``capture_model_state``.

    Returns
    -------
    None
        The model state is replaced in place.

    Raises
    ------
    RuntimeError
        If state keys or tensor shapes do not match the model.
    """
    model.load_state_dict(dict(state), strict=True)


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    validation_weighted_auc: float,
    validation_accuracy: float,
) -> Path:
    """Atomically save a compact model and optional optimizer checkpoint.

    Parameters
    ----------
    path
        Final checkpoint destination.
    model
        Model whose best state is persisted.
    optimizer
        Optional optimizer whose state enables trusted manual recovery.
    epoch
        One-based epoch associated with the validation score.
    validation_weighted_auc
        Validation ALASKA2 weighted AUC.
    validation_accuracy
        Validation binary accuracy.

    Returns
    -------
    pathlib.Path
        Final checkpoint path after atomic replacement.

    Raises
    ------
    OSError
        If the parent or temporary checkpoint cannot be written.
    RuntimeError
        If PyTorch cannot serialize the payload.

    Notes
    -----
    The temporary file is written beside the destination before ``Path.replace``.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": 1,
        "epoch": int(epoch),
        "model_state": capture_model_state(model),
        "validation_weighted_auc": float(validation_weighted_auc),
        "validation_accuracy": float(validation_accuracy),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        torch.save(payload, temporary_path)
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    device: str | torch.device = "cpu",
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    """Load a trusted project checkpoint and restore model and optional optimizer.

    Parameters
    ----------
    path
        Existing project checkpoint path.
    model
        Model receiving the stored state.
    device
        Explicit map location for serialized tensors.
    optimizer
        Optional optimizer receiving a stored optimizer state.

    Returns
    -------
    dict[str, Any]
        Validated checkpoint payload.

    Raises
    ------
    FileNotFoundError
        If the checkpoint path does not exist.
    ValueError
        If the payload is not a supported project checkpoint.
    RuntimeError
        If stored state is incompatible with the model or optimizer.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {source}")
    checkpoint = torch.load(source, map_location=torch.device(device), weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError(f"Unsupported checkpoint format: {source}")
    restore_model_state(model, checkpoint["model_state"])
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint
