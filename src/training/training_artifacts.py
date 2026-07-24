"""
===============================================================================
training_artifacts.py
===============================================================================
Typed, side-effect-free training artifact path resolution.

Responsibilities:
  - Build dataset- and model-scoped checkpoint and history locations.
  - Derive one stable run name from the model name and caller-provided run ID.
  - Distinguish single-run destinations from staged-training directories.

Design principles:
  - One resolver owns the repository path convention without creating files.

Boundaries:
  - Training loops own file writes and staged trainers own per-stage filenames.
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, overload

from ..config.config_paths import ProjectPaths

__all__ = ["SingleRunArtifactPaths", "StagedRunArtifactPaths", "resolve_artifact_paths"]


@dataclass(frozen=True, slots=True)
class SingleRunArtifactPaths:
    """Describe checkpoint and history destinations for one training run.

    Parameters
    ----------
    dataset_name
        Stable dataset identity used for the lower-case artifact directory.
    model_name
        Stable model identity used for the model artifact directory.
    run_name
        Model-prefixed run identity shared by training and artifact names.
    checkpoint_path
        Best-checkpoint destination.
    history_path
        Epoch-history CSV destination.
    """

    dataset_name: str
    model_name: str
    run_name: str
    checkpoint_path: Path
    history_path: Path


@dataclass(frozen=True, slots=True)
class StagedRunArtifactPaths:
    """Describe checkpoint and history directories for staged training.

    Parameters
    ----------
    dataset_name
        Stable dataset identity used for the lower-case artifact directory.
    model_name
        Stable model identity used for the model artifact directory.
    run_name
        Model-prefixed run identity shared by the staged artifact directories.
    checkpoint_directory
        Directory in which the staged trainer stores distinct checkpoints.
    history_directory
        Directory in which the staged trainer stores distinct histories.
    """

    dataset_name: str
    model_name: str
    run_name: str
    checkpoint_directory: Path
    history_directory: Path


def _validate_component(value: str, *, field: str) -> str:
    if not value or value != value.strip() or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"{field} must be one non-empty path component without surrounding whitespace.")
    return value


if TYPE_CHECKING:

    @overload
    def resolve_artifact_paths(
        paths: ProjectPaths,
        *,
        dataset_name: str,
        model_name: str,
        run_id: str,
        staged: Literal[False] = False,
    ) -> SingleRunArtifactPaths: ...

    @overload
    def resolve_artifact_paths(
        paths: ProjectPaths,
        *,
        dataset_name: str,
        model_name: str,
        run_id: str,
        staged: Literal[True],
    ) -> StagedRunArtifactPaths: ...


def resolve_artifact_paths(
    paths: ProjectPaths,
    *,
    dataset_name: str,
    model_name: str,
    run_id: str,
    staged: bool = False,
) -> SingleRunArtifactPaths | StagedRunArtifactPaths:
    """Resolve immutable training artifact locations without filesystem writes.

    Parameters
    ----------
    paths
        Project-root path contract containing checkpoint and report roots.
    dataset_name
        Stable dataset name; its lower-case form scopes both artifact trees.
    model_name
        Stable model directory and run-name prefix.
    run_id
        Caller-owned experiment identity appended to the model name.
    staged
        Whether to return run directories instead of single-file destinations.

    Returns
    -------
    SingleRunArtifactPaths or StagedRunArtifactPaths
        Typed paths preserving the single-run or staged-training contract.

    Raises
    ------
    ValueError
        If a dataset, model, or run identifier is not one safe path component.
    """
    dataset = _validate_component(dataset_name, field="dataset_name")
    model = _validate_component(model_name, field="model_name")
    identifier = _validate_component(run_id, field="run_id")
    run_name = f"{model}_{identifier}"
    checkpoint_model_directory = paths.checkpoints / dataset.lower() / model
    history_model_directory = paths.reports / dataset.lower() / model
    if staged:
        return StagedRunArtifactPaths(
            dataset_name=dataset,
            model_name=model,
            run_name=run_name,
            checkpoint_directory=checkpoint_model_directory / run_name,
            history_directory=history_model_directory / run_name,
        )
    return SingleRunArtifactPaths(
        dataset_name=dataset,
        model_name=model,
        run_name=run_name,
        checkpoint_path=checkpoint_model_directory / f"{run_name}_best.pt",
        history_path=history_model_directory / f"{run_name}_history.csv",
    )
