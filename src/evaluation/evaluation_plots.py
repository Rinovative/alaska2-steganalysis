"""
===============================================================================
evaluation_plots.py
===============================================================================
Structured public evaluation artifacts and diagnostic figures.

Responsibilities:
  - Resolve and validate compact repository-visible evaluation artifacts.
  - Prepare continuous and stage-aware chronological training histories.
  - Reconstruct histories, confusion matrices, ROC curves, score
    distributions, numerical metrics, and model comparisons as display data.
  - Return Matplotlib figures without displaying or saving them.

Design principles:
  - Verified structured evidence is the source of every public result view.
  - Staged histories use separate line segments on one chronological axis.
  - Rendering is separated from inference, widgets, and runtime caching.

Boundaries:
  - This module never trains a model, runs inference, recomputes test
    metrics, reads checkpoints, or requires a dataset.

Notes:
  - ALASKA2 weighted-AUC bands are shown on ROC plots for context only.
===============================================================================
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, SupportsFloat, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from ..config.config_paths import ProjectPaths, default_paths

__all__ = [
    "EFFICIENTNET_STAGE_ORDER",
    "EvaluationResults",
    "HistoryTimeline",
    "ModelEvaluation",
    "StageInterval",
    "aggregate_score_distribution",
    "comparison_table",
    "load_evaluation_results",
    "metrics_table",
    "plot_confusion_matrix",
    "plot_history",
    "plot_roc_curves",
    "plot_score_distribution",
    "plot_score_histogram",
    "prepare_history",
]

EFFICIENTNET_STAGE_ORDER: Final[tuple[str, ...]] = (
    "head_stem",
    "feature_8",
    "feature_7",
    "feature_6",
    "feature_5",
    "feature_4",
    "feature_3",
    "feature_2",
    "feature_1",
)
_HISTORY_COLUMNS: Final[frozenset[str]] = frozenset(
    {"epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_wauc"}
)
_MODEL_LAYOUT: Final[tuple[tuple[str, str, str], ...]] = (
    ("tinycnn", "tinycnn", "TinyCNN"),
    ("efficientnet_b0", "efficientnet_b0", "EfficientNet-B0"),
)
_TRAIN_COLOR: Final[str] = "#1f77b4"
_VALIDATION_COLOR: Final[str] = "#d1495b"
_SELECTED_COLOR: Final[str] = "#b7791f"


def _format_runtime_seconds(value: object) -> str:
    rounded_seconds = round(float(cast(str | SupportsFloat, value)))
    minutes, seconds = divmod(rounded_seconds, 60)
    return f"{minutes} min {seconds:02d} s"


@dataclass(frozen=True, slots=True)
class StageInterval:
    """Describe one stage interval on the global epoch axis.

    Parameters
    ----------
    name
        Maintained stage name.
    start
        First global epoch occupied by the stage.
    end
        Last global epoch occupied by the stage.
    """

    name: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class HistoryTimeline:
    """Store chronological history rows and their stage intervals.

    Parameters
    ----------
    dataframe
        Validated history with an added integer global_epoch column.
    intervals
        Ordered stage intervals; empty for a continuous non-staged history.
    """

    dataframe: pd.DataFrame
    intervals: tuple[StageInterval, ...]


@dataclass(frozen=True, slots=True)
class ModelEvaluation:
    """Store every maintained public view for one verified model.

    Parameters
    ----------
    key
        Stable model-selection value.
    display_name
        Human-readable model name.
    history
        Prepared chronological training history.
    metrics
        Compact verified test metrics.
    confusion_matrix
        Two-by-two Cover/Stego count matrix.
    roc_data
        Precomputed false-positive and true-positive rates.
    score_distribution
        Explicit score-bin edges and class counts.
    selected_epoch
        Local epoch of the selected validation state.
    selected_stage
        Fine-tuning stage of the selected state, if staged.
    selected_validation_weighted_auc
        Verified validation Weighted AUC at model selection.
    runtime_seconds
        Verified model runtime.
    """

    key: str
    display_name: str
    history: HistoryTimeline
    metrics: dict[str, int | float]
    confusion_matrix: np.ndarray
    roc_data: pd.DataFrame
    score_distribution: pd.DataFrame
    selected_epoch: int
    selected_stage: str | None
    selected_validation_weighted_auc: float
    runtime_seconds: float


@dataclass(frozen=True, slots=True)
class EvaluationResults:
    """Store one verified run's model evaluations and comparison table.

    Parameters
    ----------
    run_root
        Repository-relative resolved run directory.
    models
        Ordered TinyCNN and EfficientNet-B0 results.
    comparison
        Verified model-comparison rows.
    """

    run_root: Path
    models: tuple[ModelEvaluation, ...]
    comparison: pd.DataFrame

    def model(self, key: str) -> ModelEvaluation:
        """Return one model by its stable selection key.

        Parameters
        ----------
        key
            Model-selection value.

        Returns
        -------
        ModelEvaluation
            Matching model result.

        Raises
        ------
        KeyError
            If the run has no model with the requested key.
        """
        for result in self.models:
            if result.key == key:
                return result
        raise KeyError(f"Unknown evaluation model: {key}")


def _require_columns(dataframe: pd.DataFrame, required: set[str] | frozenset[str], context: str) -> None:
    missing = set(required) - set(dataframe.columns)
    if missing:
        raise ValueError(f"{context} is missing columns: {sorted(missing)}")


def _validated_local_epochs(dataframe: pd.DataFrame, context: str) -> pd.DataFrame:
    ordered = dataframe.sort_values("epoch", kind="stable").copy()
    epoch_values = cast(pd.Series, pd.to_numeric(cast(pd.Series, ordered["epoch"]), errors="raise"))
    epochs = epoch_values.astype(int)
    expected = list(range(1, len(ordered) + 1))
    if epochs.tolist() != expected:
        raise ValueError(f"{context} epochs must be unique and consecutive from one.")
    ordered["epoch"] = epochs
    return ordered


def prepare_history(
    dataframe: pd.DataFrame,
    *,
    stage_order: Sequence[str] | None = None,
) -> HistoryTimeline:
    """Create a continuous global epoch coordinate from retained history rows.

    Parameters
    ----------
    dataframe
        Training history containing loss, accuracy, and validation Weighted
        AUC columns.
    stage_order
        Required complete stage order for a non-cumulative staged history,
        or None for a normal continuous history.

    Returns
    -------
    HistoryTimeline
        Validated, chronologically ordered rows and exact stage intervals.

    Raises
    ------
    ValueError
        If columns, stages, or local epoch sequences violate the contract.
    """
    _require_columns(dataframe, _HISTORY_COLUMNS, "History")
    if dataframe.empty:
        raise ValueError("History must contain at least one epoch.")

    if stage_order is None:
        ordered = _validated_local_epochs(dataframe, "Continuous history")
        ordered["global_epoch"] = ordered["epoch"]
        return HistoryTimeline(dataframe=ordered.reset_index(drop=True), intervals=())

    _require_columns(dataframe, {"stage"}, "Staged history")
    maintained_order = tuple(stage_order)
    if not maintained_order or len(set(maintained_order)) != len(maintained_order):
        raise ValueError("stage_order must contain unique stage names.")
    observed = set(dataframe["stage"].astype(str))
    expected = set(maintained_order)
    if observed != expected:
        raise ValueError(
            "Staged history must contain exactly the maintained stages; "
            f"missing={sorted(expected - observed)}, unknown={sorted(observed - expected)}."
        )

    offset = 0
    segments: list[pd.DataFrame] = []
    intervals: list[StageInterval] = []
    for stage in maintained_order:
        raw_segment = dataframe.loc[dataframe["stage"].astype(str) == stage]
        segment = _validated_local_epochs(raw_segment, f"Stage {stage}")
        segment["stage"] = stage
        segment["global_epoch"] = np.arange(offset + 1, offset + len(segment) + 1)
        intervals.append(StageInterval(stage, offset + 1, offset + len(segment)))
        segments.append(segment)
        offset += len(segment)
    return HistoryTimeline(
        dataframe=pd.concat(segments, ignore_index=True),
        intervals=tuple(intervals),
    )


def _required_artifact(run_root: Path, relative: str) -> Path:
    path = run_root / relative
    if not path.is_file():
        raise FileNotFoundError(
            f"Required public evaluation artifact is missing: {relative}. "
            "Restore the Git-visible structured artifact for this verified run."
        )
    return path


def _load_metrics(path: Path) -> tuple[dict[str, int | float], np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Metrics must be a JSON object: {path}")
    required = {
        "confusion_matrix",
        "evaluation_pass_count",
        "sample_count",
        "test_accuracy",
        "test_loss",
        "test_weighted_auc",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Metrics are missing values: {sorted(missing)}")
    matrix = np.asarray(payload["confusion_matrix"], dtype=np.int64)
    if matrix.shape != (2, 2):
        raise ValueError(f"Expected a 2 by 2 confusion matrix in {path}.")
    metrics: dict[str, int | float] = {
        "evaluation_pass_count": int(cast(int | float, payload["evaluation_pass_count"])),
        "sample_count": int(cast(int | float, payload["sample_count"])),
        "test_accuracy": float(cast(int | float, payload["test_accuracy"])),
        "test_loss": float(cast(int | float, payload["test_loss"])),
        "test_weighted_auc": float(cast(int | float, payload["test_weighted_auc"])),
    }
    return metrics, matrix


def _load_roc(path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(path)
    _require_columns(dataframe, {"fpr", "tpr"}, "ROC data")
    result = dataframe.loc[:, ["fpr", "tpr"]].astype(float)
    if result.empty or not np.isfinite(result.to_numpy()).all():
        raise ValueError(f"ROC data must contain finite points: {path}")
    if ((result < 0) | (result > 1)).any(axis=None):
        raise ValueError(f"ROC rates must be within [0, 1]: {path}")
    return result


def _load_score_distribution(path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(path)
    required = ["bin_left", "bin_right", "cover_count", "stego_count"]
    _require_columns(dataframe, set(required), "Score distribution")
    result = dataframe.loc[:, required].copy()
    result = result.sort_values("bin_left", kind="stable").reset_index(drop=True)
    for column in ("bin_left", "bin_right"):
        values = cast(pd.Series, pd.to_numeric(cast(pd.Series, result[column]), errors="raise"))
        result[column] = values.astype(float)
    for column in ("cover_count", "stego_count"):
        values = cast(pd.Series, pd.to_numeric(cast(pd.Series, result[column]), errors="raise"))
        result[column] = values.astype(int)
    if result.empty or (result[["cover_count", "stego_count"]] < 0).any(axis=None):
        raise ValueError(f"Score distribution counts must be non-negative: {path}")
    left = result["bin_left"].to_numpy()
    right = result["bin_right"].to_numpy()
    if left[0] != 0.0 or right[-1] != 1.0 or not np.allclose(left[1:], right[:-1]):
        raise ValueError(f"Score distribution bins must cover [0, 1] without gaps: {path}")
    return result


def load_evaluation_results(
    run_id: str,
    *,
    paths: ProjectPaths | None = None,
) -> EvaluationResults:
    """Load all public result views for one repository-local verified run.

    Parameters
    ----------
    run_id
        Run-directory name under artifacts/alaska2.
    paths
        Optional project path contract for tests or alternate clones.

    Returns
    -------
    EvaluationResults
        Validated model results and comparison data.

    Raises
    ------
    FileNotFoundError
        If the run or any required Git-visible artifact is absent.
    ValueError
        If structured evidence violates its public schema.
    """
    project_paths = paths or default_paths()
    run_root = project_paths.artifacts / "alaska2" / run_id
    if not run_root.is_dir():
        raise FileNotFoundError(
            f"Public evaluation run is missing: artifacts/alaska2/{run_id}. "
            "Restore the Git-visible verified artifact directory."
        )
    comparison = pd.read_csv(_required_artifact(run_root, "comparison.csv"))
    comparison_required = {
        "model",
        "training_groups",
        "training_images",
        "best_validation_weighted_auc",
        "final_test_accuracy",
        "final_test_weighted_auc",
        "runtime_seconds",
    }
    _require_columns(comparison, comparison_required, "Model comparison")

    results: list[ModelEvaluation] = []
    for key, directory, display_name in _MODEL_LAYOUT:
        history_path = _required_artifact(run_root, f"{directory}/histories/training_history.csv")
        metrics_path = _required_artifact(run_root, f"{directory}/evaluation/test_metrics.json")
        roc_path = _required_artifact(run_root, f"{directory}/evaluation/roc_data.csv")
        score_path = _required_artifact(run_root, f"{directory}/evaluation/score_distribution.csv")

        raw_history = pd.read_csv(history_path)
        timeline = prepare_history(
            raw_history,
            stage_order=EFFICIENTNET_STAGE_ORDER if key == "efficientnet_b0" else None,
        )
        best_index = timeline.dataframe["val_wauc"].astype(float).idxmax()
        best_row = cast(pd.Series, timeline.dataframe.loc[best_index])
        selected_epoch = int(best_row["epoch"])
        selected_stage = str(best_row["stage"]) if key == "efficientnet_b0" else None
        selected_wauc = float(best_row["val_wauc"])

        comparison_rows = comparison.loc[comparison["model"] == display_name]
        if len(comparison_rows) != 1:
            raise ValueError(f"Comparison must contain exactly one {display_name} row.")
        comparison_row = comparison_rows.iloc[0]
        if not np.isclose(
            selected_wauc,
            float(comparison_row["best_validation_weighted_auc"]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"{display_name} history and comparison disagree on selected validation Weighted AUC.")

        metrics, matrix = _load_metrics(metrics_path)
        if not np.isclose(
            float(metrics["test_accuracy"]),
            float(comparison_row["final_test_accuracy"]),
            rtol=0.0,
            atol=1e-12,
        ) or not np.isclose(
            float(metrics["test_weighted_auc"]),
            float(comparison_row["final_test_weighted_auc"]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"{display_name} metrics and comparison values disagree.")

        results.append(
            ModelEvaluation(
                key=key,
                display_name=display_name,
                history=timeline,
                metrics=metrics,
                confusion_matrix=matrix,
                roc_data=_load_roc(roc_path),
                score_distribution=_load_score_distribution(score_path),
                selected_epoch=selected_epoch,
                selected_stage=selected_stage,
                selected_validation_weighted_auc=selected_wauc,
                runtime_seconds=float(comparison_row["runtime_seconds"]),
            )
        )
    return EvaluationResults(run_root=run_root, models=tuple(results), comparison=comparison)


def aggregate_score_distribution(
    y_probability: np.ndarray | Sequence[float],
    y_true: np.ndarray | Sequence[int | float],
    *,
    bin_edges: np.ndarray | Sequence[float] | None = None,
) -> pd.DataFrame:
    """Aggregate raw verified scores into explicit class-count bins.

    Parameters
    ----------
    y_probability
        One-dimensional Stego probabilities.
    y_true
        Aligned binary or four-class targets; nonzero values are Stego.
    bin_edges
        Strictly increasing edges spanning zero through one. Forty equal
        bins are used by default.

    Returns
    -------
    pandas.DataFrame
        Explicit left/right edges and Cover/Stego counts.

    Raises
    ------
    ValueError
        If arrays, probabilities, targets, or bin edges are invalid.
    """
    probabilities = np.asarray(y_probability, dtype=float).reshape(-1)
    targets = (np.asarray(y_true).reshape(-1) != 0).astype(np.int8)
    if probabilities.shape != targets.shape or probabilities.size == 0:
        raise ValueError("Probability and target arrays must be non-empty and aligned.")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be finite and within [0, 1].")
    edges = np.asarray(bin_edges if bin_edges is not None else np.linspace(0.0, 1.0, 41), dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or edges[0] != 0.0 or edges[-1] != 1.0 or not np.all(np.diff(edges) > 0):
        raise ValueError("bin_edges must increase strictly from 0 to 1.")
    cover_counts, _ = np.histogram(probabilities[targets == 0], bins=edges)
    stego_counts, _ = np.histogram(probabilities[targets == 1], bins=edges)
    return pd.DataFrame(
        {
            "bin_left": edges[:-1],
            "bin_right": edges[1:],
            "cover_count": cover_counts.astype(int),
            "stego_count": stego_counts.astype(int),
        }
    )


def _selected_global_epoch(
    timeline: HistoryTimeline,
    selected_epoch: int,
    selected_stage: str | None,
) -> tuple[int, float]:
    dataframe = timeline.dataframe
    mask = dataframe["epoch"].astype(int) == selected_epoch
    if timeline.intervals:
        if selected_stage is None:
            raise ValueError("A selected stage is required for staged history.")
        mask &= dataframe["stage"].astype(str) == selected_stage
    selected = dataframe.loc[mask]
    if len(selected) != 1:
        raise ValueError("Selected state must identify exactly one retained history row.")
    row = selected.iloc[0]
    return int(row["global_epoch"]), float(row["val_wauc"])


def plot_history(
    dataframe: pd.DataFrame,
    title: str = "Training history",
    *,
    selected_epoch: int | None = None,
    selected_stage: str | None = None,
    stage_order: Sequence[str] | None = None,
) -> Figure:
    """Plot loss, accuracy, and validation Weighted AUC chronologically.

    Parameters
    ----------
    dataframe
        Continuous or stage-labelled retained history.
    title
        Figure title.
    selected_epoch
        Local epoch of the selected validation state; the maximum validation
        Weighted AUC row is used when omitted.
    selected_stage
        Selected stage for a staged history.
    stage_order
        Complete chronological stage order, or None for continuous training.

    Returns
    -------
    matplotlib.figure.Figure
        Aligned three-panel history figure with the selected state marked.

    Raises
    ------
    ValueError
        If the history or selected-state identity is invalid.
    """
    timeline = prepare_history(dataframe, stage_order=stage_order)
    prepared = timeline.dataframe
    if selected_epoch is None:
        best_row = cast(pd.Series, prepared.loc[prepared["val_wauc"].astype(float).idxmax()])
        selected_epoch = int(best_row["epoch"])
        if timeline.intervals:
            selected_stage = str(best_row["stage"])
    selected_global, selected_wauc = _selected_global_epoch(timeline, selected_epoch, selected_stage)

    figure, axes = plt.subplots(1, 3, figsize=(18, 6.6), sharex=True)
    figure.suptitle(title, fontsize=15, y=0.99)
    segments = (
        [prepared.loc[prepared["stage"].astype(str) == interval.name] for interval in timeline.intervals]
        if timeline.intervals
        else [prepared]
    )
    for segment in segments:
        x = segment["global_epoch"].to_numpy()
        axes[0].plot(
            x,
            segment["train_loss"],
            color=_TRAIN_COLOR,
            linewidth=1.9,
            marker="o",
            markersize=3.2,
            gid="history-series",
        )
        axes[0].plot(
            x,
            segment["val_loss"],
            color=_VALIDATION_COLOR,
            linewidth=1.9,
            marker="o",
            markersize=3.2,
            gid="history-series",
        )
        axes[1].plot(
            x, segment["train_acc"], color=_TRAIN_COLOR, linewidth=1.9, marker="o", markersize=3.2, gid="history-series"
        )
        axes[1].plot(
            x,
            segment["val_acc"],
            color=_VALIDATION_COLOR,
            linewidth=1.9,
            marker="o",
            markersize=3.2,
            gid="history-series",
        )
        axes[2].plot(
            x,
            segment["val_wauc"],
            color=_VALIDATION_COLOR,
            linewidth=1.9,
            marker="o",
            markersize=3.2,
            gid="history-series",
        )

    if timeline.intervals:
        for interval_index, interval in enumerate(timeline.intervals):
            for axis in axes:
                if interval_index % 2:
                    axis.axvspan(
                        interval.start - 0.5,
                        interval.end + 0.5,
                        color="#64748b",
                        alpha=0.055,
                        zorder=0,
                    )
                if interval_index:
                    axis.axvline(
                        interval.start - 0.5,
                        color="#64748b",
                        linewidth=0.9,
                        linestyle="--",
                        alpha=0.75,
                        gid="stage-boundary",
                    )
                if axis is axes[0]:
                    axis.text(
                        (interval.start + interval.end) / 2,
                        0.975,
                        interval.name,
                        transform=axis.get_xaxis_transform(),
                        ha="center",
                        va="top",
                        fontsize=7.4,
                        rotation=35,
                        color="#334155",
                    )

    for axis in axes:
        axis.axvline(
            selected_global,
            color=_SELECTED_COLOR,
            linewidth=1.4,
            linestyle=":",
            alpha=0.9,
            gid="selected-state",
        )
        axis.set_xlabel("Global epoch" if timeline.intervals else "Epoch")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(alpha=0.22)
    axes[0].set_title("Loss")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("Accuracy")
    axes[1].set_ylabel("Accuracy")
    axes[2].set_title("Validation Weighted AUC")
    axes[2].set_ylabel("Weighted AUC")
    axes[2].scatter(
        [selected_global],
        [selected_wauc],
        s=62,
        color=_SELECTED_COLOR,
        edgecolor="white",
        linewidth=0.9,
        zorder=5,
        gid="selected-state-marker",
    )
    state_name = (
        f"{selected_stage}, stage epoch {selected_epoch}" if selected_stage is not None else f"epoch {selected_epoch}"
    )
    axes[2].text(
        0.03,
        0.04,
        f"Selected: {state_name}\nvalidation wAUC {selected_wauc:.6f}",
        transform=axes[2].transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="#713f12",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#fffbeb", "edgecolor": "#f59e0b", "alpha": 0.95},
    )
    axes[0].set_xlim(0.5, float(prepared["global_epoch"].max()) + 0.5)

    handles = [
        Line2D([0], [0], color=_TRAIN_COLOR, marker="o", linewidth=1.9, markersize=4, label="Train"),
        Line2D([0], [0], color=_VALIDATION_COLOR, marker="o", linewidth=1.9, markersize=4, label="Validation"),
        Line2D([0], [0], color=_SELECTED_COLOR, linestyle=":", linewidth=1.4, label="Selected state"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.005))
    figure.tight_layout(rect=(0.0, 0.07, 1.0, 0.91))
    return figure


def plot_confusion_matrix(
    matrix: np.ndarray,
    *,
    labels: tuple[str, str] = ("Cover", "Stego"),
    normalize: bool = False,
) -> Figure:
    """Plot a binary confusion matrix with optional row normalization.

    Parameters
    ----------
    matrix
        Two-by-two Cover/Stego count matrix.
    labels
        Axis labels ordered to match the matrix.
    normalize
        Whether each true-class row is normalized independently.

    Returns
    -------
    matplotlib.figure.Figure
        Annotated confusion-matrix figure.

    Raises
    ------
    ValueError
        If the input is not exactly two by two.
    """
    values = np.asarray(matrix, dtype=np.float64 if normalize else np.int64)
    if values.shape != (2, 2):
        raise ValueError("Expected a 2 by 2 confusion matrix.")
    if normalize:
        denominator = values.sum(axis=1, keepdims=True)
        values = np.divide(values, denominator, out=np.zeros_like(values), where=denominator != 0)
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(values, cmap="Blues")
    axis.set_xticks([0, 1], labels=labels)
    axis.set_yticks([0, 1], labels=labels)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Confusion matrix")
    maximum = float(values.max()) if values.size else 0.0
    for row in range(2):
        for column in range(2):
            value = values[row, column]
            text = f"{value:.2f}" if normalize else str(int(value))
            axis.text(
                column,
                row,
                text,
                ha="center",
                va="center",
                color="white" if maximum and value > maximum * 0.6 else "black",
            )
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    return figure


def plot_roc_curves(
    curves: Sequence[Mapping[str, object]],
    title: str = "ROC - ALASKA2 Weighted AUC",
) -> Figure:
    """Plot one or more precomputed ROC curves with ALASKA2 score bands.

    Parameters
    ----------
    curves
        Mappings containing fpr, tpr, wauc, and label values.
    title
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        Combined ROC figure.

    Raises
    ------
    KeyError
        If a curve mapping omits a required value.
    ValueError
        If a curve contains invalid or unaligned rate arrays.
    """
    figure, axis = plt.subplots(figsize=(10, 7))
    axis.axhspan(0.0, 0.4, color="limegreen", alpha=0.12)
    axis.axhspan(0.4, 1.0, color="lightcoral", alpha=0.12)
    for curve in curves:
        false_positive_rate = np.asarray(curve["fpr"], dtype=float).reshape(-1)
        true_positive_rate = np.asarray(curve["tpr"], dtype=float).reshape(-1)
        if false_positive_rate.shape != true_positive_rate.shape or false_positive_rate.size == 0:
            raise ValueError("ROC false-positive and true-positive rates must be non-empty and aligned.")
        weighted_auc = float(cast(float, curve["wauc"]))
        axis.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{curve['label']} (wAUC={weighted_auc:.3f})",
        )
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="False-positive rate",
        ylabel="True-positive rate",
        title=title,
    )
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    return figure


def plot_score_distribution(dataframe: pd.DataFrame) -> Figure:
    """Plot a compact class-count score distribution.

    Parameters
    ----------
    dataframe
        Explicit contiguous score-bin edges and Cover/Stego counts.

    Returns
    -------
    matplotlib.figure.Figure
        Overlaid step histograms reconstructed without raw predictions.

    Raises
    ------
    ValueError
        If the compact distribution schema or bins are invalid.
    """
    required = {"bin_left", "bin_right", "cover_count", "stego_count"}
    _require_columns(dataframe, required, "Score distribution")
    ordered = dataframe.sort_values("bin_left", kind="stable")
    left = ordered["bin_left"].to_numpy(dtype=float)
    right = ordered["bin_right"].to_numpy(dtype=float)
    if len(ordered) == 0 or left[0] != 0.0 or right[-1] != 1.0 or not np.allclose(left[1:], right[:-1]):
        raise ValueError("Score distribution bins must cover [0, 1] without gaps.")
    edges = np.concatenate([left, right[-1:]])
    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.stairs(
        ordered["cover_count"].to_numpy(dtype=int),
        edges,
        fill=True,
        alpha=0.42,
        linewidth=1.5,
        color=_TRAIN_COLOR,
        label="Cover",
    )
    axis.stairs(
        ordered["stego_count"].to_numpy(dtype=int),
        edges,
        fill=True,
        alpha=0.36,
        linewidth=1.5,
        color=_VALIDATION_COLOR,
        label="Stego",
    )
    axis.axvline(0.5, color="#475569", linestyle="--", linewidth=1.0, label="Decision threshold")
    axis.set(xlim=(0, 1), xlabel="P(Stego)", ylabel="Images", title="Binary model score distribution")
    axis.grid(axis="y", alpha=0.22)
    axis.legend()
    figure.tight_layout()
    return figure


def plot_score_histogram(
    y_probability: np.ndarray,
    y_true: np.ndarray,
    *,
    bins: int = 30,
) -> Figure:
    """Plot Cover and Stego probability distributions from aligned scores.

    Parameters
    ----------
    y_probability
        One-dimensional Stego probabilities.
    y_true
        Aligned binary or four-class targets.
    bins
        Positive histogram bin count.

    Returns
    -------
    matplotlib.figure.Figure
        Overlaid Cover and Stego score histograms.

    Raises
    ------
    ValueError
        If probability and target shapes differ or bins are invalid.
    """
    if bins <= 0:
        raise ValueError("bins must be positive.")
    distribution = aggregate_score_distribution(
        y_probability,
        y_true,
        bin_edges=np.linspace(0.0, 1.0, bins + 1),
    )
    return plot_score_distribution(distribution)


def metrics_table(result: ModelEvaluation) -> pd.DataFrame:
    """Build a compact numerical metric table for one model.

    Parameters
    ----------
    result
        Loaded verified model evaluation.

    Returns
    -------
    pandas.DataFrame
        Ordered metric labels and display-ready values.
    """
    selected_state = (
        f"{result.selected_stage}, epoch {result.selected_epoch}"
        if result.selected_stage is not None
        else f"epoch {result.selected_epoch}"
    )
    minutes, seconds = divmod(round(result.runtime_seconds), 60)
    rows: list[tuple[str, object]] = [
        ("Selected validation state", selected_state),
        ("Validation Weighted AUC", f"{result.selected_validation_weighted_auc:.6f}"),
        ("Test loss", f"{float(result.metrics['test_loss']):.6f}"),
        ("Test accuracy", f"{float(result.metrics['test_accuracy']):.6f}"),
        ("Test Weighted AUC", f"{float(result.metrics['test_weighted_auc']):.6f}"),
        ("Test images", f"{int(result.metrics['sample_count']):,}"),
        ("Recorded evaluation passes", int(result.metrics["evaluation_pass_count"])),
        ("Runtime", f"{minutes} min {seconds:02d} s"),
    ]
    return pd.DataFrame(
        {
            "Metric": [metric for metric, _ in rows],
            result.display_name: [value for _, value in rows],
        }
    ).set_index("Metric")


def comparison_table(results: EvaluationResults) -> pd.DataFrame:
    """Build the concise public model comparison without local checkpoint paths.

    Parameters
    ----------
    results
        Loaded verified evaluation run.

    Returns
    -------
    pandas.DataFrame
        Ordered display columns for both maintained models.
    """
    columns = [
        "model",
        "training_groups",
        "training_images",
        "best_validation_weighted_auc",
        "final_test_accuracy",
        "final_test_weighted_auc",
        "runtime_seconds",
    ]
    table = results.comparison.loc[:, columns].copy()
    table["runtime"] = table["runtime_seconds"].map(_format_runtime_seconds)
    table = table.drop(columns="runtime_seconds")
    return table.rename(
        columns={
            "model": "Model",
            "training_groups": "Training groups",
            "training_images": "Training images",
            "best_validation_weighted_auc": "Best validation Weighted AUC",
            "final_test_accuracy": "Test accuracy",
            "final_test_weighted_auc": "Test Weighted AUC",
            "runtime": "Runtime",
        }
    ).set_index("Model")
