"""
===============================================================================
evaluation_metrics.py
===============================================================================
Validated ALASKA2 and classification metrics.

Responsibilities:
  - Implement the ALASKA2 weighted AUC from its published piecewise ROC
    definition.
  - Convert four-class Cover/stego scores into a binary Stego probability.
  - Validate shapes, finite values, class presence, and decision thresholds.

Design principles:
  - The competition score is computed by clipping the piecewise-linear ROC
    ordinate to each weighted TPR band and integrating with NumPy's non-
    deprecated trapezoid API.

Boundaries:
  - This module evaluates already-collected arrays and performs no model
    inference.

Notes:
  - Reference definition: https://www.kaggle.com/competitions/alaska2-image-
    steganalysis/overview/evaluation TPR bands [0.0, 0.4, 1.0] have weights
    [2.0, 1.0] and normalization 1.4.
===============================================================================
"""

from __future__ import annotations

from typing import Final

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_curve

__all__ = [
    "classification_scores",
    "confusion_counts",
    "predict_binary",
    "roc_data",
    "weighted_auc",
]

_TPR_BOUNDS: Final[np.ndarray] = np.array([0.0, 0.4, 1.0], dtype=np.float64)
_WEIGHTS: Final[np.ndarray] = np.array([2.0, 1.0], dtype=np.float64)
_NORMALIZATION: Final[float] = float(np.dot(np.diff(_TPR_BOUNDS), _WEIGHTS))


def _as_binary_targets(y_true: np.ndarray) -> np.ndarray:
    targets = np.asarray(y_true).reshape(-1)
    if targets.size == 0:
        raise ValueError("y_true must not be empty.")
    if not np.isfinite(targets).all():
        raise ValueError("y_true contains non-finite values.")
    binary = (targets != 0).astype(np.int8)
    if np.unique(binary).size != 2:
        raise ValueError("Weighted AUC requires both Cover and Stego targets.")
    return binary


def _stego_scores(y_score: np.ndarray) -> np.ndarray:
    scores = np.asarray(y_score, dtype=np.float64)
    if scores.ndim == 1:
        result = scores
    elif scores.ndim == 2 and scores.shape[1] == 4:
        is_probability = np.all((scores >= 0.0) & (scores <= 1.0)) and np.allclose(scores.sum(axis=1), 1.0, atol=1e-6)
        probabilities = scores
        if not is_probability:
            shifted = scores - scores.max(axis=1, keepdims=True)
            exponentials = np.exp(shifted)
            probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
        result = probabilities[:, 1:].sum(axis=1)
    else:
        raise ValueError("y_score must have shape [N] or [N, 4].")
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError("y_score must contain finite values.")
    return result


def weighted_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute the normalized ALASKA2 weighted area under the ROC curve.

    Parameters
    ----------
    y_true
        Binary targets or four-class targets with Cover encoded as zero.
    y_score
        Stego scores shaped ``[N]`` or four-class probabilities/logits shaped ``[N, 4]``.

    Returns
    -------
    float
        Normalized competition weighted AUC in the interval from zero to one.

    Raises
    ------
    ValueError
        If arrays are empty, non-finite, mismatched, malformed, or lack both binary classes.

    Notes
    -----
    Four-class scores are reduced to total Stego probability before ROC integration.
    """
    targets = _as_binary_targets(y_true)
    scores = _stego_scores(y_score)
    if targets.shape[0] != scores.shape[0]:
        raise ValueError("y_true and y_score lengths differ.")

    false_positive_rate, true_positive_rate, _ = roc_curve(
        targets,
        scores,
        drop_intermediate=False,
    )
    total = 0.0
    for lower, upper, weight in zip(_TPR_BOUNDS[:-1], _TPR_BOUNDS[1:], _WEIGHTS, strict=True):
        band_height = np.clip(true_positive_rate - lower, 0.0, upper - lower)
        total += float(weight * np.trapezoid(band_height, false_positive_rate))
    return total / _NORMALIZATION


def predict_binary(y_score: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert Stego scores into binary Cover or Stego decisions.

    Parameters
    ----------
    y_score
        One-dimensional Stego scores or four-class probabilities/logits.
    threshold
        Inclusive Stego decision threshold in ``[0, 1]``.

    Returns
    -------
    numpy.ndarray
        One-dimensional ``int8`` binary decision array.

    Raises
    ------
    ValueError
        If the threshold or score shape and values are invalid.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1].")
    return (_stego_scores(y_score) >= threshold).astype(np.int8)


def classification_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute accuracy and macro precision, recall, and F1.

    Parameters
    ----------
    y_true
        Binary or four-class targets reduced to Cover versus Stego.
    y_pred
        Binary predictions aligned with the targets.

    Returns
    -------
    dict[str, float]
        Accuracy, precision, recall, and F1 values.

    Raises
    ------
    ValueError
        If arrays mismatch, contain invalid targets, or lack both target classes.
    """
    targets = _as_binary_targets(y_true)
    predictions = np.asarray(y_pred).reshape(-1)
    if targets.shape != predictions.shape:
        raise ValueError("y_true and y_pred shapes differ.")
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets,
        predictions,
        average="macro",
        zero_division="warn",
    )
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Build a stable two-by-two Cover/Stego confusion matrix.

    Parameters
    ----------
    y_true
        Binary or four-class targets reduced to Cover versus Stego.
    y_pred
        Binary predictions aligned with the targets.

    Returns
    -------
    numpy.ndarray
        Integer confusion matrix ordered as Cover then Stego.

    Raises
    ------
    ValueError
        If target or prediction arrays are invalid or mismatched.
    """
    targets = _as_binary_targets(y_true)
    predictions = np.asarray(y_pred).reshape(-1)
    return confusion_matrix(targets, predictions, labels=[0, 1])


def roc_data(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, np.ndarray]:
    """Collect the complete binary ROC coordinates and decision thresholds.

    Parameters
    ----------
    y_true
        Binary or four-class targets reduced to Cover versus Stego.
    y_score
        One-dimensional Stego scores or four-class probabilities/logits.

    Returns
    -------
    dict[str, numpy.ndarray]
        False-positive rates, true-positive rates, and thresholds.

    Raises
    ------
    ValueError
        If arrays are invalid, mismatched, or lack both target classes.
    """
    targets = _as_binary_targets(y_true)
    scores = _stego_scores(y_score)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        targets,
        scores,
        drop_intermediate=False,
    )
    return {
        "fpr": false_positive_rate,
        "tpr": true_positive_rate,
        "thresholds": thresholds,
    }
