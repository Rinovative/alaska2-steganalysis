"""
===============================================================================
test_metrics.py
===============================================================================
Verify golden ALASKA2 weighted-AUC values and error handling.

Responsibilities:
  - Compare binary and four-class score paths against deterministic
    expectations.
  - Reject invalid shapes, missing classes, and non-finite values.

Design principles:
  - Golden arrays make score regressions immediately visible.
  - Tests exercise the public metric contract without model inference.

Boundaries:
  - Plotting and DataLoader evaluation belong to other test modules.
  - No competition data or external service is required.
===============================================================================
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.evaluation_metrics import weighted_auc


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        (np.array([0.1, 0.2, 0.8, 0.9]), 1.0),
        (np.array([0.9, 0.8, 0.2, 0.1]), 0.0),
        (np.array([0.5, 0.5, 0.5, 0.5]), 0.5),
    ],
)
def test_binary_golden_rankings(scores: np.ndarray, expected: float) -> None:
    targets = np.array([0, 0, 1, 1])
    assert weighted_auc(targets, scores) == pytest.approx(expected)


def test_deterministic_random_golden_value() -> None:
    targets = np.tile([0, 1], 50)
    scores = np.random.default_rng(42).random(100)
    assert weighted_auc(targets, scores) == pytest.approx(0.5974285714285714)


def test_four_class_probability_and_logit_aggregation() -> None:
    targets = np.array([0, 0, 1, 2, 3, 3])
    probabilities = np.array(
        [
            [0.9, 0.04, 0.03, 0.03],
            [0.8, 0.05, 0.10, 0.05],
            [0.1, 0.8, 0.05, 0.05],
            [0.2, 0.1, 0.6, 0.1],
            [0.05, 0.05, 0.10, 0.8],
            [0.1, 0.1, 0.1, 0.7],
        ]
    )
    assert weighted_auc(targets, probabilities) == pytest.approx(1.0)
    assert weighted_auc(targets, np.log(probabilities)) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("targets", "scores", "message"),
    [
        (np.array([0, 0]), np.array([0.1, 0.2]), "both Cover and Stego"),
        (np.array([0, 1]), np.array([0.1]), "lengths differ"),
        (np.array([0, 1]), np.zeros((2, 3)), "shape"),
        (np.array([0, 1]), np.array([0.1, np.nan]), "finite"),
    ],
)
def test_metric_errors_are_actionable(targets: np.ndarray, scores: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        weighted_auc(targets, scores)
