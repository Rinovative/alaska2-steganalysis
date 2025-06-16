from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix, precision_recall_fscore_support, accuracy_score

# ────────────────────────────────────────────────────────────────
def weighted_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    ALASKA2-spezifische Weighted AUC.

    Gewichtung: TPR-Bereich [0, 0.4] doppelt so stark wie [0.4, 1].
    """
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights        = [2, 1]

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = 0.0
    for i in range(len(weights)):
        # Segmentgrenzen
        y_min, y_max = tpr_thresholds[i], tpr_thresholds[i + 1]
        mask = (tpr >= y_min) & (tpr <= y_max)
        # Segmentfläche (Trapezregel)
        seg_fpr = np.concatenate(([y_min], fpr[mask], [y_max]))
        seg_tpr = np.concatenate(([y_min], tpr[mask], [y_max]))
        seg_area = np.trapz(seg_tpr, seg_fpr)
        auc += weights[i] * seg_area
    # Normieren: Sum(weights) = 3 ⇒ teilen
    return auc / sum(weights)


# ────────────────────────────────────────────────────────────────
def classification_scores(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """
    Liefert Accuracy, Precision, Recall, F1 (macro).
    """
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# ────────────────────────────────────────────────────────────────
def confusion_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    binary: bool = False
) -> np.ndarray:
    """
    Liefert Konfusionsmatrix (k×k oder 2×2 wenn `binary=True`).

    Wenn `binary=True`, werden alle Werte >0 als Stego (1) gezählt.
    """
    if binary:
        y_true = (y_true > 0).astype(int)
        y_pred = (y_pred > 0).astype(int)
    return confusion_matrix(y_true, y_pred)
