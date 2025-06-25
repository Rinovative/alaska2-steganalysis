from __future__ import annotations

import numpy as np
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)


def weighted_auc(y_true, y_pred):
    """
    ALASKA2 – Weighted AUC
    • Binär  : y_true ∈ {0,1}, y_pred ∈ ℝ
    • 4-Klassig: y_true ∈ {0,1,2,3}, y_pred ∈ ℝ^{N×4}  (Cover=0, Stego=1–3)
    Gewichtung: TPR-Abschnitt 0-0.4 doppelt, Rest einfach.
    """
    # ── 4-Klasseneingabe → P(Stego) ──────────────────────────────────────────────
    if y_pred.ndim == 2:  # erwartet Form (N,4)
        if not np.allclose(y_pred.sum(1), 1.0, atol=1e-3):
            y_pred = softmax(y_pred, axis=1)  # Logits → Softmax
        y_pred = y_pred[:, 1:].sum(1)  # Stego-Wahrscheinlichkeit
    y_true = (y_true != 0).astype(int)  # Cover=0, Stego=1

    # ── ROC-Kurve ───────────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    fpr = np.concatenate(([0.0], fpr, [1.0]))
    tpr = np.concatenate(([0.0], tpr, [1.0]))

    # ── FPR bei TPR = 0.4 (lineare Interpolation) ──────────────────────────────
    fpr_04 = np.interp(0.4, tpr, fpr)

    # Segment 1: 0 ≤ TPR ≤ 0.4  (Gewicht 2)
    mask1 = tpr <= 0.4
    auc1 = np.trapz(np.concatenate((tpr[mask1], [0.4])), np.concatenate((fpr[mask1], [fpr_04])))

    # Segment 2: 0.4 ≤ TPR ≤ 1  (Gewicht 1)
    mask2 = tpr >= 0.4
    auc2 = np.trapz(np.concatenate(([0.4], tpr[mask2])), np.concatenate(([fpr_04], fpr[mask2])))

    return (2 * auc1 + auc2) / 3


# ────────────────────────────────────────────────────────────────
def classification_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Liefert Accuracy, Precision, Recall, F1 (macro).
    """
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# ────────────────────────────────────────────────────────────────
def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray, *, binary: bool = False) -> np.ndarray:
    """
    Liefert Konfusionsmatrix (k×k oder 2×2 wenn `binary=True`).

    Wenn `binary=True`, werden alle Werte >0 als Stego (1) gezählt.
    """
    if binary:
        y_true = (y_true > 0).astype(int)
        y_pred = (y_pred > 0).astype(int)
    return confusion_matrix(y_true, y_pred)


def roc_data(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    Gibt dict mit fpr, tpr, thresholds zurück für plot_roc_curves(...)

    Ideal als Input für z. B.:
        {"label": "ModelX", "fpr": ..., "tpr": ..., "wauc": ...}
    """
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}


def predict_binary(y_prob: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Wandelt Sigmoid-Probabilitäten in binäre Labels um.
    """
    return (y_prob > threshold).astype(int)


def evaluate_model(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5, *, label: str = "Model") -> dict:
    y_pred = predict_binary(y_prob, threshold)
    return {
        "label": label,
        **classification_scores(y_true, y_pred),
        "wauc": weighted_auc(y_true, y_prob),
    }
