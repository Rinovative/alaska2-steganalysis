from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)


# ────────────────────────────────────────────────────────────────
def _safe_softmax(logits: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    Mini-Softmax in NumPy – fällt zurück, falls die Eingabe noch Logits sind.
    """
    z = logits - logits.max(axis=axis, keepdims=True)  # stab.
    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=axis, keepdims=True)


def weighted_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    ALASKA2-spezifische Weighted AUC.

    Gewichtung: TPR-Bereich [0, 0.4] doppelt so stark wie [0.4, 1].

    Akzeptiert:
      • binär:  y_true ∈ {0,1},     y_pred ∈ ℝ   (P(Stego))
      • 4-Klassig: y_true ∈ {0,1,2,3}, y_pred ∈ ℝ^{N×4} (Logits o. Softmax)

    Kollabiert intern automatisch auf Cover vs. Stego.
    """
    # ---------- Auto-Kollaps auf binär ----------
    if y_pred.ndim == 2 and y_pred.shape[1] == 4:
        # Logits → Softmax, falls Zeilensumme ≠ 1
        if not np.allclose(y_pred.sum(1), 1.0, atol=1e-3):
            probs = _safe_softmax(y_pred, axis=1)
        else:
            probs = y_pred
        y_pred = probs[:, 1:].sum(1)  # P(Stego)

    # y_true 0/1 erstellen (Cover=0, Stego=1)
    if y_true.max() > 1 or y_true.min() < 0:
        y_true = (y_true != 0).astype(int)

    # ---------- Weighted AUC wie bisher ----------
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2, 1]

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc = 0.0
    for i, w in enumerate(weights):
        y_min, y_max = tpr_thresholds[i], tpr_thresholds[i + 1]
        mask = (tpr >= y_min) & (tpr <= y_max)
        seg_fpr = np.concatenate(([y_min], fpr[mask], [y_max]))
        seg_tpr = np.concatenate(([y_min], tpr[mask], [y_max]))
        auc += w * np.trapz(seg_tpr, seg_fpr)

    return auc / sum(weights)


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
