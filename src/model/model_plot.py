from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision.transforms.functional import to_pil_image


# ────────────────────────────────────────────────────────────────
def plot_history(df: pd.DataFrame, title="Training History") -> plt.Figure:
    """
    Zeigt Loss, Accuracy **und Weighted-AUC**.

    Erwartet Spalten: epoch, train_loss, train_acc, val_loss,
                      val_acc, val_wauc
    """
    epochs = df["epoch"]

    fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharex=True)
    fig.suptitle(title, fontsize=15)

    # ❶ Loss
    ax = axes[0]
    ax.plot(epochs, df["train_loss"], label="Train", lw=2)
    ax.plot(epochs, df["val_loss"], label="Val", lw=2)
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.grid(alpha=0.3)
    ax.legend()

    # ❷ Accuracy
    ax = axes[1]
    ax.plot(epochs, df["train_acc"], label="Train", lw=2)
    ax.plot(epochs, df["val_acc"], label="Val", lw=2)
    ax.set_title("Accuracy")
    ax.set_xlabel("Epoch")
    ax.grid(alpha=0.3)
    ax.legend()

    # ❸ Weighted AUC
    ax = axes[2]
    ax.plot(epochs, df["val_wauc"], "o-", label="Val wAUC", lw=2)
    ax.set_title("Weighted AUC")
    ax.set_xlabel("Epoch")
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ────────────────────────────────────────────────────────────────
def plot_confmat(
    cm: np.ndarray,
    *,
    labels_true: list[str],
    labels_pred: list[str],
    title: str = "Confusion Matrix",
    normalize: bool = False,
    figsize: tuple[int, int] | None = None,
    cmap: str = "Blues",
) -> plt.Figure:
    """
    Heatmap-Darstellung einer Konfusionsmatrix (auch nicht-quadratisch).
    """
    if normalize:
        cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1e-9)

    n_rows, n_cols = cm.shape
    figsize = figsize or (6 + 1.2 * n_cols, 6 + 0.4 * n_rows)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, cmap=cmap)

    ax.set_xticks(range(n_cols), labels=labels_pred, rotation=45, ha="right")
    ax.set_yticks(range(n_rows), labels=labels_true)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    # Zahlen eintragen
    for i in range(n_rows):
        for j in range(n_cols):
            val = cm[i, j]
            txt = f"{val:.2f}" if normalize else f"{int(val)}"
            ax.text(j, i, txt, ha="center", va="center", color="white" if val > cm.max() * 0.6 else "black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig


# ────────────────────────────────────────────────────────────────
def plot_roc_curves(curves: list[dict], title="ROC – Weighted AUC") -> plt.Figure:
    """
    Zeichnet ROC-Kurven, füllt Hintergrundbereiche und gibt die Figure zurück.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Hintergrund: grün unter 0.4 TPR, rot darüber
    ax.fill_between([0, 1], [0, 0], [0.4, 0.4], color="limegreen", alpha=0.15)
    ax.fill_between([0, 1], [0.4, 0.4], [1, 1], color="lightcoral", alpha=0.15)

    for c in curves:
        ax.plot(c["fpr"], c["tpr"], label=f'{c["label"]} (wAUC={c["wauc"]:.3f})')

    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(title)
    ax.legend()
    ax.grid()
    return fig


# ────────────────────────────────────────────────────────────────
def plot_score_histogram(y_prob, y_true, bins=30, title="Sigmoid Scores (P[Stego])") -> plt.Figure:
    """
    Überlagertes Histogramm der Stego-Wahrscheinlichkeiten.
    """
    y_prob = np.array(y_prob).squeeze()
    y_true = np.array(y_true).squeeze()

    fig, ax = plt.subplots()
    ax.hist(y_prob[y_true == 0], bins, alpha=0.6, label="Cover")
    ax.hist(y_prob[y_true == 1], bins, alpha=0.6, label="Stego")
    ax.set_xlabel("P(Stego)")
    ax.set_ylabel("# Bilder")
    ax.set_title(title)
    ax.legend()
    return fig


# ────────────────────────────────────────────────────────────────
def plot_sample_scores(
    image_tensor: torch.Tensor,
    prob_vec: torch.Tensor | list[float],
    class_labels: list[str],
    *,
    true_label: str | None = None,
    figsize: tuple[int, int] = (5, 3),
) -> plt.Figure:
    """
    Zeigt ein Bild und daneben Balken mit Klass-Wahrscheinlichkeiten.
    """
    probs = torch.as_tensor(prob_vec).float().cpu().numpy()
    assert len(probs) == len(class_labels), "len(prob_vec) ≠ len(labels)"

    fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1, 1.4]})

    # Bild
    img = to_pil_image(image_tensor.cpu())
    ax_img.imshow(img)
    ax_img.axis("off")

    # Balkendiagramm
    y_pos = list(range(len(class_labels)))
    bars = ax_bar.barh(y_pos, probs)
    if true_label in class_labels:
        bars[class_labels.index(true_label)].set_color("orange")

    ax_bar.set_yticks(y_pos, class_labels)
    ax_bar.set_xlim(0, 1)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Probability")
    ax_bar.set_title("Soft-max Scores")
    for i, v in enumerate(probs):
        ax_bar.text(v + 0.03, i, f"{v:.2f}", va="center")

    plt.tight_layout()
    return fig
