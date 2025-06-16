from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchvision.transforms.functional import to_pil_image


# ────────────────────────────────────────────────────────────────
def plot_history(hist_df: pd.DataFrame, title="Training History"):
    """
    Zeigt Loss, Accuracy **und Weighted-AUC**.

    Erwartet Spalten: epoch, train_loss, train_acc, val_loss,
                      val_acc, val_wauc
    """
    epochs = hist_df["epoch"]

    plt.figure(figsize=(18, 12))

    # ❶ Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, hist_df["train_loss"], label="Train")
    plt.plot(epochs, hist_df["val_loss"],   label="Val")
    plt.title("Loss");  plt.xlabel("Epoch"); plt.grid(); plt.legend()

    # ❷ Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(epochs, hist_df["train_acc"], label="Train")
    plt.plot(epochs, hist_df["val_acc"],   label="Val")
    plt.title("Accuracy"); plt.xlabel("Epoch"); plt.grid(); plt.legend()

    # ❸ Weighted AUC
    plt.subplot(2, 2, 3)
    plt.plot(epochs, hist_df["val_wauc"], 'd-', label="Val wAUC")
    plt.title("Weighted AUC"); plt.xlabel("Epoch"); plt.grid(); plt.legend()

    plt.suptitle(title)
    plt.tight_layout(); plt.show()


# ────────────────────────────────────────────────────────────────
def plot_confmat(
    cm: np.ndarray,
    labels: list[str],
    title: str = "Confusion Matrix",
    *,
    normalize: bool = False,
    figsize: tuple[int, int] | None = None,
    cmap: str = "Blues",
) -> None:
    """
    Heatmap-Darstellung einer Konfusionsmatrix.

    Args:
        cm        : Raw counts (ndarray).
        labels    : Achsen-Beschriftungen.
        title     : Plot-Titel.
        normalize : True → Prozentwerte zeilenweise.
        figsize   : Figure-Größe; auto falls None.
        cmap      : Matplotlib-Colormap.
    """
    if normalize:
        cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1e-9)

    n = cm.shape[0]
    figsize = figsize or (4 + 1.2 * n, 4 + 0.2 * n)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, cmap=cmap)

    ax.set_xticks(range(n), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(n), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    # Zahlen eintragen
    for i in range(n):
        for j in range(n):
            val = cm[i, j]
            txt = f"{val:.2f}" if normalize else f"{int(val)}"
            ax.text(j, i, txt,
                    ha="center", va="center",
                    color="white" if val > cm.max() * 0.6 else "black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

# ────────────────────────────────────────────────────────────────
def plot_roc_curves(curves: list[dict], title="ROC – Weighted AUC"):
    """
    curves = [
        {"label":"TinyCNN", "fpr":fpr1, "tpr":tpr1, "wauc":0.86},
        {"label":"EffNet",  "fpr":fpr2, "tpr":tpr2, "wauc":0.92},
        …
    ]
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,12))

    # ALASKA-Hintergrund – grün unter 0.4 TPR, rot darüber
    plt.fill_between([0, 1], [0, 0.4], [0.4, 0.4],
                     color="limegreen", alpha=.15)
    plt.fill_between([0, 1], [0.4, 0.4], [1, 1],
                     color="lightcoral", alpha=.15)

    for c in curves:
        plt.plot(c["fpr"], c["tpr"],
                 label=f'{c["label"]} (wAUC={c["wauc"]:.3f})')

    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(title); plt.legend(); plt.grid(); plt.show()

def plot_score_histogram(y_prob, y_true, bins=30, title="Softmax Scores"):
    """
    Überlagertes Histogramm der Stego-Wahrscheinlichkeiten.
    """
    plt.hist(y_prob[y_true == 0], bins, alpha=.6, label="Cover")
    plt.hist(y_prob[y_true == 1], bins, alpha=.6, label="Stego")
    plt.xlabel("P(Stego)"); plt.ylabel("# Bilder")
    plt.title(title); plt.legend(); plt.show()


def plot_sample_scores(
    image_tensor: torch.Tensor,
    prob_vec: torch.Tensor | list[float],
    class_labels: list[str],
    *,
    true_label: str | None = None,
    figsize: tuple[int, int] = (5, 3),
) -> None:
    """
    Zeigt ein Bild und daneben Balken mit Klass-Wahrscheinlichkeiten.

    Args:
        image_tensor : Tensor [C,H,W] (YCbCr oder RGB → wird als PIL angezeigt)
        prob_vec     : 1-D Tensor/List mit Soft-max-Scores (z. B. len==4)
        class_labels : Beschriftungen (z. B. ["Cover","JMiPOD","JUNIWARD","UERD"])
        true_label   : Optional – hebt die GT-Klasse farblich hervor
    """
    probs = torch.as_tensor(prob_vec).float().cpu().numpy()
    assert len(probs) == len(class_labels), "len(prob_vec) ≠ len(labels)"

    fig, ax = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1, 1.4]})

    # 1️⃣  Bild links
    img = to_pil_image(image_tensor.cpu())
    ax[0].imshow(img); ax[0].axis("off")

    # 2️⃣  Balken rechts
    y_pos = list(range(len(class_labels)))
    barlist = ax[1].barh(y_pos, probs, color="steelblue")
    if true_label is not None and true_label in class_labels:
        barlist[class_labels.index(true_label)].set_color("orange")

    ax[1].set_yticks(y_pos, class_labels)
    ax[1].set_xlim(0, 1)
    ax[1].invert_yaxis()            # oberste Klasse oben
    ax[1].set_xlabel("Probability")
    ax[1].set_title("Soft-max Scores")
    for i, v in enumerate(probs):
        ax[1].text(v + 0.03, i, f"{v:.2f}", va="center")

    plt.tight_layout()
    plt.show()
