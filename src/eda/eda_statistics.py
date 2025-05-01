from pathlib import Path

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from IPython.display import clear_output
from PIL import Image
from scipy.stats import zscore


def plot_image_mean_distribution(df: pd.DataFrame, dataset_name: str = "") -> plt.Figure:
    """
    Zeigt die Verteilung der mittleren Pixelwerte:
    - Stripplot über alle Bilder
    - Boxplot über nur die Extremfälle (oberes/unteres Quantil)

    Args:
        df (pd.DataFrame): DataFrame mit Spalten 'path' und 'label_name'.
        dataset_name (str): Optionaler Zusatz für den Plot-Titel.

    Returns:
        plt.Figure: Matplotlib-Figur mit beiden Visualisierungen.
    """
    df = df.copy()
    df["image_mean"] = df["path"].apply(lambda p: plt.imread(p).mean())

    # 5. und 95. Perzentil berechnen
    lower, upper = df["image_mean"].quantile([0.05, 0.95])
    df_extreme = df[(df["image_mean"] <= lower) | (df["image_mean"] >= upper)]

    # Zwei Unterplots: Stripplot (alle) und Boxplot (nur Extreme)
    fig, axs = plt.subplots(2, 1, figsize=(8, 12), sharex=True, gridspec_kw={"height_ratios": [1, 1]})

    # Stripplot: alle
    sns.boxplot(data=df, x="label_name", y="image_mean", ax=axs[0])
    axs[0].set_title(f"Verteilung aller mittleren Pixelwerte – {dataset_name}")
    axs[0].set_ylabel("Mittlerer Pixelwert")
    axs[0].grid(True, linestyle="--", alpha=0.5)

    # Boxplot: nur Extreme
    sns.boxplot(data=df_extreme, x="label_name", y="image_mean", ax=axs[1])
    axs[1].set_title("Extremwerte (unterstes & oberstes 5%-Quantil)")
    axs[1].set_xlabel("Klasse")
    axs[1].set_ylabel("Mittlerer Pixelwert")
    axs[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    return fig


def plot_kde(df: pd.DataFrame, dataset_name: str = "") -> plt.Figure:
    """
    Zeigt KDE-Kurven der mittleren Helligkeit (Y), Blauabweichung (Cb) und Rotabweichung (Cr) pro Bild und Klasse.
    Jeder Kanal wird in einem separaten Subplot dargestellt, inklusive eigener Legende.
    """
    channels = ["Y", "Cb", "Cr"]
    fig, axes = plt.subplots(3, 1, figsize=(8, 12))

    for idx, channel in enumerate(channels):
        ax = axes[idx]
        for cls in df["label_name"].cat.categories:
            subset = df[df["label_name"] == cls].sample(n=min(50, len(df)), random_state=42)
            values = []

            for path in subset["path"]:
                try:
                    img = Image.open(path).convert("YCbCr")
                    arr = np.array(img)
                    if channel == "Y":
                        values.append(arr[:, :, 0].mean())
                    elif channel == "Cb":
                        values.append(arr[:, :, 1].mean())
                    elif channel == "Cr":
                        values.append(arr[:, :, 2].mean())
                except Exception:
                    continue

            sns.kdeplot(values, clip=(0, 255), label=cls, ax=ax, linewidth=2, alpha=0.7)

        ax.set_title(f"KDE - Mittlerer {channel}-Kanal - {dataset_name}")
        ax.set_ylabel("Dichte")
        ax.grid(True, linestyle="--", alpha=0.5)
        if idx == 2:
            ax.set_xlabel("Mittlerer Kanalwert")
        else:
            ax.set_xlabel("")

    for ax in axes:
        ax.legend(title="Klasse")

    fig.tight_layout()
    return fig


def plot_channel_boxplots(df: pd.DataFrame, dataset_name: str = "") -> plt.Figure:
    """
    Zeigt Boxplots der mittleren Y, Cb und Cr-Kanalwerte gruppiert nach Klasse.
    """
    df_stats = []

    for cls in df["label_name"].cat.categories:
        subset = df[df["label_name"] == cls].sample(n=min(50, len(df)), random_state=42)

        for path in subset["path"]:
            try:
                img = Image.open(path).convert("YCbCr")
                arr = np.array(img)
                y_mean = arr[:, :, 0].mean()
                cb_mean = arr[:, :, 1].mean()
                cr_mean = arr[:, :, 2].mean()
                df_stats.append({"label": cls, "Y": y_mean, "Cb": cb_mean, "Cr": cr_mean})
            except Exception:
                continue

    df_long = pd.DataFrame(df_stats).melt(id_vars="label", var_name="channel", value_name="value")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df_long, x="channel", y="value", hue="label", ax=ax)
    ax.set_title(f"Boxplots mittlerer YCbCr-Kanalwerte – {dataset_name}")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


def plot_ycbcr_scatter(df: pd.DataFrame, dataset_name: str = "") -> px.scatter_3d:
    """
    Interaktiver 3D-Plot der mittleren Y, Cb, Cr-Werte pro Bild mit Plotly.

    Args:
        df (pd.DataFrame): DataFrame mit 'path' und 'label_name'
        dataset_name (str): Optionaler Titel
        force_recompute (bool): Wird ignoriert – nur für Cache-Kompatibilität

    Returns:
        plotly.graph_objs._figure.Figure: Interaktive Plotly-Figur
    """
    records = []
    for idx, row in df.iterrows():
        try:
            img = Image.open(row["path"]).convert("YCbCr")
            arr = np.array(img)
            records.append(
                {
                    "label": row["label_name"],
                    "Y": arr[:, :, 0].mean(),
                    "Cb": arr[:, :, 1].mean(),
                    "Cr": arr[:, :, 2].mean(),
                }
            )
        except Exception:
            continue

    df_ycbcr = pd.DataFrame(records)

    fig = px.scatter_3d(
        df_ycbcr,
        x="Y",
        y="Cb",
        z="Cr",
        color="label",
        title=f"YCbCr-Mittelwerte – 3D-Scatterplot ({dataset_name})",
        opacity=0.7,
        width=800,
        height=600,
        category_orders={"label": ["Cover", "JMiPOD", "JUNIWARD", "UERD"]},
    )

    fig.update_traces(marker=dict(size=4))
    fig.update_layout(scene=dict(xaxis_title="Y", yaxis_title="Cb", zaxis_title="Cr"))
    return fig


def plot_channel_correlation(df: pd.DataFrame, dataset_name: str = "") -> plt.Figure:
    """
    Zeigt die Korrelationsmatrix der mittleren Y, Cb, Cr-Werte pro Bild,
    gruppiert nach Klasse (nur über Bildmittelwerte, nicht auf Pixelebene).
    """
    df = df.copy()
    means = {"label_name": [], "Y": [], "Cb": [], "Cr": []}

    for _, row in df.iterrows():
        try:
            img = Image.open(row["path"]).convert("YCbCr")
            arr = np.array(img)
            means["label_name"].append(row["label_name"])
            means["Y"].append(arr[:, :, 0].mean())
            means["Cb"].append(arr[:, :, 1].mean())
            means["Cr"].append(arr[:, :, 2].mean())
        except Exception:
            continue

    df_means = pd.DataFrame(means)
    fig, axs = plt.subplots(1, len(df_means["label_name"].unique()), figsize=(15, 4))

    df_means["label_name"] = pd.Categorical(df_means["label_name"], categories=df["label_name"].cat.categories, ordered=True)

    for i, label in enumerate(df_means["label_name"].cat.categories):
        corr = df_means[df_means["label_name"] == label][["Y", "Cb", "Cr"]].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, square=True, ax=axs[i])
        axs[i].set_title(str(label))

    fig.suptitle(f"Korrelationsmatrizen YCbCr pro Klasse – {dataset_name}")
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    return fig


def plot_rgb_kde(df: pd.DataFrame, dataset_name: str = "", force_recompute: bool = False) -> plt.Figure:
    """
    Zeigt KDE-Kurven der mittleren R, G, B-Werte pro Bild für jede Klasse.
    Alle Kanäle werden in separaten Subplots dargestellt.
    """
    channels = ["R", "G", "B"]
    fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=False)

    for idx, channel in enumerate(channels):
        ax = axes[idx]
        for cls in df["label_name"].cat.categories:
            subset = df[df["label_name"] == cls].sample(n=min(50, len(df)), random_state=42)
            values = []

            for path in subset["path"]:
                try:
                    img = Image.open(path).convert("RGB")
                    arr = np.array(img)
                    if channel == "R":
                        values.append(arr[:, :, 0].mean())
                    elif channel == "G":
                        values.append(arr[:, :, 1].mean())
                    elif channel == "B":
                        values.append(arr[:, :, 2].mean())
                except Exception:
                    continue

            sns.kdeplot(values, clip=(0, 255), label=cls, ax=ax, linewidth=2, alpha=0.7)

        ax.set_title(f"KDE – Mittlerer {channel}-Kanal – {dataset_name}")
        ax.set_ylabel("Dichte")
        ax.grid(True, linestyle="--", alpha=0.5)
        if idx == 2:
            ax.set_xlabel("Mittlerer RGB-Wert")
        else:
            ax.set_xlabel("")

    axes[0].legend(title="Klasse")
    fig.tight_layout()
    return fig


def show_outliers_by_ychannel(df: pd.DataFrame, dataset_name: str = "", z_thresh: float = 3.0):
    """
    Zeigt Gruppen von Stego-Varianten desselben Motivs nebeneinander, wenn eine davon ein Outlier ist.

    Args:
        df (pd.DataFrame): DataFrame mit 'path' und 'label_name'.
        dataset_name (str): Optionaler Titel.
        z_thresh (float): Z-Score-Schwelle.

    Returns:
        VBox: Interaktive Anzeige der Ausreißergruppen.
    """
    df = df.copy()
    df["motif_id"] = df["path"].apply(lambda p: Path(p).name)

    y_means = []
    for path in df["path"]:
        try:
            img = Image.open(path).convert("YCbCr")
            y = np.array(img)[:, :, 0]
            y_means.append(y.mean())
        except Exception:
            y_means.append(np.nan)

    df["y_mean"] = y_means
    df = df.dropna(subset=["y_mean"])
    df["z_score"] = zscore(df["y_mean"])

    # Finde alle Motive mit mindestens einer Variante als Outlier
    outlier_ids = df[np.abs(df["z_score"]) > z_thresh]["motif_id"].unique()
    grouped = [df[df["motif_id"] == mid].sort_values("label_name") for mid in outlier_ids]
    grouped = [g for g in grouped if len(g) > 1]

    if not grouped:
        print(f"Keine Ausreißergruppen mit Z-Score > {z_thresh} gefunden.")
        return

    idx_box = widgets.BoundedIntText(value=0, min=0, max=len(grouped) - 1, description="Gruppe:")
    out = widgets.Output()

    def render(idx):
        with out:
            clear_output(wait=True)
            group = grouped[idx]
            fig, axs = plt.subplots(1, len(group), figsize=(len(group) * 3, 4))
            if len(group) == 1:
                axs = [axs]
            for ax, (_, row) in zip(axs, group.iterrows()):
                try:
                    img = Image.open(row["path"])
                    ax.imshow(img)
                except Exception:
                    ax.text(0.5, 0.5, "Fehler", ha="center", va="center")
                ax.set_title(f"{row['label_name']}\nZ={row['z_score']:.2f}")
                ax.axis("off")
            fig.suptitle(f"Motiv: {group['motif_id'].iloc[0]} – {dataset_name}", fontsize=12)
            plt.tight_layout()
            plt.show()

    def on_change(change):
        render(change["new"])

    idx_box.observe(on_change, names="value")
    render(0)

    return widgets.VBox([idx_box, out])
