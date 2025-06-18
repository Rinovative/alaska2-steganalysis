from pathlib import Path

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import clear_output
from PIL import Image
from scipy.stats import zscore

# Einheitliche Farbpalette für Klassen
LABEL_ORDER = ["Cover", "JMiPOD", "JUNIWARD", "UERD"]
PALETTE = sns.color_palette("colorblind", n_colors=4)
LABEL_COLORS = dict(zip(LABEL_ORDER, PALETTE))


def plot_pixel_histograms(df: pd.DataFrame, dataset_name: str = "", color_space: str = "YCbCr") -> plt.Figure:
    """
    Zeigt Histogramme der Pixelwerte in allen 3 Kanälen für jede Klasse (Raster: Klassenzeilen × Kanälspalten).

    Args:
        df (pd.DataFrame): Enthält 'path' und 'label_name'.
        dataset_name (str): Für Titel.
        color_space (str): "YCbCr" oder "RGB".

    Returns:
        plt.Figure: Visualisierung.
    """
    assert color_space in ["YCbCr", "RGB"], "Nur 'YCbCr' oder 'RGB' erlaubt."
    channels = ["Y", "Cb", "Cr"] if color_space == "YCbCr" else ["R", "G", "B"]
    fig, axes = plt.subplots(len(LABEL_ORDER), 3, figsize=(18, 2.8 * len(LABEL_ORDER)), sharex=True, sharey=False)

    for row_idx, cls in enumerate(LABEL_ORDER):
        subset = df[df["label_name"] == cls].sample(n=min(50, len(df)), random_state=42)
        all_pixels = {ch: [] for ch in range(3)}

        for path in subset["path"]:
            try:
                img = Image.open(path).convert(color_space)
                arr = np.array(img)
                for ch in range(3):
                    all_pixels[ch].extend(arr[:, :, ch].flatten())
            except Exception:
                continue

        for ch in range(3):
            ax = axes[row_idx, ch]
            ax.hist(all_pixels[ch], bins=50, color=LABEL_COLORS[cls], alpha=0.85)
            if row_idx == 0:
                ax.set_title(f"Kanal {channels[ch]}", fontsize=11)
            if ch == 0:
                ax.set_ylabel(cls, fontsize=11)
            if row_idx == len(LABEL_ORDER) - 1:
                ax.set_xlabel("Pixelwert")
            ax.set_xlim(0, 255)
            ax.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(f"Histogramm der Pixelwerte – {color_space} – {dataset_name}", fontsize=15)
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    return fig


def plot_image_mean_distribution(df: pd.DataFrame, dataset_name: str = "") -> plt.Figure:
    """
    Zeigt die Verteilung der mittleren Pixelwerte:
    - Boxplot über alle Bilder (farbig nach label_name)
    - Boxplot über Extremfälle (oberes/unteres Quantil, ebenfalls farbig)

    Args:
        df (pd.DataFrame): DataFrame mit 'path' und 'label_name'.
        dataset_name (str): Optionaler Titel.

    Returns:
        plt.Figure: Matplotlib-Figur.
    """
    df = df.copy()
    df["image_mean"] = df["path"].apply(lambda p: plt.imread(p).mean())

    # Extremfälle (unteres und oberes 5%-Quantil)
    lower, upper = df["image_mean"].quantile([0.05, 0.95])
    df_extreme = df[(df["image_mean"] <= lower) | (df["image_mean"] >= upper)]

    fig, axs = plt.subplots(2, 1, figsize=(8, 12), sharex=True, gridspec_kw={"height_ratios": [1, 1]})

    # Boxplot: alle
    sns.boxplot(data=df, x="label_name", y="image_mean", hue="label_name", palette=LABEL_COLORS, ax=axs[0])
    axs[0].set_title(f"Verteilung aller mittleren Pixelwerte – {dataset_name}")
    axs[0].set_ylabel("Mittlerer Pixelwert")
    axs[0].grid(True, linestyle="--", alpha=0.5)

    # Boxplot: nur Extreme
    sns.boxplot(data=df_extreme, x="label_name", y="image_mean", hue="label_name", palette=LABEL_COLORS, ax=axs[1])
    axs[1].set_title("Extremwerte (unterstes & oberstes 5%-Quantil)")
    axs[1].set_xlabel("Klasse")
    axs[1].set_ylabel("Mittlerer Pixelwert")
    axs[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    return fig


def plot_kde_and_boxplot(df: pd.DataFrame, dataset_name: str = "", color_space: str = "YCbCr") -> plt.Figure:
    """
    Kombinierte Visualisierung: KDE- und Boxplots für mittlere Kanalwerte (YCbCr oder RGB).

    Args:
        df (pd.DataFrame): Enthält 'path' und 'label_name'.
        dataset_name (str): Für Titel.
        color_space (str): "YCbCr" oder "RGB".

    Returns:
        plt.Figure: Visualisierung.
    """
    assert color_space in ["YCbCr", "RGB"], "Nur 'YCbCr' oder 'RGB' erlaubt."

    if color_space == "YCbCr":
        channels = ["Y", "Cb", "Cr"]
    else:
        channels = ["R", "G", "B"]

    # Mittelwerte extrahieren
    stats = []
    for cls in df["label_name"].cat.categories:
        subset = df[df["label_name"] == cls].sample(n=min(50, len(df)), random_state=42)
        for path in subset["path"]:
            try:
                img = Image.open(path).convert(color_space)
                arr = np.array(img)
                values = {channels[i]: arr[:, :, i].mean() for i in range(3)}
                for ch in channels:
                    stats.append({"label": cls, "channel": ch, "value": values[ch]})
            except Exception:
                continue

    df_stats = pd.DataFrame(stats)

    # Plot erstellen
    fig, axes = plt.subplots(len(channels), 2, figsize=(15, 5 * len(channels)))

    for i, ch in enumerate(channels):
        ax_kde = axes[i, 0]
        ax_box = axes[i, 1]

        # KDE: Linien ohne Füllung
        for cls in df["label_name"].cat.categories:
            values = df_stats[(df_stats["channel"] == ch) & (df_stats["label"] == cls)]["value"]
            sns.kdeplot(values, ax=ax_kde, label=cls, color=LABEL_COLORS[cls], clip=(0, 255), linewidth=2, alpha=0.95)
        ax_kde.set_title(f"KDE – Mittlerer {ch}-Kanal – {dataset_name}")
        ax_kde.set_ylabel("Dichte")
        ax_kde.set_xlabel("Mittlerer Kanalwert" if i == len(channels) - 1 else "")
        ax_kde.grid(True, linestyle="--", alpha=0.5)
        ax_kde.legend(title="Klasse")

        # Boxplot mit hue für Farbgebung
        sns.boxplot(data=df_stats[df_stats["channel"] == ch], x="channel", y="value", hue="label", palette=LABEL_COLORS, ax=ax_box)
        ax_box.set_title(f"Boxplot – Kanal {ch}")
        ax_box.set_xlabel("")
        ax_box.set_ylabel("Mittlerer Wert")
        ax_box.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(f"Kanalverteilungen – {color_space} – {dataset_name}", fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=0.93)
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


def show_outliers_by_channel(df: pd.DataFrame, dataset_name: str = "", z_thresh: float = 3.0) -> widgets.VBox:
    """
    Interaktive Anzeige von Ausreissergruppen (nach Z-Score) basierend auf einem wählbaren YCbCr-Kanal.
    Ergebnisse werden zwischengespeichert, um mehrfache Bildberechnungen zu vermeiden.
    """
    df = df.copy()
    df["motif_id"] = df["path"].apply(lambda p: Path(p).name)

    channel_map = {"Y": 0, "Cb": 1, "Cr": 2}
    dropdown_channel = widgets.Dropdown(options=list(channel_map.keys()), description="Kanal:")
    idx_input = widgets.BoundedIntText(value=0, min=0, description="Gruppe:")
    btn_prev = widgets.Button(description="←", layout=widgets.Layout(width="40px"))
    btn_next = widgets.Button(description="→", layout=widgets.Layout(width="40px"))
    btn_row = widgets.HBox([idx_input, btn_prev, btn_next])
    out = widgets.Output()

    # Caching: Kanal-Mittelwerte und Outlier-Gruppen
    channel_means_cache = {}
    grouped_all = {}

    def compute_channel_means(channel: str) -> pd.DataFrame:
        if channel in channel_means_cache:
            return channel_means_cache[channel]

        means = []
        for path in df["path"]:
            try:
                img = Image.open(path).convert("YCbCr")
                val = np.array(img)[:, :, channel_map[channel]].mean()
                means.append(val)
            except Exception:
                means.append(np.nan)

        df_copy = df.copy()
        df_copy[f"{channel}_mean"] = means
        df_copy = df_copy.dropna(subset=[f"{channel}_mean"])
        df_copy["z_score"] = zscore(df_copy[f"{channel}_mean"])

        channel_means_cache[channel] = df_copy
        return df_copy

    def compute_outliers(channel: str):
        if channel in grouped_all:
            return grouped_all[channel]

        df_chan = compute_channel_means(channel)
        outlier_ids = df_chan[np.abs(df_chan["z_score"]) > z_thresh]["motif_id"].unique()
        grouped = [df_chan[df_chan["motif_id"] == mid].sort_values("label_name") for mid in outlier_ids]
        grouped = [g for g in grouped if len(g) > 1]
        grouped_all[channel] = grouped
        return grouped

    def render(idx, channel):
        with out:
            clear_output(wait=True)
            grouped = grouped_all.get(channel, [])
            if not grouped:
                print(f"Keine Ausreissergruppen in Kanal {channel} (Z > {z_thresh}) gefunden.")
                return
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
            fig.suptitle(f"Motiv: {group['motif_id'].iloc[0]} – Kanal {channel} – {dataset_name}", fontsize=12)
            plt.tight_layout()
            plt.show()

    def update_channel(change):
        channel = change["new"]
        compute_channel_means(channel)
        grouped = compute_outliers(channel)
        idx_input.max = max(0, len(grouped) - 1)
        idx_input.value = 0
        render(0, channel)

    def on_change_idx(change):
        render(change["new"], dropdown_channel.value)

    def step(delta):
        new_val = max(0, min(idx_input.max, idx_input.value + delta))
        idx_input.value = new_val

    dropdown_channel.observe(update_channel, names="value")
    idx_input.observe(on_change_idx, names="value")
    btn_prev.on_click(lambda _: step(-1))
    btn_next.on_click(lambda _: step(1))

    # Initialisierung
    update_channel({"new": dropdown_channel.value})
    return widgets.VBox([dropdown_channel, btn_row, out])
