from pathlib import Path

import ipywidgets as widgets
import jpegio as jio
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import clear_output
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from PIL import Image
from scipy.ndimage import gaussian_filter

# Einheitliche Farbpalette für Klassen
LABEL_ORDER = ["Cover", "JMiPOD", "JUNIWARD", "UERD"]
PALETTE = sns.color_palette("colorblind", n_colors=4)
LABEL_COLORS = dict(zip(LABEL_ORDER, PALETTE))


def plot_dct_avg_and_delta(df: pd.DataFrame, dataset_name: str = "") -> plt.Figure:
    """
    Kombiniert die Heatmap der mittleren Cover-DCT-Y-Quantisierung und
    die Differenzheatmaps der Stego-Klassen zu Cover.

    Args:
        df (pd.DataFrame): DataFrame mit Spalten 'label_name' und 'q_y_00' bis 'q_y_63'.
        dataset_name (str): Optionaler Titel.

    Returns:
        plt.Figure: Matplotlib-Figur mit 4 Subplots (1x Cover, 3x Differenzen).
    """
    q_cols = [f"q_y_{i:02d}" for i in range(64)]
    classes = df["label_name"].cat.categories

    cover_mean = df[df["label_name"] == "Cover"][q_cols].mean().values.reshape(8, 8)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    # Cover
    sns.heatmap(cover_mean, cmap="YlGnBu", annot=True, fmt=".0f", ax=axes[0], cbar=False)
    axes[0].set_title("Cover – Mittelwert")

    for i, cls in enumerate(classes[1:]):
        cls_mean = df[df["label_name"] == cls][q_cols].mean().values.reshape(8, 8)
        delta = cls_mean - cover_mean
        sns.heatmap(delta, cmap="coolwarm", center=0, annot=True, fmt=".2f", ax=axes[i + 1], cbar=True)
        axes[i + 1].set_title(f"{cls} – Mittelwert Δ zu Cover")

    fig.suptitle(f"DCT-Y-Quantisierung (Mittelwert + Differenz) – {dataset_name}", fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    return fig


def plot_flip_counts(df: pd.DataFrame) -> plt.Figure:
    """
    Zählt AC-DCT-Flips pro JPEG-Komponente (Y, Cb, Cr) und zeigt Boxplots mit und ohne Ausreisser
    in einem 3x2-Raster: Zeilen = Kanäle, Spalten = mit/ohne Ausreisser.

    Args:
        df (pd.DataFrame): DataFrame mit 'label_name' und 'path'.

    Returns:
        plt.Figure: Matplotlib-Figur mit 3x2 Boxplot-Grid.
    """
    LABEL_ORDER = ["JMiPOD", "JUNIWARD", "UERD"]
    stego_df = df[df["label_name"].isin(LABEL_ORDER)]

    flip_records = {0: [], 1: [], 2: []}  # 0 = Y, 1 = Cb, 2 = Cr

    for _, row in stego_df.iterrows():
        stego_path = Path(row["path"])
        cover_path = stego_path.parent.parent / "Cover" / stego_path.name

        jpeg_cover = jio.read(str(cover_path))
        jpeg_stego = jio.read(str(stego_path))

        for i in range(3):  # Y, Cb, Cr
            cover = jpeg_cover.coef_arrays[i]
            stego = jpeg_stego.coef_arrays[i]

            mask_ac = np.ones_like(cover, dtype=bool)
            mask_ac[np.arange(0, cover.shape[0], 8)[:, None], np.arange(0, cover.shape[1], 8)] = False

            flips = np.sum((stego != cover) & mask_ac)
            flip_records[i].append({"label_name": row["label_name"], "flip_count_ac": flips})

    # Erstelle Figure
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True, sharey=False)
    component_names = ["Y-Kanal", "Cb-Kanal", "Cr-Kanal"]

    for i, records in flip_records.items():
        flip_df = pd.DataFrame(records)
        flip_df["label_name"] = pd.Categorical(flip_df["label_name"], categories=LABEL_ORDER, ordered=True)

        # mit Ausreissern
        sns.boxplot(
            data=flip_df, x="label_name", y="flip_count_ac", hue="label_name", palette=LABEL_COLORS, ax=axes[i, 0], showfliers=True, legend=False
        )
        axes[i, 0].set_title(f"{component_names[i]} – mit Ausreissern")
        axes[i, 0].set_ylabel("AC-Flips")

        # ohne Ausreisser
        sns.boxplot(
            data=flip_df, x="label_name", y="flip_count_ac", hue="label_name", palette=LABEL_COLORS, ax=axes[i, 1], showfliers=False, legend=False
        )
        axes[i, 1].set_title(f"{component_names[i]} – ohne Ausreisser")
        axes[i, 1].set_ylabel("AC-Flips")

    for ax in axes.flat:
        ax.set_xlabel("Steganographie-Verfahren")

    fig.suptitle("AC-DCT-Flips pro Bild und Kanal", fontsize=16)
    plt.tight_layout()
    return fig


def plot_flip_direction_overview(df: pd.DataFrame) -> plt.Figure:
    """
    Erstellt ein 2×1-Figure:
      • oben  : Summierte + / – Flip-Counts (gestapelte Balken)
      • unten : Gesamtsaldo (+ minus –)

    Args
    ----
    df : DataFrame mit 'label_name' und 'path'.

    Returns
    -------
    Matplotlib-Figure mit zwei Achsen.
    """
    STEGO = ["JMiPOD", "JUNIWARD", "UERD"]
    PAL_DIR = {"pos": "#64b5cd", "neg": "#d5605e"}
    totals = {lbl: {"pos": 0, "neg": 0} for lbl in STEGO}
    balance = {lbl: 0 for lbl in STEGO}

    # -------- Datensammlung -------------------------------------------------
    for _, row in df[df["label_name"].isin(STEGO)].iterrows():
        s_path = Path(row["path"])
        c_path = s_path.parent.parent / "Cover" / s_path.name

        cover = jio.read(str(c_path)).coef_arrays[0].astype(np.int32)
        stego = jio.read(str(s_path)).coef_arrays[0].astype(np.int32)

        mask = np.ones_like(cover, bool)
        mask[0::8, 0::8] = False  # DC ausblenden
        delta = (stego - cover)[mask].ravel()

        pos_cnt = int((delta > 0).sum())
        neg_cnt = int((delta < 0).sum())

        lbl = row["label_name"]
        totals[lbl]["pos"] += pos_cnt
        totals[lbl]["neg"] += neg_cnt
        balance[lbl] += pos_cnt - neg_cnt

    # -------- Figure & Subplots (2 × 1) -------------------------------------
    fig, (ax_tot, ax_bal) = plt.subplots(2, 1, figsize=(8, 8), sharex=False, gridspec_kw={"hspace": 0.35})

    # (1) Summierte + / – Flips ---------------------------------------------
    plot_df = pd.DataFrame(totals).T.melt(ignore_index=False, var_name="direction", value_name="count").reset_index(names="label_name")

    sns.barplot(data=plot_df, x="label_name", y="count", hue="direction", palette=PAL_DIR, order=STEGO, hue_order=["pos", "neg"], ax=ax_tot)
    handles, _ = ax_tot.get_legend_handles_labels()
    legend_labels = {"pos": "+ Flips", "neg": "– Flips"}  # sichere explizite Zuordnung
    new_labels = [legend_labels.get(h.get_label(), h.get_label()) for h in handles]

    ax_tot.legend(handles=handles, labels=new_labels, title="Δ-Vorzeichen")
    ax_tot.set_title("Σ positiver / negativer Flips (Y-AC)")
    ax_tot.set_xlabel("")
    ax_tot.set_ylabel("Summe AC-Flips")

    # (2) Saldo (+ − –) ------------------------------------------------------
    diffs = [balance[lbl] for lbl in STEGO]
    colors = [LABEL_COLORS[lbl] for lbl in STEGO]

    ax_bal.bar(STEGO, diffs, color=colors)
    ax_bal.axhline(0, color="0.3")
    ax_bal.set_title("Differenz: Anzahl +1-Flips minus −1-Flips")
    ax_bal.set_xlabel("Steganographie-Verfahren")
    ax_bal.set_ylabel("Differenz")

    return fig


def plot_flip_position_heatmap(df: pd.DataFrame, channel: int = 0, dataset_name: str = "") -> plt.Figure:
    """
    Zeigt die Häufigkeit von AC-Flips nach DCT-Position (8×8-Index) pro Stego-Verfahren
    für einen gewählten JPEG-Kanal (Y=0, Cb=1, Cr=2) als Heatmaps.

    Args:
        df (pd.DataFrame): DataFrame mit 'path' und 'label_name'
        channel (int): 0 = Y, 1 = Cb, 2 = Cr
        dataset_name (str): Optionaler Zusatztitel

    Returns:
        plt.Figure: Matplotlib-Figur mit 3 Heatmaps (JMiPOD, JUNIWARD, UERD)
    """
    STEGO = ["JMiPOD", "JUNIWARD", "UERD"]
    flip_maps = {lbl: np.zeros((8, 8), dtype=np.uint32) for lbl in STEGO}

    for _, row in df[df["label_name"].isin(STEGO)].iterrows():
        s_path = Path(row["path"])
        c_path = s_path.parent.parent / "Cover" / s_path.name

        coef_s = jio.read(str(s_path)).coef_arrays[channel].astype(np.int32)
        coef_c = jio.read(str(c_path)).coef_arrays[channel].astype(np.int32)

        delta = coef_s - coef_c

        # DC-Maske entfernen
        mask_ac = np.ones_like(delta, dtype=bool)
        mask_ac[0::8, 0::8] = False

        flips = ((delta != 0) & mask_ac).astype(np.uint8)

        # Position pro Block bestimmen
        h, w = delta.shape
        for by in range(0, h, 8):
            for bx in range(0, w, 8):
                block = flips[by : by + 8, bx : bx + 8]
                flip_maps[row["label_name"]] += block

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    vmax = max(np.max(m) for m in flip_maps.values())

    for ax, lbl in zip(axes, STEGO):
        sns.heatmap(flip_maps[lbl], ax=ax, cmap="YlOrRd", vmin=0, vmax=vmax, annot=False, cbar=True)
        ax.set_title(f"{lbl} – Flip-Häufigkeit nach DCT-Index")
        ax.set_xlabel("DCT-x (u)")
        ax.set_ylabel("DCT-y (v)")

    fig.suptitle(f"Verteilung der AC-Flips nach DCT-Position – Kanal: {['Y', 'Cb', 'Cr'][channel]} – {dataset_name}", fontsize=14)
    plt.tight_layout()
    return fig


def plot_cover_stego_flipmask(df: pd.DataFrame, dataset_name: str = "", init_channel: int = 0) -> widgets.VBox:
    """
    Zeigt für ein Motiv vier Bilder nebeneinander:
    - Cover
    - JMiPOD, JUNIWARD, UERD mit Flipmasken (Stego – Cover)

    Interaktiv umschaltbar zwischen
    ▸ drei JPEG-Kanälen  (Y, Cb, Cr)
    ▸ zwei Darstellungsmodi ("overlay", "heatmap")
    """
    df = df.copy()
    df["filename"] = df["path"].apply(lambda p: Path(p).name)
    df["base_name"] = df["filename"].str.extract(r"(\d+)\.jpg")

    LABELS = ["Cover", "JMiPOD", "JUNIWARD", "UERD"]
    channel_map = {"Y": 0, "Cb": 1, "Cr": 2}

    # ­— Widgets -------------------------------------------------------
    channel_selector = widgets.Dropdown(
        options=list(channel_map.keys()),
        value=list(channel_map.keys())[init_channel],
        description="Kanal:",
        layout=widgets.Layout(width="140px"),
    )
    mode_selector = widgets.Dropdown(
        options=["heatmap", "overlay"],
        value="heatmap",
        description="Modus:",
        layout=widgets.Layout(width="175px"),
    )

    complete_groups = df.groupby("base_name")["label_name"].nunique().loc[lambda g: g == 4].index.sort_values().tolist()

    idx_input = widgets.BoundedIntText(value=0, min=0, max=len(complete_groups) - 1, description="Index:")
    btn_prev = widgets.Button(description="←", layout=widgets.Layout(width="40px"))
    btn_next = widgets.Button(description="→", layout=widgets.Layout(width="40px"))
    btn_row = widgets.HBox([idx_input, btn_prev, btn_next])
    out = widgets.Output()

    # ­— Kernfunktion --------------------------------------------------
    def show(idx: int, channel_key: str, mode: str):
        base_id = complete_groups[idx]
        group = df[df["base_name"] == base_id]

        with out:
            clear_output(wait=True)
            if group["label_name"].nunique() < 4:
                print(f"Inkomplette Bildgruppe: {base_id}")
                return

            ch = channel_map[channel_key]
            paths = {lbl: group[group["label_name"] == lbl]["path"].iloc[0] for lbl in LABELS}

            jpeg_cover = jio.read(str(paths["Cover"]))
            coef_cover = jpeg_cover.coef_arrays[ch].astype(np.int32)

            cover_img = Image.open(paths["Cover"]).convert("RGB")
            cover_img = cover_img.resize(coef_cover.shape[::-1])

            fig = plt.figure(figsize=(22, 6), constrained_layout=True)
            spec = gridspec.GridSpec(ncols=5, nrows=1, figure=fig, width_ratios=[1, 1, 1, 1, 0.03])  # letzte Spalte: Colorbar
            axes = [fig.add_subplot(spec[0, i]) for i in range(4)]
            cax = fig.add_subplot(spec[0, 4])

            fig.suptitle(
                f"AC-Flip-Masken – ID {base_id} – Kanal: {channel_key} – " f"{dataset_name} – Modus: {mode}",
                fontsize=16,
            )

            # Cover ----------------------------------------------------
            axes[0].imshow(cover_img)
            axes[0].set_title("Cover")
            axes[0].axis("off")

            # Stego-Varianten -----------------------------------------
            for ax, lbl in zip(axes[1:], LABELS[1:]):
                jpeg_stego = jio.read(str(paths[lbl]))
                coef_stego = jpeg_stego.coef_arrays[ch].astype(np.int32)

                # AC-Koeff.-Differenz (DC ausschließen)
                mask = np.ones_like(coef_cover, dtype=bool)
                mask[0::8, 0::8] = False
                delta = (coef_stego - coef_cover) * mask

                ax.imshow(cover_img)

                if mode == "overlay":
                    flipmask = (delta != 0).astype(np.uint8)
                    pos_y, pos_x = np.where(delta == 1)
                    neg_y, neg_x = np.where(delta == -1)

                    ax.imshow(flipmask, cmap="Greys", alpha=0.2)
                    ax.scatter(pos_x, pos_y, s=1.0, c="red", alpha=0.5)
                    ax.scatter(neg_x, neg_y, s=1.0, c="blue", alpha=0.5)

                elif mode == "heatmap":
                    heat = gaussian_filter(np.abs(delta).astype(float), sigma=5)
                    heat = np.power(heat, 1.8)  # Kontrast
                    heat_norm = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)

                    ax.imshow(heat_norm, cmap="magma", vmin=0, vmax=1, alpha=0.6)

                ax.set_title(lbl)
                ax.axis("off")

            # Farb­legende: immer nur **eine** -------------------------
            cax.clear()  # alte Leiste entfernen

            if mode == "overlay":
                sm = ScalarMappable(
                    norm=Normalize(vmin=-2, vmax=2),
                    cmap="seismic",
                )
                sm.set_array([])
                fig.colorbar(sm, cax=cax, label=f"Δ (AC, {channel_key})")

            elif mode == "heatmap":
                sm = ScalarMappable(
                    norm=Normalize(vmin=0, vmax=1),
                    cmap="magma",
                )
                sm.set_array([])
                fig.colorbar(sm, cax=cax, label="|Δ| (norm)")

            plt.show()

    # ­— Navigation ----------------------------------------------------
    def go_relative(delta: int):
        idx_input.value = max(0, min(len(complete_groups) - 1, idx_input.value + delta))

    btn_prev.on_click(lambda _: go_relative(-1))
    btn_next.on_click(lambda _: go_relative(1))

    # Widget-Callbacks -------------------------------------------------
    def refresh(*_):
        show(idx_input.value, channel_selector.value, mode_selector.value)

    idx_input.observe(lambda changes: refresh(), names="value")
    channel_selector.observe(lambda changes: refresh(), names="value")
    mode_selector.observe(lambda changes: refresh(), names="value")

    # Erstes Rendering -------------------------------------------------
    show(0, channel_selector.value, mode_selector.value)

    controls = widgets.HBox([btn_row, channel_selector, mode_selector])
    return widgets.VBox([controls, out])
