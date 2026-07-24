"""
===============================================================================
eda_dct.py
===============================================================================
JPEG coefficient exploratory diagnostics.

Responsibilities:
  - Compare luminance quantization and cover/stego coefficient changes.
  - Summarize flip counts, directions, and within-block positions.
  - Render interactive cover/stego flip masks.

Design principles:
  - Class identities come from the dataframe so ALASKA2 and the synthetic
    compatibility workflow share one stable public label contract.

Boundaries:
  - German plot text is retained for the protected academic notebook. These
    functions are exploratory and do not transform training data.

Notes:
  - All comparisons expect complete source groups validated by data.index.
===============================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import ipywidgets as widgets
import jpegio as jio
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from PIL import Image
from scipy.ndimage import gaussian_filter

from .eda_controls import FigureController
from .eda_style import label_colors, label_order

__all__ = [
    "make_cover_stego_flipmask_widget",
    "plot_cover_stego_flipmask",
    "plot_dct_avg_and_delta",
    "plot_flip_counts",
    "plot_flip_direction_overview",
    "plot_flip_position_heatmap",
]


def plot_dct_avg_and_delta(df: pd.DataFrame, dataset_name: str = "") -> Figure:
    """Plot mean Cover luminance quantization and class-wise deltas.

    Parameters
    ----------
    df
        Metadata dataframe containing labels and 64 luminance quantization columns.
    dataset_name
        Human-readable dataset name included in the figure title.

    Returns
    -------
    matplotlib.figure.Figure
        Cover table and one delta heatmap per stego class.

    Raises
    ------
    ValueError
        If required labels or quantization columns are unavailable.
    """
    q_cols = [f"q_y_{i:02d}" for i in range(64)]
    classes = label_order(df)

    cover_rows = cast(pd.DataFrame, df.loc[df["label_name"] == "Cover", q_cols])
    cover_mean = cast(pd.Series, cover_rows.mean()).to_numpy(dtype=float).reshape(8, 8)

    fig, axes = plt.subplots(1, len(classes), figsize=(5 * len(classes), 4))

    # Cover
    sns.heatmap(cover_mean, cmap="YlGnBu", annot=True, fmt=".0f", ax=axes[0], cbar=False)
    axes[0].set_title("Cover – Mittelwert")

    for i, cls in enumerate(classes[1:]):
        class_rows = cast(pd.DataFrame, df.loc[df["label_name"] == cls, q_cols])
        cls_mean = cast(pd.Series, class_rows.mean()).to_numpy(dtype=float).reshape(8, 8)
        delta = cls_mean - cover_mean
        sns.heatmap(delta, cmap="coolwarm", center=0, annot=True, fmt=".2f", ax=axes[i + 1], cbar=True)
        axes[i + 1].set_title(f"{cls} – Mittelwert Δ zu Cover")

    fig.suptitle(f"DCT-Y-Quantisierung (Mittelwert + Differenz) – {dataset_name}", fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    return fig


def plot_flip_counts(df: pd.DataFrame) -> Figure:
    """Plot per-image AC coefficient-change counts with and without outliers.

    Parameters
    ----------
    df
        Dataframe containing image paths and class labels.

    Returns
    -------
    matplotlib.figure.Figure
        Y, Cb, and Cr box plots for every observed stego class.

    Raises
    ------
    ValueError
        If class labels or matched Cover files are unavailable.
    """
    stego_labels = [label for label in label_order(df) if label != "Cover"]
    colors = label_colors(df)
    stego_df = df[df["label_name"].isin(stego_labels)]

    flip_records = {0: [], 1: [], 2: []}  # 0 = Y, 1 = Cb, 2 = Cr

    for _, row in stego_df.iterrows():
        stego_path = Path(str(row["path"]))
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

    # One row per JPEG component.
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True, sharey=False)
    component_names = ["Y-Kanal", "Cb-Kanal", "Cr-Kanal"]

    for i, records in flip_records.items():
        flip_df = pd.DataFrame(records)
        flip_df["label_name"] = pd.Categorical(flip_df["label_name"], categories=stego_labels, ordered=True)

        # Include outliers.
        sns.boxplot(
            data=flip_df,
            x="label_name",
            y="flip_count_ac",
            hue="label_name",
            palette=colors,
            ax=axes[i, 0],
            showfliers=True,
            legend=False,
        )
        axes[i, 0].set_title(f"{component_names[i]} – mit Ausreissern")
        axes[i, 0].set_ylabel("AC-Flips")

        # Hide outliers.
        sns.boxplot(
            data=flip_df,
            x="label_name",
            y="flip_count_ac",
            hue="label_name",
            palette=colors,
            ax=axes[i, 1],
            showfliers=False,
            legend=False,
        )
        axes[i, 1].set_title(f"{component_names[i]} – ohne Ausreisser")
        axes[i, 1].set_ylabel("AC-Flips")

    for ax in axes.flat:
        ax.set_xlabel("Steganographie-Verfahren")

    fig.suptitle("AC-DCT-Flips pro Bild und Kanal", fontsize=16)
    plt.tight_layout()
    return fig


def plot_flip_direction_overview(df: pd.DataFrame) -> Figure:
    """Plot positive and negative luminance AC changes and their balance.

    Parameters
    ----------
    df
        Dataframe containing image paths and class labels.

    Returns
    -------
    matplotlib.figure.Figure
        Direction totals and signed balance panels.

    Raises
    ------
    ValueError
        If class labels or matched Cover files are unavailable.
    """
    stego_labels = [label for label in label_order(df) if label != "Cover"]
    PAL_DIR = {"pos": "#64b5cd", "neg": "#d5605e"}
    totals = {lbl: {"pos": 0, "neg": 0} for lbl in stego_labels}
    balance = {lbl: 0 for lbl in stego_labels}

    # Collect coefficient-change directions.
    for _, row in df[df["label_name"].isin(stego_labels)].iterrows():
        s_path = Path(str(row["path"]))
        c_path = s_path.parent.parent / "Cover" / s_path.name

        cover = jio.read(str(c_path)).coef_arrays[0].astype(np.int32)
        stego = jio.read(str(s_path)).coef_arrays[0].astype(np.int32)

        mask = np.ones_like(cover, bool)
        mask[0::8, 0::8] = False  # Exclude DC coefficients.
        delta = (stego - cover)[mask].ravel()

        pos_cnt = int((delta > 0).sum())
        neg_cnt = int((delta < 0).sum())

        label = str(row["label_name"])
        totals[label]["pos"] += pos_cnt
        totals[label]["neg"] += neg_cnt
        balance[label] += pos_cnt - neg_cnt

    # Render direction totals and balance in separate panels.
    fig, (ax_tot, ax_bal) = plt.subplots(2, 1, figsize=(8, 8), sharex=False, gridspec_kw={"hspace": 0.35})

    # Summed positive and negative changes.
    plot_df = (
        pd.DataFrame(totals)
        .T.melt(ignore_index=False, var_name="direction", value_name="count")
        .reset_index(names="label_name")
    )

    sns.barplot(
        data=plot_df,
        x="label_name",
        y="count",
        hue="direction",
        palette=PAL_DIR,
        order=stego_labels,
        hue_order=["pos", "neg"],
        ax=ax_tot,
    )
    handles, _ = ax_tot.get_legend_handles_labels()
    legend_labels = {"pos": "+ Flips", "neg": "– Flips"}  # Explicit direction mapping.
    new_labels = [legend_labels.get(h.get_label(), h.get_label()) for h in handles]

    ax_tot.legend(handles=handles, labels=new_labels, title="Δ-Vorzeichen")
    ax_tot.set_title("Σ positiver / negativer Flips (Y-AC)")
    ax_tot.set_xlabel("")
    ax_tot.set_ylabel("Summe AC-Flips")

    # Signed balance: positive changes minus negative changes.
    diffs = [balance[lbl] for lbl in stego_labels]
    colors = [label_colors(df)[lbl] for lbl in stego_labels]

    ax_bal.bar(stego_labels, diffs, color=colors)
    ax_bal.axhline(0, color="0.3")
    ax_bal.set_title("Differenz: Anzahl +1-Flips minus −1-Flips")
    ax_bal.set_xlabel("Steganographie-Verfahren")
    ax_bal.set_ylabel("Differenz")

    return fig


def plot_flip_position_heatmap(df: pd.DataFrame, channel: int = 0, dataset_name: str = "") -> Figure:
    """Plot AC change counts by within-block position and stego class.

    Parameters
    ----------
    df
        Dataframe containing image paths and class labels.
    channel
        JPEG component index: zero for Y, one for Cb, or two for Cr.
    dataset_name
        Human-readable dataset name included in the figure title.

    Returns
    -------
    matplotlib.figure.Figure
        One within-block heatmap per observed stego class.

    Raises
    ------
    ValueError
        If the channel, labels, or matched Cover files are invalid.
    """
    if channel not in {0, 1, 2}:
        raise ValueError("channel must be 0 (Y), 1 (Cb), or 2 (Cr).")
    stego_labels = [label for label in label_order(df) if label != "Cover"]
    flip_maps = {lbl: np.zeros((8, 8), dtype=np.uint32) for lbl in stego_labels}

    for _, row in df[df["label_name"].isin(stego_labels)].iterrows():
        s_path = Path(str(row["path"]))
        c_path = s_path.parent.parent / "Cover" / s_path.name

        coef_s = jio.read(str(s_path)).coef_arrays[channel].astype(np.int32)
        coef_c = jio.read(str(c_path)).coef_arrays[channel].astype(np.int32)

        delta = coef_s - coef_c

        # Exclude DC coefficients.
        mask_ac = np.ones_like(delta, dtype=bool)
        mask_ac[0::8, 0::8] = False

        flips = ((delta != 0) & mask_ac).astype(np.uint8)

        # Accumulate the within-block position.
        h, w = delta.shape
        for by in range(0, h, 8):
            for bx in range(0, w, 8):
                block = flips[by : by + 8, bx : bx + 8]
                flip_maps[str(row["label_name"])] += block

    # Render one heatmap per observed stego class.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    vmax = max(float(np.max(values)) for values in flip_maps.values())

    for ax, lbl in zip(axes, stego_labels, strict=True):
        sns.heatmap(flip_maps[lbl], ax=ax, cmap="YlOrRd", vmin=0, vmax=vmax, annot=False, cbar=True)
        ax.set_title(f"{lbl} – Flip-Häufigkeit nach DCT-Index")
        ax.set_xlabel("DCT-x (u)")
        ax.set_ylabel("DCT-y (v)")

    fig.suptitle(
        f"Verteilung der AC-Flips nach DCT-Position – Kanal: {['Y', 'Cb', 'Cr'][channel]} – {dataset_name}", fontsize=14
    )
    plt.tight_layout()
    return fig


def plot_cover_stego_flipmask(
    df: pd.DataFrame,
    dataset_name: str = "",
    *,
    source_index: int = 0,
    channel: int = 0,
    mode: str = "heatmap",
) -> Figure:
    """Plot one Cover and coefficient-change-mask comparison.

    Parameters
    ----------
    df
        Dataframe containing image paths and class labels.
    dataset_name
        Human-readable dataset name included in the figure title.
    source_index
        Zero-based complete source-group index.
    channel
        Component index: zero for Y, one for Cb, or two for Cr.
    mode
        Either ``heatmap`` or ``overlay``.

    Returns
    -------
    matplotlib.figure.Figure
        Flip-mask figure without displaying it.

    Raises
    ------
    ValueError
        If a selection or the complete source-group contract is invalid.
    """
    frame, labels, complete_groups = _flipmask_state(df)
    return _plot_cover_stego_flipmask_group(
        frame,
        labels,
        complete_groups,
        dataset_name=dataset_name,
        source_index=source_index,
        channel=channel,
        mode=mode,
    )


def _plot_cover_stego_flipmask_group(
    frame: pd.DataFrame,
    labels: list[str],
    complete_groups: list[object],
    *,
    dataset_name: str,
    source_index: int,
    channel: int,
    mode: str,
) -> Figure:
    if channel not in {0, 1, 2}:
        raise ValueError("channel must be 0 (Y), 1 (Cb), or 2 (Cr).")
    if mode not in {"heatmap", "overlay"}:
        raise ValueError("mode must be either 'heatmap' or 'overlay'.")
    if not 0 <= source_index < len(complete_groups):
        raise ValueError("source_index is outside the complete source-group range.")
    channel_key = ("Y", "Cb", "Cr")[channel]
    base_id = complete_groups[source_index]
    group = frame[frame["base_name"] == base_id]
    paths = {label: str(cast(pd.Series, group.loc[group["label_name"] == label, "path"]).iloc[0]) for label in labels}
    jpeg_cover = jio.read(paths["Cover"])
    coefficient_cover = jpeg_cover.coef_arrays[channel].astype(np.int32)
    with Image.open(paths["Cover"]) as opened_cover:
        cover_image = opened_cover.convert("RGB").resize(coefficient_cover.shape[::-1])

    figure = plt.figure(figsize=(22, 6), constrained_layout=True)
    try:
        spec = gridspec.GridSpec(
            ncols=len(labels) + 1,
            nrows=1,
            figure=figure,
            width_ratios=[*[1] * len(labels), 0.03],
        )
        axes = [figure.add_subplot(spec[0, index]) for index in range(len(labels))]
        color_axis = figure.add_subplot(spec[0, len(labels)])
        figure.suptitle(
            f"AC-Flip-Masken – ID {base_id} – Kanal: {channel_key} – {dataset_name} – Modus: {mode}",
            fontsize=16,
        )
        axes[0].imshow(cover_image)
        axes[0].set_title("Cover")
        axes[0].axis("off")

        for axis, label in zip(axes[1:], labels[1:], strict=True):
            jpeg_stego = jio.read(paths[label])
            coefficient_stego = jpeg_stego.coef_arrays[channel].astype(np.int32)
            mask = np.ones_like(coefficient_cover, dtype=bool)
            mask[0::8, 0::8] = False
            delta = (coefficient_stego - coefficient_cover) * mask
            axis.imshow(cover_image)
            if mode == "overlay":
                flipmask = (delta != 0).astype(np.uint8)
                positive_y, positive_x = np.where(delta == 1)
                negative_y, negative_x = np.where(delta == -1)
                axis.imshow(flipmask, cmap="Greys", alpha=0.2)
                axis.scatter(positive_x, positive_y, s=1.0, c="red", alpha=0.5)
                axis.scatter(negative_x, negative_y, s=1.0, c="blue", alpha=0.5)
            else:
                heat = np.power(gaussian_filter(np.abs(delta).astype(float), sigma=5), 1.8)
                heat_norm = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)
                axis.imshow(heat_norm, cmap="magma", vmin=0, vmax=1, alpha=0.6)
            axis.set_title(label)
            axis.axis("off")

        if mode == "overlay":
            scalar_mappable = ScalarMappable(norm=Normalize(vmin=-2, vmax=2), cmap="seismic")
            scalar_mappable.set_array([])
            figure.colorbar(scalar_mappable, cax=color_axis, label=f"Δ (AC, {channel_key})")
        else:
            scalar_mappable = ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap="magma")
            scalar_mappable.set_array([])
            figure.colorbar(scalar_mappable, cax=color_axis, label="|Δ| (norm)")
        return figure
    except Exception:
        plt.close(figure)
        raise


def make_cover_stego_flipmask_widget(
    df: pd.DataFrame,
    dataset_name: str = "",
    init_channel: int = 0,
) -> FigureController:
    """Build display-free controls for the pure flip-mask renderer."""
    if init_channel not in {0, 1, 2}:
        raise ValueError("init_channel must be 0 (Y), 1 (Cb), or 2 (Cr).")
    frame, labels, complete_groups = _flipmask_state(df)
    channel_map = {"Y": 0, "Cb": 1, "Cr": 2}
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

    idx_input = widgets.BoundedIntText(value=0, min=0, max=len(complete_groups) - 1, description="Index:")
    btn_prev = widgets.Button(description="←", layout=widgets.Layout(width="40px"))
    btn_next = widgets.Button(description="→", layout=widgets.Layout(width="40px"))
    btn_row = widgets.HBox([idx_input, btn_prev, btn_next])
    controls = widgets.HBox([btn_row, channel_selector, mode_selector])
    controller = FigureController([controls])

    def go_relative(delta: int) -> None:
        idx_input.value = max(0, min(len(complete_groups) - 1, idx_input.value + delta))

    def selected_text(widget: widgets.Dropdown) -> str:
        value = widget.value
        if not isinstance(value, str):
            raise TypeError("Widget selection must be a string.")
        return value

    def refresh(_: dict[str, object] | None = None) -> None:
        controller.set_figure(
            _plot_cover_stego_flipmask_group(
                frame,
                labels,
                complete_groups,
                dataset_name=dataset_name,
                source_index=idx_input.value,
                channel=channel_map[selected_text(channel_selector)],
                mode=selected_text(mode_selector),
            )
        )

    def previous(_: widgets.Button) -> None:
        go_relative(-1)

    def next_source(_: widgets.Button) -> None:
        go_relative(1)

    controller.register_observer(idx_input, refresh, names="value")
    controller.register_observer(channel_selector, refresh, names="value")
    controller.register_observer(mode_selector, refresh, names="value")
    controller.on_click(btn_prev, previous)
    controller.on_click(btn_next, next_source)
    try:
        refresh()
    except Exception:
        controller.close()
        raise
    return controller


def _flipmask_state(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[object]]:
    frame = df.copy()
    frame["filename"] = frame["path"].apply(lambda path: Path(path).name)
    frame["base_name"] = frame["filename"].str.extract(r"(\d+)\.jpg")
    labels = label_order(frame)
    group_counts = cast(pd.Series, frame.groupby("base_name")["label_name"].nunique())
    complete_groups = group_counts.loc[group_counts == len(labels)].index.sort_values().tolist()
    if not complete_groups:
        raise ValueError("No complete Cover/Stego source groups are available for the flip-mask widget.")
    return frame, labels, complete_groups
