"""
===============================================================================
eda_channels.py
===============================================================================
Spatial color-channel exploratory analysis.

Responsibilities:
  - Plot decoded pixel and per-image channel distributions.
  - Compare class-wise channel correlations.
  - Provide an interactive outlier browser for the academic notebook.

Design principles:
  - Scientific labels and colors come from the dataframe, preserving one
    consistent class contract for ALASKA2 and the synthetic compatibility
    workflow.

Boundaries:
  - German plot text is intentionally retained because these figures belong
    to the protected German-language notebook.

Notes:
  - Image read failures are skipped only in exploratory plots and never in
    model data.
===============================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from PIL import Image
from scipy.stats import zscore

from .eda_controls import FigureController
from .eda_style import label_colors, label_order

__all__ = [
    "plot_channel_correlation",
    "plot_image_mean_distribution",
    "plot_kde_and_boxplot",
    "plot_outliers_by_channel",
    "plot_pixel_histograms",
    "show_outliers_by_channel",
]


def plot_pixel_histograms(df: pd.DataFrame, dataset_name: str = "", color_space: str = "YCbCr") -> Figure:
    """Plot sampled decoded-pixel histograms by class and color channel.

    Parameters
    ----------
    df
        Dataframe containing image paths and class labels.
    dataset_name
        Human-readable dataset name included in the figure title.
    color_space
        Decoded color space, either ``YCbCr`` or ``RGB``.

    Returns
    -------
    matplotlib.figure.Figure
        Grid containing one three-channel histogram row per class.

    Raises
    ------
    ValueError
        If the color space or class-label contract is invalid.

    Notes
    -----
    Unreadable files are skipped only for this exploratory visualization.
    """
    order = label_order(df)
    colors = label_colors(df)
    if color_space not in {"YCbCr", "RGB"}:
        raise ValueError("color_space must be 'YCbCr' or 'RGB'.")
    channels = ["Y", "Cb", "Cr"] if color_space == "YCbCr" else ["R", "G", "B"]
    fig, axes = plt.subplots(len(order), 3, figsize=(18, 2.8 * len(order)), sharex=True, sharey=False)

    for row_idx, cls in enumerate(order):
        subset = df[df["label_name"] == cls]
        subset = subset.sample(n=min(50, len(subset)), random_state=42)
        all_pixels = {ch: [] for ch in range(3)}

        for path in subset["path"]:
            try:
                with Image.open(path) as opened:
                    arr = np.array(opened.convert(color_space))
                for ch in range(3):
                    all_pixels[ch].extend(arr[:, :, ch].flatten())
            except Exception:
                continue

        for ch in range(3):
            ax = axes[row_idx, ch]
            ax.hist(all_pixels[ch], bins=50, color=colors[cls], alpha=0.85)
            if row_idx == 0:
                ax.set_title(f"Kanal {channels[ch]}", fontsize=11)
            if ch == 0:
                ax.set_ylabel(cls, fontsize=11)
            if row_idx == len(order) - 1:
                ax.set_xlabel("Pixelwert")
            ax.set_xlim(0, 255)
            ax.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(f"Histogramm der Pixelwerte – {color_space} – {dataset_name}", fontsize=15)
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    return fig


def plot_image_mean_distribution(df: pd.DataFrame, dataset_name: str = "") -> Figure:
    """Plot full and tail distributions of mean decoded image values.

    Parameters
    ----------
    df
        Dataframe containing image paths and class labels.
    dataset_name
        Human-readable dataset name included in the figure title.

    Returns
    -------
    matplotlib.figure.Figure
        Two-panel class-wise box-plot figure.

    Raises
    ------
    ValueError
        If class labels are unavailable for palette construction.
    """
    colors = label_colors(df)
    df = df.copy()
    df["image_mean"] = df["path"].apply(lambda p: plt.imread(p).mean())

    # Lower and upper five-percent tails.
    lower, upper = df["image_mean"].quantile([0.05, 0.95])
    df_extreme = cast(pd.DataFrame, df.loc[(df["image_mean"] <= lower) | (df["image_mean"] >= upper)])

    fig, axs = plt.subplots(2, 1, figsize=(8, 12), sharex=True, gridspec_kw={"height_ratios": [1, 1]})

    # Full sample.
    sns.boxplot(data=df, x="label_name", y="image_mean", hue="label_name", palette=colors, ax=axs[0])
    axs[0].set_title(f"Verteilung aller mittleren Pixelwerte – {dataset_name}")
    axs[0].set_ylabel("Mittlerer Pixelwert")
    axs[0].grid(True, linestyle="--", alpha=0.5)

    # Distribution tails only.
    sns.boxplot(data=df_extreme, x="label_name", y="image_mean", hue="label_name", palette=colors, ax=axs[1])
    axs[1].set_title("Extremwerte (unterstes & oberstes 5%-Quantil)")
    axs[1].set_xlabel("Klasse")
    axs[1].set_ylabel("Mittlerer Pixelwert")
    axs[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    return fig


def plot_kde_and_boxplot(df: pd.DataFrame, dataset_name: str = "", color_space: str = "YCbCr") -> Figure:
    """Plot density and box summaries of per-image channel means.

    Parameters
    ----------
    df
        Dataframe containing image paths and class labels.
    dataset_name
        Human-readable dataset name included in the figure title.
    color_space
        Decoded color space, either ``YCbCr`` or ``RGB``.

    Returns
    -------
    matplotlib.figure.Figure
        Density and box-plot columns for each decoded channel.

    Raises
    ------
    ValueError
        If the color space or class-label contract is invalid.

    Notes
    -----
    Unreadable files are skipped only for this exploratory visualization.
    """
    colors = label_colors(df)
    if color_space not in {"YCbCr", "RGB"}:
        raise ValueError("color_space must be 'YCbCr' or 'RGB'.")

    channels = ["Y", "Cb", "Cr"] if color_space == "YCbCr" else ["R", "G", "B"]

    # Extract per-image means.
    stats = []
    for cls in df["label_name"].cat.categories:
        subset = df[df["label_name"] == cls]
        subset = subset.sample(n=min(50, len(subset)), random_state=42)
        for path in subset["path"]:
            try:
                with Image.open(path) as opened:
                    arr = np.array(opened.convert(color_space))
                values = {channels[i]: arr[:, :, i].mean() for i in range(3)}
                stats.extend({"label": cls, "channel": ch, "value": values[ch]} for ch in channels)
            except Exception:
                continue

    df_stats = pd.DataFrame(stats)

    # Render one density and one box plot per channel.
    fig, axes = plt.subplots(len(channels), 2, figsize=(15, 5 * len(channels)))

    for i, ch in enumerate(channels):
        ax_kde = axes[i, 0]
        ax_box = axes[i, 1]

        # Density curves without fill.
        for cls in df["label_name"].cat.categories:
            values = cast(
                pd.Series,
                df_stats.loc[(df_stats["channel"] == ch) & (df_stats["label"] == cls), "value"],
            )
            sns.kdeplot(x=values, ax=ax_kde, label=cls, color=colors[cls], clip=(0, 255), linewidth=2, alpha=0.95)
        ax_kde.set_title(f"KDE – Mittlerer {ch}-Kanal – {dataset_name}")
        ax_kde.set_ylabel("Dichte")
        ax_kde.set_xlabel("Mittlerer Kanalwert" if i == len(channels) - 1 else "")
        ax_kde.grid(True, linestyle="--", alpha=0.5)
        ax_kde.legend(title="Klasse")

        # Use hue to preserve class colors.
        sns.boxplot(
            data=cast(pd.DataFrame, df_stats.loc[df_stats["channel"] == ch]),
            x="channel",
            y="value",
            hue="label",
            palette=colors,
            ax=ax_box,
        )
        ax_box.set_title(f"Boxplot – Kanal {ch}")
        ax_box.set_xlabel("")
        ax_box.set_ylabel("Mittlerer Wert")
        ax_box.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(f"Kanalverteilungen – {color_space} – {dataset_name}", fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=0.93)
    return fig


def plot_channel_correlation(df: pd.DataFrame, dataset_name: str = "") -> Figure:
    """Plot class-wise correlations of per-image Y, Cb, and Cr means.

    Parameters
    ----------
    df
        Dataframe containing image paths and class labels.
    dataset_name
        Human-readable dataset name included in the figure title.

    Returns
    -------
    matplotlib.figure.Figure
        One correlation heatmap per observed class.

    Raises
    ------
    ValueError
        If no valid class labels or decoded channel rows remain.
    """
    df = df.copy()
    means = {"label_name": [], "Y": [], "Cb": [], "Cr": []}

    for _, row in df.iterrows():
        try:
            with Image.open(str(row["path"])) as opened:
                arr = np.array(opened.convert("YCbCr"))
            means["label_name"].append(row["label_name"])
            means["Y"].append(arr[:, :, 0].mean())
            means["Cb"].append(arr[:, :, 1].mean())
            means["Cr"].append(arr[:, :, 2].mean())
        except Exception:
            continue

    df_means = pd.DataFrame(means)
    fig, axs = plt.subplots(1, len(df_means["label_name"].unique()), figsize=(15, 4))

    df_means["label_name"] = pd.Categorical(
        df_means["label_name"], categories=df["label_name"].cat.categories, ordered=True
    )

    for i, label in enumerate(df_means["label_name"].cat.categories):
        values = cast(pd.DataFrame, df_means.loc[df_means["label_name"] == label, ["Y", "Cb", "Cr"]])
        corr = values.corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, square=True, ax=axs[i])
        axs[i].set_title(str(label))

    fig.suptitle(f"Korrelationsmatrizen YCbCr pro Klasse – {dataset_name}")
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    return fig


def show_outliers_by_channel(df: pd.DataFrame, dataset_name: str = "", z_thresh: float = 3.0) -> widgets.VBox:
    """Build display-free controls for channel-mean outlier figures.

    Parameters
    ----------
    df
        Dataframe containing image paths and class labels.
    dataset_name
        Human-readable dataset name included in the figure title.
    z_thresh
        Absolute z-score threshold used to select outliers.

    Returns
    -------
    ipywidgets.VBox
        Channel selector, navigation controls, and rendered output.

    Raises
    ------
    ValueError
        If the label contract is invalid or the threshold is non-positive.
    """
    if z_thresh <= 0:
        raise ValueError("z_thresh must be positive.")
    channel_map = {"Y": 0, "Cb": 1, "Cr": 2}
    dropdown_channel = widgets.Dropdown(options=list(channel_map.keys()), description="Kanal:")
    idx_input = widgets.BoundedIntText(value=0, min=0, description="Gruppe:")
    btn_prev = widgets.Button(description="←", layout=widgets.Layout(width="40px"))
    btn_next = widgets.Button(description="→", layout=widgets.Layout(width="40px"))
    btn_row = widgets.HBox([idx_input, btn_prev, btn_next])
    controller = FigureController([dropdown_channel, btn_row])

    # Cache channel means and outlier groups for widget navigation.
    channel_means_cache: dict[str, pd.DataFrame] = {}
    grouped_all: dict[str, list[pd.DataFrame]] = {}

    def compute_channel_means(channel: str) -> pd.DataFrame:
        if channel in channel_means_cache:
            return channel_means_cache[channel]
        channel_means_cache[channel] = _channel_means(df, channel, channel_map[channel])
        return channel_means_cache[channel]

    def compute_outliers(channel: str) -> list[pd.DataFrame]:
        if channel in grouped_all:
            return grouped_all[channel]
        df_chan = compute_channel_means(channel)
        grouped_all[channel] = _outlier_groups(df_chan, z_thresh)
        return grouped_all[channel]

    def render() -> None:
        channel = str(dropdown_channel.value)
        controller.set_figure(
            _plot_outlier_group(
                compute_outliers(channel),
                dataset_name=dataset_name,
                z_thresh=z_thresh,
                channel=channel,
                group_index=idx_input.value,
            )
        )

    changing_channel = False

    def update_channel(change: dict[str, object]) -> None:
        nonlocal changing_channel
        channel = str(change["new"])
        grouped = compute_outliers(channel)
        changing_channel = True
        try:
            idx_input.max = max(0, len(grouped) - 1)
            idx_input.value = 0
        finally:
            changing_channel = False
        render()

    def on_change_idx(_: dict[str, object]) -> None:
        if not changing_channel:
            render()

    def step(delta: int) -> None:
        idx_input.value = max(0, min(idx_input.max, idx_input.value + delta))

    def previous(_: widgets.Button) -> None:
        step(-1)

    def next_group(_: widgets.Button) -> None:
        step(1)

    controller.register_observer(dropdown_channel, update_channel, names="value")
    controller.register_observer(idx_input, on_change_idx, names="value")
    controller.on_click(btn_prev, previous)
    controller.on_click(btn_next, next_group)

    try:
        update_channel({"new": dropdown_channel.value})
    except Exception:
        controller.close()
        raise
    return controller


def plot_outliers_by_channel(
    df: pd.DataFrame,
    dataset_name: str = "",
    z_thresh: float = 3.0,
    *,
    channel: str = "Y",
    group_index: int = 0,
) -> Figure:
    """Plot one channel-mean outlier group without displaying it."""
    channel_map = {"Y": 0, "Cb": 1, "Cr": 2}
    if channel not in channel_map:
        raise ValueError(f"Unknown channel: {channel!r}.")
    channel_means = _channel_means(df, channel, channel_map[channel])
    return _plot_outlier_group(
        _outlier_groups(channel_means, z_thresh),
        dataset_name=dataset_name,
        z_thresh=z_thresh,
        channel=channel,
        group_index=group_index,
    )


def _channel_means(df: pd.DataFrame, channel: str, channel_index: int) -> pd.DataFrame:
    if channel_index not in {0, 1, 2}:
        raise ValueError("channel_index must identify Y, Cb, or Cr.")
    means: list[float] = []
    for path in df["path"]:
        try:
            with Image.open(path) as opened:
                value = float(np.array(opened.convert("YCbCr"))[:, :, channel_index].mean())
            means.append(value)
        except Exception:
            means.append(np.nan)
    frame = df.copy()
    frame["motif_id"] = frame["path"].apply(lambda path: Path(path).name)
    frame[f"{channel}_mean"] = means
    frame = frame.dropna(subset=[f"{channel}_mean"])
    z_scores = np.asarray(zscore(frame[f"{channel}_mean"]), dtype=np.float64)
    if z_scores.ndim != 1 or z_scores.shape[0] != len(frame):
        raise ValueError("Channel z-scores must be one-dimensional and match the dataframe length.")
    frame["z_score"] = pd.Series(z_scores, index=frame.index, dtype="float64")
    return frame


def _outlier_groups(channel_means: pd.DataFrame, z_thresh: float) -> list[pd.DataFrame]:
    if z_thresh <= 0:
        raise ValueError("z_thresh must be positive.")
    z_scores = cast(pd.Series, channel_means["z_score"])
    outlier_ids = cast(pd.Series, channel_means.loc[np.abs(z_scores) > z_thresh, "motif_id"]).unique()
    groups = [
        cast(pd.DataFrame, channel_means.loc[channel_means["motif_id"] == motif_id]).sort_values("label_name")
        for motif_id in outlier_ids
    ]
    return [group for group in groups if len(group) > 1]


def _plot_outlier_group(
    groups: list[pd.DataFrame],
    *,
    dataset_name: str,
    z_thresh: float,
    channel: str,
    group_index: int,
) -> Figure:
    if not groups:
        figure, axis = plt.subplots(figsize=(8, 3))
        axis.text(
            0.5,
            0.5,
            f"Keine Ausreissergruppen in Kanal {channel} (Z > {z_thresh}) gefunden.",
            ha="center",
            va="center",
        )
        axis.axis("off")
        return figure
    if not 0 <= group_index < len(groups):
        raise ValueError("group_index is outside the outlier-group range.")
    group = groups[group_index]
    figure, axes = plt.subplots(1, len(group), figsize=(len(group) * 3, 4))
    flattened_axes = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for axis, (_, row) in zip(flattened_axes, group.iterrows(), strict=False):
        try:
            with Image.open(str(row["path"])) as opened:
                axis.imshow(opened.convert("RGB"))
        except Exception:
            axis.text(0.5, 0.5, "Fehler", ha="center", va="center")
        axis.set_title(f"{row['label_name']}\nZ={row['z_score']:.2f}")
        axis.axis("off")
    figure.suptitle(f"Motiv: {group['motif_id'].iloc[0]} – Kanal {channel} – {dataset_name}", fontsize=12)
    figure.tight_layout()
    return figure
