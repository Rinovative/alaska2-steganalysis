"""
===============================================================================
eda_examples.py
===============================================================================
Interactive spatial image examples.

Responsibilities:
  - Browse class-specific image grids.
  - Compare every complete variant of one source identity side by side.

Design principles:
  - Labels are discovered from the dataframe, preserving the shared public
    class contract.

Boundaries:
  - German widget text is retained for the protected academic notebook.

Notes:
  - Input indexes are expected to have complete source groups.
===============================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import ipywidgets as widgets
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from PIL import Image

from .eda_controls import FigureController
from .eda_style import label_order

__all__ = [
    "make_cover_stego_comparison_widget",
    "make_image_grid_widget",
    "plot_cover_stego_comparison",
    "plot_image_grid",
]


def plot_image_grid(
    df: pd.DataFrame,
    dataset_name: str = "",
    rows: int = 4,
    cols: int = 4,
    *,
    class_name: str | None = None,
    page_index: int = 0,
) -> Figure:
    """Plot one page of class-specific image examples.

    Parameters
    ----------
    df
        Dataframe containing image paths and class labels.
    dataset_name
        Human-readable dataset name included in the figure title.
    rows
        Positive grid row count.
    cols
        Positive grid column count.

    class_name
        Selected class, defaulting to the first observed category.
    page_index
        Zero-based page to plot.

    Returns
    -------
    matplotlib.figure.Figure
        Image-grid figure without displaying it.

    Raises
    ------
    ValueError
        If rows, columns, the class selection, or page index is invalid.
    """
    classes, grouped, max_steps = _image_grid_state(df, rows, cols)
    selected_class = classes[0] if class_name is None else class_name
    return _plot_image_grid_page(
        grouped,
        max_steps,
        dataset_name=dataset_name,
        rows=rows,
        cols=cols,
        class_name=selected_class,
        page_index=page_index,
    )


def _plot_image_grid_page(
    grouped: dict[str, pd.DataFrame],
    max_steps: dict[str, int],
    *,
    dataset_name: str,
    rows: int,
    cols: int,
    class_name: str,
    page_index: int,
) -> Figure:
    if class_name not in grouped:
        raise ValueError(f"Unknown image-grid class: {class_name!r}.")
    if not 0 <= page_index <= max_steps[class_name]:
        raise ValueError("page_index is outside the selected class range.")

    imgs_per_page = rows * cols
    subset = grouped[class_name]
    start = page_index * imgs_per_page
    rows_subset = subset.iloc[start : start + imgs_per_page]

    figure, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.2))
    figure.suptitle(f"{class_name} - Seite {page_index} - {dataset_name}", fontsize=14)
    flattened_axes = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for axis in flattened_axes:
        axis.axis("off")
    for axis, (_, row) in zip(flattened_axes, rows_subset.iterrows(), strict=False):
        try:
            with Image.open(row["path"]) as opened:
                axis.imshow(opened.convert("RGB"))
        except Exception:
            axis.text(0.5, 0.5, "Fehler", ha="center", va="center")
    figure.tight_layout()
    return figure


def make_image_grid_widget(
    df: pd.DataFrame,
    dataset_name: str = "",
    rows: int = 4,
    cols: int = 4,
) -> FigureController:
    """Build display-free controls for the pure image-grid renderer."""
    classes, grouped, max_steps = _image_grid_state(df, rows, cols)
    initial_class = classes[0]
    dropdown = widgets.Dropdown(options=classes, description="Klasse:")
    idx_box = widgets.BoundedIntText(value=0, min=0, max=max_steps[initial_class], description="Seite:")
    btn_prev = widgets.Button(description="←", layout=widgets.Layout(width="40px"))
    btn_next = widgets.Button(description="→", layout=widgets.Layout(width="40px"))
    controls = widgets.HBox([idx_box, btn_prev, btn_next])
    controller = FigureController([dropdown, controls])
    changing_class = False

    def render() -> None:
        controller.set_figure(
            _plot_image_grid_page(
                grouped,
                max_steps,
                dataset_name=dataset_name,
                rows=rows,
                cols=cols,
                class_name=str(dropdown.value),
                page_index=idx_box.value,
            )
        )

    def on_change_class(change: dict[str, object]) -> None:
        nonlocal changing_class
        changing_class = True
        try:
            idx_box.max = max_steps[str(change["new"])]
            idx_box.value = 0
        finally:
            changing_class = False
        render()

    def on_change_idx(_: dict[str, object]) -> None:
        if not changing_class:
            render()

    def step(delta: int) -> None:
        idx_box.value = max(0, min(idx_box.max, idx_box.value + delta))

    def previous(_: widgets.Button) -> None:
        step(-1)

    def next_page(_: widgets.Button) -> None:
        step(1)

    controller.register_observer(dropdown, on_change_class, names="value")
    controller.register_observer(idx_box, on_change_idx, names="value")
    controller.on_click(btn_prev, previous)
    controller.on_click(btn_next, next_page)
    try:
        render()
    except Exception:
        controller.close()
        raise
    return controller


def _image_grid_state(
    df: pd.DataFrame,
    rows: int,
    cols: int,
) -> tuple[list[str], dict[str, pd.DataFrame], dict[str, int]]:
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive.")
    frame = df.copy()
    frame["label_name"] = frame["label_name"].astype("category")
    classes = [str(value) for value in frame["label_name"].cat.categories.tolist()]
    if not classes:
        raise ValueError("At least one image-grid class is required.")
    grouped = {
        class_name: cast(pd.DataFrame, frame.loc[frame["label_name"] == class_name]).reset_index(drop=True)
        for class_name in classes
    }
    imgs_per_page = rows * cols
    max_steps = {cls: len(grouped[cls]) // imgs_per_page for cls in classes}
    return classes, grouped, max_steps


def plot_cover_stego_comparison(
    df: pd.DataFrame,
    dataset_name: str = "",
    *,
    source_index: int = 0,
) -> Figure:
    """Plot every variant of one complete source group.

    Parameters
    ----------
    df
        Dataframe containing image paths and class labels.
    dataset_name
        Human-readable dataset name included in the figure title.

    source_index
        Zero-based complete source-group index.

    Returns
    -------
    matplotlib.figure.Figure
        Comparison figure without displaying it.

    Raises
    ------
    ValueError
        If no complete Cover/stego source group is available.
    """
    frame, labels, complete_groups = _comparison_state(df)
    return _plot_cover_stego_comparison_group(
        frame,
        labels,
        complete_groups,
        dataset_name=dataset_name,
        source_index=source_index,
    )


def _plot_cover_stego_comparison_group(
    frame: pd.DataFrame,
    labels: list[str],
    complete_groups: list[object],
    *,
    dataset_name: str,
    source_index: int,
) -> Figure:
    if not 0 <= source_index < len(complete_groups):
        raise ValueError("source_index is outside the complete source-group range.")
    base_id = complete_groups[source_index]
    group = frame[frame["base_name"] == base_id]
    paths = {label: str(cast(pd.Series, group.loc[group["label_name"] == label, "path"]).iloc[0]) for label in labels}
    figure = plt.figure(figsize=(22, 6), constrained_layout=True)
    spec = gridspec.GridSpec(ncols=len(labels), nrows=1, figure=figure)
    axes = [figure.add_subplot(spec[0, index]) for index in range(len(labels))]
    figure.suptitle(f"Vergleich – ID {base_id} – {dataset_name}", fontsize=16)
    for axis, label in zip(axes, labels, strict=True):
        try:
            with Image.open(paths[label]) as opened:
                axis.imshow(opened.convert("RGB"))
            axis.set_title(label)
        except Exception:
            axis.text(0.5, 0.5, "Fehler", ha="center", va="center")
        axis.axis("off")
    return figure


def make_cover_stego_comparison_widget(df: pd.DataFrame, dataset_name: str = "") -> FigureController:
    """Build display-free controls for the pure source comparison renderer."""
    frame, labels, complete_groups = _comparison_state(df)
    idx_input = widgets.BoundedIntText(value=0, min=0, max=len(complete_groups) - 1, description="Index:")
    btn_prev = widgets.Button(description="←", layout=widgets.Layout(width="40px"))
    btn_next = widgets.Button(description="→", layout=widgets.Layout(width="40px"))
    controls = widgets.HBox([idx_input, btn_prev, btn_next])
    controller = FigureController([controls])

    def render(_: dict[str, object] | None = None) -> None:
        controller.set_figure(
            _plot_cover_stego_comparison_group(
                frame,
                labels,
                complete_groups,
                dataset_name=dataset_name,
                source_index=idx_input.value,
            )
        )

    def step(delta: int) -> None:
        idx_input.value = max(0, min(idx_input.max, idx_input.value + delta))

    def previous(_: widgets.Button) -> None:
        step(-1)

    def next_source(_: widgets.Button) -> None:
        step(1)

    controller.register_observer(idx_input, render, names="value")
    controller.on_click(btn_prev, previous)
    controller.on_click(btn_next, next_source)
    try:
        render()
    except Exception:
        controller.close()
        raise
    return controller


def _comparison_state(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[object]]:
    frame = df.copy()
    frame["filename"] = frame["path"].apply(lambda path: Path(path).name)
    frame["base_name"] = frame["filename"].str.extract(r"(\d+)\.jpg")
    labels = label_order(frame)
    group_counts = cast(pd.Series, frame.groupby("base_name")["label_name"].nunique())
    complete_groups = group_counts.loc[group_counts == len(labels)].index.sort_values().tolist()
    if not complete_groups:
        raise ValueError("No complete Cover/Stego source groups are available for comparison.")
    return frame, labels, complete_groups
