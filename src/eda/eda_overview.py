"""
===============================================================================
eda_overview.py
===============================================================================
Dataset structure and distribution summaries.

Responsibilities:
  - Format a concise dataframe overview for notebook display.
  - Plot class and JPEG quality distributions.

Design principles:
  - Functions return text or figures and do not display or save them.

Boundaries:
  - German figure text is retained for the protected academic notebook.

Notes:
  - Metadata columns are supplied by data.metadata.
===============================================================================
"""

from __future__ import annotations

import io

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

__all__ = ["plot_class_distribution", "plot_jpeg_quality_distribution", "show_dataset_overview"]


def show_dataset_overview(df: pd.DataFrame, dataset_name: str = "") -> str:
    """Format selected structure and descriptive statistics for notebook display.

    Parameters
    ----------
    df
        Metadata dataframe containing the documented overview columns.
    dataset_name
        Human-readable dataset name included in the figure title.

    Returns
    -------
    str
        Markdown-oriented dataframe structure and statistics text.

    Raises
    ------
    KeyError
        If required overview or metadata columns are missing.
    """
    output = io.StringIO()

    print(f"## Datensatzübersicht: {dataset_name}", file=output)

    # Selected-column structure.
    print("\n### Struktur ausgewählter Spalten", file=output)
    selected_cols = ["path", "label_name", "jpeg_quality", "width", "height", "q_y_00", "q_y_63"]
    df[selected_cols].info(buf=output)

    # Overall summary statistics.
    print("\n### Gesamte Statistik ausgewählter Spalten", file=output)
    summary_cols = [
        "jpeg_quality",
        "width",
        "height",
        "mode",
        "q_y_00",
        "q_y_01",
        "q_y_02",
        "q_y_03",
        "q_y_10",
        "q_y_11",
        "q_y_20",
        "q_y_21",
        "q_y_30",
        "q_y_31",
        "q_y_40",
        "q_y_41",
        "q_y_50",
        "q_y_51",
        "q_y_60",
        "q_y_61",
        "q_y_62",
        "q_y_63",
    ]
    print(df[summary_cols].describe().to_string(), file=output)

    # Class-wise summary statistics.
    if "label_name" in df.columns:
        print("\n### Gruppierte Statistik nach Klasse ausgewählter Spalten", file=output)

        # Keep the grouped summary concise.
        summary_cols = ["width", "height", "mode", "q_y_00", "q_y_63"]
        df_summary = df[[*summary_cols, "label_name"]]

        grouped = df_summary.groupby("label_name", observed=False).describe()
        print(grouped.to_string(), file=output)

    return output.getvalue()


def plot_class_distribution(df: pd.DataFrame, dataset_name: str = "") -> Figure:
    """Plot image counts for every observed class.

    Parameters
    ----------
    df
        Dataframe containing image paths and class labels.
    dataset_name
        Human-readable dataset name included in the figure title.

    Returns
    -------
    matplotlib.figure.Figure
        Class-count bar chart.

    Raises
    ------
    KeyError
        If the dataframe has no ``label_name`` column.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    df["label_name"].value_counts().sort_index().plot(kind="bar", ax=ax, color="skyblue", edgecolor="black")
    ax.set_title(f"Klassenverteilung – {dataset_name}")
    ax.set_xlabel("Klasse")
    ax.set_ylabel("Anzahl Bilder")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    return fig


def plot_jpeg_quality_distribution(df: pd.DataFrame, dataset_name: str = "") -> Figure:
    """Plot JPEG quality-level counts by observed class.

    Parameters
    ----------
    df
        Metadata dataframe containing class and JPEG quality columns.
    dataset_name
        Human-readable dataset name included in the figure title.

    Returns
    -------
    matplotlib.figure.Figure
        Grouped quality-level bar chart.

    Raises
    ------
    KeyError
        If class or JPEG quality columns are missing.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    df.groupby(["jpeg_quality", "label_name"], observed=False).size().unstack().plot(kind="bar", ax=ax)
    ax.set_title(f"JPEG-Qualitätsverteilung pro Klasse – {dataset_name}")
    ax.set_xlabel("JPEG-Qualitätsstufe")
    ax.set_ylabel("Anzahl Bilder")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(title="Klasse")
    return fig
