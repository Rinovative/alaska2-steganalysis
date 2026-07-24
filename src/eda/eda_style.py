"""
===============================================================================
eda_style.py
===============================================================================
Shared, dataset-aware EDA presentation style.

Responsibilities:
  - Preserve categorical label order when present.
  - Support the shared ALASKA2 and synthetic compatibility labels.
  - Build a stable colorblind palette for the observed classes.

Design principles:
  - Style follows the dataframe rather than a hard-coded scientific identity.

Boundaries:
  - This module contains no plotting or data extraction.

Notes:
  - Cover is placed first when the input is not already an ordered
    categorical.
===============================================================================
"""

from __future__ import annotations

import pandas as pd
import seaborn as sns

__all__ = ["label_colors", "label_order"]


def label_order(dataframe: pd.DataFrame) -> list[str]:
    """Return non-empty labels in intentional display order.

    Parameters
    ----------
    dataframe
        Dataframe containing the public ``label_name`` column.

    Returns
    -------
    list[str]
        Observed class names with Cover first unless categorical order is explicit.

    Raises
    ------
    ValueError
        If the label column is missing or contains no usable labels.
    """
    if "label_name" not in dataframe:
        raise ValueError("dataframe must contain label_name.")
    series = dataframe["label_name"]
    if isinstance(series.dtype, pd.CategoricalDtype):
        categories = [str(value) for value in series.cat.categories if value in set(series.dropna())]
    else:
        categories = sorted(str(value) for value in series.dropna().unique())
        if "Cover" in categories:
            categories.remove("Cover")
            categories.insert(0, "Cover")
    if not categories:
        raise ValueError("No labels available for EDA.")
    return categories


def label_colors(dataframe: pd.DataFrame) -> dict[str, tuple[float, float, float]]:
    """Build a stable colorblind palette keyed by observed label.

    Parameters
    ----------
    dataframe
        Dataframe providing label order and observed classes.

    Returns
    -------
    dict[str, tuple[float, float, float]]
        RGB color triple for every observed class.

    Raises
    ------
    ValueError
        If no valid labels are available.
    """
    order = label_order(dataframe)
    return dict(zip(order, sns.color_palette("colorblind", n_colors=len(order)), strict=True))
