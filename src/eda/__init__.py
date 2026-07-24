"""Exploratory analysis interfaces for the academic notebook.

Provides:
- channels: spatial color-channel statistics and outlier views
- dct: JPEG coefficient-change diagnostics
- examples: interactive image grids and paired comparisons
- overview: dataset structure and distribution figures
- style: dataset-aware label ordering and plotting colors
"""

from __future__ import annotations

from . import eda_channels as channels
from . import eda_dct as dct
from . import eda_examples as examples
from . import eda_overview as overview
from . import eda_style as style

__all__ = ["channels", "dct", "examples", "overview", "style"]
