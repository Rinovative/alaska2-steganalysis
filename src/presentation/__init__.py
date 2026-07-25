"""Notebook presentation and figure-cache interfaces.

Provides:
- cache: validated dataset-specific figure reads and writes
- widgets: lazy dropdown and tab widgets for notebook orchestration
"""

from __future__ import annotations

from . import presentation_cache as cache
from . import presentation_widgets as widgets

__all__ = ["cache", "widgets"]
