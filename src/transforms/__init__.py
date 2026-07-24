"""JPEG-grid-aware spatial transformation interfaces.

Provides:
- shuffle: spatial tile-shuffle augmentation with explicit semantics
- spatial: aligned random and source-deterministic crop transforms
"""

from __future__ import annotations

from . import transforms_shuffle as shuffle
from . import transforms_spatial as spatial

__all__ = ["shuffle", "spatial"]
