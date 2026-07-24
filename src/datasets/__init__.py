"""Dataset construction and loading interfaces.

Provides:
- dct: JPEG coefficient and aligned fusion dataset implementations
- images: spatial image dataset implementations
- loaders: train, validation, and test DataLoader construction
"""

from __future__ import annotations

from . import datasets_dct as dct
from . import datasets_images as images
from . import datasets_loaders as loaders

__all__ = ["dct", "images", "loaders"]
