"""Baseline model architecture and fine-tuning interfaces.

Provides:
- efficientnet: EfficientNet-B0 adapter for decoded YCbCr input
- freezing: explicit stage definitions and trainability controls
- tinycnn: compact luminance-channel baseline
"""

from __future__ import annotations

from . import models_efficientnet as efficientnet
from . import models_freezing as freezing
from . import models_tinycnn as tinycnn

__all__ = ["efficientnet", "freezing", "tinycnn"]
