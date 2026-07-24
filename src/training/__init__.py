"""Training, checkpoint, and staged fine-tuning interfaces.

Provides:
- artifacts: typed single-run and staged artifact path resolution
- checkpoint: device-safe atomic checkpoint save and load helpers
- loop: binary training with one-pass validation and best restoration
- staged: EfficientNet stage orchestration with global best-state handoff
"""

from __future__ import annotations

from . import training_artifacts as artifacts
from . import training_checkpoint as checkpoint
from . import training_loop as loop
from . import training_staged as staged

__all__ = ["artifacts", "checkpoint", "loop", "staged"]
