"""Dataset preparation, validation, and integrity interfaces.

Provides:
- index: group-complete file indexing and target construction
- metadata: validated JPEG metadata extraction
- preflight: read-only ALASKA2 training-readiness validation
- preparation: complete indexing and dataset-specific split orchestration
- split: leakage-safe grouped train, validation, and test splitting
- synthetic: explicit PD12M proxy acquisition and generation
"""

from __future__ import annotations

from . import data_index as index
from . import data_metadata as metadata
from . import data_preflight as preflight
from . import data_preparation as preparation
from . import data_split as split
from . import data_synthetic as synthetic

__all__ = ["index", "metadata", "preflight", "preparation", "split", "synthetic"]
