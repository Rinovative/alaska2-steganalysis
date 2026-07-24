"""Project configuration and runtime validation interfaces.

Provides:
- device: explicit device resolution and CUDA preflight checks
- paths: repository-relative paths and dataset selection
- runtime: reproducibility settings and DataLoader worker seeding
"""

from __future__ import annotations

from . import config_device as device
from . import config_paths as paths
from . import config_runtime as runtime

__all__ = ["device", "paths", "runtime"]
