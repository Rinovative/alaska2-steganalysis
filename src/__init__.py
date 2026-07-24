"""Application interfaces for the ALASKA2 steganalysis baseline.

Provides:
- config: portable paths, devices, and reproducibility settings
- data: dataset discovery, validation, metadata, splitting, and synthetic data
- datasets: PyTorch dataset and DataLoader adapters
- eda: exploratory analysis used by the academic notebook
- evaluation: metrics, inference, and diagnostic figures
- models: baseline neural-network architectures and freezing controls
- presentation: curated-figure caching and notebook widgets
- training: training loops, checkpoints, and staged fine-tuning
- transforms: JPEG-grid-aware spatial transformations
"""

from __future__ import annotations

from . import config, data, datasets, eda, evaluation, models, presentation, training, transforms

__all__ = [
    "config",
    "data",
    "datasets",
    "eda",
    "evaluation",
    "models",
    "presentation",
    "training",
    "transforms",
]
