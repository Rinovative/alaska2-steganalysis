"""
===============================================================================
config_runtime.py
===============================================================================
Central reproducibility and worker configuration.

Responsibilities:
  - Seed Python, NumPy, and PyTorch from one explicit value.
  - Construct independent, repeatable PyTorch generators.
  - Seed data-loader workers from PyTorch's per-worker seed.

Design principles:
  - Determinism is opt-in and explicit. Mutating global random state happens
    only when seed_everything is called by an entry point such as the
    notebook.

Boundaries:
  - This module does not select devices, datasets, models, or output
    directories.

Notes:
  - Perfect bitwise reproducibility can still depend on the PyTorch build and
    hardware.
===============================================================================
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch

__all__ = ["ReproducibilityConfig", "make_generator", "seed_everything", "seed_worker"]


@dataclass(frozen=True, slots=True)
class ReproducibilityConfig:
    """Configure project-wide random seeding and deterministic PyTorch behavior.

    Parameters
    ----------
    seed
        Seed shared by Python, NumPy, PyTorch, and worker generators.
    deterministic_algorithms
        Whether PyTorch deterministic algorithms are requested.
    warn_only
        Whether unavailable deterministic implementations warn instead of raising.
    """

    seed: int = 42
    deterministic_algorithms: bool = True
    warn_only: bool = True


def seed_everything(config: ReproducibilityConfig | int = 42) -> None:
    """Seed supported random generators and configure deterministic PyTorch execution.

    Parameters
    ----------
    config
        Reproducibility configuration or a shorthand integer seed.

    Returns
    -------
    None
        Global random state and PyTorch backend flags are configured in place.

    Notes
    -----
    Bitwise identity can still depend on the installed PyTorch build and accelerator hardware.
    """
    if isinstance(config, int):
        config = ReproducibilityConfig(seed=config)
    os.environ["PYTHONHASHSEED"] = str(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(config.deterministic_algorithms, warn_only=config.warn_only)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = not config.deterministic_algorithms
        torch.backends.cudnn.deterministic = config.deterministic_algorithms


def make_generator(seed: int) -> torch.Generator:
    """Create an independently seeded CPU generator for a DataLoader.

    Parameters
    ----------
    seed
        Seed applied only to the returned generator.

    Returns
    -------
    torch.Generator
        Independent CPU generator with deterministic state.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:
    """Seed Python and NumPy inside a PyTorch DataLoader worker.

    Parameters
    ----------
    worker_id
        Worker identifier supplied by PyTorch; the value is not used directly.

    Returns
    -------
    None
        Worker-local Python and NumPy random states are updated.
    """
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
