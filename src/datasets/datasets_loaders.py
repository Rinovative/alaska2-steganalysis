"""
===============================================================================
datasets_loaders.py
===============================================================================
Reproducible loader construction and statistics.

Responsibilities:
  - Build distinct train, validation, and test DataLoader objects.
  - Apply safe worker options for CPU and accelerator execution.
  - Construct spatial image datasets from an existing split contract.
  - Compute per-channel image statistics in one loader pass.

Design principles:
  - Each loader receives its own seeded generator. Worker-only options are
    omitted when num_workers is zero.

Boundaries:
  - This module consumes existing splits and transforms; it does not define
    split policy, transforms, or models.

Notes:
  - Pinned memory and non-blocking transfers help only when CUDA is
    available.
===============================================================================
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, TypeVar

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ..config.config_runtime import make_generator, seed_worker
from ..data.data_split import DatasetSplits
from ..transforms.transforms_spatial import AlignedDeterministicCrop
from .datasets_images import ImageDataset, ImageTransform

__all__ = [
    "LoaderBundle",
    "build_image_loaders",
    "build_loaders",
    "compute_channel_statistics",
    "compute_image_channel_statistics",
]

Sample = TypeVar("Sample")


@dataclass(frozen=True, slots=True)
class LoaderBundle:
    """Store distinct train, validation, and test DataLoaders.

    Parameters
    ----------
    train
        Shuffled training loader.
    validation
        Deterministic validation loader.
    test
        Deterministic final-test loader.
    """

    train: DataLoader
    validation: DataLoader
    test: DataLoader


def build_loaders(
    train_dataset: Dataset,
    validation_dataset: Dataset,
    test_dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int | None = None,
    seed: int = 42,
    prefetch_factor: int = 2,
    pin_memory: bool | None = None,
) -> LoaderBundle:
    """Construct independent reproducible train, validation, and test DataLoaders.

    Parameters
    ----------
    train_dataset
        Dataset used only by the shuffled training loader.
    validation_dataset
        Distinct dataset used only for model selection.
    test_dataset
        Distinct dataset reserved for final evaluation.
    batch_size
        Positive samples per batch.
    num_workers
        Worker count or ``None`` for a bounded CPU-derived default.
    seed
        Base seed; validation and test use deterministic offsets.
    prefetch_factor
        Worker-side batches prefetched when workers are enabled.
    pin_memory
        Pinned-memory policy or ``None`` to follow CUDA availability.

    Returns
    -------
    LoaderBundle
        Three independent loaders with split-appropriate shuffling.

    Raises
    ------
    ValueError
        If datasets alias each other or loader arguments are invalid.

    Notes
    -----
    Worker-only options are disabled when ``num_workers`` is zero.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if prefetch_factor <= 0:
        raise ValueError("prefetch_factor must be positive.")
    if train_dataset is validation_dataset or train_dataset is test_dataset or validation_dataset is test_dataset:
        raise ValueError("Train, validation, and test datasets must be distinct objects.")
    workers = min(8, max(0, (os.cpu_count() or 1) // 2)) if num_workers is None else num_workers
    if workers < 0:
        raise ValueError("num_workers must be non-negative.")
    use_pin_memory = torch.cuda.is_available() if pin_memory is None else pin_memory

    def create_loader(dataset: Dataset, *, shuffle: bool, loader_seed: int) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=use_pin_memory,
            worker_init_fn=seed_worker,
            persistent_workers=workers > 0,
            prefetch_factor=prefetch_factor if workers > 0 else None,
            generator=make_generator(loader_seed),
        )

    return LoaderBundle(
        train=create_loader(train_dataset, shuffle=True, loader_seed=seed),
        validation=create_loader(validation_dataset, shuffle=False, loader_seed=seed + 1),
        test=create_loader(test_dataset, shuffle=False, loader_seed=seed + 2),
    )


def compute_channel_statistics(loader: DataLoader) -> tuple[list[float], list[float]]:
    """Compute population channel mean and standard deviation in one loader pass.

    Parameters
    ----------
    loader
        Loader yielding image batches shaped ``[N, C, H, W]``.

    Returns
    -------
    tuple[list[float], list[float]]
        Per-channel population means and standard deviations.

    Raises
    ------
    ValueError
        If a batch shape is invalid or the loader yields no pixels.
    """
    channel_sum: torch.Tensor | None = None
    channel_square_sum: torch.Tensor | None = None
    pixel_count = 0
    with torch.inference_mode():
        for images, _ in loader:
            if images.ndim != 4:
                raise ValueError("Expected image batches with shape [N, C, H, W].")
            images = images.to(dtype=torch.float64)
            batch_sum = images.sum(dim=(0, 2, 3))
            batch_square_sum = images.square().sum(dim=(0, 2, 3))
            channel_sum = batch_sum if channel_sum is None else channel_sum + batch_sum
            channel_square_sum = (
                batch_square_sum if channel_square_sum is None else channel_square_sum + batch_square_sum
            )
            pixel_count += images.shape[0] * images.shape[2] * images.shape[3]
    if channel_sum is None or channel_square_sum is None or pixel_count == 0:
        raise ValueError("Cannot compute statistics from an empty loader.")
    mean = channel_sum / pixel_count
    variance = (channel_square_sum / pixel_count - mean.square()).clamp_min(0)
    return mean.tolist(), variance.sqrt().tolist()


def build_image_loaders(
    splits: DatasetSplits,
    *,
    color_mode: Literal["RGB", "YCbCr", "Y"],
    train_transform: ImageTransform,
    evaluation_transform: ImageTransform,
    identity_crop: AlignedDeterministicCrop,
    batch_size: int,
    num_workers: int | None = None,
    seed: int = 42,
    prefetch_factor: int = 2,
) -> LoaderBundle:
    """Construct split-isolated spatial image datasets and loaders.

    Parameters
    ----------
    splits
        Existing source-isolated dataframe partitions.
    color_mode
        Explicit decoded image mode used by all three datasets.
    train_transform
        Training-only PIL-to-tensor transform.
    evaluation_transform
        Shared validation and test PIL-to-tensor transform.
    identity_crop
        Source-aware deterministic crop for validation and test.
    batch_size
        Positive samples per batch.
    num_workers
        Worker count or None for the standard bounded default.
    seed
        Base loader and worker seed.
    prefetch_factor
        Worker-side batches prefetched when workers are enabled.

    Returns
    -------
    LoaderBundle
        Distinct training, validation, and final-test loaders.

    Raises
    ------
    ValueError
        If dataset or loader arguments violate their maintained contracts.
    """
    train_dataset = ImageDataset(
        splits.train,
        color_mode=color_mode,
        target_column="label_bin",
        transform=train_transform,
    )
    validation_dataset = ImageDataset(
        splits.validation,
        color_mode=color_mode,
        target_column="label_bin",
        transform=evaluation_transform,
        identity_crop=identity_crop,
    )
    test_dataset = ImageDataset(
        splits.test,
        color_mode=color_mode,
        target_column="label_bin",
        transform=evaluation_transform,
        identity_crop=identity_crop,
    )
    return build_loaders(
        train_dataset,
        validation_dataset,
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        prefetch_factor=prefetch_factor,
    )


def compute_image_channel_statistics(
    dataframe: pd.DataFrame,
    *,
    color_mode: Literal["RGB", "YCbCr", "Y"],
    transform: ImageTransform,
    identity_crop: AlignedDeterministicCrop,
    batch_size: int,
    num_workers: int = 0,
) -> tuple[list[float], list[float]]:
    """Compute decoded image-channel statistics from one dataframe partition.

    Parameters
    ----------
    dataframe
        Training rows containing paths, source identities, and binary labels.
    color_mode
        Explicit decoded image mode.
    transform
        PIL-to-tensor transform without normalization.
    identity_crop
        Source-aware deterministic crop defining the measured pixels.
    batch_size
        Positive samples per statistics batch.
    num_workers
        Non-negative DataLoader worker count.

    Returns
    -------
    tuple[list[float], list[float]]
        Per-channel population means and standard deviations.

    Raises
    ------
    ValueError
        If arguments, rows, tensors, or the resulting loader are invalid.
    """
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative.")
    dataset = ImageDataset(
        dataframe,
        color_mode=color_mode,
        target_column="label_bin",
        transform=transform,
        identity_crop=identity_crop,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    return compute_channel_statistics(loader)
