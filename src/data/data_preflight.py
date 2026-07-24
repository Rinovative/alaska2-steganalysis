"""
===============================================================================
data_preflight.py
===============================================================================
Validate a local ALASKA2 dataset before GPU training begins.

Responsibilities:
  - Validate the canonical four-class layout, file counts, and JPEG readability.
  - Reuse the project index and target logic to construct a real sample dataset.
  - Prove deterministic grouped splits with no source leakage.

Design principles:
  - Validation is deterministic, read-only, and fails with actionable diagnostics.
  - Existing indexing, dataset, and split contracts remain the single source of truth.

Boundaries:
  - The preflight never downloads, generates, modifies, or trains on images.
  - Dataset selection and synthetic proxy preparation remain separate concerns.
===============================================================================
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torchvision.transforms import ToTensor

from ..config.config_paths import CLASS_LABELS, default_paths
from ..datasets.datasets_images import ImageDataset
from .data_index import (
    DuplicateSourceError,
    IncompleteGroupError,
    add_targets,
    build_file_index,
    discover_jpeg_files,
)
from .data_split import DatasetSplits, assert_split_isolation, split_by_source

__all__ = ["DatasetPreflightError", "DatasetPreflightReport", "main", "run_dataset_preflight"]


class DatasetPreflightError(ValueError):
    """Raised when local dataset content is unsafe or invalid for training."""


@dataclass(frozen=True, slots=True)
class DatasetPreflightReport:
    """Describe a successfully validated ALASKA2 dataset.

    Parameters
    ----------
    root
        Canonical dataset root that was inspected.
    class_counts
        Number of supported JPEG files discovered in each class.
    source_groups
        Number of complete Cover/stego source identities.
    image_count
        Total indexed image count across all classes.
    split_group_counts
        Number of distinct source identities in each model partition.
    sample_shape
        Tensor shape produced by the constructed luminance sample dataset.
    sample_target
        Binary target produced for the constructed sample.
    """

    root: Path
    class_counts: dict[str, int]
    source_groups: int
    image_count: int
    split_group_counts: dict[str, int]
    sample_shape: tuple[int, ...]
    sample_target: float

    def format_summary(self) -> str:
        """Format the successful preflight as a terminal-readable summary.

        Returns
        -------
        str
            Multi-line dataset, class, split, and sample report.
        """
        class_text = ", ".join(f"{class_name}={count}" for class_name, count in self.class_counts.items())
        split_text = ", ".join(f"{split_name}={count}" for split_name, count in self.split_group_counts.items())
        return "\n".join(
            (
                "Dataset preflight passed.",
                "Selected source: ALASKA2",
                f"Dataset root: {self.root}",
                f"Class counts: {class_text}",
                f"Complete source groups: {self.source_groups}",
                f"Indexed images: {self.image_count}",
                f"Split source groups: {split_text}",
                f"Sample tensor: shape={self.sample_shape}, binary_target={self.sample_target:.1f}",
            )
        )


def _validate_readable_jpegs(files_by_class: Mapping[str, Mapping[str, Path]]) -> None:
    failures: list[str] = []
    for class_name, files in files_by_class.items():
        for source_id, path in files.items():
            try:
                with Image.open(path) as image:
                    image.load()
                    if image.format != "JPEG":
                        raise DatasetPreflightError(f"Decoded format is {image.format!r}, not JPEG")
            except (OSError, UnidentifiedImageError, DatasetPreflightError) as error:
                failures.append(f"{class_name}/{source_id}: {path.name}: {error}")
    if failures:
        examples = "; ".join(failures[:5])
        raise DatasetPreflightError(f"{len(failures)} unreadable or invalid JPEG files detected; examples: {examples}")


def _source_ids(frame: pd.DataFrame) -> tuple[str, ...]:
    values = cast(pd.Series, frame["source_id"]).astype(str).drop_duplicates().tolist()
    return tuple(values)


def _split_group_counts(splits: DatasetSplits) -> dict[str, int]:
    return {
        "train": len(set(_source_ids(splits.train))),
        "validation": len(set(_source_ids(splits.validation))),
        "test": len(set(_source_ids(splits.test))),
    }


def run_dataset_preflight(
    root: str | Path,
    *,
    class_labels: Mapping[str, int] = CLASS_LABELS,
    seed: int = 42,
) -> DatasetPreflightReport:
    """Validate an ALASKA2 root and construct deterministic training contracts.

    Parameters
    ----------
    root
        Dataset root containing ``Cover``, ``JMiPOD``, ``JUNIWARD``, and ``UERD``.
    class_labels
        Ordered class-to-target mapping used by indexing and sample construction.
    seed
        Seed used for both reproducibility checks of the grouped split.

    Returns
    -------
    DatasetPreflightReport
        Structured counts, split sizes, and sample details.

    Raises
    ------
    FileNotFoundError
        If the dataset root or a required class directory is missing or empty.
    DuplicateSourceError
        If a class contains ambiguous files with the same source stem.
    IncompleteGroupError
        If Cover and stego source identities do not match exactly.
    DatasetPreflightError
        If a JPEG is unreadable, targets are invalid, sample construction fails, or
        grouped split reproducibility cannot be proven.
    ValueError
        If an inherited indexing or split contract is invalid.
    """
    dataset_root = Path(root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"ALASKA2 root does not exist: {dataset_root}")

    files_by_class = {class_name: discover_jpeg_files(dataset_root / class_name) for class_name in class_labels}
    _validate_readable_jpegs(files_by_class)

    index = build_file_index(dataset_root, class_labels, on_incomplete="raise")
    numeric = add_targets(index, class_labels)
    label_names = cast(pd.Series, numeric["label_name"]).astype(str)
    binary_targets = cast(pd.Series, numeric["label_bin"]).astype(float)
    if not binary_targets.loc[label_names == "Cover"].eq(0.0).all():
        raise DatasetPreflightError("Cover rows do not all map to binary target 0.")
    if not binary_targets.loc[label_names != "Cover"].eq(1.0).all():
        raise DatasetPreflightError("Stego rows do not all map to binary target 1.")

    first_splits = split_by_source(numeric, seed=seed)
    second_splits = split_by_source(numeric, seed=seed)
    assert_split_isolation(first_splits)
    for first, second in zip(
        (first_splits.train, first_splits.validation, first_splits.test),
        (second_splits.train, second_splits.validation, second_splits.test),
        strict=True,
    ):
        if _source_ids(first) != _source_ids(second):
            raise DatasetPreflightError("Grouped split creation is not reproducible for the configured seed.")

    sample_frame = cast(pd.DataFrame, numeric.iloc[[0]]).reset_index(drop=True)
    sample_dataset = ImageDataset(
        sample_frame,
        color_mode="Y",
        target_column="label_bin",
        transform=ToTensor(),
    )
    sample_tensor, sample_target = sample_dataset[0]
    if not isinstance(sample_tensor, torch.Tensor) or sample_tensor.ndim != 3:
        raise DatasetPreflightError("Constructed sample dataset did not return a [C, H, W] tensor.")
    if float(sample_target.item()) not in {0.0, 1.0}:
        raise DatasetPreflightError("Constructed sample dataset returned an invalid binary target.")

    return DatasetPreflightReport(
        root=dataset_root,
        class_counts={class_name: len(files) for class_name, files in files_by_class.items()},
        source_groups=len(files_by_class["Cover"]),
        image_count=len(numeric),
        split_group_counts=_split_group_counts(first_splits),
        sample_shape=tuple(int(dimension) for dimension in sample_tensor.shape),
        sample_target=float(sample_target.item()),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line ALASKA2 dataset preflight.

    Parameters
    ----------
    argv
        Optional argument sequence. ``None`` reads process arguments.

    Returns
    -------
    int
        Zero on success and one when the dataset is not training-ready.
    """
    parser = argparse.ArgumentParser(description="Validate the local ALASKA2 training dataset.")
    parser.add_argument(
        "--root",
        type=Path,
        default=default_paths().alaska2,
        help="ALASKA2 root containing the four class directories.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Grouped split reproducibility seed.")
    arguments = parser.parse_args(argv)
    try:
        report = run_dataset_preflight(arguments.root, seed=arguments.seed)
    except (
        DatasetPreflightError,
        DuplicateSourceError,
        IncompleteGroupError,
        FileNotFoundError,
        ValueError,
    ) as error:
        print(f"Dataset preflight failed: {error}", file=sys.stderr)
        return 1
    print(report.format_summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
