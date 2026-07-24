"""
===============================================================================
data_index.py
===============================================================================
Build leakage-safe indexes for matched Cover and stego JPEG files.

Responsibilities:
  - Discover supported JPEG files and detect ambiguous source identities.
  - Retain only complete Cover/stego groups under an explicit class contract.
  - Add multiclass and binary targets without mutating caller data.

Design principles:
  - Source stems define grouping identity across filename extensions.
  - Sampling operates on complete groups and never on individual class rows.

Boundaries:
  - Image decoding and corruption checks belong to data_preflight.py.
  - Train, validation, and test partitioning belongs to data_split.py.
===============================================================================
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from random import Random
from typing import Final, Literal, cast

import pandas as pd

__all__ = [
    "SUPPORTED_JPEG_SUFFIXES",
    "DuplicateSourceError",
    "IncompleteGroupError",
    "add_targets",
    "build_file_index",
    "discover_jpeg_files",
    "select_source_groups",
    "validate_complete_groups",
]

SUPPORTED_JPEG_SUFFIXES: Final[frozenset[str]] = frozenset({".jpg", ".jpeg"})


class DuplicateSourceError(ValueError):
    """Raised when one class contains multiple files for the same source stem."""


class IncompleteGroupError(ValueError):
    """Raised when one or more source identities lack a requested class variant."""


def discover_jpeg_files(directory: str | Path) -> dict[str, Path]:
    """Discover one supported JPEG file per source stem.

    Parameters
    ----------
    directory
        Class directory to inspect without recursion.

    Returns
    -------
    dict[str, pathlib.Path]
        Mapping from source stem to its concrete JPEG path.

    Raises
    ------
    FileNotFoundError
        If the class directory does not exist or contains no supported JPEGs.
    DuplicateSourceError
        If more than one supported file has the same source stem.

    Notes
    -----
    Supported suffixes are ``.jpg`` and ``.jpeg`` with case-insensitive matching.
    """
    class_directory = Path(directory).expanduser().resolve()
    if not class_directory.is_dir():
        raise FileNotFoundError(f"Missing class directory: {class_directory}")

    discovered: dict[str, Path] = {}
    candidates = sorted(
        (
            path
            for path in class_directory.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_JPEG_SUFFIXES
        ),
        key=lambda path: (path.stem, path.name),
    )
    for path in candidates:
        source_id = path.stem
        previous = discovered.get(source_id)
        if previous is not None:
            raise DuplicateSourceError(
                f"Ambiguous source_id={source_id!r} in {class_directory}: {previous.name!r} and {path.name!r}."
            )
        discovered[source_id] = path
    if not discovered:
        suffixes = ", ".join(sorted(SUPPORTED_JPEG_SUFFIXES))
        raise FileNotFoundError(f"No supported JPEG files ({suffixes}) found in class directory: {class_directory}")
    return discovered


def build_file_index(
    dataset_root: str | Path,
    class_labels: Mapping[str, int],
    *,
    on_incomplete: Literal["raise", "drop"] = "raise",
) -> pd.DataFrame:
    """Build a group-complete image index.

    Parameters
    ----------
    dataset_root
        Directory containing one folder per class.
    class_labels
        Ordered class-to-integer mapping. ``Cover`` must map to zero.
    on_incomplete
        Whether incomplete identities raise an error or are explicitly dropped.

    Returns
    -------
    pandas.DataFrame
        Dataframe with ``path``, ``source_id``, and ``label_name`` columns.

    Raises
    ------
    FileNotFoundError
        If a class directory is absent or contains no supported JPEG files.
    DuplicateSourceError
        If a class contains ambiguous files for one source stem.
    IncompleteGroupError
        If class source sets differ and ``on_incomplete`` is ``"raise"``.
    ValueError
        If the class mapping or incomplete-group policy is invalid.
    """
    if not class_labels or class_labels.get("Cover") != 0:
        raise ValueError("class_labels must contain Cover mapped to 0.")
    if len(set(class_labels.values())) != len(class_labels):
        raise ValueError("class label values must be unique.")
    if on_incomplete not in {"raise", "drop"}:
        raise ValueError("on_incomplete must be 'raise' or 'drop'.")

    root = Path(dataset_root).expanduser().resolve()
    files_by_class = {class_name: discover_jpeg_files(root / class_name) for class_name in class_labels}
    source_sets = [set(files) for files in files_by_class.values()]
    common_sources = set.intersection(*source_sets)
    union_sources = set.union(*source_sets)
    incomplete = union_sources - common_sources
    if incomplete and on_incomplete == "raise":
        examples = ", ".join(sorted(incomplete)[:5])
        counts = ", ".join(f"{class_name}={len(files)}" for class_name, files in files_by_class.items())
        raise IncompleteGroupError(
            f"{len(incomplete)} incomplete source groups detected ({counts}); examples: {examples}"
        )
    if not common_sources:
        raise IncompleteGroupError("No complete Cover/stego source groups were found.")

    source_ids = sorted(common_sources)
    records = [
        {
            "path": str(files_by_class[class_name][source_id]),
            "source_id": source_id,
            "label_name": class_name,
        }
        for source_id in source_ids
        for class_name in class_labels
    ]
    result = pd.DataFrame.from_records(records)
    validate_complete_groups(result, expected_classes=tuple(class_labels))
    return result


def select_source_groups(
    dataframe: pd.DataFrame,
    *,
    group_count: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Select an exact reproducible sample of complete source groups.

    Parameters
    ----------
    dataframe
        Group-complete rows containing a source_id column.
    group_count
        Exact positive number of source identities to select.
    seed
        Local sampling seed that does not modify global random state.

    Returns
    -------
    pandas.DataFrame
        Reset-index copy containing every row for each selected source.

    Raises
    ------
    ValueError
        If the source column, group completeness, or requested count is invalid.

    Notes
    -----
    This explicit secondary selection keeps exploratory budgets separate from
    full-dataset indexing and model split construction.
    """
    if "source_id" not in dataframe:
        raise ValueError("dataframe must contain a source_id column.")
    source_counts = dataframe.groupby("source_id", sort=False).size()
    if source_counts.empty or source_counts.nunique() != 1:
        raise ValueError("Every source_id must contain one equally sized complete group.")
    source_ids = sorted(str(value) for value in source_counts.index)
    if group_count <= 0 or group_count > len(source_ids):
        raise ValueError(f"group_count must be in [1, {len(source_ids)}].")
    selected_ids = Random(seed).sample(source_ids, group_count)
    selected = cast(pd.DataFrame, dataframe.loc[dataframe["source_id"].astype(str).isin(selected_ids)])
    result = selected.reset_index(drop=True)
    if result["source_id"].nunique() != group_count:
        raise ValueError(f"Expected {group_count} selected source groups.")
    return result


def validate_complete_groups(
    dataframe: pd.DataFrame,
    *,
    expected_classes: tuple[str, ...],
) -> None:
    """Validate one row per expected class for every source identity.

    Parameters
    ----------
    dataframe
        Image index containing source, class, and path columns.
    expected_classes
        Complete class set required for every source identity.

    Returns
    -------
    None
        The function returns after every complete-group postcondition passes.

    Raises
    ------
    ValueError
        If required columns are missing or the index is empty.
    IncompleteGroupError
        If a source has missing or duplicate class rows.
    """
    required = {"source_id", "label_name", "path"}
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(f"Index is missing required columns: {sorted(missing)}")
    if dataframe.empty:
        raise ValueError("Index must not be empty.")
    expected = set(expected_classes)
    for source_id, group in dataframe.groupby("source_id", sort=False):
        labels = cast(pd.Series, group["label_name"]).astype(str).tolist()
        if len(labels) != len(expected) or set(labels) != expected:
            raise IncompleteGroupError(f"Incomplete or duplicate group for source_id={source_id!r}: {labels}")


def add_targets(dataframe: pd.DataFrame, class_labels: Mapping[str, int]) -> pd.DataFrame:
    """Add multiclass and binary targets to a copy of an image index.

    Parameters
    ----------
    dataframe
        Image index containing a ``label_name`` column.
    class_labels
        Mapping from each supported class name to a unique integer.

    Returns
    -------
    pandas.DataFrame
        Copy with integer ``label`` and float ``label_bin`` columns.

    Raises
    ------
    ValueError
        If the index contains class names absent from ``class_labels``.
    """
    label_names = cast(pd.Series, dataframe["label_name"]).astype(str)
    unknown = set(label_names) - set(class_labels)
    if unknown:
        raise ValueError(f"Unknown class names: {sorted(unknown)}")
    result = dataframe.copy()
    result["label"] = label_names.map(class_labels).astype("int64")
    result["label_bin"] = (result["label"] != 0).astype("float32")
    return result
