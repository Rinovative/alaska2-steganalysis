"""
===============================================================================
data_split.py
===============================================================================
Leakage-safe grouped dataset splitting.

Responsibilities:
  - Partition source identities into independent train, validation, and test
    sets.
  - Preserve complete cover/stego groups.
  - Validate split proportions and postconditions.

Design principles:
  - Rows never drive partitioning; the explicit source_id does. Returned
    frames are reset for predictable downstream indexing.

Boundaries:
  - This module does not create targets, read images, or construct data
    loaders.

Notes:
  - The split is deterministic for a fixed source-id set and seed.
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import cast

import pandas as pd
from sklearn.model_selection import train_test_split

__all__ = [
    "DatasetSplits",
    "ReservoirSubsets",
    "assert_split_isolation",
    "split_by_source",
    "split_reservoir_subsets",
]


@dataclass(frozen=True, slots=True)
class DatasetSplits:
    """Store independent source-grouped model partitions.

    Parameters
    ----------
    train
        Training rows containing complete source groups.
    validation
        Validation rows containing disjoint complete source groups.
    test
        Final test rows containing disjoint complete source groups.
    """

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True, slots=True)
class ReservoirSubsets:
    """Store exact reservoirs, selected model subsets, and final test rows.

    Parameters
    ----------
    training_reservoir
        Complete rows assigned to the training reservoir.
    validation_reservoir
        Complete rows assigned to the validation reservoir.
    selected_training
        Exact training-budget rows selected from the training reservoir.
    selected_validation
        Exact validation-budget rows selected from the validation reservoir.
    final_test
        Complete final-test reservoir, never reduced by a model budget.
    """

    training_reservoir: pd.DataFrame
    validation_reservoir: pd.DataFrame
    selected_training: pd.DataFrame
    selected_validation: pd.DataFrame
    final_test: pd.DataFrame

    def reservoir_splits(self) -> DatasetSplits:
        """Return the three mutually isolated full reservoirs.

        Returns
        -------
        DatasetSplits
            Training reservoir, validation reservoir, and final test split.
        """
        return DatasetSplits(
            train=self.training_reservoir,
            validation=self.validation_reservoir,
            test=self.final_test,
        )

    def model_splits(self) -> DatasetSplits:
        """Return the selected training, validation, and untouched test frames.

        Returns
        -------
        DatasetSplits
            Exact frames consumed identically by each model workflow.
        """
        return DatasetSplits(
            train=self.selected_training,
            validation=self.selected_validation,
            test=self.final_test,
        )


def split_by_source(
    dataframe: pd.DataFrame,
    *,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> DatasetSplits:
    """Partition complete source identities into train, validation, and test rows.

    Parameters
    ----------
    dataframe
        Indexed rows containing a ``source_id`` column.
    train_fraction
        Fraction of source identities assigned to training.
    validation_fraction
        Fraction assigned to validation.
    test_fraction
        Fraction reserved for final testing.
    seed
        Deterministic scikit-learn split seed.

    Returns
    -------
    DatasetSplits
        Three reset-index dataframes with source-level isolation.

    Raises
    ------
    ValueError
        If columns, fractions, group counts, or isolation postconditions are invalid.
    """
    if "source_id" not in dataframe:
        raise ValueError("dataframe must contain a source_id column.")
    fractions = (train_fraction, validation_fraction, test_fraction)
    if any(value <= 0 for value in fractions) or abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("Split fractions must be positive and sum to 1.0.")

    source_ids = sorted(str(value) for value in dataframe["source_id"].unique().tolist())
    if len(source_ids) < 3:
        raise ValueError("At least three source groups are required for three partitions.")

    train_validation_ids_raw, test_ids_raw = train_test_split(
        source_ids,
        test_size=test_fraction,
        random_state=seed,
        shuffle=True,
    )
    train_validation_ids = cast(list[str], train_validation_ids_raw)
    test_ids = cast(list[str], test_ids_raw)
    relative_validation = validation_fraction / (train_fraction + validation_fraction)
    train_ids_raw, validation_ids_raw = train_test_split(
        train_validation_ids,
        test_size=relative_validation,
        random_state=seed,
        shuffle=True,
    )

    train_ids = cast(list[str], train_ids_raw)
    validation_ids = cast(list[str], validation_ids_raw)

    def subset(ids: list[str]) -> pd.DataFrame:
        selected = cast(pd.DataFrame, dataframe.loc[dataframe["source_id"].isin(ids)])
        return selected.reset_index(drop=True)

    splits = DatasetSplits(
        train=subset(train_ids),
        validation=subset(validation_ids),
        test=subset(test_ids),
    )
    assert_split_isolation(splits)
    return splits


def assert_split_isolation(splits: DatasetSplits) -> None:
    """Validate non-empty partitions with no source identity overlap.

    Parameters
    ----------
    splits
        Grouped train, validation, and test partitions to inspect.

    Returns
    -------
    None
        The function returns after all isolation postconditions pass.

    Raises
    ------
    ValueError
        If a partition is empty or any source appears in multiple partitions.
    """
    id_sets = {
        "train": set(splits.train["source_id"]),
        "validation": set(splits.validation["source_id"]),
        "test": set(splits.test["source_id"]),
    }
    if any(not values for values in id_sets.values()):
        raise ValueError("Train, validation, and test partitions must all be non-empty.")
    overlap = (
        (id_sets["train"] & id_sets["validation"])
        | (id_sets["train"] & id_sets["test"])
        | (id_sets["validation"] & id_sets["test"])
    )
    if overlap:
        raise ValueError(f"Source leakage across splits: {sorted(overlap)[:5]}")


def split_reservoir_subsets(
    dataframe: pd.DataFrame,
    *,
    training_reservoir_groups: int,
    validation_reservoir_groups: int,
    test_groups: int,
    training_subset_groups: int,
    validation_subset_groups: int,
    seed: int = 42,
) -> ReservoirSubsets:
    """Build exact grouped subsets from deterministic shuffled reservoirs.

    Parameters
    ----------
    dataframe
        Complete source-group rows containing a source_id column.
    training_reservoir_groups
        Exact source groups in the initial training reservoir.
    validation_reservoir_groups
        Exact source groups in the initial validation reservoir.
    test_groups
        Exact source groups reserved for the final test partition.
    training_subset_groups
        Source groups sampled from the training reservoir.
    validation_subset_groups
        Source groups sampled from the validation reservoir.
    seed
        Seed used once for reservoir ordering and by a separate subset
        sampler.

    Returns
    -------
    ReservoirSubsets
        Full training and validation reservoirs, their exact selected subsets,
        and the complete final test rows.

    Raises
    ------
    ValueError
        If counts, group completeness, or isolation violate the contract.

    Notes
    -----
    The separate subset RNG reproduces the maintained ALASKA2 run contract:
    it samples the sorted training reservoir first and the sorted validation
    reservoir second.
    """
    if "source_id" not in dataframe:
        raise ValueError("dataframe must contain a source_id column.")
    counts = (
        training_reservoir_groups,
        validation_reservoir_groups,
        test_groups,
        training_subset_groups,
        validation_subset_groups,
    )
    if any(value <= 0 for value in counts):
        raise ValueError("Reservoir and subset group counts must be positive.")
    if training_subset_groups > training_reservoir_groups:
        raise ValueError("Training subset cannot exceed its reservoir.")
    if validation_subset_groups > validation_reservoir_groups:
        raise ValueError("Validation subset cannot exceed its reservoir.")

    source_counts = dataframe.groupby("source_id", sort=False).size()
    if source_counts.empty or source_counts.nunique() != 1:
        raise ValueError("Every source_id must contain one equally sized complete group.")
    images_per_group = int(source_counts.iloc[0])
    source_ordering = sorted(str(value) for value in source_counts.index)
    expected_groups = training_reservoir_groups + validation_reservoir_groups + test_groups
    if len(source_ordering) != expected_groups:
        raise ValueError(f"Expected {expected_groups} complete source groups, found {len(source_ordering)}.")

    reservoir_rng = Random(seed)
    reservoir_rng.shuffle(source_ordering)
    training_end = training_reservoir_groups
    validation_end = training_end + validation_reservoir_groups
    training_reservoir = source_ordering[:training_end]
    validation_reservoir = source_ordering[training_end:validation_end]
    final_test_ids = source_ordering[validation_end:]

    subset_rng = Random(seed)
    training_ids = subset_rng.sample(sorted(training_reservoir), training_subset_groups)
    validation_ids = subset_rng.sample(sorted(validation_reservoir), validation_subset_groups)

    def subset(source_ids: list[str], expected_count: int) -> pd.DataFrame:
        selected = cast(pd.DataFrame, dataframe.loc[dataframe["source_id"].astype(str).isin(source_ids)])
        selected = selected.reset_index(drop=True)
        if len(set(source_ids)) != expected_count or len(selected) != expected_count * images_per_group:
            raise ValueError(f"Expected {expected_count} complete selected source groups.")
        return selected

    result = ReservoirSubsets(
        training_reservoir=subset(training_reservoir, training_reservoir_groups),
        validation_reservoir=subset(validation_reservoir, validation_reservoir_groups),
        selected_training=subset(training_ids, training_subset_groups),
        selected_validation=subset(validation_ids, validation_subset_groups),
        final_test=subset(final_test_ids, test_groups),
    )
    assert_split_isolation(result.reservoir_splits())
    assert_split_isolation(result.model_splits())
    training_reservoir_ids = set(result.training_reservoir["source_id"].astype(str))
    validation_reservoir_ids = set(result.validation_reservoir["source_id"].astype(str))
    if not set(result.selected_training["source_id"].astype(str)) <= training_reservoir_ids:
        raise ValueError("Selected training groups must belong to the training reservoir.")
    if not set(result.selected_validation["source_id"].astype(str)) <= validation_reservoir_ids:
        raise ValueError("Selected validation groups must belong to the validation reservoir.")
    return result
