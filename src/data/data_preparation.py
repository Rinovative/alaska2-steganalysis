"""
===============================================================================
data_preparation.py
===============================================================================
Maintain the notebook-facing dataset preparation contract.

Responsibilities:
  - Resolve the real ALASKA2 or synthetic grouped split policy.
  - Build one complete target-bearing index before any secondary selection.
  - Expose full reservoirs, selected model subsets, and the untouched test set.
  - Select a separate dataset-specific grouped population for EDA.

Design principles:
  - Exact ALASKA2 group budgets are immutable integers; the synthetic proxy
    uses grouped fractions without ALASKA2-specific absolute assumptions.

Boundaries:
  - JPEG metadata extraction remains an optional EDA concern.
  - DataLoaders, model training, and evaluation consume the prepared frames.
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..config.config_paths import DatasetSelection
from .data_index import add_targets, build_file_index, select_source_groups
from .data_split import DatasetSplits, split_by_source, split_reservoir_subsets

__all__ = [
    "Alaska2SplitConfig",
    "GroupedSplitConfig",
    "PreparedDataset",
    "prepare_dataset",
    "resolve_split_config",
    "select_eda_population",
]


@dataclass(frozen=True, slots=True)
class Alaska2SplitConfig:
    """Configure the exact verified ALASKA2 reservoir and subset contract.

    Parameters
    ----------
    training_reservoir_groups
        Exact complete groups in the training reservoir.
    validation_reservoir_groups
        Exact complete groups in the validation reservoir.
    final_test_groups
        Exact complete groups kept for final testing.
    selected_training_groups
        Exact model-training groups selected from the training reservoir.
    selected_validation_groups
        Exact model-validation groups selected from the validation reservoir.
    seed
        Seed shared by deterministic reservoir and subset selection.
    """

    training_reservoir_groups: int = 60_000
    validation_reservoir_groups: int = 7_500
    final_test_groups: int = 7_500
    selected_training_groups: int = 6_000
    selected_validation_groups: int = 750
    seed: int = 42

    def __post_init__(self) -> None:
        counts = (
            self.training_reservoir_groups,
            self.validation_reservoir_groups,
            self.final_test_groups,
            self.selected_training_groups,
            self.selected_validation_groups,
        )
        if any(value <= 0 for value in counts):
            raise ValueError("ALASKA2 reservoir and subset group counts must be positive.")
        if self.selected_training_groups > self.training_reservoir_groups:
            raise ValueError("Selected training groups cannot exceed the training reservoir.")
        if self.selected_validation_groups > self.validation_reservoir_groups:
            raise ValueError("Selected validation groups cannot exceed the validation reservoir.")


@dataclass(frozen=True, slots=True)
class GroupedSplitConfig:
    """Configure grouped fractional splitting for the synthetic fallback.

    Parameters
    ----------
    train_fraction
        Fraction of complete source groups assigned to training.
    validation_fraction
        Fraction of complete source groups assigned to validation.
    test_fraction
        Fraction of complete source groups assigned to final testing.
    seed
        Deterministic grouped split seed.
    """

    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    seed: int = 42

    def __post_init__(self) -> None:
        fractions = (self.train_fraction, self.validation_fraction, self.test_fraction)
        if any(value <= 0 for value in fractions) or abs(sum(fractions) - 1.0) > 1e-9:
            raise ValueError("Grouped split fractions must be positive and sum to 1.0.")


@dataclass(frozen=True, slots=True)
class PreparedDataset:
    """Expose every scientifically distinct preparation frame.

    Parameters
    ----------
    index
        Complete target-bearing dataset index before split or budget selection.
    training_reservoir
        Full grouped training reservoir.
    validation_reservoir
        Full grouped validation reservoir.
    selected_training
        Rows consumed by model training.
    selected_validation
        Rows consumed during model selection.
    final_test
        Untouched rows reserved for final evaluation.
    """

    index: pd.DataFrame
    training_reservoir: pd.DataFrame
    validation_reservoir: pd.DataFrame
    selected_training: pd.DataFrame
    selected_validation: pd.DataFrame
    final_test: pd.DataFrame

    @property
    def model_splits(self) -> DatasetSplits:
        """Return the one shared model-facing split membership."""
        return DatasetSplits(
            train=self.selected_training,
            validation=self.selected_validation,
            test=self.final_test,
        )

    def summary_frame(self) -> pd.DataFrame:
        """Summarize source-group and image counts for every preparation stage.

        Returns
        -------
        pandas.DataFrame
            Ordered stage names with unambiguous group and image counts.
        """
        frames = (
            ("complete_index", self.index),
            ("training_reservoir", self.training_reservoir),
            ("selected_training", self.selected_training),
            ("validation_reservoir", self.validation_reservoir),
            ("selected_validation", self.selected_validation),
            ("final_test", self.final_test),
        )
        return pd.DataFrame.from_records(
            (
                {
                    "stage": name,
                    "source_groups": int(frame["source_id"].nunique()),
                    "images": len(frame),
                }
                for name, frame in frames
            )
        )


def resolve_split_config(
    selection: DatasetSelection,
    *,
    seed: int = 42,
) -> Alaska2SplitConfig | GroupedSplitConfig:
    """Resolve the dataset-specific split policy without mixing data sources.

    Parameters
    ----------
    selection
        One real or synthetic dataset selected by the path configuration.
    seed
        Deterministic split and subset seed.

    Returns
    -------
    Alaska2SplitConfig or GroupedSplitConfig
        Exact ALASKA2 budgets or grouped fallback fractions.
    """
    if selection.synthetic:
        return GroupedSplitConfig(seed=seed)
    return Alaska2SplitConfig(seed=seed)


def select_eda_population(
    complete_index: pd.DataFrame,
    *,
    selection: DatasetSelection,
    alaska2_group_count: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Select the dataset-specific grouped population used only for EDA.

    Parameters
    ----------
    complete_index
        Complete target-bearing dataset index before model split selection.
    selection
        Resolved dataset identity determining the EDA population policy.
    alaska2_group_count
        Exact complete-source-group budget used only for real ALASKA2 data.
    seed
        Deterministic ALASKA2 group-selection seed.

    Returns
    -------
    pandas.DataFrame
        A grouped ALASKA2 sample or a complete grouped copy of the PD12M index.

    Raises
    ------
    ValueError
        If the dataset identity is inconsistent or the group budget is invalid.

    Notes
    -----
    The input and all prepared split frames remain unchanged. JPEG metadata is
    intentionally excluded and remains an optional downstream EDA operation.
    """
    if alaska2_group_count <= 0:
        raise ValueError("alaska2_group_count must be positive.")
    if selection.synthetic:
        if selection.name != "PD12M":
            raise ValueError("Synthetic EDA selection requires the PD12M dataset identity.")
        return select_source_groups(
            complete_index,
            group_count=int(complete_index["source_id"].nunique()),
            seed=seed,
        )
    if selection.name != "ALASKA2":
        raise ValueError("Real EDA selection requires the ALASKA2 dataset identity.")
    return select_source_groups(
        complete_index,
        group_count=alaska2_group_count,
        seed=seed,
    )


def prepare_dataset(
    selection: DatasetSelection,
    *,
    split_config: Alaska2SplitConfig | GroupedSplitConfig,
) -> PreparedDataset:
    """Index all complete groups, construct targets, and apply one split policy.

    Parameters
    ----------
    selection
        Resolved dataset root, identity, and class-label contract.
    split_config
        Exact ALASKA2 or grouped synthetic split configuration.

    Returns
    -------
    PreparedDataset
        Complete target-bearing index and every distinct split-stage frame.

    Raises
    ------
    ValueError
        If the selected dataset and split policy are scientifically incompatible.

    Notes
    -----
    The complete index is always built before model-budget selection. JPEG
    metadata is intentionally excluded because it is required only for EDA.
    """
    if selection.synthetic != isinstance(split_config, GroupedSplitConfig):
        raise ValueError("Synthetic data requires grouped fractional splits; ALASKA2 requires exact group budgets.")

    complete_index = build_file_index(
        selection.root,
        selection.class_labels,
        on_incomplete="raise",
    )
    numeric_index = add_targets(complete_index, selection.class_labels)
    numeric_index["label_name"] = pd.Categorical(
        numeric_index["label_name"],
        categories=list(selection.class_labels),
        ordered=True,
    )

    if isinstance(split_config, Alaska2SplitConfig):
        split_result = split_reservoir_subsets(
            numeric_index,
            training_reservoir_groups=split_config.training_reservoir_groups,
            validation_reservoir_groups=split_config.validation_reservoir_groups,
            test_groups=split_config.final_test_groups,
            training_subset_groups=split_config.selected_training_groups,
            validation_subset_groups=split_config.selected_validation_groups,
            seed=split_config.seed,
        )
        return PreparedDataset(
            index=numeric_index,
            training_reservoir=split_result.training_reservoir,
            validation_reservoir=split_result.validation_reservoir,
            selected_training=split_result.selected_training,
            selected_validation=split_result.selected_validation,
            final_test=split_result.final_test,
        )

    grouped = split_by_source(
        numeric_index,
        train_fraction=split_config.train_fraction,
        validation_fraction=split_config.validation_fraction,
        test_fraction=split_config.test_fraction,
        seed=split_config.seed,
    )
    return PreparedDataset(
        index=numeric_index,
        training_reservoir=grouped.train,
        validation_reservoir=grouped.validation,
        selected_training=grouped.train,
        selected_validation=grouped.validation,
        final_test=grouped.test,
    )
