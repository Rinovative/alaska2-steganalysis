"""
===============================================================================
test_data_preparation.py
===============================================================================
Verify the notebook-facing full-index and experiment-budget contract.

Responsibilities:
  - Prove ALASKA2 is indexed completely before exact reservoir selection.
  - Validate reservoir isolation, subset inclusion, and untouched final tests.
  - Exercise the grouped synthetic fallback and separate grouped EDA sampling.

Design principles:
  - Small complete JPEG fixtures reproduce group semantics without licensed data.
  - Exact integer counts make every scientific population directly observable.

Boundaries:
  - Tests do not train models, evaluate predictions, or inspect the real dataset.
  - JPEG metadata is read only from temporary fixture images.
===============================================================================
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from src.config.config_paths import CLASS_LABELS, DatasetSelection
from src.data import data_preparation as preparation
from src.data.data_index import build_file_index
from src.data.data_metadata import add_jpeg_metadata
from src.data.data_preparation import (
    Alaska2SplitConfig,
    GroupedSplitConfig,
    prepare_dataset,
    resolve_split_config,
    select_eda_population,
)
from src.data.data_split import ReservoirSubsets, assert_split_isolation, split_reservoir_subsets


def _write_complete_dataset(root: Path, groups: int) -> None:
    for label in CLASS_LABELS:
        for index in range(groups):
            path = root / label / f"{index:05d}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (16, 16), (index, 40, 80)).save(
                path,
                format="JPEG",
                quality=90,
            )


def _selection(root: Path, *, synthetic: bool) -> DatasetSelection:
    return DatasetSelection(
        name="PD12M" if synthetic else "ALASKA2",
        display_name="synthetic PD12M proxy" if synthetic else "ALASKA2",
        root=root,
        class_labels=CLASS_LABELS.copy(),
        synthetic=synthetic,
    )


def _ids(dataframe: pd.DataFrame) -> set[str]:
    return set(dataframe["source_id"].astype(str))


def _assert_complete_groups(dataframe: pd.DataFrame, expected_groups: int) -> None:
    counts = dataframe.groupby("source_id")["label_name"].nunique()
    assert dataframe["source_id"].nunique() == expected_groups
    assert counts.eq(len(CLASS_LABELS)).to_numpy().all()


def test_alaska2_preparation_indexes_all_groups_before_exact_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "ALASKA2"
    _write_complete_dataset(root, groups=10)
    observed_split_groups: list[int] = []
    original_split = split_reservoir_subsets

    def tracked_split(
        dataframe: pd.DataFrame,
        *,
        training_reservoir_groups: int,
        validation_reservoir_groups: int,
        test_groups: int,
        training_subset_groups: int,
        validation_subset_groups: int,
        seed: int = 42,
    ) -> ReservoirSubsets:
        observed_split_groups.append(int(dataframe["source_id"].nunique()))
        return original_split(
            dataframe,
            training_reservoir_groups=training_reservoir_groups,
            validation_reservoir_groups=validation_reservoir_groups,
            test_groups=test_groups,
            training_subset_groups=training_subset_groups,
            validation_subset_groups=validation_subset_groups,
            seed=seed,
        )

    monkeypatch.setattr(preparation, "split_reservoir_subsets", tracked_split)
    prepared = prepare_dataset(
        _selection(root, synthetic=False),
        split_config=Alaska2SplitConfig(
            training_reservoir_groups=6,
            validation_reservoir_groups=2,
            final_test_groups=2,
            selected_training_groups=3,
            selected_validation_groups=1,
            seed=42,
        ),
    )

    assert observed_split_groups == [10]
    assert "subsample_fraction" not in inspect.signature(build_file_index).parameters
    assert len(prepared.index) == 40
    assert prepared.index["source_id"].nunique() == 10
    assert prepared.index["label_name"].dtype.name == "category"
    assert set(prepared.index["label"]) == set(CLASS_LABELS.values())
    assert set(prepared.index["label_bin"]) == {0.0, 1.0}

    _assert_complete_groups(prepared.training_reservoir, 6)
    _assert_complete_groups(prepared.validation_reservoir, 2)
    _assert_complete_groups(prepared.selected_training, 3)
    _assert_complete_groups(prepared.selected_validation, 1)
    _assert_complete_groups(prepared.final_test, 2)
    assert_split_isolation(
        ReservoirSubsets(
            training_reservoir=prepared.training_reservoir,
            validation_reservoir=prepared.validation_reservoir,
            selected_training=prepared.selected_training,
            selected_validation=prepared.selected_validation,
            final_test=prepared.final_test,
        ).reservoir_splits()
    )
    assert _ids(prepared.selected_training) <= _ids(prepared.training_reservoir)
    assert _ids(prepared.selected_validation) <= _ids(prepared.validation_reservoir)
    assert len(prepared.final_test) == 8

    tinycnn_splits = prepared.model_splits
    efficientnet_splits = prepared.model_splits
    for tiny_frame, efficient_frame in zip(
        (tinycnn_splits.train, tinycnn_splits.validation, tinycnn_splits.test),
        (efficientnet_splits.train, efficientnet_splits.validation, efficientnet_splits.test),
        strict=True,
    ):
        assert _ids(tiny_frame) == _ids(efficient_frame)

    assert prepared.summary_frame().to_dict(orient="records") == [
        {"stage": "complete_index", "source_groups": 10, "images": 40},
        {"stage": "training_reservoir", "source_groups": 6, "images": 24},
        {"stage": "selected_training", "source_groups": 3, "images": 12},
        {"stage": "validation_reservoir", "source_groups": 2, "images": 8},
        {"stage": "selected_validation", "source_groups": 1, "images": 4},
        {"stage": "final_test", "source_groups": 2, "images": 8},
    ]


def test_synthetic_preparation_uses_grouped_fractions_without_absolute_budgets(
    tmp_path: Path,
) -> None:
    root = tmp_path / "PD12M"
    _write_complete_dataset(root, groups=20)
    selection = _selection(root, synthetic=True)
    split_config = resolve_split_config(selection, seed=7)

    assert isinstance(split_config, GroupedSplitConfig)
    prepared = prepare_dataset(selection, split_config=split_config)
    _assert_complete_groups(prepared.index, 20)
    _assert_complete_groups(prepared.training_reservoir, 16)
    _assert_complete_groups(prepared.validation_reservoir, 2)
    _assert_complete_groups(prepared.final_test, 2)
    assert _ids(prepared.selected_training) == _ids(prepared.training_reservoir)
    assert _ids(prepared.selected_validation) == _ids(prepared.validation_reservoir)
    assert_split_isolation(prepared.model_splits)

    with pytest.raises(ValueError, match="Synthetic data requires grouped"):
        prepare_dataset(
            selection,
            split_config=Alaska2SplitConfig(
                training_reservoir_groups=16,
                validation_reservoir_groups=2,
                final_test_groups=2,
                selected_training_groups=8,
                selected_validation_groups=1,
            ),
        )


def test_alaska2_eda_population_is_deterministic_grouped_and_separate(tmp_path: Path) -> None:
    root = tmp_path / "ALASKA2"
    _write_complete_dataset(root, groups=10)
    selection = _selection(root, synthetic=False)
    prepared = prepare_dataset(
        selection,
        split_config=Alaska2SplitConfig(
            training_reservoir_groups=6,
            validation_reservoir_groups=2,
            final_test_groups=2,
            selected_training_groups=3,
            selected_validation_groups=1,
        ),
    )
    model_membership_before = tuple(
        _ids(frame)
        for frame in (
            prepared.selected_training,
            prepared.selected_validation,
            prepared.final_test,
        )
    )
    complete_before = prepared.index.copy(deep=True)

    eda_index = select_eda_population(
        prepared.index,
        selection=selection,
        alaska2_group_count=4,
        seed=42,
    )
    repeated = select_eda_population(
        prepared.index,
        selection=selection,
        alaska2_group_count=4,
        seed=42,
    )
    metadata = add_jpeg_metadata(eda_index, strict=True, show_progress=False)

    pd.testing.assert_frame_equal(eda_index, repeated)
    pd.testing.assert_frame_equal(prepared.index, complete_before)
    _assert_complete_groups(metadata, 4)
    assert len(metadata) == 16
    assert metadata["width"].eq(16).to_numpy().all()
    assert (
        tuple(
            _ids(frame)
            for frame in (
                prepared.selected_training,
                prepared.selected_validation,
                prepared.final_test,
            )
        )
        == model_membership_before
    )


def test_pd12m_eda_population_is_a_complete_grouped_copy(tmp_path: Path) -> None:
    root = tmp_path / "PD12M"
    _write_complete_dataset(root, groups=5)
    selection = _selection(root, synthetic=True)
    prepared = prepare_dataset(
        selection,
        split_config=GroupedSplitConfig(train_fraction=0.6, validation_fraction=0.2, test_fraction=0.2),
    )
    complete_before = prepared.index.copy(deep=True)
    split_membership_before = tuple(
        _ids(frame)
        for frame in (
            prepared.selected_training,
            prepared.selected_validation,
            prepared.final_test,
        )
    )

    eda_index = select_eda_population(
        prepared.index,
        selection=selection,
        alaska2_group_count=4,
        seed=42,
    )

    _assert_complete_groups(eda_index, 5)
    pd.testing.assert_frame_equal(eda_index, complete_before)
    assert eda_index is not prepared.index
    pd.testing.assert_frame_equal(prepared.index, complete_before)
    assert (
        tuple(
            _ids(frame)
            for frame in (
                prepared.selected_training,
                prepared.selected_validation,
                prepared.final_test,
            )
        )
        == split_membership_before
    )


def test_eda_population_rejects_invalid_dataset_identity_and_budget(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _write_complete_dataset(root, groups=2)
    index = build_file_index(root, CLASS_LABELS)
    invalid = DatasetSelection(
        name="OTHER",
        display_name="Other",
        root=root,
        class_labels=CLASS_LABELS.copy(),
        synthetic=False,
    )

    with pytest.raises(ValueError, match="positive"):
        select_eda_population(
            index,
            selection=_selection(root, synthetic=False),
            alaska2_group_count=0,
        )
    with pytest.raises(ValueError, match="ALASKA2"):
        select_eda_population(index, selection=invalid, alaska2_group_count=1)
