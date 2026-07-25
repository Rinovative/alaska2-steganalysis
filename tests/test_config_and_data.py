"""
===============================================================================
test_config_and_data.py
===============================================================================
Verify portable paths, image indexes, metadata, and grouped splits.

Responsibilities:
  - Exercise repository-relative path and explicit dataset-selection
    contracts.
  - Validate target construction, JPEG metadata, and source-isolated splits.

Design principles:
  - Temporary datasets keep filesystem behavior deterministic and self-
    contained.
  - Assertions target public data contracts rather than implementation
    details.

Boundaries:
  - Synthetic proxy generation and network access are tested elsewhere.
  - Large licensed datasets and model training remain outside unit-test
    scope.
===============================================================================
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from src.config.config_paths import CLASS_LABELS, ProjectPaths, select_dataset
from src.data.data_index import IncompleteGroupError, add_targets, build_file_index
from src.data.data_metadata import JPEGMetadataError, add_jpeg_metadata
from src.data.data_split import assert_split_isolation, split_by_source, split_reservoir_subsets


def _write_jpeg(path: Path, color: tuple[int, int, int] = (20, 40, 60)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color).save(path, format="JPEG", quality=90)


def _complete_dataset(root: Path, groups: int = 10) -> None:
    for label in CLASS_LABELS:
        for index in range(groups):
            _write_jpeg(root / label / f"{index:05d}.jpg")


def test_project_paths_are_cwd_independent_and_create_only_runtime_dirs(tmp_path: Path) -> None:
    paths = ProjectPaths(tmp_path)
    assert paths.data == tmp_path / "data"
    assert paths.alaska2 == tmp_path / "data" / "ALASKA2"
    assert paths.pd12m == tmp_path / "data" / "PD12M"
    assert paths.assets == tmp_path / "assets"
    assert paths.dataset_cache("alaska2") == tmp_path / "cache" / "alaska2"
    assert paths.dataset_cache("pd12m") == tmp_path / "cache" / "pd12m"
    paths.create_runtime_directories()
    assert paths.cache.is_dir()
    assert paths.checkpoints.is_dir()
    assert paths.reports.is_dir()
    assert paths.artifacts == tmp_path / "artifacts"
    assert not paths.assets.exists()
    assert not paths.data.exists()


def test_dataset_selection_uses_flat_isolated_roots(tmp_path: Path) -> None:
    paths = ProjectPaths(tmp_path)
    for class_name in CLASS_LABELS:
        _write_jpeg(paths.pd12m / class_name / "00001.jpg")
    synthetic = select_dataset(paths, source="synthetic")
    automatic_proxy = select_dataset(paths, source="auto")
    assert synthetic.root == paths.pd12m
    assert synthetic.synthetic
    assert synthetic.cache_namespace == "pd12m"
    assert automatic_proxy.root == paths.pd12m
    assert automatic_proxy.cache_namespace == "pd12m"
    assert "JMiPOD" in synthetic.class_labels
    assert "nsF5" not in synthetic.class_labels

    for class_name in CLASS_LABELS:
        _write_jpeg(paths.alaska2 / class_name / "00001.jpg")
    automatic = select_dataset(paths, source="auto")
    assert automatic.root == paths.alaska2
    assert not automatic.synthetic
    assert automatic.cache_namespace == "alaska2"
    assert select_dataset(paths, source="synthetic").root == paths.pd12m


def test_index_targets_and_grouped_split_are_complete_and_isolated(tmp_path: Path) -> None:
    root = tmp_path / "proxy"
    _complete_dataset(root)
    index = build_file_index(root, CLASS_LABELS)
    assert len(index) == 40
    assert index.groupby("source_id")["label_name"].nunique().eq(4).to_numpy().all()
    numeric = add_targets(index, CLASS_LABELS)
    assert set(numeric.loc[numeric["label_bin"] == 0, "label_name"]) == {"Cover"}

    splits = split_by_source(
        numeric,
        train_fraction=0.6,
        validation_fraction=0.2,
        test_fraction=0.2,
        seed=7,
    )
    assert_split_isolation(splits)
    for frame in (splits.train, splits.validation, splits.test):
        assert frame.groupby("source_id")["label_name"].nunique().eq(4).to_numpy().all()


def test_fixed_reservoir_subsets_reproduce_exact_group_counts() -> None:
    rows = [{"source_id": f"{source_id:02d}", "variant": variant} for source_id in range(10) for variant in range(4)]
    dataframe = pd.DataFrame(rows)
    first = split_reservoir_subsets(
        dataframe,
        training_reservoir_groups=6,
        validation_reservoir_groups=2,
        test_groups=2,
        training_subset_groups=3,
        validation_subset_groups=1,
        seed=42,
    )
    second = split_reservoir_subsets(
        dataframe,
        training_reservoir_groups=6,
        validation_reservoir_groups=2,
        test_groups=2,
        training_subset_groups=3,
        validation_subset_groups=1,
        seed=42,
    )

    assert [
        frame["source_id"].nunique()
        for frame in (
            first.training_reservoir,
            first.validation_reservoir,
            first.selected_training,
            first.selected_validation,
            first.final_test,
        )
    ] == [6, 2, 3, 1, 2]
    assert first.selected_training["source_id"].tolist() == second.selected_training["source_id"].tolist()
    assert first.selected_validation["source_id"].tolist() == second.selected_validation["source_id"].tolist()
    assert first.final_test["source_id"].tolist() == second.final_test["source_id"].tolist()
    assert_split_isolation(first.reservoir_splits())
    assert_split_isolation(first.model_splits())


def test_incomplete_groups_are_rejected_or_explicitly_dropped(tmp_path: Path) -> None:
    root = tmp_path / "proxy"
    _complete_dataset(root, groups=3)
    (root / "JMiPOD" / "00002.jpg").unlink()
    with pytest.raises(IncompleteGroupError, match="incomplete"):
        build_file_index(root, CLASS_LABELS)
    dropped = build_file_index(root, CLASS_LABELS, on_incomplete="drop")
    assert set(dropped["source_id"]) == {"00000", "00001"}


def test_jpeg_metadata_and_malformed_error_modes(tmp_path: Path) -> None:
    good = tmp_path / "good.jpg"
    bad = tmp_path / "bad.jpg"
    _write_jpeg(good)
    bad.write_text("not a jpeg", encoding="utf-8")
    good_frame = pd.DataFrame({"path": [str(good)]})
    metadata = add_jpeg_metadata(good_frame)
    assert metadata.loc[0, "width"] == 32
    assert metadata.loc[0, "height"] == 32
    assert metadata.loc[0, "metadata_error"] is None
    assert all(f"q_y_{index:02d}" in metadata for index in range(64))

    with pytest.raises(JPEGMetadataError):
        add_jpeg_metadata(pd.DataFrame({"path": [str(bad)]}))
    non_strict = add_jpeg_metadata(pd.DataFrame({"path": [str(bad)]}), strict=False)
    assert non_strict.loc[0, "width"] == -1
    assert isinstance(non_strict.loc[0, "metadata_error"], str)
