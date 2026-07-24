"""
===============================================================================
test_preflights.py
===============================================================================
Verify ALASKA2 dataset and CUDA readiness checks without external resources.

Responsibilities:
  - Exercise successful and failing dataset preflight contracts with tiny JPEGs.
  - Prove that explicit ALASKA2 selection never falls back to the synthetic proxy.
  - Exercise the required-CUDA failure path in a CPU-only test environment.

Design principles:
  - Fixtures remain small, deterministic, and isolated under pytest temporary paths.
  - Tests never download datasets, generate stego images, or require a real GPU.

Boundaries:
  - Real container GPU passthrough requires a separate local smoke test.
  - Full ALASKA2 scale and training behavior are intentionally outside this suite.
===============================================================================
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from src.config.config_device import GPUPreflightError, resolve_device, run_gpu_preflight
from src.config.config_paths import CLASS_LABELS, ProjectPaths, select_dataset
from src.data.data_index import DuplicateSourceError, IncompleteGroupError
from src.data.data_preflight import DatasetPreflightError, run_dataset_preflight


def _write_jpeg(path: Path, *, color: tuple[int, int, int] = (20, 40, 60)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color).save(path, format="JPEG", quality=90)


def _complete_dataset(root: Path, *, groups: int = 12) -> None:
    for class_name in CLASS_LABELS:
        for index in range(groups):
            _write_jpeg(root / class_name / f"{index:05d}.jpg")


def test_dataset_preflight_validates_counts_targets_sample_and_splits(tmp_path: Path) -> None:
    root = tmp_path / "ALASKA2"
    _complete_dataset(root)

    report = run_dataset_preflight(root, seed=17)

    assert report.root == root.resolve()
    assert report.class_counts == {class_name: 12 for class_name in CLASS_LABELS}
    assert report.source_groups == 12
    assert report.image_count == 48
    assert sum(report.split_group_counts.values()) == 12
    assert report.sample_shape == (1, 16, 16)
    assert report.sample_target == 0.0


def test_dataset_preflight_rejects_missing_class_directory(tmp_path: Path) -> None:
    for class_name in ("Cover", "JMiPOD", "JUNIWARD"):
        _write_jpeg(tmp_path / class_name / "00001.jpg")

    with pytest.raises(FileNotFoundError, match="UERD"):
        run_dataset_preflight(tmp_path)


def test_dataset_preflight_rejects_mismatched_source_stems(tmp_path: Path) -> None:
    _complete_dataset(tmp_path)
    (tmp_path / "JMiPOD" / "00011.jpg").unlink()

    with pytest.raises(IncompleteGroupError, match="incomplete"):
        run_dataset_preflight(tmp_path)


def test_dataset_preflight_rejects_corrupted_jpeg(tmp_path: Path) -> None:
    _complete_dataset(tmp_path)
    (tmp_path / "UERD" / "00005.jpg").write_text("not a JPEG", encoding="utf-8")

    with pytest.raises(DatasetPreflightError, match="unreadable"):
        run_dataset_preflight(tmp_path)


def test_dataset_preflight_rejects_ambiguous_extensions(tmp_path: Path) -> None:
    _complete_dataset(tmp_path)
    _write_jpeg(tmp_path / "Cover" / "00001.jpeg")

    with pytest.raises(DuplicateSourceError, match="Ambiguous"):
        run_dataset_preflight(tmp_path)


def test_explicit_alaska2_selection_never_uses_synthetic_fallback(tmp_path: Path) -> None:
    paths = ProjectPaths(tmp_path)
    _complete_dataset(paths.pd12m, groups=3)

    with pytest.raises(FileNotFoundError, match="No synthetic fallback"):
        select_dataset(paths, source="alaska2")
    assert select_dataset(paths, source="synthetic").synthetic


def test_required_cuda_path_fails_actionably_without_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(GPUPreflightError, match="GPU Dev Container"):
        resolve_device("cuda")
    with pytest.raises(GPUPreflightError, match="alaska2-gpu-preflight"):
        run_gpu_preflight()
