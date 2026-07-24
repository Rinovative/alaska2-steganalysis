"""
===============================================================================
test_transforms_and_datasets.py
===============================================================================
Verify crop, tile-shuffle, dataset, fusion, and loader contracts.

Responsibilities:
  - Exercise source-aligned crops and honest spatial tile permutation.
  - Validate image handles, DCT reconciliation, and distinct DataLoader
    behavior.

Design principles:
  - Small coordinate fixtures make alignment and mutation errors observable.
  - CPU-only tests avoid dependence on worker processes or accelerators.

Boundaries:
  - Scientific split policy belongs to the data package tests.
  - Full-resolution ALASKA2 performance remains outside this suite.
===============================================================================
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch.utils.data import TensorDataset
from torchvision import transforms as vision_transforms

from src.data.data_split import DatasetSplits
from src.datasets.datasets_dct import FusionDataset
from src.datasets.datasets_images import ImageDataset
from src.datasets.datasets_loaders import build_image_loaders, build_loaders
from src.transforms.transforms_shuffle import RandomTileShuffle
from src.transforms.transforms_spatial import AlignedDeterministicCrop, AlignedRandomCrop


def _write_coordinate_jpeg(path: Path, size: int = 32) -> None:
    values = np.arange(size * size, dtype=np.uint8).reshape(size, size)
    image = np.stack([values, values, values], axis=-1)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path, format="JPEG", quality=100, subsampling=0)


def test_deterministic_crop_pairs_by_source_identity_not_pixels() -> None:
    crop = AlignedDeterministicCrop(size=16, block_size=8, seed=42)
    first = Image.new("RGB", (32, 32), "black")
    second = Image.new("RGB", (32, 32), "white")
    assert crop.box_for(first.size, "00001.jpg") == crop.box_for(second.size, "00001.jpg")
    assert crop.box_for(first.size, Path("variant/00001.jpg")) == crop.box_for(first.size, "00001.jpg")


@pytest.mark.parametrize(
    ("constructor", "message"),
    [
        (lambda: AlignedDeterministicCrop(size=15, block_size=8).box_for((32, 32), "a.jpg"), "divisible"),
        (lambda: AlignedRandomCrop(size=40, block_size=8)(Image.new("RGB", (32, 32))), "smaller"),
    ],
)
def test_crop_size_errors_are_explicit(constructor, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        constructor()


def test_tile_shuffle_preserves_complete_tiles(monkeypatch: pytest.MonkeyPatch) -> None:
    image = Image.new("L", (4, 4))
    pixels = image.load()
    assert pixels is not None
    for y in range(4):
        for x in range(4):
            pixels[x, y] = (x // 2) + 2 * (y // 2)
    monkeypatch.setattr("random.shuffle", lambda values: values.reverse())
    shuffled = np.asarray(RandomTileShuffle(tiles_per_axis=2)(image))
    assert set(np.unique(shuffled)) == {0, 1, 2, 3}
    assert np.all(shuffled[:2, :2] == 3)
    with pytest.raises(ValueError, match="divisible"):
        RandomTileShuffle(tiles_per_axis=3)(image)


def test_image_dataset_applies_identity_crop_and_closes_file(tmp_path: Path) -> None:
    path = tmp_path / "Cover" / "00001.jpg"
    _write_coordinate_jpeg(path)
    frame = pd.DataFrame({"path": [str(path)], "source_id": ["00001.jpg"], "label_bin": [0.0]})
    dataset = ImageDataset(
        frame,
        color_mode="Y",
        target_column="label_bin",
        transform=vision_transforms.ToTensor(),
        identity_crop=AlignedDeterministicCrop(16, seed=4),
    )
    image, label = dataset[0]
    assert image.shape == (1, 16, 16)
    assert label.dtype == torch.float32
    path.unlink()


def test_fusion_reads_coefficients_once_and_reconciles_chroma(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "Cover" / "00001.jpg"
    _write_coordinate_jpeg(path, size=16)
    arrays = [
        np.arange(256, dtype=np.int32).reshape(16, 16),
        np.arange(64, dtype=np.int32).reshape(8, 8),
        np.arange(64, dtype=np.int32).reshape(8, 8) + 100,
    ]
    originals = [array.copy() for array in arrays]
    calls = 0

    def fake_read(_: str):
        nonlocal calls
        calls += 1
        return SimpleNamespace(coef_arrays=arrays)

    monkeypatch.setattr("src.datasets.datasets_dct.jpegio.read", fake_read)
    frame = pd.DataFrame({"path": [str(path)], "source_id": ["00001.jpg"], "label": [0]})
    dataset = FusionDataset(
        frame,
        image_transform=vision_transforms.ToTensor(),
        dct_channels=3,
        identity_crop=AlignedDeterministicCrop(16, seed=1),
    )
    (image, coefficients), label = dataset[0]
    assert calls == 1
    assert image.shape == (3, 16, 16)
    assert coefficients.shape == (3, 16, 16)
    assert label.dtype == torch.long
    for actual, expected in zip(arrays, originals, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_loader_bundle_has_three_distinct_loader_contracts() -> None:
    tensors = torch.arange(24, dtype=torch.float32).reshape(6, 1, 2, 2)
    labels = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.float32)
    bundle = build_loaders(
        TensorDataset(tensors, labels),
        TensorDataset(tensors.clone(), labels.clone()),
        TensorDataset(tensors.clone(), labels.clone()),
        batch_size=2,
        num_workers=0,
        seed=9,
    )
    assert len({id(bundle.train), id(bundle.validation), id(bundle.test)}) == 3
    assert bundle.train.sampler.__class__.__name__ == "RandomSampler"
    assert bundle.validation.sampler.__class__.__name__ == "SequentialSampler"
    assert bundle.test.sampler.__class__.__name__ == "SequentialSampler"


def test_image_loader_factory_constructs_distinct_split_datasets(tmp_path: Path) -> None:
    frames: list[pd.DataFrame] = []
    for index, split_name in enumerate(("train", "validation", "test")):
        image_path = tmp_path / split_name / "Cover" / f"{index:05d}.jpg"
        _write_coordinate_jpeg(image_path)
        frames.append(
            pd.DataFrame(
                {
                    "path": [str(image_path)],
                    "source_id": [f"{index:05d}"],
                    "label_bin": [0.0],
                }
            )
        )
    bundle = build_image_loaders(
        DatasetSplits(train=frames[0], validation=frames[1], test=frames[2]),
        color_mode="Y",
        train_transform=vision_transforms.ToTensor(),
        evaluation_transform=vision_transforms.ToTensor(),
        identity_crop=AlignedDeterministicCrop(16, seed=42),
        batch_size=1,
        num_workers=0,
        seed=42,
    )

    assert bundle.train.dataset is not bundle.validation.dataset
    assert bundle.validation.dataset is not bundle.test.dataset
    validation_images, _ = next(iter(bundle.validation))
    test_images, _ = next(iter(bundle.test))
    assert validation_images.shape == test_images.shape == (1, 1, 16, 16)
