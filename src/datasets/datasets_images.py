"""
===============================================================================
datasets_images.py
===============================================================================
Spatial JPEG datasets for PyTorch.

Responsibilities:
  - Load one JPEG exactly once per sample and close its file handle.
  - Apply source-identity-aware evaluation crops before ordinary transforms.
  - Produce labels with the dtype required by binary or multiclass losses.

Design principles:
  - Color conversion is explicit. Identity-aware crops are a separate
    contract rather than being hidden inside torchvision Compose.

Boundaries:
  - This module does not split dataframes or create data loaders.

Notes:
  - Supported modes are RGB, YCbCr, and Y (luminance only).
===============================================================================
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, cast

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..transforms.transforms_spatial import AlignedDeterministicCrop

__all__ = ["ImageDataset"]

ImageTransform = Callable[[Image.Image], torch.Tensor]


class ImageDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Load decoded JPEG samples with explicit color and crop contracts.

    Parameters
    ----------
    dataframe
        Rows containing path, source identity, and target columns.
    color_mode
        Decoded output mode: RGB, YCbCr, or luminance-only Y.
    target_column
        Dataframe column converted to the label tensor.
    transform
        Required PIL-to-tensor transform.
    identity_crop
        Optional source-deterministic evaluation crop.

    Raises
    ------
    ValueError
        If columns, color mode, target dtype, or transform contract is invalid.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        color_mode: Literal["RGB", "YCbCr", "Y"],
        target_column: str = "label",
        transform: ImageTransform | None = None,
        identity_crop: AlignedDeterministicCrop | None = None,
    ) -> None:
        required = {"path", "source_id", target_column}
        missing = required - set(dataframe.columns)
        if missing:
            raise ValueError(f"Dataset dataframe is missing columns: {sorted(missing)}")
        if color_mode not in {"RGB", "YCbCr", "Y"}:
            raise ValueError(f"Unsupported color_mode: {color_mode}")
        self.dataframe = dataframe.reset_index(drop=True)
        self.color_mode = color_mode
        self.target_column = target_column
        self.transform = transform
        self.identity_crop = identity_crop
        kind = self.dataframe[target_column].dtype.kind
        if kind == "f":
            self.label_dtype = torch.float32
        elif kind in {"i", "u", "b"}:
            self.label_dtype = torch.long
        else:
            raise ValueError(f"Unsupported target dtype: {self.dataframe[target_column].dtype}")

    def __len__(self) -> int:
        """Return sample count."""
        return len(self.dataframe)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load, close, transform, and label one spatial JPEG sample.

        Parameters
        ----------
        index
            Zero-based dataframe row position.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Image tensor and scalar target tensor.

        Raises
        ------
        ValueError
            If no transform is configured.
        TypeError
            If the transform does not return a tensor.
        OSError
            If Pillow cannot decode the selected image.
        """
        row = self.dataframe.iloc[index]
        with Image.open(row["path"]) as opened:
            converted = opened.convert("YCbCr" if self.color_mode in {"YCbCr", "Y"} else "RGB")
            image = converted.copy()
        if self.identity_crop is not None:
            image = self.identity_crop(image, source_id=str(row["source_id"]))
        if self.color_mode == "Y":
            image = image.split()[0]
        if self.transform is None:
            raise ValueError("ImageDataset requires a transform that returns a tensor.")
        tensor = cast(torch.Tensor, self.transform(image))
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Image transform must return a torch.Tensor.")
        label = torch.tensor(row[self.target_column], dtype=self.label_dtype)
        return tensor, label
