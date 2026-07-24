"""
===============================================================================
datasets_dct.py
===============================================================================
DCT coefficient and fusion datasets.

Responsibilities:
  - Read each JPEG coefficient structure once per sample.
  - Copy coefficient arrays before masking or normalization.
  - Reconcile chroma subsampling and decoded spatial shapes for fusion
    tensors.
  - Share source-identity crop coordinates across spatial and coefficient
    modalities.

Design principles:
  - Raw DCT datasets preserve native channel dimensions. Fusion datasets
    explicitly resample coefficient planes with nearest-neighbour
    interpolation so concatenation is well-defined and spatially aligned at
    the decoded-image scale.

Boundaries:
  - Nearest-neighbour reconciliation is a modeling adapter; it is not an
    inverse DCT and does not claim to recover missing chroma samples.

Notes:
  - jpegio loads coefficient arrays eagerly and does not expose a persistent
    file handle.
===============================================================================
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, cast

import jpegio
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as torch_functional
from PIL import Image
from torch.utils.data import Dataset

from ..transforms.transforms_spatial import AlignedDeterministicCrop, CropBox, crop_image

__all__ = ["DCTCoefficientDataset", "FusionDataset"]

TensorTransform = Callable[[torch.Tensor], torch.Tensor]
ImageTransform = Callable[[Image.Image], torch.Tensor]


def _label_dtype(series: pd.Series) -> torch.dtype:
    kind = series.dtype.kind
    if kind == "f":
        return torch.float32
    if kind in {"i", "u", "b"}:
        return torch.long
    raise ValueError(f"Unsupported target dtype: {series.dtype}")


def _coefficient_tensor(
    array: np.ndarray,
    *,
    ac_only: bool,
    standardize: bool,
) -> torch.Tensor:
    coefficients = np.array(array, dtype=np.float32, copy=True)
    if coefficients.ndim != 2 or min(coefficients.shape) == 0:
        raise ValueError(f"Invalid JPEG coefficient shape: {coefficients.shape}")
    if ac_only:
        mask = np.ones(coefficients.shape, dtype=np.float32)
        mask[::8, ::8] = 0.0
        coefficients = coefficients * mask
    if standardize:
        mean = float(coefficients.mean())
        deviation = float(coefficients.std())
        coefficients = (coefficients - mean) / max(deviation, 1e-8)
    return torch.from_numpy(coefficients).unsqueeze(0)


def _scaled_crop(
    tensor: torch.Tensor,
    box: CropBox,
    *,
    image_size: tuple[int, int],
    output_size: tuple[int, int],
) -> torch.Tensor:
    image_width, image_height = image_size
    coefficient_height, coefficient_width = tensor.shape[-2:]
    left = round(box.left * coefficient_width / image_width)
    right = round(box.right * coefficient_width / image_width)
    top = round(box.top * coefficient_height / image_height)
    bottom = round(box.bottom * coefficient_height / image_height)
    right = max(left + 1, min(right, coefficient_width))
    bottom = max(top + 1, min(bottom, coefficient_height))
    cropped = tensor[:, top:bottom, left:right].unsqueeze(0)
    return torch_functional.interpolate(cropped, size=output_size, mode="nearest").squeeze(0)


class DCTCoefficientDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Load one native-resolution JPEG coefficient channel and its target.

    Parameters
    ----------
    dataframe
        Rows containing image paths and the requested target column.
    channel
        JPEG component index: zero for Y, one for Cb, or two for Cr.
    target_column
        Dataframe column converted to a PyTorch label tensor.
    ac_only
        Whether all block DC coefficients are zeroed in the copied array.
    standardize
        Whether each coefficient plane is standardized independently.
    transform
        Optional tensor-to-tensor transform applied after conversion.

    Raises
    ------
    ValueError
        If required columns, channel, target dtype, or coefficient shape is invalid.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        channel: Literal[0, 1, 2] = 0,
        target_column: str = "label",
        ac_only: bool = False,
        standardize: bool = False,
        transform: TensorTransform | None = None,
    ) -> None:
        required = {"path", target_column}
        missing = required - set(dataframe.columns)
        if missing:
            raise ValueError(f"Dataset dataframe is missing columns: {sorted(missing)}")
        if channel not in {0, 1, 2}:
            raise ValueError("channel must be 0 (Y), 1 (Cb), or 2 (Cr).")
        self.dataframe = dataframe.reset_index(drop=True)
        self.channel = channel
        self.target_column = target_column
        self.ac_only = ac_only
        self.standardize = standardize
        self.transform = transform
        self.label_dtype = _label_dtype(cast(pd.Series, self.dataframe[target_column]))

    def __len__(self) -> int:
        """Return sample count."""
        return len(self.dataframe)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load and transform one coefficient plane and label.

        Parameters
        ----------
        index
            Zero-based dataframe row position.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Coefficient tensor and scalar target tensor.

        Raises
        ------
        ValueError
            If the JPEG lacks the requested coefficient component.
        """
        row = self.dataframe.iloc[index]
        jpeg = jpegio.read(str(row["path"]))
        if self.channel >= len(jpeg.coef_arrays):
            raise ValueError(f"JPEG lacks channel {self.channel}: {row['path']}")
        tensor = _coefficient_tensor(
            jpeg.coef_arrays[self.channel],
            ac_only=self.ac_only,
            standardize=self.standardize,
        )
        if self.transform is not None:
            tensor = self.transform(tensor)
        label = torch.tensor(row[self.target_column], dtype=self.label_dtype)
        return tensor, label


class FusionDataset(Dataset[tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]]):
    """Load aligned decoded YCbCr and JPEG coefficient tensors for one source.

    Parameters
    ----------
    dataframe
        Rows containing path, source identity, and target columns.
    image_transform
        Transform producing a spatial ``[C, H, W]`` tensor.
    dct_channels
        Whether one luminance or all three coefficient planes are returned.
    target_column
        Dataframe column converted to the label tensor.
    identity_crop
        Optional source-deterministic crop shared by both modalities.
    ac_only
        Whether block DC coefficients are removed from copied planes.
    standardize_dct
        Whether each coefficient plane is standardized independently.

    Raises
    ------
    ValueError
        If columns, channel count, target dtype, or JPEG components are invalid.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        image_transform: ImageTransform,
        dct_channels: Literal[1, 3] = 3,
        target_column: str = "label",
        identity_crop: AlignedDeterministicCrop | None = None,
        ac_only: bool = True,
        standardize_dct: bool = True,
    ) -> None:
        required = {"path", "source_id", target_column}
        missing = required - set(dataframe.columns)
        if missing:
            raise ValueError(f"Dataset dataframe is missing columns: {sorted(missing)}")
        if dct_channels not in {1, 3}:
            raise ValueError("dct_channels must be 1 or 3.")
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_transform = image_transform
        self.dct_channels = dct_channels
        self.target_column = target_column
        self.identity_crop = identity_crop
        self.ac_only = ac_only
        self.standardize_dct = standardize_dct
        self.label_dtype = _label_dtype(cast(pd.Series, self.dataframe[target_column]))

    def __len__(self) -> int:
        """Return sample count."""
        return len(self.dataframe)

    def __getitem__(self, index: int) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Load one aligned spatial/coefficient pair and target.

        Parameters
        ----------
        index
            Zero-based dataframe row position.

        Returns
        -------
        tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]
            Spatial tensor, DCT tensor, and scalar target tensor.

        Raises
        ------
        TypeError
            If the spatial transform does not return a three-dimensional tensor.
        ValueError
            If the JPEG lacks the requested coefficient components.
        """
        row = self.dataframe.iloc[index]
        path = str(row["path"])
        with Image.open(path) as opened:
            image_size = opened.size
            image = opened.convert("YCbCr").copy()
        box: CropBox | None = None
        if self.identity_crop is not None:
            box = self.identity_crop.box_for(image_size, str(row["source_id"]))
            image = crop_image(image, box)
        image_tensor = cast(torch.Tensor, self.image_transform(image))
        if not isinstance(image_tensor, torch.Tensor) or image_tensor.ndim != 3:
            raise TypeError("image_transform must return a [C, H, W] tensor.")

        jpeg = jpegio.read(path)
        requested_channels = range(self.dct_channels)
        if len(jpeg.coef_arrays) < self.dct_channels:
            raise ValueError(f"JPEG has {len(jpeg.coef_arrays)} coefficient channels: {path}")
        target_shape = (image_tensor.shape[-2], image_tensor.shape[-1])
        coefficient_tensors: list[torch.Tensor] = []
        for channel in requested_channels:
            tensor = _coefficient_tensor(
                jpeg.coef_arrays[channel],
                ac_only=self.ac_only,
                standardize=self.standardize_dct,
            )
            if box is not None:
                tensor = _scaled_crop(
                    tensor,
                    box,
                    image_size=image_size,
                    output_size=target_shape,
                )
            elif tensor.shape[-2:] != target_shape:
                tensor = torch_functional.interpolate(
                    tensor.unsqueeze(0),
                    size=target_shape,
                    mode="nearest",
                ).squeeze(0)
            coefficient_tensors.append(tensor)

        dct_tensor = torch.cat(coefficient_tensors, dim=0)
        label = torch.tensor(row[self.target_column], dtype=self.label_dtype)
        return (image_tensor, dct_tensor), label
