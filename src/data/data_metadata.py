"""
===============================================================================
data_metadata.py
===============================================================================
Extract validated JPEG metadata for EDA.

Responsibilities:
  - Read image dimensions, mode, luminance quantization, and a quality proxy.
  - Close every image handle promptly.
  - Surface malformed files instead of silently fabricating valid metadata.

Design principles:
  - Extraction is row-preserving and returns a copy. A non-strict mode
    records explicit error text for exploratory audits.

Boundaries:
  - The quality value is a coarse proxy for the known ALASKA2 quality levels,
    not a general JPEG quality estimator.

Notes:
  - The 64 quantization values are stored as q_y_00 through q_y_63.
===============================================================================
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm.auto import tqdm

__all__ = ["JPEGMetadataError", "add_jpeg_metadata"]


class JPEGMetadataError(ValueError):
    """Raised when JPEG metadata cannot be extracted in strict mode."""


def _quality_proxy(values: list[int]) -> int:
    mean = float(np.mean(values))
    if mean < 8:
        return 95
    if mean < 20:
        return 90
    return 75


def add_jpeg_metadata(
    dataframe: pd.DataFrame,
    *,
    strict: bool = True,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Add validated dimensions, mode, quality proxy, and luminance quantization values.

    Parameters
    ----------
    dataframe
        Image index containing a ``path`` column.
    strict
        Whether the first unreadable or non-quantized image raises immediately.
    show_progress
        Whether to display a tqdm progress bar.

    Returns
    -------
    pandas.DataFrame
        Row-preserving copy augmented with JPEG metadata and error details.

    Raises
    ------
    ValueError
        If the input has no ``path`` column.
    JPEGMetadataError
        If strict extraction encounters an unreadable or invalid JPEG.

    Notes
    -----
    The reported quality is a coarse proxy for the known ALASKA2 quality levels.
    """
    if "path" not in dataframe:
        raise ValueError("dataframe must contain a path column.")

    rows: list[dict[str, object]] = []
    paths = dataframe["path"].tolist()
    iterator = tqdm(paths, desc="Extracting JPEG metadata", disable=not show_progress)
    for raw_path in iterator:
        path = Path(raw_path)
        row: dict[str, object] = {
            "jpeg_quality": -1,
            "width": -1,
            "height": -1,
            "mode": "unknown",
            "metadata_error": None,
            **{f"q_y_{index:02d}": -1 for index in range(64)},
        }
        try:
            with Image.open(path) as image:
                image.load()
                if image.format != "JPEG":
                    raise JPEGMetadataError(f"Not a JPEG file: {path}")
                quantization = getattr(image, "quantization", None)
                if not quantization or 0 not in quantization or len(quantization[0]) < 64:
                    raise JPEGMetadataError(f"Missing luminance quantization table: {path}")
                values = [int(value) for value in quantization[0][:64]]
                row.update(
                    jpeg_quality=_quality_proxy(values),
                    width=image.width,
                    height=image.height,
                    mode=image.mode,
                )
                row.update({f"q_y_{index:02d}": value for index, value in enumerate(values)})
        except (OSError, UnidentifiedImageError, JPEGMetadataError) as error:
            if strict:
                raise JPEGMetadataError(str(error)) from error
            row["metadata_error"] = str(error)
        rows.append(row)

    return pd.concat(
        [dataframe.reset_index(drop=True), pd.DataFrame.from_records(rows)],
        axis=1,
    )
