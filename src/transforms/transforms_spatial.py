"""
===============================================================================
transforms_spatial.py
===============================================================================
JPEG-grid-aligned crop transforms.

Responsibilities:
  - Sample random training crops on an explicit block grid.
  - Derive validation/test coordinates from source identity, not decoded
    pixels.
  - Reject undersized images and invalid crop or block dimensions.

Design principles:
  - The deterministic transform separates coordinate generation from image
    cropping so spatial and DCT modalities can share one box.

Boundaries:
  - These transforms preserve alignment to the decoded image origin. They do
    not claim to reconstruct or manipulate JPEG DCT coefficients.

Notes:
  - A shared source filename yields the same box for Cover and every stego
    variant.
===============================================================================
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

__all__ = ["AlignedDeterministicCrop", "AlignedRandomCrop", "CropBox", "crop_image"]


@dataclass(frozen=True, slots=True)
class CropBox:
    """Represent a pixel crop with exclusive right and lower coordinates.

    Parameters
    ----------
    left
        Inclusive horizontal start.
    top
        Inclusive vertical start.
    right
        Exclusive horizontal end.
    bottom
        Exclusive vertical end.
    """

    left: int
    top: int
    right: int
    bottom: int

    @property
    def size(self) -> tuple[int, int]:
        """Return crop width and height.

        Returns
        -------
        tuple[int, int]
            Width and height derived from exclusive coordinates.
        """
        return self.right - self.left, self.bottom - self.top


def _validate(size: int, block_size: int, image_size: tuple[int, int]) -> tuple[int, int]:
    if size <= 0 or block_size <= 0:
        raise ValueError("size and block_size must be positive.")
    if size % block_size:
        raise ValueError("size must be divisible by block_size.")
    width, height = image_size
    if width < size or height < size:
        raise ValueError(f"Image size {image_size} is smaller than requested crop {size}.")
    return (width - size) // block_size, (height - size) // block_size


def crop_image(image: Image.Image, box: CropBox) -> Image.Image:
    """Crop a PIL image with an already validated crop box.

    Parameters
    ----------
    image
        Source image.
    box
        Pixel crop coordinates.

    Returns
    -------
    PIL.Image.Image
        Cropped image view returned by Pillow.
    """
    return image.crop((box.left, box.top, box.right, box.bottom))


class AlignedRandomCrop:
    """Sample random training crops aligned to an explicit pixel grid.

    Parameters
    ----------
    size
        Square crop side length in pixels.
    block_size
        Alignment grid spacing in pixels.
    """

    def __init__(self, size: int, block_size: int = 8) -> None:
        self.size = size
        self.block_size = block_size

    def box_for(self, image_size: tuple[int, int]) -> CropBox:
        """Sample one block-aligned crop box for an image size.

        Parameters
        ----------
        image_size
            Image width and height.

        Returns
        -------
        CropBox
            Random valid crop coordinates.

        Raises
        ------
        ValueError
            If crop dimensions, alignment, or image size are invalid.
        """
        x_steps, y_steps = _validate(self.size, self.block_size, image_size)
        left = random.randint(0, x_steps) * self.block_size
        top = random.randint(0, y_steps) * self.block_size
        return CropBox(left, top, left + self.size, top + self.size)

    def __call__(self, image: Image.Image) -> Image.Image:
        """Crop an image at a sampled aligned training position.

        Parameters
        ----------
        image
            PIL image large enough for the configured crop.

        Returns
        -------
        PIL.Image.Image
            Random block-aligned crop.

        Raises
        ------
        ValueError
            If the image or configured dimensions are invalid.
        """
        return crop_image(image, self.box_for(image.size))


class AlignedDeterministicCrop:
    """Derive block-aligned evaluation crops from source identity.

    Parameters
    ----------
    size
        Square crop side length in pixels.
    block_size
        Alignment grid spacing in pixels.
    seed
        Project seed included in the stable identity digest.

    Notes
    -----
    Cover and stego variants sharing a source identity receive identical coordinates.
    """

    def __init__(self, size: int, block_size: int = 8, seed: int = 42) -> None:
        self.size = size
        self.block_size = block_size
        self.seed = seed

    def box_for(self, image_size: tuple[int, int], source_id: str | Path) -> CropBox:
        """Compute a stable block-aligned crop box for one source identity.

        Parameters
        ----------
        image_size
            Image width and height.
        source_id
            Stable source stem or path name shared by all variants.

        Returns
        -------
        CropBox
            Deterministic valid crop coordinates.

        Raises
        ------
        ValueError
            If crop dimensions, alignment, or image size are invalid.
        """
        x_steps, y_steps = _validate(self.size, self.block_size, image_size)
        identity = Path(source_id).name
        digest = hashlib.blake2b(
            f"{self.seed}:{identity}".encode(),
            digest_size=16,
            person=b"alaska2-crop",
        ).digest()
        x_index = int.from_bytes(digest[:8], "little") % (x_steps + 1)
        y_index = int.from_bytes(digest[8:], "little") % (y_steps + 1)
        left = x_index * self.block_size
        top = y_index * self.block_size
        return CropBox(left, top, left + self.size, top + self.size)

    def __call__(self, image: Image.Image, *, source_id: str | Path) -> Image.Image:
        """Crop an image at source-identity-derived coordinates.

        Parameters
        ----------
        image
            PIL image large enough for the configured crop.
        source_id
            Stable identity shared by Cover and stego variants.

        Returns
        -------
        PIL.Image.Image
            Deterministic block-aligned crop.

        Raises
        ------
        ValueError
            If the image or configured dimensions are invalid.
        """
        return crop_image(image, self.box_for(image.size, source_id))
