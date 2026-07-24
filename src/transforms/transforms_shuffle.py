"""
===============================================================================
transforms_shuffle.py
===============================================================================
Spatial tile-shuffle augmentation.

Responsibilities:
  - Divide an image into a declared number of tiles per axis.
  - Randomly permute tiles while preserving each tile's decoded pixels.
  - Fail clearly when image dimensions do not divide into the tile grid.

Design principles:
  - The API names what the transform actually does. It makes no claim that
    decoded spatial tiles preserve JPEG coefficients after recompression.

Boundaries:
  - This transform changes semantic layout and is intended only for training.

Notes:
  - For a 256-pixel crop and tiles_per_axis=8, each tile is 32 by 32 pixels.
===============================================================================
"""

from __future__ import annotations

import random

from PIL import Image

__all__ = ["RandomTileShuffle"]


class RandomTileShuffle:
    """Randomly permute a regular decoded-image tile grid.

    Parameters
    ----------
    tiles_per_axis
        Number of equal tiles along both image axes; must exceed one.

    Raises
    ------
    ValueError
        If the tile count is less than two.

    Notes
    -----
    The transform preserves decoded tile pixels but makes no DCT-preservation claim.
    """

    def __init__(self, tiles_per_axis: int = 8) -> None:
        if tiles_per_axis <= 1:
            raise ValueError("tiles_per_axis must be greater than one.")
        self.tiles_per_axis = tiles_per_axis

    def __call__(self, image: Image.Image) -> Image.Image:
        """Permute all spatial tiles in one image.

        Parameters
        ----------
        image
            PIL image whose dimensions divide evenly into the tile grid.

        Returns
        -------
        PIL.Image.Image
            New image with every complete tile placed exactly once.

        Raises
        ------
        ValueError
            If either image dimension is not divisible by the tile count.
        """
        width, height = image.size
        if width % self.tiles_per_axis or height % self.tiles_per_axis:
            raise ValueError(f"Image size {image.size} must be divisible by tiles_per_axis={self.tiles_per_axis}.")
        tile_width = width // self.tiles_per_axis
        tile_height = height // self.tiles_per_axis
        tiles = [
            image.crop(
                (
                    x * tile_width,
                    y * tile_height,
                    (x + 1) * tile_width,
                    (y + 1) * tile_height,
                )
            )
            for y in range(self.tiles_per_axis)
            for x in range(self.tiles_per_axis)
        ]
        random.shuffle(tiles)
        result = Image.new(image.mode, image.size)
        for index, tile in enumerate(tiles):
            left = (index % self.tiles_per_axis) * tile_width
            top = (index // self.tiles_per_axis) * tile_height
            result.paste(tile, (left, top))
        return result
