"""
===============================================================================
models_tinycnn.py
===============================================================================
Compact luminance-channel baseline.

Responsibilities:
  - Define the original five-convolution TinyCNN baseline.
  - Preserve logits output for BCEWithLogitsLoss.
  - Validate the expected single-channel input contract.

Design principles:
  - The architecture remains intentionally small and contains no dataset or
    training policy.

Boundaries:
  - This model consumes decoded spatial luminance; it is not a DCT-domain
    network.

Notes:
  - Convolutions intentionally use no padding to preserve the historical
    baseline.
===============================================================================
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["TinyCNN"]


class TinyCNN(nn.Module):
    """Implement the five-convolution luminance-channel binary baseline.

    Notes
    -----
    Convolutions intentionally use no padding to preserve the academic baseline.
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._block(1, 8, stride=1),
            self._block(8, 16, stride=1),
            self._block(16, 32, stride=1),
            self._block(32, 64, stride=2),
            self._block(64, 128, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, 1)

    @staticmethod
    def _block(in_channels: int, out_channels: int, *, stride: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute one binary logit per luminance image.

        Parameters
        ----------
        inputs
            Batch tensor shaped ``[N, 1, H, W]``.

        Returns
        -------
        torch.Tensor
            Logit tensor shaped ``[N, 1]``.

        Raises
        ------
        ValueError
            If inputs are not a four-dimensional single-channel batch.
        """
        if inputs.ndim != 4 or inputs.shape[1] != 1:
            raise ValueError("TinyCNN expects input shape [N, 1, H, W].")
        features = self.features(inputs)
        return self.classifier(self.pool(features).flatten(1))
