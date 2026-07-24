"""
===============================================================================
models_efficientnet.py
===============================================================================
EfficientNet-B0 decoded-YCbCr adapter.

Responsibilities:
  - Construct torchvision EfficientNet-B0 with optional pretrained weights.
  - Reinitialize the input stem for YCbCr semantics.
  - Replace the classifier with one binary logit.

Design principles:
  - Weight download is a constructor choice, never an import-time side
    effect.

Boundaries:
  - This module defines architecture only; freezing and optimization live
    elsewhere.

Notes:
  - The decoded YCbCr tensor still has three channels, but its statistics
    differ from ImageNet RGB, so the stem is intentionally reinitialized.
===============================================================================
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

__all__ = ["EfficientNetB0YCbCr"]


class EfficientNetB0YCbCr(nn.Module):
    """Adapt torchvision EfficientNet-B0 to decoded YCbCr binary steganalysis.

    Parameters
    ----------
    weights
        Optional torchvision pretrained weights; ``None`` avoids any download.

    Notes
    -----
    The three-channel input stem and one-logit classifier are reinitialized.
    """

    def __init__(self, *, weights: EfficientNet_B0_Weights | None = None) -> None:
        super().__init__()
        self.backbone = efficientnet_b0(weights=weights)
        old_stem = self.backbone.features[0][0]
        replacement = nn.Conv2d(
            3,
            old_stem.out_channels,
            kernel_size=old_stem.kernel_size,
            stride=old_stem.stride,
            padding=old_stem.padding,
            dilation=old_stem.dilation,
            groups=old_stem.groups,
            bias=False,
        )
        nn.init.kaiming_normal_(replacement.weight, mode="fan_out")
        self.backbone.features[0][0] = replacement
        input_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(input_features, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute one binary logit per decoded YCbCr image.

        Parameters
        ----------
        inputs
            Batch tensor shaped ``[N, 3, H, W]``.

        Returns
        -------
        torch.Tensor
            Logit tensor shaped ``[N, 1]``.

        Raises
        ------
        ValueError
            If inputs are not a four-dimensional three-channel batch.
        """
        if inputs.ndim != 4 or inputs.shape[1] != 3:
            raise ValueError("EfficientNetB0YCbCr expects input shape [N, 3, H, W].")
        return self.backbone(inputs)
