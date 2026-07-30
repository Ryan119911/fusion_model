"""Differentiable geometry-to-brush-appearance refinement network."""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from models.character_generator import ConvBlock, DownBlock, UpBlock


STYLE_REFINER_CHECKPOINT_FORMAT = "kaishu_style_refiner_v1"


class StyleRefinerUNet(nn.Module):
    """Predict ink appearance while retaining an explicit geometry residual."""

    def __init__(
        self,
        input_channels: int = 4,
        base_channels: int = 24,
        depth: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        widths = [base_channels * (2**level) for level in range(depth + 1)]
        self.input_channels = int(input_channels)
        self.input_block = ConvBlock(input_channels, widths[0])
        self.down_blocks = nn.ModuleList(
            [DownBlock(widths[index], widths[index + 1]) for index in range(depth)]
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.up_blocks = nn.ModuleList(
            [
                UpBlock(widths[index], widths[index - 1], widths[index - 1])
                for index in range(depth, 0, -1)
            ]
        )
        self.output_layer = nn.Conv2d(widths[0], 1, kernel_size=1)
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4 or features.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected [B,{self.input_channels},H,W], got {tuple(features.shape)}"
            )
        values = self.input_block(features)
        skips = [values]
        for down in self.down_blocks:
            values = down(values)
            skips.append(values)
        values = self.dropout(values)
        for up, skip in zip(self.up_blocks, reversed(skips[:-1])):
            values = up(values, skip)
        # Channel zero is the geometry mask. A bounded logit residual makes
        # initialization an identity mapping, rather than an arbitrary gray image.
        geometry = features[:, :1].clamp(1e-4, 1.0 - 1e-4)
        base_logits = torch.logit(geometry)
        residual = 3.0 * torch.tanh(self.output_layer(values))
        return torch.sigmoid(base_logits + residual)


def build_style_refiner(
    config: Optional[Dict[str, Any]] = None, **overrides: Any
) -> StyleRefinerUNet:
    values = dict(config or {})
    values.update({key: value for key, value in overrides.items() if value is not None})
    return StyleRefinerUNet(**values)
