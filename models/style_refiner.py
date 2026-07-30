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
        support_mode: str = "mask_only",
    ):
        super().__init__()
        if support_mode not in {"mask_only", "mask_or_soft"}:
            raise ValueError(
                "support_mode must be 'mask_only' or 'mask_or_soft'"
            )
        widths = [base_channels * (2**level) for level in range(depth + 1)]
        self.input_channels = int(input_channels)
        self.support_mode = support_mode
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
        # Channel zero is a hard geometry mask. Keep it as an explicit output
        # gate so appearance learning can never conceal a wrong trajectory by
        # creating ink elsewhere. Inside the support, a moderate base logit and
        # wide residual range allow the network to learn real gray ink values;
        # using logit(binary_mask) here would saturate and suppress gradients.
        geometry = features[:, :1].clamp(0.0, 1.0)
        if self.support_mode == "mask_or_soft":
            if self.input_channels < 4:
                raise ValueError(
                    "mask_or_soft support requires the soft-geometry channel"
                )
            geometry = torch.maximum(
                geometry, features[:, 3:4].clamp(0.0, 1.0)
            )
        residual = 8.0 * torch.tanh(self.output_layer(values))
        appearance = torch.sigmoid(4.0 + residual)
        return geometry * appearance


def build_style_refiner(
    config: Optional[Dict[str, Any]] = None, **overrides: Any
) -> StyleRefinerUNet:
    values = dict(config or {})
    values.update({key: value for key, value in overrides.items() if value is not None})
    return StyleRefinerUNet(**values)
