from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int) -> int:
    groups = min(8, channels)
    while channels % groups != 0:
        groups -= 1
    return groups


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        groups = _group_count(out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(inputs))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        inputs = self.up(inputs)
        if inputs.shape[-2:] != skip.shape[-2:]:
            inputs = F.interpolate(
                inputs,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return self.conv(torch.cat([skip, inputs], dim=1))


class CharacterUNet(nn.Module):
    """Generate a complete glyph from complete spatial trajectory maps."""

    def __init__(
        self,
        input_channels: int = 5,
        base_channels: int = 32,
        out_channels: int = 1,
        image_size: int = 128,
        depth: int = 4,
        dropout: float = 0.1,
        use_tanh: bool = False,
    ):
        super().__init__()
        if input_channels < 1:
            raise ValueError("input_channels must be positive")
        if depth < 1:
            raise ValueError("depth must be positive")
        if image_size % (2 ** depth) != 0:
            raise ValueError("image_size must be divisible by 2 ** depth")

        self.input_channels = input_channels
        self.image_size = image_size
        widths = [base_channels * (2 ** level) for level in range(depth + 1)]
        self.input_block = ConvBlock(input_channels, widths[0])
        self.down_blocks = nn.ModuleList(
            [DownBlock(widths[index], widths[index + 1]) for index in range(depth)]
        )
        self.bottleneck_dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.up_blocks = nn.ModuleList(
            [
                UpBlock(widths[index], widths[index - 1], widths[index - 1])
                for index in range(depth, 0, -1)
            ]
        )
        self.output_layer = nn.Conv2d(widths[0], out_channels, kernel_size=1)
        self.output_activation = nn.Tanh() if use_tanh else nn.Sigmoid()
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, trajectory_maps: torch.Tensor) -> torch.Tensor:
        if trajectory_maps.ndim != 4:
            raise ValueError(
                f"trajectory_maps must have shape [B,C,H,W], got {tuple(trajectory_maps.shape)}"
            )
        if trajectory_maps.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} input channels, got {trajectory_maps.shape[1]}"
            )

        features = self.input_block(trajectory_maps)
        skips = [features]
        for down in self.down_blocks:
            features = down(features)
            skips.append(features)
        features = self.bottleneck_dropout(features)
        for up, skip in zip(self.up_blocks, reversed(skips[:-1])):
            features = up(features, skip)
        return self.output_activation(self.output_layer(features))


def build_character_generator(
    config: Optional[Dict[str, Any]] = None,
    **overrides,
) -> CharacterUNet:
    """Compatibility factory name; the implementation is a pure U-Net."""
    values = dict(config or {})
    values.update({key: value for key, value in overrides.items() if value is not None})
    return CharacterUNet(**values)


def build_character_unet(
    config: Optional[Dict[str, Any]] = None,
    **overrides,
) -> CharacterUNet:
    return build_character_generator(config, **overrides)
