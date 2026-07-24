from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


CHARACTER_CHECKPOINT_FORMAT = "character_unet_v5"


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
        input_channels: int = 6,
        base_channels: int = 32,
        out_channels: int = 1,
        image_size: int = 128,
        depth: int = 4,
        dropout: float = 0.1,
        use_tanh: bool = False,
        prior_strength: float = 0.75,
        prior_channel: int = 1,
        prior_threshold: float = 0.70,
        prior_sharpness: float = 10.0,
    ):
        super().__init__()
        if input_channels < 1:
            raise ValueError("input_channels must be positive")
        if depth < 1:
            raise ValueError("depth must be positive")
        if image_size % (2 ** depth) != 0:
            raise ValueError("image_size must be divisible by 2 ** depth")
        if prior_channel < 0 or prior_channel >= input_channels:
            raise ValueError("prior_channel must index an input trajectory channel")
        if prior_strength < 0:
            raise ValueError("prior_strength must be non-negative")
        if not 0.0 < prior_threshold < 1.0:
            raise ValueError("prior_threshold must satisfy 0 < value < 1")
        if prior_sharpness <= 0:
            raise ValueError("prior_sharpness must be positive")

        self.input_channels = input_channels
        self.image_size = image_size
        self.use_tanh = use_tanh
        self.prior_channel = int(prior_channel)
        self.prior_threshold = float(prior_threshold)
        self.prior_sharpness = float(prior_sharpness)
        # Unlike the old fixed smooth proximity bias, v5 can reduce this
        # gate when the structure supervision demands a narrower stroke.
        initial_gain = max(float(prior_strength), 1e-4)
        raw_gain = torch.log(torch.expm1(torch.tensor(initial_gain)))
        self.prior_gain_raw = nn.Parameter(raw_gain)
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
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Begin with the explicit trajectory prior instead of a random gray
        # field. Gradients still train this head normally from the first step.
        nn.init.zeros_(self.output_layer.weight)
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)

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
        logits = self.output_layer(features)
        if self.use_tanh:
            return torch.tanh(logits)

        # Initialize from a narrow trajectory neighborhood, but keep its gain
        # learnable. The threshold is intentionally above 0.5 so the prior no
        # longer hard-codes the wide gray ribbon observed with v3.
        proximity = trajectory_maps[
            :, self.prior_channel : self.prior_channel + 1
        ].clamp(0.0, 1.0)
        prior_gain = F.softplus(self.prior_gain_raw)
        prior_logits = self.prior_sharpness * (
            proximity - self.prior_threshold
        )
        logits = logits + prior_gain * prior_logits
        return torch.sigmoid(logits)


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
