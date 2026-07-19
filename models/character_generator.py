from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from models.bbsmg import ConvDecoder, MLPEncoder


class CharacterGenerator(nn.Module):
    """Generate one complete character from all of its stroke features at once."""

    def __init__(
        self,
        input_dim: int = 10,
        latent_dim: int = 128,
        base_channels: int = 64,
        out_channels: int = 1,
        image_size: int = 128,
        max_strokes: int = 64,
        transformer_layers: int = 2,
        attention_heads: int = 4,
        dropout: float = 0.1,
        use_tanh: bool = False,
    ):
        super().__init__()
        if latent_dim % attention_heads != 0:
            raise ValueError("latent_dim must be divisible by attention_heads")
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.max_strokes = max_strokes

        self.stroke_encoder = MLPEncoder(input_dim=input_dim, latent_dim=latent_dim)
        self.position_embedding = nn.Embedding(max_strokes, latent_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=attention_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=transformer_layers)
        self.pool = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
        )
        self.decoder = ConvDecoder(
            latent_dim=latent_dim,
            base_channels=base_channels,
            out_channels=out_channels,
            image_size=image_size,
            use_tanh=use_tanh,
        )
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

    def forward(self, stroke_features: torch.Tensor, stroke_mask: torch.Tensor) -> torch.Tensor:
        if stroke_features.ndim != 3:
            raise ValueError(
                f"stroke_features must have shape [B,S,D], got {tuple(stroke_features.shape)}"
            )
        batch_size, stroke_count, input_dim = stroke_features.shape
        if input_dim != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {input_dim}")
        if stroke_count > self.max_strokes:
            raise ValueError(f"Input has {stroke_count} strokes, max_strokes={self.max_strokes}")
        if stroke_mask.shape != (batch_size, stroke_count):
            raise ValueError("stroke_mask must have shape [B,S]")

        mask = stroke_mask.to(device=stroke_features.device, dtype=torch.bool)
        if not torch.all(mask.any(dim=1)):
            raise ValueError("Every character must contain at least one valid stroke")

        tokens = self.stroke_encoder(stroke_features.reshape(-1, input_dim)).reshape(
            batch_size, stroke_count, self.latent_dim
        )
        positions = torch.arange(stroke_count, device=stroke_features.device)
        tokens = tokens + self.position_embedding(positions).unsqueeze(0)
        tokens = self.transformer(tokens, src_key_padding_mask=~mask)

        mask_f = mask.unsqueeze(-1).to(tokens.dtype)
        mean_pool = (tokens * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
        max_pool = tokens.masked_fill(~mask.unsqueeze(-1), torch.finfo(tokens.dtype).min).amax(dim=1)
        character_code = self.pool(torch.cat([mean_pool, max_pool], dim=-1))
        return self.decoder(character_code)


def build_character_generator(config: Optional[Dict[str, Any]] = None, **overrides) -> CharacterGenerator:
    values = dict(config or {})
    values.update({key: value for key, value in overrides.items() if value is not None})
    return CharacterGenerator(**values)
