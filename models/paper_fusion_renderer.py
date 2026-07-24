"""Differentiable Dynamic-Brush + B-BSMG forward renderer.

This module is intentionally separate from the legacy renderer.  Its posture
semantics match the papers: H [mm], alpha [rad], beta [rad].  The trajectory
heading is computed from fixed x/y, while gamma is not an input because the
axisymmetric prototype cannot identify it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.bbsmg import build_bbsmg, normalize_bbsmg_inputs
from models.paper_bbsm import (
    clamp_posture_torch,
    geometry_to_posture_torch,
    posture_to_geometry_torch,
)


@dataclass(frozen=True)
class PaperDynamicConfig:
    width_inertia: float = 0.02
    drag_inertia: float = 0.02
    offset_fraction: float = 0.25
    pixels_per_model_unit: float = 20.0
    inverse_regularization: float = 1e-4
    patch_floor: float = 0.05
    footprint_scale: float = 0.5


def _infer_model_config(state: Dict[str, torch.Tensor]) -> Dict[str, int]:
    first = state["encoder.net.0.weight"]
    latent_key = max(
        (
            key
            for key in state
            if key.startswith("encoder.net.") and key.endswith(".weight")
        ),
        key=lambda key: int(key.split(".")[2]),
    )
    latent_dim = int(state[latent_key].shape[0])
    base_channels = int(state["decoder.fc.weight"].shape[0] // (8 * 8 * 8))
    return {
        "input_dim": int(first.shape[1]),
        "latent_dim": latent_dim,
        "base_channels": base_channels,
    }


class PaperFusionRenderer(nn.Module):
    """Render a complete character while preserving the supplied x/y path."""

    def __init__(
        self,
        bbsmg: nn.Module,
        input_normalization: Dict[str, Any],
        image_size: int = 128,
        dynamic: PaperDynamicConfig | None = None,
        point_batch_size: int = 128,
    ):
        super().__init__()
        if int(input_normalization.get("input_dim", -1)) != 5:
            raise ValueError(
                "PaperFusionRenderer requires a 5D paper_bbsmg_v1 checkpoint"
            )
        self.bbsmg = bbsmg.eval()
        for parameter in self.bbsmg.parameters():
            parameter.requires_grad_(False)
        self.input_normalization = input_normalization
        self.image_size = int(image_size)
        self.dynamic = dynamic or PaperDynamicConfig()
        if not 0.0 <= self.dynamic.patch_floor < 1.0:
            raise ValueError("patch_floor must be in [0,1)")
        if not 0.0 <= self.dynamic.width_inertia <= 1.0:
            raise ValueError("width_inertia must be in [0,1]")
        if not 0.0 <= self.dynamic.drag_inertia <= 1.0:
            raise ValueError("drag_inertia must be in [0,1]")
        if self.dynamic.footprint_scale <= 0.0:
            raise ValueError("footprint_scale must be positive")
        self.point_batch_size = int(point_batch_size)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str | torch.device = "cpu",
        image_size: int = 128,
        dynamic: PaperDynamicConfig | None = None,
        point_batch_size: int = 128,
    ) -> "PaperFusionRenderer":
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state = checkpoint.get("model_state", checkpoint)
        config = checkpoint.get("model_config") or _infer_model_config(state)
        if int(config["input_dim"]) != 5:
            raise ValueError(
                "Incompatible checkpoint: expected paper B-BSMG input_dim=5"
            )
        model = build_bbsmg(
            input_dim=5,
            latent_dim=int(config["latent_dim"]),
            base_channels=int(config["base_channels"]),
            out_channels=1,
            image_size=image_size,
            use_tanh=False,
        )
        model.load_state_dict(state)
        normalization = checkpoint.get("input_normalization")
        if normalization is None:
            raise ValueError("Checkpoint does not contain input_normalization")
        feature_names = normalization.get("feature_names")
        expected_features = [
            "H_mm",
            "alpha_rad",
            "beta_rad",
            "x0_px",
            "y0_px",
        ]
        if (
            checkpoint.get("format") != "paper_bbsmg_v1"
            and feature_names != expected_features
        ):
            raise ValueError(
                "Checkpoint is not marked as paper_bbsmg_v1 and does not "
                "declare the required paper posture features"
            )
        if feature_names is not None and feature_names != expected_features:
            raise ValueError("Checkpoint posture features do not match paper semantics")
        return cls(
            model.to(device),
            normalization,
            image_size=image_size,
            dynamic=dynamic,
            point_batch_size=point_batch_size,
        ).to(device)

    @staticmethod
    def trajectory_heading(
        xy: torch.Tensor, stroke_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        headings = torch.zeros(
            xy.shape[0], dtype=xy.dtype, device=xy.device
        )
        step_lengths = torch.zeros_like(headings)
        for stroke_id in torch.unique_consecutive(stroke_ids):
            indices = torch.nonzero(stroke_ids == stroke_id, as_tuple=False).flatten()
            points = xy[indices]
            if len(indices) == 1:
                continue
            delta = points[1:] - points[:-1]
            angles = torch.atan2(delta[:, 1], delta[:, 0])
            angles = torch.cat([angles[:1], angles], dim=0)
            lengths = torch.linalg.vector_norm(delta, dim=-1)
            lengths = torch.cat([torch.zeros_like(lengths[:1]), lengths], dim=0)
            headings[indices] = angles
            step_lengths[indices] = lengths
        return headings, step_lengths

    def dynamic_posture(
        self,
        posture: torch.Tensor,
        stroke_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Apply paper-style first-order width/drag dynamics."""
        instant = posture_to_geometry_torch(posture)
        result = torch.empty_like(instant)
        kw = float(self.dynamic.width_inertia)
        kd = float(self.dynamic.drag_inertia)
        for stroke_id in torch.unique_consecutive(stroke_ids):
            indices = torch.nonzero(stroke_ids == stroke_id, as_tuple=False).flatten()
            previous_width = instant[indices[0], 2]
            previous_drag = instant[indices[0], 0] + instant[indices[0], 1]
            for index in indices:
                current = instant[index]
                width = previous_width * kw + current[2] * (1.0 - kw)
                drag_target = current[0] + current[1]
                drag = previous_drag * kd + drag_target * (1.0 - kd)
                heel_ratio = current[1] / (drag_target + 1e-8)
                result[index] = torch.stack(
                    [drag * (1.0 - heel_ratio), drag * heel_ratio, width]
                )
                previous_width, previous_drag = width, drag
        virtual = geometry_to_posture_torch(
            result,
            reference=posture,
            regularization=self.dynamic.inverse_regularization,
        )
        return clamp_posture_torch(virtual)

    def _rotate_about(
        self,
        images: torch.Tensor,
        centers_px: torch.Tensor,
        angles: torch.Tensor,
    ) -> torch.Tensor:
        _, _, height, width = images.shape
        yy, xx = torch.meshgrid(
            torch.linspace(0.0, height - 1.0, height, device=images.device),
            torch.linspace(0.0, width - 1.0, width, device=images.device),
            indexing="ij",
        )
        dx = xx[None] - centers_px[:, 0, None, None]
        dy = yy[None] - centers_px[:, 1, None, None]
        cosine = torch.cos(angles)[:, None, None]
        sine = torch.sin(angles)[:, None, None]
        scale = float(self.dynamic.footprint_scale)
        source_x = (
            (cosine * dx + sine * dy) / scale
            + centers_px[:, 0, None, None]
        )
        source_y = (
            (-sine * dx + cosine * dy) / scale
            + centers_px[:, 1, None, None]
        )
        grid = torch.stack(
            [
                2.0 * source_x / max(width - 1, 1) - 1.0,
                2.0 * source_y / max(height - 1, 1) - 1.0,
            ],
            dim=-1,
        )
        return F.grid_sample(
            images,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

    def forward(
        self,
        xy_canvas: torch.Tensor,
        posture: torch.Tensor,
        stroke_ids: torch.Tensor,
    ) -> torch.Tensor:
        if xy_canvas.ndim != 2 or xy_canvas.shape[1] != 2:
            raise ValueError("xy_canvas must have shape [N,2]")
        if posture.shape != (xy_canvas.shape[0], 3):
            raise ValueError("posture must have shape [N,3]")
        if stroke_ids.shape != (xy_canvas.shape[0],):
            raise ValueError("stroke_ids must have shape [N]")
        if len(xy_canvas) == 0:
            return torch.zeros(
                (1, 1, self.image_size, self.image_size),
                dtype=posture.dtype,
                device=posture.device,
            )

        states = self.compute_dynamic_states(xy_canvas, posture, stroke_ids)
        virtual_posture = states["virtual_posture"]
        contact_xy = states["contact_xy"]
        heading = states["heading"]
        raw_params = torch.cat([virtual_posture, contact_xy], dim=-1)
        normalized = normalize_bbsmg_inputs(
            raw_params, self.input_normalization
        )

        transmittance = torch.ones(
            (1, 1, self.image_size, self.image_size),
            dtype=posture.dtype,
            device=posture.device,
        )
        for start in range(0, len(normalized), self.point_batch_size):
            stop = min(start + self.point_batch_size, len(normalized))
            patches = self.bbsmg(normalized[start:stop]).clamp(0.0, 1.0)
            floor = float(self.dynamic.patch_floor)
            if floor > 0.0:
                patches = F.relu(patches - floor) / max(1.0 - floor, 1e-6)
            patches = self._rotate_about(
                patches, contact_xy[start:stop], heading[start:stop]
            )
            chunk_transmittance = torch.prod(
                (1.0 - patches).clamp_min(1e-6), dim=0, keepdim=True
            )
            transmittance = transmittance * chunk_transmittance
        return (1.0 - transmittance).clamp(0.0, 1.0)

    def compute_dynamic_states(
        self,
        xy_canvas: torch.Tensor,
        posture: torch.Tensor,
        stroke_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Expose the paper dynamic state bridge for calibration diagnostics."""
        heading, step_length = self.trajectory_heading(xy_canvas, stroke_ids)
        virtual_posture = self.dynamic_posture(posture, stroke_ids)
        geometry = posture_to_geometry_torch(virtual_posture)
        free_offset = self.dynamic.offset_fraction * (
            geometry[:, 0] + geometry[:, 1]
        )
        effective_scale = (
            float(self.dynamic.pixels_per_model_unit)
            * float(self.dynamic.footprint_scale)
        )
        held_offset = step_length / effective_scale
        offset = torch.minimum(free_offset, held_offset)
        direction = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
        contact_xy = (
            xy_canvas
            - offset[:, None]
            * effective_scale
            * direction
        )
        return {
            "heading": heading,
            "step_length_px": step_length,
            "virtual_posture": virtual_posture,
            "geometry": geometry,
            "offset_model_unit": offset,
            "contact_xy": contact_xy,
        }
