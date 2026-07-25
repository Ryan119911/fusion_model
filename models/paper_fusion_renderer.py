"""Differentiable Dynamic-Brush + B-BSMG forward renderer.

This module is intentionally separate from the legacy renderer.  Its posture
semantics match the papers: H [mm], alpha [rad], beta [rad].  The trajectory
heading is computed from fixed x/y, while gamma is not an input because the
axisymmetric prototype cannot identify it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.bbsmg import build_bbsmg, normalize_bbsmg_inputs
from models.paper_bbsm import (
    PAPER_ANGLE_BASES,
    PAPER_ANGLE_BASIS_DEGREE_FITTED,
    PAPER_ANGLE_BASIS_RADIAN,
    clamp_posture_torch,
    geometry_to_posture_torch,
    posture_to_geometry_torch,
)
from models.paper_calibration import (
    DYNAMIC_PROFILES,
    LEGACY_OFFSET_PROFILE,
    WANG2020_PROFILE,
    wang2020_dimension_progress_torch,
    wang2020_offset_drag_ratio_torch,
)


@dataclass(frozen=True)
class PaperDynamicConfig:
    width_inertia: float = 0.02
    drag_inertia: float = 0.02
    calibration_profile: str = WANG2020_PROFILE
    offset_fraction: float = 0.25
    pixels_per_model_unit: float = 20.0
    inverse_regularization: float = 1e-4
    patch_floor: float = 0.05
    footprint_scale: float = 0.22
    render_max_step_px: float = 2.0


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
                "PaperFusionRenderer requires a 5D paper B-BSMG checkpoint"
            )
        self.bbsmg = bbsmg.eval()
        for parameter in self.bbsmg.parameters():
            parameter.requires_grad_(False)
        self.input_normalization = input_normalization
        self.regression_angle_basis = input_normalization.get(
            "regression_angle_basis", PAPER_ANGLE_BASIS_RADIAN
        )
        if self.regression_angle_basis not in PAPER_ANGLE_BASES:
            raise ValueError(
                "Checkpoint has an unsupported regression_angle_basis: "
                f"{self.regression_angle_basis!r}"
            )
        self.image_size = int(image_size)
        self.dynamic = dynamic or PaperDynamicConfig()
        if not 0.0 <= self.dynamic.patch_floor < 1.0:
            raise ValueError("patch_floor must be in [0,1)")
        if not 0.0 <= self.dynamic.width_inertia <= 1.0:
            raise ValueError("width_inertia must be in [0,1]")
        if not 0.0 <= self.dynamic.drag_inertia <= 1.0:
            raise ValueError("drag_inertia must be in [0,1]")
        if self.dynamic.calibration_profile not in DYNAMIC_PROFILES:
            raise ValueError(
                "calibration_profile must be one of "
                f"{DYNAMIC_PROFILES}, got {self.dynamic.calibration_profile!r}"
            )
        if self.dynamic.offset_fraction < 0.0:
            raise ValueError("offset_fraction must be non-negative")
        if self.dynamic.footprint_scale <= 0.0:
            raise ValueError("footprint_scale must be positive")
        if self.dynamic.render_max_step_px <= 0.0:
            raise ValueError("render_max_step_px must be positive")
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
        checkpoint_format = checkpoint.get("format")
        regression_angle_basis = normalization.get(
            "regression_angle_basis", PAPER_ANGLE_BASIS_RADIAN
        )
        expected_formats = {
            PAPER_ANGLE_BASIS_RADIAN: "paper_bbsmg_v1",
            PAPER_ANGLE_BASIS_DEGREE_FITTED: "paper_bbsmg_degree_fitted_v2",
        }
        if regression_angle_basis not in expected_formats:
            raise ValueError(
                "Checkpoint declares an unsupported regression angle basis: "
                f"{regression_angle_basis!r}"
            )
        if (
            checkpoint_format != expected_formats[regression_angle_basis]
            and feature_names != expected_features
        ):
            raise ValueError(
                "Checkpoint format and paper posture features are incompatible"
            )
        allowed_formats = {expected_formats[regression_angle_basis]}
        if regression_angle_basis == PAPER_ANGLE_BASIS_RADIAN:
            allowed_formats.add(None)
        if checkpoint_format not in allowed_formats:
            raise ValueError(
                "Checkpoint format does not match regression_angle_basis: "
                f"format={checkpoint_format!r}, "
                f"basis={regression_angle_basis!r}"
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
        """Apply Wang Eq. (6)/(7) dynamics to B-BSM footprint geometry."""
        result = self._dynamic_geometry(posture, stroke_ids)
        virtual = geometry_to_posture_torch(
            result,
            reference=posture,
            regularization=self.dynamic.inverse_regularization,
            angle_basis=self.regression_angle_basis,
        )
        return clamp_posture_torch(virtual)

    def _dynamic_geometry(
        self,
        posture: torch.Tensor,
        stroke_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return dynamic ``(Lt,Lh,Lr)`` while carrying state across strokes.

        Wang et al. condition a stroke on the final brush state of the previous
        stroke.  Width and drag therefore remain stateful across the complete
        character.  The first sample starts from zero deformation, and the
        reported K=0.02 makes the first update almost instantaneous.
        """
        instant = posture_to_geometry_torch(
            posture, angle_basis=self.regression_angle_basis
        )
        if self.dynamic.calibration_profile == WANG2020_PROFILE:
            zero_angle_posture = torch.stack(
                [
                    posture[:, 0],
                    torch.zeros_like(posture[:, 1]),
                    torch.zeros_like(posture[:, 2]),
                ],
                dim=-1,
            )
            zero_angle_geometry = posture_to_geometry_torch(
                zero_angle_posture,
                angle_basis=self.regression_angle_basis,
            )
            endpoints = torch.as_tensor(
                [[11.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
                dtype=posture.dtype,
                device=posture.device,
            )
            endpoint_geometry = posture_to_geometry_torch(
                endpoints, angle_basis=self.regression_angle_basis
            )
            progress = wang2020_dimension_progress_torch(posture[:, 0])
            endpoint_drag = endpoint_geometry[:, 0] + endpoint_geometry[:, 1]
            mapped_drag = endpoint_drag[0] + progress["drag_progress"] * (
                endpoint_drag[1] - endpoint_drag[0]
            )
            mapped_half_width = endpoint_geometry[0, 2] + progress[
                "width_progress"
            ] * (endpoint_geometry[1, 2] - endpoint_geometry[0, 2])
            angle_delta = instant - zero_angle_geometry
            drag_targets = (
                mapped_drag + angle_delta[:, 0] + angle_delta[:, 1]
            ).clamp_min(1e-6)
            half_width_targets = (
                mapped_half_width + angle_delta[:, 2]
            ).clamp_min(1e-6)
        else:
            drag_targets = instant[:, 0] + instant[:, 1]
            half_width_targets = instant[:, 2]
        kw = float(self.dynamic.width_inertia)
        kd = float(self.dynamic.drag_inertia)
        previous_width = torch.zeros_like(instant[0, 2])
        previous_drag = torch.zeros_like(instant[0, 0])
        result = []
        for index in range(len(instant)):
            current = instant[index]
            # B-BSM Lr is the half-width; Wang w is the full mark width.
            width_target = 2.0 * half_width_targets[index]
            drag_target = drag_targets[index]
            width = previous_width * kw + width_target * (1.0 - kw)
            drag = previous_drag * kd + drag_target * (1.0 - kd)
            instant_drag = current[0] + current[1]
            heel_ratio = current[1] / (instant_drag + 1e-8)
            result.append(
                torch.stack(
                    [
                        drag * (1.0 - heel_ratio),
                        drag * heel_ratio,
                        0.5 * width,
                    ]
                )
            )
            previous_width, previous_drag = width, drag
        return torch.stack(result, dim=0)

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

        render_xy, render_posture, render_stroke_ids = (
            self.densify_for_rendering(xy_canvas, posture, stroke_ids)
        )
        states = self.compute_dynamic_states(
            render_xy, render_posture, render_stroke_ids
        )
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

    def densify_for_rendering(
        self,
        xy_canvas: torch.Tensor,
        posture: torch.Tensor,
        stroke_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Continuously sweep the footprint without changing exported x/y.

        Segment counts depend only on the fixed x/y path. Posture remains
        differentiable because every inserted pose is a linear interpolation
        of the two original endpoint poses.
        """
        xy_parts = []
        posture_parts = []
        id_parts = []
        max_step = float(self.dynamic.render_max_step_px)
        for stroke_id in torch.unique_consecutive(stroke_ids):
            indices = torch.nonzero(
                stroke_ids == stroke_id, as_tuple=False
            ).flatten()
            points = xy_canvas[indices]
            poses = posture[indices]
            if len(indices) == 0:
                continue
            xy_parts.append(points[:1])
            posture_parts.append(poses[:1])
            id_parts.append(stroke_ids[indices[:1]])
            for point_index in range(len(indices) - 1):
                distance = float(
                    torch.linalg.vector_norm(
                        points[point_index + 1] - points[point_index]
                    )
                    .detach()
                    .cpu()
                )
                steps = max(int(np.ceil(distance / max_step)), 1)
                t = torch.arange(
                    1,
                    steps + 1,
                    dtype=xy_canvas.dtype,
                    device=xy_canvas.device,
                ) / float(steps)
                xy_parts.append(
                    torch.lerp(
                        points[point_index][None],
                        points[point_index + 1][None],
                        t[:, None],
                    )
                )
                posture_parts.append(
                    torch.lerp(
                        poses[point_index][None],
                        poses[point_index + 1][None],
                        t[:, None],
                    )
                )
                id_parts.append(
                    torch.full(
                        (steps,),
                        int(stroke_id.item()),
                        dtype=stroke_ids.dtype,
                        device=stroke_ids.device,
                    )
                )
        return (
            torch.cat(xy_parts, dim=0),
            torch.cat(posture_parts, dim=0),
            torch.cat(id_parts, dim=0),
        )

    def compute_dynamic_states(
        self,
        xy_canvas: torch.Tensor,
        posture: torch.Tensor,
        stroke_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Apply Wang Eq. (1), (6)-(9) and expose every dynamic state.

        The contact root remains fixed while the handle displacement is below
        the free offset and snaps to the free-offset boundary afterwards.
        At a stroke transition the previous deformation/orientation is carried
        over, but the root is rebased at the new handle location because the
        unobserved pen-up motion is not part of the input trajectory.
        """
        tangent_heading, step_length = self.trajectory_heading(
            xy_canvas, stroke_ids
        )
        geometry = self._dynamic_geometry(posture, stroke_ids)
        virtual_posture = geometry_to_posture_torch(
            geometry,
            reference=posture,
            regularization=self.dynamic.inverse_regularization,
            angle_basis=self.regression_angle_basis,
        )
        virtual_posture = clamp_posture_torch(virtual_posture)
        dynamic_drag = geometry[:, 0] + geometry[:, 1]
        if self.dynamic.calibration_profile == WANG2020_PROFILE:
            offset_ratio = wang2020_offset_drag_ratio_torch(posture[:, 0])
            free_offset = offset_ratio * dynamic_drag
        elif self.dynamic.calibration_profile == LEGACY_OFFSET_PROFILE:
            offset_ratio = torch.full_like(
                dynamic_drag, float(self.dynamic.offset_fraction)
            )
            free_offset = offset_ratio * dynamic_drag
        else:  # Constructor validation keeps this branch unreachable.
            raise RuntimeError(
                f"Unsupported dynamic profile {self.dynamic.calibration_profile!r}"
            )
        effective_scale = (
            float(self.dynamic.pixels_per_model_unit)
            * float(self.dynamic.footprint_scale)
        )
        offsets = []
        held_offsets = []
        headings = []
        roots = []
        previous_offset = torch.zeros_like(free_offset[0])
        previous_heading = tangent_heading[0]
        previous_root = xy_canvas[0]
        previous_stroke = stroke_ids[0]
        eps = torch.finfo(xy_canvas.dtype).eps
        for index in range(len(xy_canvas)):
            is_first = index == 0
            is_new_stroke = (
                not is_first
                and bool((stroke_ids[index] != previous_stroke).detach().cpu())
            )
            if is_first or is_new_stroke:
                if is_new_stroke:
                    previous_root = (
                        xy_canvas[index]
                        - previous_offset
                        * effective_scale
                        * torch.stack(
                            [
                                torch.cos(previous_heading),
                                torch.sin(previous_heading),
                            ]
                        )
                    )
                held = previous_offset
                current_offset = torch.minimum(free_offset[index], held)
                current_heading = previous_heading
            else:
                root_to_handle = xy_canvas[index] - previous_root
                held = (
                    torch.linalg.vector_norm(root_to_handle) / effective_scale
                )
                current_offset = torch.minimum(free_offset[index], held)
                candidate_heading = torch.atan2(
                    root_to_handle[1], root_to_handle[0]
                )
                current_heading = torch.where(
                    held > eps, candidate_heading, previous_heading
                )
            direction = torch.stack(
                [torch.cos(current_heading), torch.sin(current_heading)]
            )
            current_root = (
                xy_canvas[index]
                - current_offset * effective_scale * direction
            )
            offsets.append(current_offset)
            held_offsets.append(held)
            headings.append(current_heading)
            roots.append(current_root)
            previous_offset = current_offset
            previous_heading = current_heading
            previous_root = current_root
            previous_stroke = stroke_ids[index]
        offset = torch.stack(offsets)
        held_offset = torch.stack(held_offsets)
        heading = torch.stack(headings)
        contact_xy = torch.stack(roots)
        return {
            "heading": heading,
            "trajectory_heading": tangent_heading,
            "step_length_px": step_length,
            "virtual_posture": virtual_posture,
            "geometry": geometry,
            "offset_ratio": offset_ratio,
            "free_offset_model_unit": free_offset,
            "held_offset_model_unit": held_offset,
            "offset_model_unit": offset,
            "contact_xy": contact_xy,
        }
