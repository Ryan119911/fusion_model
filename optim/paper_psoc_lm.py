"""PSOC/CGL parameterization and autograd LM for paper-pose inversion."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from models.paper_bbsm import PAPER_POSTURE_MAX, PAPER_POSTURE_MIN
from models.paper_fusion_renderer import PaperFusionRenderer
from optim.chebyshev import barycentric_weights, cgl_nodes, normalize_time_grid


def cgl_interpolation_matrix(order: int, num_samples: int) -> np.ndarray:
    """Linear matrix mapping ascending CGL node values to sample values."""
    if order < 1:
        raise ValueError("order must be >= 1")
    if num_samples < 1:
        raise ValueError("num_samples must be >= 1")
    nodes = cgl_nodes(order)[::-1]
    weights = barycentric_weights(order)[::-1]
    times = normalize_time_grid(num_samples)
    matrix = np.zeros((num_samples, order + 1), dtype=np.float32)
    for row, value in enumerate(times):
        difference = value - nodes
        exact = np.flatnonzero(np.abs(difference) < 1e-12)
        if len(exact):
            matrix[row, int(exact[0])] = 1.0
        else:
            terms = weights / difference
            matrix[row] = (terms / terms.sum()).astype(np.float32)
    return matrix


def trajectory_difference_residuals(
    values: torch.Tensor,
    point_indices: Sequence[np.ndarray],
    first_difference_weight: float,
    second_difference_weight: float,
) -> List[torch.Tensor]:
    """Build within-stroke point-space continuity residuals.

    ``values`` is a one-dimensional decoded trajectory quantity. Differences
    never cross a stroke boundary. The weights operate on the normalized
    quantity rather than physical units so they remain independent of the H
    range used by the prototype.
    """
    residuals: List[torch.Tensor] = []
    for indices in point_indices:
        stroke_values = values[
            torch.as_tensor(indices, dtype=torch.long, device=values.device)
        ]
        if first_difference_weight > 0 and len(indices) >= 2:
            residuals.append(
                first_difference_weight**0.5
                * (stroke_values[1:] - stroke_values[:-1])
            )
        if second_difference_weight > 0 and len(indices) >= 3:
            residuals.append(
                second_difference_weight**0.5
                * (
                    stroke_values[2:]
                    - 2.0 * stroke_values[1:-1]
                    + stroke_values[:-2]
                )
            )
    return residuals


@dataclass
class PaperLMResult:
    xy_canvas: np.ndarray
    posture: np.ndarray
    rendered_image: np.ndarray
    success: bool
    steps: int
    initial_cost: float
    final_cost: float
    message: str
    history: Dict[str, List[float]] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class PaperPSOCLM:
    """Optimize bounded per-stroke CGL pose and optional planar offsets."""

    FIELD_NAMES = ("H", "alpha", "beta")

    def __init__(
        self,
        renderer: PaperFusionRenderer,
        order: int = 3,
        optimization_size: int = 16,
        smoothness_weights: Sequence[float] = (0.02, 0.10, 0.10),
        posture_prior_weights: Sequence[float] = (0.001, 0.05, 0.05),
        render_stride: int = 1,
        jacobian_mode: str = "finite_difference",
        finite_difference_eps: float = 1e-2,
        field_mode: str = "auto",
        min_relative_median_sensitivity: float = 0.45,
        terminal_lift_weight: float = 0.0,
        terminal_lift_nodes: int = 1,
        optimize_xy: bool = False,
        xy_max_offset_px: float = 6.0,
        xy_smoothness_weight: float = 0.10,
        xy_prior_weight: float = 0.05,
        h_point_velocity_weight: float = 0.0,
        h_point_acceleration_weight: float = 0.0,
        cap_order_to_points: bool = False,
    ):
        if order < 1:
            raise ValueError("order must be >= 1")
        if optimization_size < 8:
            raise ValueError("optimization_size must be >= 8")
        self.renderer = renderer
        self.order = int(order)
        self.optimization_size = int(optimization_size)
        self.smoothness_weights = self._validate_field_weights(
            smoothness_weights, "smoothness_weights"
        )
        self.posture_prior_weights = self._validate_field_weights(
            posture_prior_weights, "posture_prior_weights"
        )
        self.render_stride = max(int(render_stride), 1)
        if jacobian_mode not in {"finite_difference", "autograd"}:
            raise ValueError(
                "jacobian_mode must be 'finite_difference' or 'autograd'"
            )
        if finite_difference_eps <= 0:
            raise ValueError("finite_difference_eps must be positive")
        if field_mode not in {"auto", "all", "h_only", "xy_only"}:
            raise ValueError(
                "field_mode must be 'auto', 'all', 'h_only', or 'xy_only'"
            )
        if not 0.0 <= min_relative_median_sensitivity <= 1.0:
            raise ValueError(
                "min_relative_median_sensitivity must be in [0,1]"
            )
        self.jacobian_mode = jacobian_mode
        self.finite_difference_eps = float(finite_difference_eps)
        self.field_mode = field_mode
        self.min_relative_median_sensitivity = float(
            min_relative_median_sensitivity
        )
        if terminal_lift_weight < 0:
            raise ValueError("terminal_lift_weight must be non-negative")
        if terminal_lift_nodes < 1:
            raise ValueError("terminal_lift_nodes must be >= 1")
        self.terminal_lift_weight = float(terminal_lift_weight)
        self.terminal_lift_nodes = int(terminal_lift_nodes)
        if xy_max_offset_px <= 0:
            raise ValueError("xy_max_offset_px must be positive")
        if xy_smoothness_weight < 0 or xy_prior_weight < 0:
            raise ValueError("x/y regularization weights must be non-negative")
        self.optimize_xy = bool(optimize_xy)
        self.xy_max_offset_px = float(xy_max_offset_px)
        self.xy_smoothness_weight = float(xy_smoothness_weight)
        self.xy_prior_weight = float(xy_prior_weight)
        if h_point_velocity_weight < 0 or h_point_acceleration_weight < 0:
            raise ValueError(
                "H point-space regularization weights must be non-negative"
            )
        self.h_point_velocity_weight = float(h_point_velocity_weight)
        self.h_point_acceleration_weight = float(
            h_point_acceleration_weight
        )
        self.cap_order_to_points = bool(cap_order_to_points)
        if self.field_mode == "xy_only" and not self.optimize_xy:
            raise ValueError("field_mode='xy_only' requires optimize_xy=True")

    @staticmethod
    def _validate_field_weights(
        values: Sequence[float], name: str
    ) -> np.ndarray:
        weights = np.asarray(values, dtype=np.float32)
        if weights.shape != (3,):
            raise ValueError(f"{name} must contain H/alpha/beta weights")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError(f"{name} must be finite and non-negative")
        return weights

    @property
    def device(self) -> torch.device:
        return next(self.renderer.bbsmg.parameters()).device

    def _build_layout(
        self, stroke_ids: np.ndarray
    ) -> tuple[
        List[torch.Tensor],
        List[np.ndarray],
        List[int],
        np.ndarray,
    ]:
        matrices: List[torch.Tensor] = []
        point_indices: List[np.ndarray] = []
        effective_orders: List[int] = []
        active_node_masks: List[np.ndarray] = []
        node_count = self.order + 1
        for stroke_id in np.unique(stroke_ids):
            indices = np.flatnonzero(stroke_ids == stroke_id)
            point_indices.append(indices)
            effective_order = self.order
            if self.cap_order_to_points:
                effective_order = min(self.order, max(len(indices) - 1, 0))
            if effective_order == 0:
                compact_matrix = np.ones((len(indices), 1), dtype=np.float32)
            else:
                compact_matrix = cgl_interpolation_matrix(
                    effective_order, len(indices)
                )
            matrix = np.zeros(
                (len(indices), node_count), dtype=np.float32
            )
            matrix[:, : effective_order + 1] = compact_matrix
            active_mask = np.zeros(node_count, dtype=bool)
            active_mask[: effective_order + 1] = True
            effective_orders.append(effective_order)
            active_node_masks.append(active_mask)
            matrices.append(torch.as_tensor(matrix, device=self.device))
        return (
            matrices,
            point_indices,
            effective_orders,
            np.stack(active_node_masks, axis=0),
        )

    def _decode(
        self,
        decision: torch.Tensor,
        matrices: Sequence[torch.Tensor],
        point_indices: Sequence[np.ndarray],
        point_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_count = self.order + 1
        lower = torch.as_tensor(
            PAPER_POSTURE_MIN, dtype=decision.dtype, device=decision.device
        )
        upper = torch.as_tensor(
            PAPER_POSTURE_MAX, dtype=decision.dtype, device=decision.device
        )
        node_logits = decision.view(len(matrices), 3, node_count)
        normalized_nodes = torch.sigmoid(node_logits)
        normalized_points = torch.empty(
            (point_count, 3), dtype=decision.dtype, device=decision.device
        )
        for stroke_index, (matrix, indices) in enumerate(
            zip(matrices, point_indices)
        ):
            # Interpolating already-bounded node values can overshoot between
            # CGL points. Interpolate logits first, then map every trajectory
            # point through sigmoid so the physical limits are guaranteed.
            point_logits = (
                matrix.to(dtype=decision.dtype) @ node_logits[stroke_index].T
            )
            values = torch.sigmoid(point_logits)
            normalized_points[
                torch.as_tensor(indices, device=decision.device)
            ] = values
        posture = lower + normalized_points * (upper - lower)
        return posture, normalized_nodes

    def _decode_xy_offsets(
        self,
        decision: torch.Tensor,
        matrices: Sequence[torch.Tensor],
        point_indices: Sequence[np.ndarray],
        point_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode bounded canvas offsets without polynomial overshoot."""
        node_count = self.order + 1
        node_logits = decision.view(len(matrices), 2, node_count)
        normalized_nodes = torch.tanh(node_logits)
        normalized_points = torch.empty(
            (point_count, 2), dtype=decision.dtype, device=decision.device
        )
        for stroke_index, (matrix, indices) in enumerate(
            zip(matrices, point_indices)
        ):
            point_logits = (
                matrix.to(dtype=decision.dtype) @ node_logits[stroke_index].T
            )
            normalized_points[
                torch.as_tensor(indices, device=decision.device)
            ] = torch.tanh(point_logits)
        return (
            normalized_points * self.xy_max_offset_px,
            normalized_nodes,
        )

    def _render_indices(self, point_indices: Sequence[np.ndarray]) -> np.ndarray:
        selected: List[int] = []
        for indices in point_indices:
            chosen = indices[:: self.render_stride].tolist()
            if len(indices) and int(indices[-1]) not in chosen:
                chosen.append(int(indices[-1]))
            selected.extend(chosen)
        return np.asarray(selected, dtype=np.int64)

    def optimize(
        self,
        xy_canvas: np.ndarray,
        stroke_ids: np.ndarray,
        target_image: np.ndarray,
        initial_h_mm: float = 15.5,
        initial_alpha_rad: float = 0.0,
        initial_beta_rad: float = 0.0,
        damping: float = 0.05,
        max_steps: int = 15,
        pixel_weight: float = 3.0,
        initial_posture: np.ndarray | None = None,
    ) -> PaperLMResult:
        xy = torch.as_tensor(
            xy_canvas, dtype=torch.float32, device=self.device
        )
        ids = torch.as_tensor(
            stroke_ids, dtype=torch.long, device=self.device
        )
        target = torch.as_tensor(
            target_image, dtype=torch.float32, device=self.device
        ).view(1, 1, target_image.shape[-2], target_image.shape[-1])
        target_small = F.interpolate(
            target,
            size=(self.optimization_size, self.optimization_size),
            mode="bilinear",
            align_corners=False,
        )
        (
            matrices,
            point_indices,
            effective_orders,
            active_node_mask_np,
        ) = self._build_layout(stroke_ids)
        render_indices_np = self._render_indices(point_indices)
        render_indices = torch.as_tensor(
            render_indices_np, dtype=torch.long, device=self.device
        )

        default_initial = np.asarray(
            [initial_h_mm, initial_alpha_rad, initial_beta_rad],
            dtype=np.float32,
        )
        if initial_posture is None:
            initial_points = np.tile(default_initial, (len(xy), 1))
            initial_posture_source = "command_line_defaults"
        else:
            initial_points = np.asarray(initial_posture, dtype=np.float32)
            if initial_points.shape != (len(xy), 3):
                raise ValueError(
                    "initial_posture must have shape [trajectory_points, 3]"
                )
            initial_posture_source = "initial_pose_csv"
        if np.any(initial_points < PAPER_POSTURE_MIN) or np.any(
            initial_points > PAPER_POSTURE_MAX
        ):
            raise ValueError(
                "Initial posture is outside H=11-20 mm, alpha=0-10 deg, "
                "beta=0-5 deg"
            )
        normalized_initial_points = (
            initial_points - PAPER_POSTURE_MIN
        ) / (
            PAPER_POSTURE_MAX - PAPER_POSTURE_MIN
        )
        # A value exactly on a bound has zero useful logistic derivative.
        normalized_for_audit = np.clip(
            normalized_initial_points, 0.02, 0.98
        )
        point_logits = np.log(
            normalized_for_audit / (1.0 - normalized_for_audit)
        )
        initial_node_logits = np.zeros(
            (len(matrices), 3, self.order + 1), dtype=np.float32
        )
        for stroke_index, (matrix, indices, effective_order) in enumerate(
            zip(matrices, point_indices, effective_orders)
        ):
            compact_matrix = (
                matrix[:, : effective_order + 1].detach().cpu().numpy()
            )
            fitted, _, _, _ = np.linalg.lstsq(
                compact_matrix,
                point_logits[indices],
                rcond=None,
            )
            initial_node_logits[
                stroke_index, :, : effective_order + 1
            ] = fitted.T
        posture_decision = torch.as_tensor(
            initial_node_logits.reshape(-1),
            dtype=torch.float32,
            device=self.device,
        )
        posture_decision_count = int(posture_decision.numel())
        xy_decision_count = (
            len(matrices) * 2 * (self.order + 1)
            if self.optimize_xy
            else 0
        )
        if xy_decision_count:
            decision = torch.cat(
                [
                    posture_decision,
                    torch.zeros(
                        xy_decision_count,
                        dtype=posture_decision.dtype,
                        device=posture_decision.device,
                    ),
                ]
            )
        else:
            decision = posture_decision
        initial_decision = decision.detach().clone()
        prior = torch.as_tensor(
            1.0
            / (
                1.0
                + np.exp(-np.clip(initial_node_logits, -30.0, 30.0))
            ),
            dtype=decision.dtype,
            device=decision.device,
        )
        initial_points_tensor = torch.as_tensor(
            initial_points, dtype=decision.dtype, device=decision.device
        )
        node_count = self.order + 1
        layout = np.arange(posture_decision_count, dtype=np.int64).reshape(
            len(matrices), 3, node_count
        )
        field_columns = {
            field_name: torch.as_tensor(
                layout[:, field_index, :][active_node_mask_np],
                dtype=torch.long,
                device=decision.device,
            )
            for field_index, field_name in enumerate(self.FIELD_NAMES)
        }
        all_posture_columns = torch.cat(
            [field_columns[name] for name in self.FIELD_NAMES]
        )
        if xy_decision_count:
            xy_layout = np.arange(
                posture_decision_count,
                posture_decision_count + xy_decision_count,
                dtype=np.int64,
            ).reshape(len(matrices), 2, node_count)
            xy_mask = np.broadcast_to(
                active_node_mask_np[:, None, :], xy_layout.shape
            )
            xy_columns = torch.as_tensor(
                xy_layout[xy_mask],
                dtype=torch.long,
                device=decision.device,
            )
        else:
            xy_columns = torch.empty(
                0, dtype=torch.long, device=decision.device
            )
        active_node_mask = torch.as_tensor(
            active_node_mask_np,
            dtype=decision.dtype,
            device=decision.device,
        )
        active_node_pair_mask = (
            active_node_mask[:, 1:] * active_node_mask[:, :-1]
        )
        smoothness_weights = torch.as_tensor(
            self.smoothness_weights,
            dtype=decision.dtype,
            device=decision.device,
        ).view(1, 3, 1)
        posture_prior_weights = torch.as_tensor(
            self.posture_prior_weights,
            dtype=decision.dtype,
            device=decision.device,
        ).view(1, 3, 1)

        def residual_fn(vector: torch.Tensor) -> torch.Tensor:
            posture, nodes = self._decode(
                vector[:posture_decision_count],
                matrices,
                point_indices,
                len(xy),
            )
            rendered_xy = xy
            xy_nodes = None
            if self.optimize_xy:
                xy_offsets, xy_nodes = self._decode_xy_offsets(
                    vector[posture_decision_count:],
                    matrices,
                    point_indices,
                    len(xy),
                )
                rendered_xy = xy + xy_offsets
            rendered = self.renderer(
                rendered_xy[render_indices],
                posture[render_indices],
                ids[render_indices],
            )
            rendered_small = F.interpolate(
                rendered,
                size=(self.optimization_size, self.optimization_size),
                mode="bilinear",
                align_corners=False,
            )
            weights = 1.0 + float(pixel_weight) * target_small
            residuals = [
                ((rendered_small - target_small) * torch.sqrt(weights)).flatten()
            ]
            if bool(torch.any(smoothness_weights > 0)):
                residuals.append(
                    (
                        torch.sqrt(smoothness_weights)
                        * (nodes[:, :, 1:] - nodes[:, :, :-1])
                        * active_node_pair_mask[:, None, :]
                    ).flatten()
                )
            if bool(torch.any(posture_prior_weights > 0)):
                residuals.append(
                    (
                        torch.sqrt(posture_prior_weights) * (nodes - prior)
                        * active_node_mask[:, None, :]
                    ).flatten()
                )
            if xy_nodes is not None and self.xy_smoothness_weight > 0:
                residuals.append(
                    (
                        self.xy_smoothness_weight**0.5
                        * (xy_nodes[:, :, 1:] - xy_nodes[:, :, :-1])
                        * active_node_pair_mask[:, None, :]
                    ).flatten()
                )
            if xy_nodes is not None and self.xy_prior_weight > 0:
                residuals.append(
                    (
                        self.xy_prior_weight**0.5
                        * xy_nodes
                        * active_node_mask[:, None, :]
                    ).flatten()
                )
            normalized_h_points = (
                posture[:, 0] - float(PAPER_POSTURE_MIN[0])
            ) / float(PAPER_POSTURE_MAX[0] - PAPER_POSTURE_MIN[0])
            residuals.extend(
                trajectory_difference_residuals(
                    normalized_h_points,
                    point_indices,
                    self.h_point_velocity_weight,
                    self.h_point_acceleration_weight,
                )
            )
            if self.terminal_lift_weight > 0:
                # Wang Eq. (19) penalizes terminal z so the brush tends to
                # lift.  In this bridge, normalized H=0 is H_min=11 mm.
                terminal_h = torch.cat(
                    [
                        nodes[
                            stroke_index,
                            0,
                            max(
                                effective_order
                                + 1
                                - self.terminal_lift_nodes,
                                0,
                            ) : effective_order + 1,
                        ]
                        for stroke_index, effective_order in enumerate(
                            effective_orders
                        )
                    ]
                )
                residuals.append(
                    (
                        self.terminal_lift_weight**0.5 * terminal_h
                    ).flatten()
                )
            return torch.cat(residuals)

        def evaluate_cost(vector: torch.Tensor) -> float:
            with torch.no_grad():
                residual = residual_fn(vector)
                return 0.5 * float(torch.dot(residual, residual).item())

        def evaluate_full_resolution_mse(vector: torch.Tensor) -> float:
            """Evaluate the user-facing image metric without downsampling."""
            with torch.no_grad():
                posture, _ = self._decode(
                    vector[:posture_decision_count],
                    matrices,
                    point_indices,
                    len(xy),
                )
                for field_index, field_name in enumerate(self.FIELD_NAMES):
                    if field_name not in active_fields:
                        posture[:, field_index] = initial_points_tensor[
                            :, field_index
                        ]
                rendered_xy = xy
                if self.optimize_xy:
                    xy_offsets, _ = self._decode_xy_offsets(
                        vector[posture_decision_count:],
                        matrices,
                        point_indices,
                        len(xy),
                    )
                    rendered_xy = xy + xy_offsets
                rendered = self.renderer(rendered_xy, posture, ids)
                return float(F.mse_loss(rendered, target).item())

        def finite_difference_jacobian(
            vector: torch.Tensor,
            base_residual: torch.Tensor,
            columns_to_evaluate: torch.Tensor,
        ) -> torch.Tensor:
            """Memory-bounded numerical Jacobian, matching the Wang LM flow."""
            columns = []
            total = int(columns_to_evaluate.numel())
            with torch.no_grad():
                for offset, column_tensor in enumerate(columns_to_evaluate):
                    column = int(column_tensor)
                    step = self.finite_difference_eps * (
                        1.0 + abs(float(vector[column]))
                    )
                    trial = vector.clone()
                    trial[column] += step
                    derivative = (
                        residual_fn(trial) - base_residual
                    ) / step
                    columns.append(derivative)
                    if (offset + 1) % 10 == 0 or offset + 1 == total:
                        print(
                            f"[JACOBIAN] column {offset + 1}/{total}",
                            flush=True,
                        )
            return torch.stack(columns, dim=1)

        def autograd_jacobian(
            vector: torch.Tensor, columns_to_evaluate: torch.Tensor
        ) -> torch.Tensor:
            """Optional high-memory path for GPUs with substantially more VRAM."""
            differentiable = vector.detach().requires_grad_(True)
            full = torch.autograd.functional.jacobian(
                residual_fn, differentiable, vectorize=False
            ).detach()
            return full[:, columns_to_evaluate]

        def summarize_sensitivity(
            jacobian: torch.Tensor,
            columns_evaluated: torch.Tensor,
            vector: torch.Tensor,
        ) -> Dict[str, Dict[str, float]]:
            """Summarize image sensitivity for the evaluated decision columns."""
            pixel_rows = self.optimization_size * self.optimization_size
            column_norms = torch.linalg.vector_norm(
                jacobian[:pixel_rows], dim=0
            )
            normalized_nodes = torch.sigmoid(vector)
            sigmoid_slope = (
                normalized_nodes * (1.0 - normalized_nodes)
            ).clamp_min(1e-4)
            sensitivity: Dict[str, Dict[str, float]] = {}
            medians = {}
            means = {}
            values_by_field = {}
            for field_name in self.FIELD_NAMES:
                member_mask = torch.isin(
                    columns_evaluated, field_columns[field_name]
                )
                if not bool(torch.any(member_mask)):
                    continue
                selected_columns = columns_evaluated[member_mask]
                values = (
                    column_norms[member_mask]
                    / sigmoid_slope[selected_columns]
                )
                values_by_field[field_name] = values
                medians[field_name] = values.median()
                means[field_name] = values.mean()
            max_median = max(
                float(value) for value in medians.values()
            ) if medians else 1.0
            max_mean = max(
                float(value) for value in means.values()
            ) if means else 1.0
            max_median = max(max_median, 1e-12)
            max_mean = max(max_mean, 1e-12)
            for field_name, values in values_by_field.items():
                sensitivity[field_name] = {
                    "mean_l2_per_normalized_range": float(values.mean()),
                    "median_l2_per_normalized_range": float(values.median()),
                    "max_l2_per_normalized_range": float(values.max()),
                    "relative_mean": float(means[field_name]) / max_mean,
                    "relative_median": (
                        float(medians[field_name]) / max_median
                    ),
                }
            return sensitivity

        def columns_for_fields(field_names: Sequence[str]) -> torch.Tensor:
            if not field_names:
                return torch.empty(
                    0, dtype=torch.long, device=decision.device
                )
            return torch.cat([field_columns[name] for name in field_names])

        def fix_inactive_fields(
            vector: torch.Tensor, active_field_names: Sequence[str]
        ) -> torch.Tensor:
            fixed = vector.clone()
            active = set(active_field_names)
            for field_name in self.FIELD_NAMES:
                if field_name in active:
                    continue
                fixed[field_columns[field_name]] = initial_decision[
                    field_columns[field_name]
                ]
            return fixed

        audit_sensitivity: Dict[str, Dict[str, float]] = {}
        if self.field_mode == "auto":
            with torch.no_grad():
                audit_residual = residual_fn(decision)
            print(
                "[OBSERVABILITY] auditing H/alpha/beta before gated LM",
                flush=True,
            )
            if self.jacobian_mode == "finite_difference":
                audit_jacobian = finite_difference_jacobian(
                    decision, audit_residual, all_posture_columns
                )
            else:
                audit_jacobian = autograd_jacobian(
                    decision, all_posture_columns
                )
            audit_sensitivity = summarize_sensitivity(
                audit_jacobian, all_posture_columns, decision
            )
            active_fields = ["H"]
            for field_name in ("alpha", "beta"):
                relative = audit_sensitivity[field_name][
                    "relative_median"
                ]
                if relative >= self.min_relative_median_sensitivity:
                    active_fields.append(field_name)
            print(
                "[OBSERVABILITY] active_fields="
                f"{','.join(active_fields)}, threshold="
                f"{self.min_relative_median_sensitivity:.3f}",
                flush=True,
            )
        elif self.field_mode == "h_only":
            active_fields = ["H"]
        elif self.field_mode == "xy_only":
            active_fields = []
        else:
            active_fields = list(self.FIELD_NAMES)
        decision = fix_inactive_fields(decision, active_fields)
        active_columns = columns_for_fields(active_fields)
        if self.optimize_xy:
            active_columns = torch.cat([active_columns, xy_columns])

        current_cost = evaluate_cost(decision)
        initial_cost = current_cost
        current_full_mse = evaluate_full_resolution_mse(decision)
        initial_full_mse = current_full_mse
        best_full_mse = current_full_mse
        best_decision = decision.detach().clone()
        best_step = 0
        mu = float(damping)
        history = {
            "cost": [current_cost],
            "damping": [mu],
            "full_resolution_mse": [current_full_mse],
        }
        success = False
        message = "Maximum steps reached"
        completed_steps = 0
        last_jacobian = None
        last_decision = decision.detach()
        last_columns = active_columns

        for step in range(1, int(max_steps) + 1):
            decision = decision.detach()
            with torch.no_grad():
                residual = residual_fn(decision)
            print(
                f"[JACOBIAN {step:03d}] mode={self.jacobian_mode}, "
                f"active_variables={active_columns.numel()}/"
                f"{decision.numel()}, residuals={residual.numel()}",
                flush=True,
            )
            if self.jacobian_mode == "finite_difference":
                jacobian = finite_difference_jacobian(
                    decision, residual, active_columns
                )
            else:
                jacobian = autograd_jacobian(decision, active_columns)
            last_jacobian = jacobian
            last_decision = decision.detach()
            last_columns = active_columns
            gradient = jacobian.T @ residual
            if float(torch.linalg.vector_norm(gradient, ord=float("inf"))) < 1e-6:
                success = True
                message = "Gradient tolerance reached"
                completed_steps = step - 1
                break
            normal = jacobian.T @ jacobian
            diagonal = torch.diag(normal).clamp_min(1e-8)
            system = normal + mu * torch.diag(diagonal)
            try:
                delta = torch.linalg.solve(system, -gradient)
            except RuntimeError:
                delta = torch.linalg.lstsq(system, -gradient[:, None]).solution[:, 0]
            if float(torch.linalg.vector_norm(delta)) < 1e-5 * (
                float(
                    torch.linalg.vector_norm(
                        decision.detach()[active_columns]
                    )
                )
                + 1e-5
            ):
                success = True
                message = "Step tolerance reached"
                completed_steps = step - 1
                break
            trial = decision.detach().clone()
            trial[active_columns] += delta.detach()
            trial_cost = evaluate_cost(trial)
            if np.isfinite(trial_cost) and trial_cost < current_cost:
                improvement = current_cost - trial_cost
                decision = trial
                current_cost = trial_cost
                current_full_mse = evaluate_full_resolution_mse(decision)
                if current_full_mse < best_full_mse:
                    best_full_mse = current_full_mse
                    best_decision = decision.detach().clone()
                    best_step = step
                mu = max(mu * 0.3, 1e-8)
                if improvement < 1e-7 * (1.0 + current_cost):
                    success = True
                    message = "Function tolerance reached"
                    completed_steps = step
                    history["cost"].append(current_cost)
                    history["damping"].append(mu)
                    history["full_resolution_mse"].append(current_full_mse)
                    break
            else:
                decision = decision.detach()
                mu = min(mu * 10.0, 1e8)
            completed_steps = step
            history["cost"].append(current_cost)
            history["damping"].append(mu)
            history["full_resolution_mse"].append(current_full_mse)
            print(
                f"[LM {step:03d}] cost={current_cost:.6f}, "
                f"full_mse={current_full_mse:.6f}, damping={mu:.6g}",
                flush=True,
            )

        terminal_cost = current_cost
        terminal_full_mse = current_full_mse
        selected_cost = evaluate_cost(best_decision)
        with torch.no_grad():
            posture, _ = self._decode(
                best_decision[:posture_decision_count],
                matrices,
                point_indices,
                len(xy),
            )
            for field_index, field_name in enumerate(self.FIELD_NAMES):
                if field_name not in active_fields:
                    posture[:, field_index] = initial_points_tensor[
                        :, field_index
                    ]
            optimized_xy = xy
            xy_offsets = torch.zeros_like(xy)
            if self.optimize_xy:
                xy_offsets, _ = self._decode_xy_offsets(
                    best_decision[posture_decision_count:],
                    matrices,
                    point_indices,
                    len(xy),
                )
                optimized_xy = xy + xy_offsets
            rendered = self.renderer(optimized_xy, posture, ids)[0, 0]
        diagnostics: Dict[str, Any] = {
            "checkpoint_selection": {
                "metric": "full_resolution_plain_mse",
                "initial_mse": initial_full_mse,
                "best_mse": best_full_mse,
                "best_step": best_step,
                "terminal_mse": terminal_full_mse,
                "terminal_regularized_cost": terminal_cost,
                "selected_regularized_cost": selected_cost,
                "returned_best_checkpoint": best_step != completed_steps,
            },
            "regularization": {
                "initial_posture_source": initial_posture_source,
                "field_order": list(self.FIELD_NAMES),
                "smoothness_weights": self.smoothness_weights.tolist(),
                "posture_prior_weights": self.posture_prior_weights.tolist(),
                "terminal_lift_weight": self.terminal_lift_weight,
                "terminal_lift_nodes": self.terminal_lift_nodes,
                "terminal_lift_weight_source": (
                    "user_simulation_setting"
                    if self.terminal_lift_weight > 0
                    else "disabled_paper_did_not_report_beta_k"
                ),
                "xy_enabled": self.optimize_xy,
                "xy_max_offset_px": self.xy_max_offset_px,
                "xy_smoothness_weight": self.xy_smoothness_weight,
                "xy_prior_weight": self.xy_prior_weight,
                "h_point_velocity_weight": self.h_point_velocity_weight,
                "h_point_acceleration_weight": (
                    self.h_point_acceleration_weight
                ),
            },
            "cgl_layout": {
                "requested_order": self.order,
                "cap_order_to_points": self.cap_order_to_points,
                "point_counts_per_stroke": [
                    int(len(indices)) for indices in point_indices
                ],
                "effective_orders_per_stroke": effective_orders,
                "active_nodes_per_stroke": [
                    int(order + 1) for order in effective_orders
                ],
                "active_node_count": int(active_node_mask_np.sum()),
                "allocated_node_count": int(active_node_mask_np.size),
            },
            "observability_gate": {
                "mode": self.field_mode,
                "min_relative_median_sensitivity": (
                    self.min_relative_median_sensitivity
                ),
                "optimized_fields": (
                    ["x", "y"] if self.optimize_xy else []
                )
                + list(active_fields),
                "fixed_fields": [
                    name for name in self.FIELD_NAMES
                    if name not in active_fields
                ]
                + ([] if self.optimize_xy else ["x", "y"]),
                "initial_image_jacobian_sensitivity": audit_sensitivity,
            },
        }
        if last_jacobian is not None:
            diagnostics["image_jacobian_sensitivity"] = (
                summarize_sensitivity(
                    last_jacobian, last_columns, last_decision
                )
            )

        posture_np = posture.cpu().numpy()
        normalized_posture = (posture_np - PAPER_POSTURE_MIN) / (
            PAPER_POSTURE_MAX - PAPER_POSTURE_MIN
        )
        h_first_differences = []
        h_second_differences = []
        per_stroke_h_continuity = []
        for stroke_id, indices in zip(np.unique(stroke_ids), point_indices):
            h_values = posture_np[indices, 0]
            first = np.diff(h_values)
            second = np.diff(h_values, n=2)
            h_first_differences.extend(first.tolist())
            h_second_differences.extend(second.tolist())
            per_stroke_h_continuity.append(
                {
                    "stroke_id": int(stroke_id),
                    "point_count": int(len(indices)),
                    "max_abs_step_mm": (
                        float(np.max(np.abs(first))) if len(first) else 0.0
                    ),
                    "max_abs_second_difference_mm": (
                        float(np.max(np.abs(second))) if len(second) else 0.0
                    ),
                }
            )

        def difference_summary(values: Sequence[float]) -> Dict[str, float]:
            array = np.asarray(values, dtype=np.float32)
            if not len(array):
                return {
                    "mean_abs_mm": 0.0,
                    "rms_mm": 0.0,
                    "max_abs_mm": 0.0,
                }
            return {
                "mean_abs_mm": float(np.mean(np.abs(array))),
                "rms_mm": float(np.sqrt(np.mean(array**2))),
                "max_abs_mm": float(np.max(np.abs(array))),
            }

        diagnostics["trajectory_continuity"] = {
            "quantity": "H_mm",
            "difference_domain": "successive input trajectory samples",
            "first_difference": difference_summary(h_first_differences),
            "second_difference": difference_summary(h_second_differences),
            "per_stroke": per_stroke_h_continuity,
            "physical_time_note": (
                "Input samples have no timestamps; these are point-index "
                "continuity diagnostics, not calibrated mm/s or mm/s^2."
            ),
        }
        diagnostics["bound_fraction_within_1pct"] = {
            field_name: {
                "lower": float(
                    np.mean(normalized_posture[:, field_index] <= 0.01)
                ),
                "upper": float(
                    np.mean(normalized_posture[:, field_index] >= 0.99)
                ),
            }
            for field_index, field_name in enumerate(("H", "alpha", "beta"))
        }
        xy_offsets_np = xy_offsets.cpu().numpy()
        normalized_xy_offset = xy_offsets_np / self.xy_max_offset_px
        diagnostics["xy_optimization"] = {
            "enabled": self.optimize_xy,
            "max_offset_px": self.xy_max_offset_px,
            "smoothness_weight": self.xy_smoothness_weight,
            "prior_weight": self.xy_prior_weight,
            "max_abs_change_px": float(np.abs(xy_offsets_np).max()),
            "mean_point_displacement_px": float(
                np.linalg.norm(xy_offsets_np, axis=1).mean()
            ),
            "rms_point_displacement_px": float(
                np.sqrt(np.mean(np.sum(xy_offsets_np**2, axis=1)))
            ),
            "component_bound_fraction_within_1pct": float(
                np.mean(np.abs(normalized_xy_offset) >= 0.99)
            ),
        }
        decisions = {}
        source_sensitivity = (
            audit_sensitivity
            or diagnostics.get("image_jacobian_sensitivity", {})
        )
        for field_name in self.FIELD_NAMES:
            optimized = field_name in active_fields
            boundary = diagnostics["bound_fraction_within_1pct"][field_name]
            boundary_total = boundary["lower"] + boundary["upper"]
            relative = source_sensitivity.get(field_name, {}).get(
                "relative_median"
            )
            if not optimized:
                confidence = "low"
                reason = (
                    "fixed_for_xy_only_ablation"
                    if self.field_mode == "xy_only"
                    else "fixed_below_observability_threshold"
                )
            elif boundary_total > 0.25:
                confidence = "low"
                reason = "optimized_but_boundary_saturated"
            elif field_name == "H":
                confidence = "medium_simulation"
                reason = "optimized_observable_simulation_parameter"
            else:
                confidence = "medium_simulation"
                reason = "optimized_above_observability_threshold"
            decisions[field_name] = {
                "optimized": optimized,
                "source": (
                    "lm_optimized"
                    if optimized
                    else (
                        "initial_pose_csv"
                        if initial_posture_source == "initial_pose_csv"
                        else "initial_default"
                    )
                ),
                "confidence": confidence,
                "reason": reason,
                "initial_relative_median_sensitivity": relative,
                "boundary_fraction": boundary_total if optimized else None,
                "fixed_value_on_physical_boundary": (
                    None if optimized else boundary_total > 0.0
                ),
            }
        decisions["gamma"] = {
            "optimized": False,
            "source": "fixed_model_default",
            "confidence": "low",
            "reason": "unobservable_in_axisymmetric_brush_model",
            "initial_relative_median_sensitivity": None,
            "boundary_fraction": 0.0,
            "fixed_value_on_physical_boundary": False,
        }
        for field_name, field_index in (("x", 0), ("y", 1)):
            component_boundary = float(
                np.mean(
                    np.abs(normalized_xy_offset[:, field_index]) >= 0.99
                )
            )
            decisions[field_name] = {
                "optimized": self.optimize_xy,
                "source": (
                    "lm_optimized_bounded_offset"
                    if self.optimize_xy
                    else "input_trajectory"
                ),
                "confidence": (
                    "low_simulation" if self.optimize_xy else "input"
                ),
                "reason": (
                    "bounded_planar_image_registration"
                    if self.optimize_xy
                    else "fixed_by_user_trajectory"
                ),
                "initial_relative_median_sensitivity": None,
                "boundary_fraction": (
                    component_boundary if self.optimize_xy else None
                ),
                "fixed_value_on_physical_boundary": None,
            }
        diagnostics["field_decisions"] = decisions
        return PaperLMResult(
            xy_canvas=optimized_xy.cpu().numpy(),
            posture=posture_np,
            rendered_image=rendered.cpu().numpy(),
            success=success,
            steps=completed_steps,
            initial_cost=initial_cost,
            final_cost=selected_cost,
            message=message,
            history=history,
            diagnostics=diagnostics,
        )
