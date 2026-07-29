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


def summarize_joint_identifiability(
    pixel_jacobian: torch.Tensor,
    field_column_positions: Dict[str, torch.Tensor],
    relative_rank_threshold: float = 1e-3,
    min_rank_fraction: float = 0.90,
    max_condition_number: float = 1e2,
    max_canonical_correlation: float = 0.95,
) -> Dict[str, Any]:
    """Diagnose whether selected pose columns are jointly distinguishable.

    Individual column SNR only establishes that a perturbation changes the
    image. This audit additionally detects rank loss and overlapping field
    subspaces, which allow H/alpha/beta/gamma to compensate for one another.
    Columns are expected to be scaled to one normalized physical range.
    """
    if pixel_jacobian.ndim != 2:
        raise ValueError("pixel_jacobian must have shape [pixels, columns]")
    column_count = int(pixel_jacobian.shape[1])
    if column_count == 0:
        return {
            "selected_columns": 0,
            "effective_rank": 0,
            "rank_fraction": 0.0,
            "condition_number": None,
            "max_field_canonical_correlation": None,
            "max_correlated_field_pair": None,
            "jointly_identifiable": False,
            "reason": "no_selected_pose_columns",
        }
    if not 0.0 < relative_rank_threshold < 1.0:
        raise ValueError("relative_rank_threshold must be in (0,1)")
    if not 0.0 < min_rank_fraction <= 1.0:
        raise ValueError("min_rank_fraction must be in (0,1]")
    if max_condition_number <= 1.0:
        raise ValueError("max_condition_number must be > 1")
    if not 0.0 <= max_canonical_correlation <= 1.0:
        raise ValueError("max_canonical_correlation must be in [0,1]")

    norms = torch.linalg.vector_norm(pixel_jacobian, dim=0).clamp_min(1e-12)
    directions = pixel_jacobian / norms
    singular = torch.linalg.svdvals(directions)
    largest = float(singular[0])
    rank_threshold = largest * relative_rank_threshold
    effective_rank = int((singular >= rank_threshold).sum())
    rank_fraction = effective_rank / max(column_count, 1)
    smallest = float(singular[-1])
    condition_number = (
        largest / smallest
        if smallest > torch.finfo(singular.dtype).eps
        else float("inf")
    )

    field_bases: Dict[str, torch.Tensor] = {}
    field_ranks: Dict[str, int] = {}
    for field_name, positions in field_column_positions.items():
        positions = positions.to(
            dtype=torch.long, device=pixel_jacobian.device
        )
        if int(positions.numel()) == 0:
            field_ranks[field_name] = 0
            continue
        field_matrix = directions[:, positions]
        basis, field_singular, _ = torch.linalg.svd(
            field_matrix, full_matrices=False
        )
        field_cutoff = float(field_singular[0]) * relative_rank_threshold
        field_rank = int((field_singular >= field_cutoff).sum())
        field_ranks[field_name] = field_rank
        field_bases[field_name] = basis[:, :field_rank]

    pairwise: Dict[str, float] = {}
    names = list(field_bases)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            overlap = (
                field_bases[left_name].T @ field_bases[right_name]
            )
            correlation = float(torch.linalg.svdvals(overlap)[0])
            pairwise[f"{left_name}__{right_name}"] = min(correlation, 1.0)
    max_pair = max(pairwise, key=pairwise.get) if pairwise else None
    max_correlation = pairwise[max_pair] if max_pair is not None else None

    rank_ok = rank_fraction >= min_rank_fraction
    condition_ok = condition_number <= max_condition_number
    correlation_ok = (
        max_correlation is None
        or max_correlation <= max_canonical_correlation
    )
    jointly_identifiable = rank_ok and condition_ok and correlation_ok
    failed = []
    if not rank_ok:
        failed.append("rank_fraction_below_threshold")
    if not condition_ok:
        failed.append("condition_number_above_threshold")
    if not correlation_ok:
        failed.append("field_subspaces_overlap")
    return {
        "selected_columns": column_count,
        "effective_rank": effective_rank,
        "rank_fraction": rank_fraction,
        "condition_number": condition_number,
        "stable_rank": float((singular**2).sum() / singular[0] ** 2),
        "relative_rank_threshold": relative_rank_threshold,
        "min_rank_fraction": min_rank_fraction,
        "max_condition_number": max_condition_number,
        "max_allowed_field_canonical_correlation": (
            max_canonical_correlation
        ),
        "field_effective_ranks": field_ranks,
        "pairwise_field_canonical_correlation": pairwise,
        "max_field_canonical_correlation": max_correlation,
        "max_correlated_field_pair": max_pair,
        "largest_singular_value": largest,
        "smallest_singular_value": smallest,
        "jointly_identifiable": jointly_identifiable,
        "reason": (
            "passed_joint_jacobian_audit"
            if jointly_identifiable
            else ",".join(failed)
        ),
    }


def prune_jointly_aliased_fields(
    pixel_jacobian: torch.Tensor,
    field_column_positions: Dict[str, torch.Tensor],
    field_scores: Dict[str, float],
    preserve_fields: Sequence[str] = ("H",),
    relative_rank_threshold: float = 1e-3,
    min_rank_fraction: float = 0.90,
    max_condition_number: float = 1e2,
    max_canonical_correlation: float = 0.95,
) -> Dict[str, Any]:
    """Greedily remove the weaker field from an ambiguous pose pair.

    The routine operates at field granularity. It never fabricates a unique
    pose: pruning stops when the retained Jacobian passes the conservative
    joint audit or only one field remains. ``preserve_fields`` are removed
    last when an equally safe non-preserved candidate exists.
    """
    kept = [
        name
        for name, positions in field_column_positions.items()
        if int(positions.numel()) > 0
    ]
    preserved = set(preserve_fields)
    removed: List[Dict[str, Any]] = []

    def audit(field_names: Sequence[str]) -> Dict[str, Any]:
        if not field_names:
            return summarize_joint_identifiability(
                pixel_jacobian[:, :0],
                {},
                relative_rank_threshold=relative_rank_threshold,
                min_rank_fraction=min_rank_fraction,
                max_condition_number=max_condition_number,
                max_canonical_correlation=max_canonical_correlation,
            )
        global_positions = torch.cat(
            [field_column_positions[name] for name in field_names]
        )
        global_positions = torch.sort(global_positions).values
        local_positions = {
            name: torch.nonzero(
                torch.isin(
                    global_positions, field_column_positions[name]
                ),
                as_tuple=False,
            ).flatten()
            for name in field_names
        }
        return summarize_joint_identifiability(
            pixel_jacobian[:, global_positions],
            local_positions,
            relative_rank_threshold=relative_rank_threshold,
            min_rank_fraction=min_rank_fraction,
            max_condition_number=max_condition_number,
            max_canonical_correlation=max_canonical_correlation,
        )

    diagnostics = audit(kept)
    while not diagnostics["jointly_identifiable"] and len(kept) > 1:
        correlated_pair = diagnostics.get("max_correlated_field_pair")
        correlation = diagnostics.get(
            "max_field_canonical_correlation"
        )
        if (
            correlated_pair
            and correlation is not None
            and correlation > max_canonical_correlation
        ):
            candidates = correlated_pair.split("__")
            reason = "field_subspaces_overlap"
        else:
            candidates = list(kept)
            reason = "joint_rank_or_condition_failed"
        non_preserved = [
            name for name in candidates if name not in preserved
        ]
        removable = non_preserved or candidates
        dropped = min(
            removable,
            key=lambda name: (float(field_scores.get(name, 0.0)), name),
        )
        removed.append(
            {
                "field": dropped,
                "reason": reason,
                "field_score": float(field_scores.get(dropped, 0.0)),
                "correlated_pair": (
                    correlated_pair if reason == "field_subspaces_overlap"
                    else None
                ),
                "pair_canonical_correlation": (
                    float(correlation)
                    if reason == "field_subspaces_overlap"
                    else None
                ),
            }
        )
        kept.remove(dropped)
        diagnostics = audit(kept)
    return {
        "kept_fields": kept,
        "removed_fields": removed,
        "final_joint_identifiability": diagnostics,
        "passed": bool(diagnostics["jointly_identifiable"]),
        "preserve_fields": list(preserve_fields),
    }


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
    gamma: np.ndarray | None = None


class PaperPSOCLM:
    """Optimize bounded per-stroke CGL pose and optional planar offsets."""

    FIELD_NAMES = ("H", "alpha", "beta")
    POSTURE_BOUND_EPS = 0.02

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
        optimize_gamma: bool = False,
        gamma_max_abs_rad: float = np.pi,
        gamma_smoothness_weight: float = 0.10,
        gamma_prior_weight: float = 0.05,
        observability_gate_mode: str = "field_relative",
        observability_noise_rmse: float | None = None,
        min_observability_snr: float = 1.0,
        joint_gate_action: str = "report",
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
        if gamma_max_abs_rad <= 0 or gamma_max_abs_rad > np.pi:
            raise ValueError("gamma_max_abs_rad must be in (0, pi]")
        if gamma_smoothness_weight < 0 or gamma_prior_weight < 0:
            raise ValueError(
                "gamma regularization weights must be non-negative"
            )
        self.optimize_gamma = bool(optimize_gamma)
        self.gamma_max_abs_rad = float(gamma_max_abs_rad)
        self.gamma_smoothness_weight = float(gamma_smoothness_weight)
        self.gamma_prior_weight = float(gamma_prior_weight)
        if observability_gate_mode not in {
            "field_relative",
            "node_snr",
        }:
            raise ValueError(
                "observability_gate_mode must be field_relative or node_snr"
            )
        if observability_noise_rmse is not None and observability_noise_rmse <= 0:
            raise ValueError("observability_noise_rmse must be positive")
        if min_observability_snr <= 0:
            raise ValueError("min_observability_snr must be positive")
        if joint_gate_action not in {"report", "prune"}:
            raise ValueError("joint_gate_action must be report or prune")
        self.observability_gate_mode = observability_gate_mode
        checkpoint_rmse = getattr(
            renderer, "checkpoint_validation_rmse", None
        )
        self.observability_noise_rmse = (
            float(observability_noise_rmse)
            if observability_noise_rmse is not None
            else (
                float(checkpoint_rmse)
                if checkpoint_rmse is not None
                else None
            )
        )
        self.min_observability_snr = float(min_observability_snr)
        self.joint_gate_action = joint_gate_action
        if (
            self.field_mode == "auto"
            and self.observability_gate_mode == "node_snr"
            and self.observability_noise_rmse is None
        ):
            raise ValueError(
                "node_snr observability requires checkpoint plain_mse or "
                "an explicit observability_noise_rmse"
            )
        if self.optimize_gamma and np.isclose(
            renderer.dynamic.longitudinal_scale,
            renderer.dynamic.transverse_scale,
            rtol=0.0,
            atol=1e-8,
        ):
            raise ValueError(
                "Gamma requires a non-axisymmetric footprint: longitudinal "
                "and transverse scales must differ"
            )
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
        epsilon = float(self.POSTURE_BOUND_EPS)
        normalized_nodes = (
            (torch.sigmoid(node_logits) - epsilon)
            / (1.0 - 2.0 * epsilon)
        ).clamp(0.0, 1.0)
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
            values = (
                (torch.sigmoid(point_logits) - epsilon)
                / (1.0 - 2.0 * epsilon)
            ).clamp(0.0, 1.0)
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

    def _decode_gamma(
        self,
        decision: torch.Tensor,
        matrices: Sequence[torch.Tensor],
        point_indices: Sequence[np.ndarray],
        point_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode bounded axial angles without CGL overshoot."""
        node_count = self.order + 1
        node_logits = decision.view(len(matrices), node_count)
        normalized_nodes = torch.tanh(node_logits)
        normalized_points = torch.empty(
            point_count, dtype=decision.dtype, device=decision.device
        )
        for stroke_index, (matrix, indices) in enumerate(
            zip(matrices, point_indices)
        ):
            point_logits = (
                matrix.to(dtype=decision.dtype) @ node_logits[stroke_index]
            )
            normalized_points[
                torch.as_tensor(indices, device=decision.device)
            ] = torch.tanh(point_logits)
        return (
            normalized_points * self.gamma_max_abs_rad,
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
        initial_gamma_rad: float = 0.0,
        damping: float = 0.05,
        max_steps: int = 15,
        pixel_weight: float = 3.0,
        initial_posture: np.ndarray | None = None,
        initial_gamma: np.ndarray | None = None,
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
        if initial_gamma is None:
            initial_gamma_points = np.full(
                len(xy), float(initial_gamma_rad), dtype=np.float32
            )
        else:
            initial_gamma_points = np.asarray(
                initial_gamma, dtype=np.float32
            )
            if initial_gamma_points.shape != (len(xy),):
                raise ValueError(
                    "initial_gamma must have shape [trajectory_points]"
                )
        if not np.all(np.isfinite(initial_gamma_points)) or np.any(
            np.abs(initial_gamma_points) > self.gamma_max_abs_rad + 1e-6
        ):
            raise ValueError(
                "Initial gamma exceeds configured symmetric angular bounds"
            )
        normalized_initial_points = (
            initial_points - PAPER_POSTURE_MIN
        ) / (
            PAPER_POSTURE_MAX - PAPER_POSTURE_MIN
        )
        epsilon = float(self.POSTURE_BOUND_EPS)
        normalized_for_audit = (
            epsilon
            + (1.0 - 2.0 * epsilon)
            * np.clip(normalized_initial_points, 0.0, 1.0)
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
        gamma_decision_count = (
            len(matrices) * (self.order + 1)
            if self.optimize_gamma
            else 0
        )
        xy_decision_count = (
            len(matrices) * 2 * (self.order + 1)
            if self.optimize_xy
            else 0
        )
        decision_parts = [posture_decision]
        initial_gamma_node_logits = np.zeros(
            (len(matrices), self.order + 1), dtype=np.float32
        )
        if gamma_decision_count:
            normalized_gamma = np.clip(
                initial_gamma_points / self.gamma_max_abs_rad,
                -0.98,
                0.98,
            )
            gamma_point_logits = np.arctanh(normalized_gamma)
            for stroke_index, (
                matrix,
                indices,
                effective_order,
            ) in enumerate(zip(matrices, point_indices, effective_orders)):
                compact_matrix = (
                    matrix[:, : effective_order + 1].detach().cpu().numpy()
                )
                fitted, _, _, _ = np.linalg.lstsq(
                    compact_matrix,
                    gamma_point_logits[indices],
                    rcond=None,
                )
                initial_gamma_node_logits[
                    stroke_index, : effective_order + 1
                ] = fitted
            decision_parts.append(
                torch.as_tensor(
                    initial_gamma_node_logits.reshape(-1),
                    dtype=posture_decision.dtype,
                    device=posture_decision.device,
                )
            )
        if xy_decision_count:
            decision_parts.append(
                torch.zeros(
                    xy_decision_count,
                    dtype=posture_decision.dtype,
                    device=posture_decision.device,
                )
            )
        decision = torch.cat(decision_parts)
        initial_decision = decision.detach().clone()
        prior_sigmoid = 1.0 / (
            1.0 + np.exp(-np.clip(initial_node_logits, -30.0, 30.0))
        )
        prior = torch.as_tensor(
            np.clip(
                (prior_sigmoid - epsilon) / (1.0 - 2.0 * epsilon),
                0.0,
                1.0,
            ),
            dtype=decision.dtype,
            device=decision.device,
        )
        initial_points_tensor = torch.as_tensor(
            initial_points, dtype=decision.dtype, device=decision.device
        )
        initial_gamma_tensor = torch.as_tensor(
            initial_gamma_points,
            dtype=decision.dtype,
            device=decision.device,
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
        gamma_start = posture_decision_count
        gamma_stop = gamma_start + gamma_decision_count
        if gamma_decision_count:
            gamma_layout = np.arange(
                gamma_start, gamma_stop, dtype=np.int64
            ).reshape(len(matrices), node_count)
            gamma_columns = torch.as_tensor(
                gamma_layout[active_node_mask_np],
                dtype=torch.long,
                device=decision.device,
            )
        else:
            gamma_columns = torch.empty(
                0, dtype=torch.long, device=decision.device
            )
        if xy_decision_count:
            xy_layout = np.arange(
                gamma_stop,
                gamma_stop + xy_decision_count,
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
        gamma_prior = torch.tanh(
            torch.as_tensor(
                initial_gamma_node_logits,
                dtype=decision.dtype,
                device=decision.device,
            )
        )
        gated_active_fields: set[str] | None = None

        def residual_fn(vector: torch.Tensor) -> torch.Tensor:
            posture, nodes = self._decode(
                vector[:posture_decision_count],
                matrices,
                point_indices,
                len(xy),
            )
            if gated_active_fields is not None:
                for field_index, field_name in enumerate(self.FIELD_NAMES):
                    if field_name not in gated_active_fields:
                        posture[:, field_index] = initial_points_tensor[
                            :, field_index
                        ]
            rendered_xy = xy
            xy_nodes = None
            gamma_points = initial_gamma_tensor
            gamma_nodes = None
            if self.optimize_gamma:
                gamma_points, gamma_nodes = self._decode_gamma(
                    vector[gamma_start:gamma_stop],
                    matrices,
                    point_indices,
                    len(xy),
                )
                if (
                    gated_active_fields is not None
                    and "gamma" not in gated_active_fields
                ):
                    gamma_points = initial_gamma_tensor
            if self.optimize_xy:
                xy_offsets, xy_nodes = self._decode_xy_offsets(
                    vector[gamma_stop:],
                    matrices,
                    point_indices,
                    len(xy),
                )
                rendered_xy = xy + xy_offsets
            render_args = (
                rendered_xy[render_indices],
                posture[render_indices],
                ids[render_indices],
            )
            rendered = (
                self.renderer(*render_args, gamma_points[render_indices])
                if self.optimize_gamma
                else self.renderer(*render_args)
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
            if gamma_nodes is not None and self.gamma_smoothness_weight > 0:
                residuals.append(
                    (
                        self.gamma_smoothness_weight**0.5
                        * (gamma_nodes[:, 1:] - gamma_nodes[:, :-1])
                        * active_node_pair_mask
                    ).flatten()
                )
            if gamma_nodes is not None and self.gamma_prior_weight > 0:
                residuals.append(
                    (
                        self.gamma_prior_weight**0.5
                        * (gamma_nodes - gamma_prior)
                        * active_node_mask
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
                gamma_points = initial_gamma_tensor
                if self.optimize_gamma:
                    gamma_points, _ = self._decode_gamma(
                        vector[gamma_start:gamma_stop],
                        matrices,
                        point_indices,
                        len(xy),
                    )
                    if "gamma" not in active_fields:
                        gamma_points = initial_gamma_tensor
                if self.optimize_xy:
                    xy_offsets, _ = self._decode_xy_offsets(
                        vector[gamma_stop:],
                        matrices,
                        point_indices,
                        len(xy),
                    )
                    rendered_xy = xy + xy_offsets
                rendered = (
                    self.renderer(
                        rendered_xy, posture, ids, gamma_points
                    )
                    if self.optimize_gamma
                    else self.renderer(rendered_xy, posture, ids)
                )
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

        def full_range_normalized_pixel_jacobian(
            jacobian: torch.Tensor,
            columns_evaluated: torch.Tensor,
            vector: torch.Tensor,
        ) -> torch.Tensor:
            """Scale pixel Jacobian columns to one full normalized range."""
            pixel_rows = self.optimization_size * self.optimization_size
            normalized_nodes = torch.sigmoid(vector)
            parameter_slope = (
                normalized_nodes * (1.0 - normalized_nodes)
                / (1.0 - 2.0 * float(self.POSTURE_BOUND_EPS))
            ).clamp_min(1e-4)
            if gamma_decision_count:
                gamma_normalized = torch.tanh(
                    vector[gamma_start:gamma_stop]
                )
                parameter_slope[gamma_start:gamma_stop] = (
                    1.0 - gamma_normalized**2
                ).clamp_min(1e-4)
            return (
                jacobian[:pixel_rows]
                / parameter_slope[columns_evaluated][None, :]
            )

        def normalized_column_sensitivity(
            jacobian: torch.Tensor,
            columns_evaluated: torch.Tensor,
            vector: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Return per-column L2 and pixel RMS for a normalized full range."""
            pixel_rows = self.optimization_size * self.optimization_size
            normalized_pixel_jacobian = (
                full_range_normalized_pixel_jacobian(
                    jacobian, columns_evaluated, vector
                )
            )
            column_norms = torch.linalg.vector_norm(
                normalized_pixel_jacobian, dim=0
            )
            l2 = column_norms
            return l2, l2 / float(pixel_rows) ** 0.5

        def summarize_sensitivity(
            jacobian: torch.Tensor,
            columns_evaluated: torch.Tensor,
            vector: torch.Tensor,
        ) -> Dict[str, Dict[str, float]]:
            """Summarize image sensitivity for the evaluated decision columns."""
            column_l2, column_rms = normalized_column_sensitivity(
                jacobian, columns_evaluated, vector
            )
            sensitivity: Dict[str, Dict[str, float]] = {}
            medians = {}
            means = {}
            values_by_field = {}
            rms_by_field = {}
            audit_field_names = list(self.FIELD_NAMES) + (
                ["gamma"] if self.optimize_gamma else []
            )
            audit_columns = {
                **field_columns,
                "gamma": gamma_columns,
            }
            for field_name in audit_field_names:
                member_mask = torch.isin(
                    columns_evaluated, audit_columns[field_name]
                )
                if not bool(torch.any(member_mask)):
                    continue
                selected_columns = columns_evaluated[member_mask]
                values = column_l2[member_mask]
                values_by_field[field_name] = values
                rms_by_field[field_name] = column_rms[member_mask]
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
                rms_values = rms_by_field[field_name]
                sensitivity[field_name] = {
                    "mean_l2_per_normalized_range": float(values.mean()),
                    "median_l2_per_normalized_range": float(values.median()),
                    "max_l2_per_normalized_range": float(values.max()),
                    "relative_mean": float(means[field_name]) / max_mean,
                    "relative_median": (
                        float(medians[field_name]) / max_median
                    ),
                    "mean_rms_per_pixel_per_normalized_range": float(
                        rms_values.mean()
                    ),
                    "median_rms_per_pixel_per_normalized_range": float(
                        rms_values.median()
                    ),
                    "max_rms_per_pixel_per_normalized_range": float(
                        rms_values.max()
                    ),
                }
                if self.observability_noise_rmse is not None:
                    snr = rms_values / self.observability_noise_rmse
                    sensitivity[field_name].update(
                        {
                            "mean_snr": float(snr.mean()),
                            "median_snr": float(snr.median()),
                            "max_snr": float(snr.max()),
                        }
                    )
            return sensitivity

        def columns_for_fields(field_names: Sequence[str]) -> torch.Tensor:
            if not field_names:
                return torch.empty(
                    0, dtype=torch.long, device=decision.device
                )
            sources = {**field_columns, "gamma": gamma_columns}
            return torch.cat([sources[name] for name in field_names])

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
            if self.optimize_gamma and "gamma" not in active:
                fixed[gamma_columns] = initial_decision[gamma_columns]
            return fixed

        audit_sensitivity: Dict[str, Dict[str, float]] = {}
        node_gate_diagnostics: Dict[str, Dict[str, Any]] = {}
        joint_identifiability: Dict[str, Any] = {}
        joint_pruning: Dict[str, Any] = {}
        selected_columns_by_field: Dict[str, torch.Tensor] = {}
        available_field_names = list(self.FIELD_NAMES) + (
            ["gamma"] if self.optimize_gamma else []
        )
        available_columns = {
            **field_columns,
            "gamma": gamma_columns,
        }
        if self.field_mode == "auto":
            with torch.no_grad():
                audit_residual = residual_fn(decision)
            print(
                "[OBSERVABILITY] auditing H/alpha/beta"
                + ("/gamma" if self.optimize_gamma else "")
                + " before gated LM",
                flush=True,
            )
            all_audit_columns = all_posture_columns
            if self.optimize_gamma:
                all_audit_columns = torch.cat(
                    [all_audit_columns, gamma_columns]
                )
            if self.jacobian_mode == "finite_difference":
                audit_jacobian = finite_difference_jacobian(
                    decision, audit_residual, all_audit_columns
                )
            else:
                audit_jacobian = autograd_jacobian(
                    decision, all_audit_columns
                )
            audit_sensitivity = summarize_sensitivity(
                audit_jacobian, all_audit_columns, decision
            )
            if self.observability_gate_mode == "node_snr":
                _, audit_column_rms = normalized_column_sensitivity(
                    audit_jacobian, all_audit_columns, decision
                )
                active_fields = []
                for field_name in available_field_names:
                    member_mask = torch.isin(
                        all_audit_columns,
                        available_columns[field_name],
                    )
                    field_columns_evaluated = all_audit_columns[member_mask]
                    field_snr = (
                        audit_column_rms[member_mask]
                        / float(self.observability_noise_rmse)
                    )
                    selected = field_columns_evaluated[
                        field_snr >= self.min_observability_snr
                    ]
                    selected_columns_by_field[field_name] = selected
                    if int(selected.numel()) > 0:
                        active_fields.append(field_name)
                    node_gate_diagnostics[field_name] = {
                        "evaluated_nodes": int(
                            field_columns_evaluated.numel()
                        ),
                        "selected_nodes": int(selected.numel()),
                        "selected_fraction": float(
                            selected.numel()
                            / max(field_columns_evaluated.numel(), 1)
                        ),
                        "min_snr": float(field_snr.min()),
                        "median_snr": float(field_snr.median()),
                        "max_snr": float(field_snr.max()),
                        "selected_decision_columns": [
                            int(value) for value in selected.cpu().tolist()
                        ],
                    }
                selected_audit_mask = torch.zeros(
                    len(all_audit_columns),
                    dtype=torch.bool,
                    device=decision.device,
                )
                for selected in selected_columns_by_field.values():
                    selected_audit_mask |= torch.isin(
                        all_audit_columns, selected
                    )
                selected_audit_columns = all_audit_columns[
                    selected_audit_mask
                ]
                normalized_audit_jacobian = (
                    full_range_normalized_pixel_jacobian(
                        audit_jacobian, all_audit_columns, decision
                    )[:, selected_audit_mask]
                )
                field_positions = {
                    field_name: torch.nonzero(
                        torch.isin(
                            selected_audit_columns,
                            selected_columns_by_field[field_name],
                        ),
                        as_tuple=False,
                    ).flatten()
                    for field_name in available_field_names
                }
                joint_identifiability = summarize_joint_identifiability(
                    normalized_audit_jacobian,
                    field_positions,
                )
                if (
                    self.joint_gate_action == "prune"
                    and not joint_identifiability[
                        "jointly_identifiable"
                    ]
                ):
                    joint_pruning = prune_jointly_aliased_fields(
                        normalized_audit_jacobian,
                        field_positions,
                        {
                            name: float(
                                audit_sensitivity.get(name, {}).get(
                                    "median_snr", 0.0
                                )
                            )
                            for name in active_fields
                        },
                    )
                    retained = set(joint_pruning["kept_fields"])
                    removed_by_name = {
                        item["field"]: item
                        for item in joint_pruning["removed_fields"]
                    }
                    for field_name in list(active_fields):
                        if field_name in retained:
                            continue
                        selected_columns_by_field[field_name] = (
                            torch.empty(
                                0,
                                dtype=torch.long,
                                device=decision.device,
                            )
                        )
                        node_gate_diagnostics[field_name][
                            "snr_selected_nodes_before_joint_pruning"
                        ] = node_gate_diagnostics[field_name][
                            "selected_nodes"
                        ]
                        node_gate_diagnostics[field_name][
                            "selected_nodes"
                        ] = 0
                        node_gate_diagnostics[field_name][
                            "selected_fraction"
                        ] = 0.0
                        node_gate_diagnostics[field_name][
                            "selected_decision_columns"
                        ] = []
                        node_gate_diagnostics[field_name][
                            "pruned_by_joint_gate"
                        ] = True
                        node_gate_diagnostics[field_name][
                            "joint_pruning_reason"
                        ] = removed_by_name[field_name]
                    active_fields = [
                        name for name in active_fields if name in retained
                    ]
                    joint_identifiability = joint_pruning[
                        "final_joint_identifiability"
                    ]
                    print(
                        "[JOINT PRUNE] kept="
                        f"{','.join(active_fields) or 'none'}, removed="
                        + ",".join(removed_by_name)
                        + ", passed="
                        f"{joint_pruning['passed']}",
                        flush=True,
                    )
                counts = ",".join(
                    f"{name}:{node_gate_diagnostics[name]['selected_nodes']}/"
                    f"{node_gate_diagnostics[name]['evaluated_nodes']}"
                    for name in available_field_names
                )
                print(
                    "[OBSERVABILITY] node_snr selected="
                    f"{counts}, noise_rmse="
                    f"{self.observability_noise_rmse:.6f}, "
                    f"min_snr={self.min_observability_snr:.3f}",
                    flush=True,
                )
                condition = joint_identifiability["condition_number"]
                correlation = joint_identifiability[
                    "max_field_canonical_correlation"
                ]
                condition_text = (
                    "n/a" if condition is None else f"{condition:.3e}"
                )
                correlation_text = (
                    "n/a" if correlation is None else f"{correlation:.6f}"
                )
                print(
                    "[JOINT AUDIT] rank="
                    f"{joint_identifiability['effective_rank']}/"
                    f"{joint_identifiability['selected_columns']}, "
                    f"condition={condition_text}, "
                    f"max_field_correlation={correlation_text}, "
                    "identifiable="
                    f"{joint_identifiability['jointly_identifiable']}",
                    flush=True,
                )
            else:
                active_fields = ["H"]
                candidates = ["alpha", "beta"] + (
                    ["gamma"] if self.optimize_gamma else []
                )
                for field_name in candidates:
                    relative = audit_sensitivity[field_name][
                        "relative_median"
                    ]
                    if relative >= self.min_relative_median_sensitivity:
                        active_fields.append(field_name)
                selected_columns_by_field = {
                    name: available_columns[name]
                    for name in active_fields
                }
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
            active_fields = list(self.FIELD_NAMES) + (
                ["gamma"] if self.optimize_gamma else []
            )
        if not selected_columns_by_field:
            selected_columns_by_field = {
                name: available_columns[name] for name in active_fields
            }
        decision = fix_inactive_fields(decision, active_fields)
        gated_active_fields = set(active_fields)
        selected_parts = [
            selected_columns_by_field[name]
            for name in active_fields
            if int(selected_columns_by_field[name].numel()) > 0
        ]
        active_columns = (
            torch.cat(selected_parts)
            if selected_parts
            else torch.empty(
                0, dtype=torch.long, device=decision.device
            )
        )
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
        effective_max_steps = (
            int(max_steps) if int(active_columns.numel()) > 0 else 0
        )
        message = (
            "Maximum steps reached"
            if effective_max_steps > 0 or int(max_steps) == 0
            else "No observable decision variables passed the gate"
        )
        if int(max_steps) > 0 and effective_max_steps == 0:
            print(
                "[OBSERVABILITY] no decision columns passed; skipping LM",
                flush=True,
            )
        completed_steps = 0
        last_jacobian = None
        last_decision = decision.detach()
        last_columns = active_columns

        for step in range(1, effective_max_steps + 1):
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
            optimized_gamma = initial_gamma_tensor
            if self.optimize_gamma:
                optimized_gamma, _ = self._decode_gamma(
                    best_decision[gamma_start:gamma_stop],
                    matrices,
                    point_indices,
                    len(xy),
                )
                if "gamma" not in active_fields:
                    optimized_gamma = initial_gamma_tensor
            if self.optimize_xy:
                xy_offsets, _ = self._decode_xy_offsets(
                    best_decision[gamma_stop:],
                    matrices,
                    point_indices,
                    len(xy),
                )
                optimized_xy = xy + xy_offsets
            rendered = (
                self.renderer(
                    optimized_xy, posture, ids, optimized_gamma
                )
                if self.optimize_gamma
                else self.renderer(optimized_xy, posture, ids)
            )[0, 0]
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
                "gamma_enabled": self.optimize_gamma,
                "gamma_max_abs_rad": self.gamma_max_abs_rad,
                "gamma_smoothness_weight": self.gamma_smoothness_weight,
                "gamma_prior_weight": self.gamma_prior_weight,
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
                "gate_mode": self.observability_gate_mode,
                "joint_gate_action": self.joint_gate_action,
                "min_relative_median_sensitivity": (
                    self.min_relative_median_sensitivity
                ),
                "noise_rmse": self.observability_noise_rmse,
                "noise_rmse_source": (
                    "explicit_or_checkpoint_validation_plain_mse"
                    if self.observability_noise_rmse is not None
                    else None
                ),
                "min_observability_snr": self.min_observability_snr,
                "optimized_fields": (
                    ["x", "y"] if self.optimize_xy else []
                )
                + list(active_fields),
                "fixed_fields": [
                    name for name in self.FIELD_NAMES
                    if name not in active_fields
                ]
                + (
                    ["gamma"]
                    if self.optimize_gamma and "gamma" not in active_fields
                    else []
                )
                + ([] if self.optimize_xy else ["x", "y"]),
                "partially_optimized_fields": [
                    name
                    for name in active_fields
                    if int(selected_columns_by_field[name].numel())
                    < int(available_columns[name].numel())
                ],
                "selected_node_columns": node_gate_diagnostics,
                "initial_image_jacobian_sensitivity": audit_sensitivity,
                "joint_identifiability": joint_identifiability,
                "joint_pruning": joint_pruning,
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
        joint_audit_applied = bool(joint_identifiability)
        joint_pose_identifiable = bool(
            joint_identifiability.get("jointly_identifiable", True)
        )
        joint_pruned_field_names = {
            item["field"]
            for item in joint_pruning.get("removed_fields", [])
        }
        for field_name in self.FIELD_NAMES:
            optimized = field_name in active_fields
            selected_node_count = int(
                selected_columns_by_field.get(
                    field_name,
                    torch.empty(0, device=decision.device),
                ).numel()
            )
            available_node_count = int(
                available_columns[field_name].numel()
            )
            boundary = diagnostics["bound_fraction_within_1pct"][field_name]
            boundary_total = boundary["lower"] + boundary["upper"]
            relative = source_sensitivity.get(field_name, {}).get(
                "relative_median"
            )
            if not optimized:
                confidence = "low"
                reason = (
                    "fixed_by_joint_identifiability_pruning"
                    if field_name in joint_pruned_field_names
                    else (
                        "fixed_for_xy_only_ablation"
                        if self.field_mode == "xy_only"
                        else "fixed_below_observability_threshold"
                    )
                )
            elif joint_audit_applied and not joint_pose_identifiable:
                confidence = "low"
                reason = "optimized_but_jointly_nonidentifiable"
            elif boundary_total > 0.25:
                confidence = "low"
                reason = "optimized_but_boundary_saturated"
            elif field_name == "H":
                confidence = "medium_simulation"
                reason = (
                    "partially_optimized_above_snr"
                    if selected_node_count < available_node_count
                    else "optimized_observable_simulation_parameter"
                )
            else:
                confidence = "medium_simulation"
                reason = (
                    "partially_optimized_above_snr"
                    if selected_node_count < available_node_count
                    else "optimized_above_observability_threshold"
                )
            decisions[field_name] = {
                "optimized": optimized,
                "source": (
                    (
                        "lm_optimized_selected_nodes"
                        if selected_node_count < available_node_count
                        else "lm_optimized"
                    )
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
                "initial_median_snr": source_sensitivity.get(
                    field_name, {}
                ).get("median_snr"),
                "optimized_nodes": selected_node_count,
                "available_nodes": available_node_count,
                "optimized_node_fraction": (
                    selected_node_count / max(available_node_count, 1)
                ),
                "jointly_identifiable": (
                    joint_pose_identifiable
                    if joint_audit_applied and optimized
                    else None
                ),
                "boundary_fraction": boundary_total if optimized else None,
                "fixed_value_on_physical_boundary": (
                    None if optimized else boundary_total > 0.0
                ),
            }
        gamma_optimized = "gamma" in active_fields
        gamma_selected_nodes = int(
            selected_columns_by_field.get(
                "gamma", torch.empty(0, device=decision.device)
            ).numel()
        )
        gamma_available_nodes = int(gamma_columns.numel())
        gamma_boundary = float(
            np.mean(
                np.abs(
                    optimized_gamma.detach().cpu().numpy()
                    / self.gamma_max_abs_rad
                )
                >= 0.99
            )
        )
        gamma_relative = source_sensitivity.get("gamma", {}).get(
            "relative_median"
        )
        decisions["gamma"] = {
            "optimized": gamma_optimized,
            "source": (
                (
                    "lm_optimized_selected_nodes"
                    if gamma_selected_nodes < gamma_available_nodes
                    else "lm_optimized"
                )
                if gamma_optimized
                else (
                    "initial_pose_csv"
                    if initial_posture_source == "initial_pose_csv"
                    else "initial_default"
                )
            ),
            "confidence": (
                "medium_simulation"
                if (
                    gamma_optimized
                    and gamma_boundary <= 0.25
                    and (
                        not joint_audit_applied
                        or joint_pose_identifiable
                    )
                )
                else "low"
            ),
            "reason": (
                "optimized_but_jointly_nonidentifiable"
                if (
                    gamma_optimized
                    and joint_audit_applied
                    and not joint_pose_identifiable
                )
                else
                (
                    "partially_optimized_above_snr"
                    if gamma_selected_nodes < gamma_available_nodes
                    else "optimized_above_observability_threshold"
                )
                if gamma_optimized and gamma_boundary <= 0.25
                else (
                    "optimized_but_boundary_saturated"
                    if gamma_optimized
                    else (
                        "fixed_below_observability_threshold"
                        if (
                            self.optimize_gamma
                            and "gamma" not in joint_pruned_field_names
                        )
                        else (
                            "fixed_by_joint_identifiability_pruning"
                            if "gamma" in joint_pruned_field_names
                            else "gamma_channel_disabled"
                        )
                    )
                )
            ),
            "initial_relative_median_sensitivity": gamma_relative,
            "initial_median_snr": source_sensitivity.get(
                "gamma", {}
            ).get("median_snr"),
            "optimized_nodes": gamma_selected_nodes,
            "available_nodes": gamma_available_nodes,
            "optimized_node_fraction": (
                gamma_selected_nodes / max(gamma_available_nodes, 1)
                if gamma_available_nodes
                else 0.0
            ),
            "jointly_identifiable": (
                joint_pose_identifiable
                if joint_audit_applied and gamma_optimized
                else None
            ),
            "boundary_fraction": gamma_boundary if gamma_optimized else None,
            "fixed_value_on_physical_boundary": (
                None if gamma_optimized else gamma_boundary > 0.0
            ),
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
            gamma=optimized_gamma.cpu().numpy(),
        )
