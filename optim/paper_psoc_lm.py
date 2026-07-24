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


@dataclass
class PaperLMResult:
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
    """Optimize H/alpha/beta CGL nodes while holding x/y exactly fixed."""

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
        self.jacobian_mode = jacobian_mode
        self.finite_difference_eps = float(finite_difference_eps)

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
    ) -> tuple[List[torch.Tensor], List[np.ndarray]]:
        matrices: List[torch.Tensor] = []
        point_indices: List[np.ndarray] = []
        for stroke_id in np.unique(stroke_ids):
            indices = np.flatnonzero(stroke_ids == stroke_id)
            point_indices.append(indices)
            matrix = cgl_interpolation_matrix(self.order, len(indices))
            matrices.append(torch.as_tensor(matrix, device=self.device))
        return matrices, point_indices

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
        matrices, point_indices = self._build_layout(stroke_ids)
        render_indices_np = self._render_indices(point_indices)
        render_indices = torch.as_tensor(
            render_indices_np, dtype=torch.long, device=self.device
        )

        initial = np.asarray(
            [initial_h_mm, initial_alpha_rad, initial_beta_rad],
            dtype=np.float32,
        )
        if np.any(initial < PAPER_POSTURE_MIN) or np.any(
            initial > PAPER_POSTURE_MAX
        ):
            raise ValueError(
                "Initial posture is outside H=11-20 mm, alpha=0-10 deg, "
                "beta=0-5 deg"
            )
        normalized = (initial - PAPER_POSTURE_MIN) / (
            PAPER_POSTURE_MAX - PAPER_POSTURE_MIN
        )
        # A value exactly on a bound has zero useful logistic derivative.
        normalized = np.clip(normalized, 0.02, 0.98)
        logits = np.log(normalized / (1.0 - normalized))
        decision = torch.as_tensor(
            np.tile(
                logits[None, :, None],
                (len(matrices), 1, self.order + 1),
            ).reshape(-1),
            dtype=torch.float32,
            device=self.device,
        )
        prior = torch.sigmoid(decision.detach()).view(
            len(matrices), 3, self.order + 1
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
                vector, matrices, point_indices, len(xy)
            )
            rendered = self.renderer(
                xy[render_indices],
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
                    ).flatten()
                )
            if bool(torch.any(posture_prior_weights > 0)):
                residuals.append(
                    (
                        torch.sqrt(posture_prior_weights) * (nodes - prior)
                    ).flatten()
                )
            return torch.cat(residuals)

        def evaluate_cost(vector: torch.Tensor) -> float:
            with torch.no_grad():
                residual = residual_fn(vector)
                return 0.5 * float(torch.dot(residual, residual).item())

        def finite_difference_jacobian(
            vector: torch.Tensor, base_residual: torch.Tensor
        ) -> torch.Tensor:
            """Memory-bounded numerical Jacobian, matching the Wang LM flow."""
            columns = []
            with torch.no_grad():
                for column in range(vector.numel()):
                    step = self.finite_difference_eps * (
                        1.0 + abs(float(vector[column]))
                    )
                    trial = vector.clone()
                    trial[column] += step
                    derivative = (
                        residual_fn(trial) - base_residual
                    ) / step
                    columns.append(derivative)
                    if (column + 1) % 10 == 0 or column + 1 == vector.numel():
                        print(
                            f"[JACOBIAN] column {column + 1}/"
                            f"{vector.numel()}",
                            flush=True,
                        )
            return torch.stack(columns, dim=1)

        def autograd_jacobian(vector: torch.Tensor) -> torch.Tensor:
            """Optional high-memory path for GPUs with substantially more VRAM."""
            differentiable = vector.detach().requires_grad_(True)
            return torch.autograd.functional.jacobian(
                residual_fn, differentiable, vectorize=False
            ).detach()

        current_cost = evaluate_cost(decision)
        initial_cost = current_cost
        mu = float(damping)
        history = {"cost": [current_cost], "damping": [mu]}
        success = False
        message = "Maximum steps reached"
        completed_steps = 0
        last_jacobian = None
        last_decision = decision.detach()

        for step in range(1, int(max_steps) + 1):
            decision = decision.detach()
            with torch.no_grad():
                residual = residual_fn(decision)
            print(
                f"[JACOBIAN {step:03d}] mode={self.jacobian_mode}, "
                f"variables={decision.numel()}, residuals={residual.numel()}",
                flush=True,
            )
            if self.jacobian_mode == "finite_difference":
                jacobian = finite_difference_jacobian(decision, residual)
            else:
                jacobian = autograd_jacobian(decision)
            last_jacobian = jacobian
            last_decision = decision.detach()
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
                float(torch.linalg.vector_norm(decision.detach())) + 1e-5
            ):
                success = True
                message = "Step tolerance reached"
                completed_steps = step - 1
                break
            trial = decision.detach() + delta.detach()
            trial_cost = evaluate_cost(trial)
            if np.isfinite(trial_cost) and trial_cost < current_cost:
                improvement = current_cost - trial_cost
                decision = trial
                current_cost = trial_cost
                mu = max(mu * 0.3, 1e-8)
                if improvement < 1e-7 * (1.0 + current_cost):
                    success = True
                    message = "Function tolerance reached"
                    completed_steps = step
                    history["cost"].append(current_cost)
                    history["damping"].append(mu)
                    break
            else:
                decision = decision.detach()
                mu = min(mu * 10.0, 1e8)
            completed_steps = step
            history["cost"].append(current_cost)
            history["damping"].append(mu)
            print(
                f"[LM {step:03d}] cost={current_cost:.6f}, damping={mu:.6g}",
                flush=True,
            )

        with torch.no_grad():
            posture, _ = self._decode(
                decision.detach(), matrices, point_indices, len(xy)
            )
            rendered = self.renderer(xy, posture, ids)[0, 0]
        diagnostics: Dict[str, Any] = {
            "regularization": {
                "field_order": ["H", "alpha", "beta"],
                "smoothness_weights": self.smoothness_weights.tolist(),
                "posture_prior_weights": self.posture_prior_weights.tolist(),
            }
        }
        if last_jacobian is not None:
            # Use only image residual rows. Dividing out the sigmoid derivative
            # reports sensitivity per unit of normalized physical range instead
            # of sensitivity to the unconstrained optimization logit.
            pixel_rows = self.optimization_size * self.optimization_size
            pixel_jacobian = last_jacobian[:pixel_rows]
            column_norms = torch.linalg.vector_norm(
                pixel_jacobian, dim=0
            ).view(len(matrices), 3, self.order + 1)
            normalized_nodes = torch.sigmoid(last_decision).view(
                len(matrices), 3, self.order + 1
            )
            sigmoid_slope = (
                normalized_nodes * (1.0 - normalized_nodes)
            ).clamp_min(1e-4)
            normalized_sensitivity = column_norms / sigmoid_slope
            field_means = normalized_sensitivity.mean(dim=(0, 2))
            field_medians = torch.stack(
                [
                    normalized_sensitivity[:, field_index, :].median()
                    for field_index in range(3)
                ]
            )
            max_mean = float(field_means.max().clamp_min(1e-12))
            max_median = float(field_medians.max().clamp_min(1e-12))
            sensitivity = {}
            for field_index, field_name in enumerate(("H", "alpha", "beta")):
                values = normalized_sensitivity[:, field_index, :]
                sensitivity[field_name] = {
                    "mean_l2_per_normalized_range": float(values.mean()),
                    "median_l2_per_normalized_range": float(values.median()),
                    "max_l2_per_normalized_range": float(values.max()),
                    "relative_mean": float(field_means[field_index]) / max_mean,
                    "relative_median": (
                        float(field_medians[field_index]) / max_median
                    ),
                }
            diagnostics["image_jacobian_sensitivity"] = sensitivity

        posture_np = posture.cpu().numpy()
        normalized_posture = (posture_np - PAPER_POSTURE_MIN) / (
            PAPER_POSTURE_MAX - PAPER_POSTURE_MIN
        )
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
        return PaperLMResult(
            posture=posture_np,
            rendered_image=rendered.cpu().numpy(),
            success=success,
            steps=completed_steps,
            initial_cost=initial_cost,
            final_cost=current_cost,
            message=message,
            history=history,
            diagnostics=diagnostics,
        )
