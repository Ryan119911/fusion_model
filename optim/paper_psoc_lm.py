"""PSOC/CGL parameterization and autograd LM for paper-pose inversion."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

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


class PaperPSOCLM:
    """Optimize H/alpha/beta CGL nodes while holding x/y exactly fixed."""

    def __init__(
        self,
        renderer: PaperFusionRenderer,
        order: int = 3,
        optimization_size: int = 16,
        smoothness_weight: float = 0.02,
        posture_prior_weight: float = 0.001,
        render_stride: int = 1,
    ):
        if order < 1:
            raise ValueError("order must be >= 1")
        if optimization_size < 8:
            raise ValueError("optimization_size must be >= 8")
        self.renderer = renderer
        self.order = int(order)
        self.optimization_size = int(optimization_size)
        self.smoothness_weight = float(smoothness_weight)
        self.posture_prior_weight = float(posture_prior_weight)
        self.render_stride = max(int(render_stride), 1)

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
        normalized_nodes = torch.sigmoid(
            decision.view(len(matrices), 3, node_count)
        )
        normalized_points = torch.empty(
            (point_count, 3), dtype=decision.dtype, device=decision.device
        )
        for stroke_index, (matrix, indices) in enumerate(
            zip(matrices, point_indices)
        ):
            values = matrix.to(dtype=decision.dtype) @ normalized_nodes[
                stroke_index
            ].T
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
            if self.smoothness_weight > 0:
                for stroke_nodes in nodes:
                    residuals.append(
                        np.sqrt(self.smoothness_weight)
                        * (stroke_nodes[:, 1:] - stroke_nodes[:, :-1]).flatten()
                    )
            if self.posture_prior_weight > 0:
                residuals.append(
                    np.sqrt(self.posture_prior_weight)
                    * (nodes - prior).flatten()
                )
            return torch.cat(residuals)

        def evaluate_cost(vector: torch.Tensor) -> float:
            with torch.no_grad():
                residual = residual_fn(vector)
                return 0.5 * float(torch.dot(residual, residual).item())

        def build_jacobian(vector: torch.Tensor) -> torch.Tensor:
            """Build J column-wise to keep GPU memory bounded."""
            columns = []
            try:
                for column in range(vector.numel()):
                    tangent = torch.zeros_like(vector)
                    tangent[column] = 1.0
                    _, derivative = torch.func.jvp(
                        residual_fn, (vector,), (tangent,)
                    )
                    columns.append(derivative)
                return torch.stack(columns, dim=1)
            except (RuntimeError, AttributeError):
                # Reverse-mode fallback is slower but supports older PyTorch ops.
                return torch.autograd.functional.jacobian(
                    residual_fn, vector, vectorize=False
                )

        current_cost = evaluate_cost(decision)
        initial_cost = current_cost
        mu = float(damping)
        history = {"cost": [current_cost], "damping": [mu]}
        success = False
        message = "Maximum steps reached"
        completed_steps = 0

        for step in range(1, int(max_steps) + 1):
            decision = decision.detach().requires_grad_(True)
            residual = residual_fn(decision)
            jacobian = build_jacobian(decision).detach()
            gradient = jacobian.T @ residual.detach()
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
        return PaperLMResult(
            posture=posture.cpu().numpy(),
            rendered_image=rendered.cpu().numpy(),
            success=success,
            steps=completed_steps,
            initial_cost=initial_cost,
            final_cost=current_cost,
            message=message,
            history=history,
        )
