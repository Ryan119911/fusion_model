from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from models.fusion_renderer import FusionRenderer
from models.geometry import resample_stroke, trajectory_bounds
from optim.chebyshev import fit_nodes_from_sequence, parameterize_1d
from optim.lm import LMResult, lm_solve
from utils.feature_schema import STROKE10_POSE_V2
from utils.image_preprocessing import DEFAULT_CANVAS_PADDING, load_character_image
from utils.types import CharacterTrajectory, StrokeTrajectory, TrajectoryPoint


@dataclass
class TrajectoryOptimizationResult:
    order: int
    lm_result: LMResult
    optimized_sample: CharacterTrajectory
    target_image: np.ndarray
    rendered_image: np.ndarray
    initial_image: np.ndarray
    initial_score: float
    final_score: float
    optimize_angles: bool


def load_target_image(path: str, image_size: int = 128) -> np.ndarray:
    return load_character_image(path, image_size, DEFAULT_CANVAS_PADDING)


def _stroke_arrays(stroke: StrokeTrajectory) -> Tuple[np.ndarray, np.ndarray]:
    points = stroke.sorted_points()
    xyz = np.asarray([[p.x, p.y, p.z] for p in points], dtype=np.float64)
    angles = np.asarray(
        [[p.alpha, p.beta, p.gamma] for p in points], dtype=np.float64
    )
    if len(angles):
        angles = np.unwrap(angles, axis=0)
    return xyz, angles


def build_decision(
    sample: CharacterTrajectory,
    order: int,
    optimize_angles: bool,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    vectors: List[np.ndarray] = []
    specifications: List[Dict[str, Any]] = []
    cursor = 0
    for stroke in sample.sorted_strokes():
        xyz, angles = _stroke_arrays(stroke)
        if len(xyz) == 0:
            continue
        components = [xyz[:, 0], xyz[:, 1], xyz[:, 2]]
        if optimize_angles:
            components.extend([angles[:, 0], angles[:, 1], angles[:, 2]])
        nodes = [fit_nodes_from_sequence(values, order) for values in components]
        vector = np.concatenate(nodes)
        specifications.append(
            {
                "stroke": stroke,
                "start": cursor,
                "end": cursor + len(vector),
                "components": len(components),
                "nodes_per_component": order + 1,
            }
        )
        vectors.append(vector)
        cursor += len(vector)
    if not vectors:
        raise ValueError("Template trajectory is empty")
    return np.concatenate(vectors).astype(np.float64), specifications


def decode_decision(
    template: CharacterTrajectory,
    decision: np.ndarray,
    specifications: List[Dict[str, Any]],
    render_samples: int,
    optimize_angles: bool,
    bounds: Dict[str, Tuple[float, float]],
) -> CharacterTrajectory:
    strokes: List[StrokeTrajectory] = []
    for specification in specifications:
        base = resample_stroke(specification["stroke"], render_samples)
        base_points = base.sorted_points()
        segment = decision[specification["start"]:specification["end"]]
        count = specification["nodes_per_component"]
        components = [
            parameterize_1d(segment[index * count:(index + 1) * count], render_samples)
            for index in range(specification["components"])
        ]
        components[0] = np.clip(components[0], *bounds["x"])
        components[1] = np.clip(components[1], *bounds["y"])
        components[2] = np.clip(components[2], *bounds["z"])
        if optimize_angles:
            for index in range(3, 6):
                components[index] = np.clip(
                    components[index], *bounds[f"angle_{index - 3}"]
                )
        points: List[TrajectoryPoint] = []
        for index, base_point in enumerate(base_points):
            angles = (
                [components[3][index], components[4][index], components[5][index]]
                if optimize_angles
                else [base_point.alpha, base_point.beta, base_point.gamma]
            )
            points.append(
                TrajectoryPoint(
                    stroke_id=base.stroke_id,
                    point_id=index,
                    x=float(components[0][index]),
                    y=float(components[1][index]),
                    z=float(components[2][index]),
                    alpha=float(angles[0]),
                    beta=float(angles[1]),
                    gamma=float(angles[2]),
                    state=base_point.state,
                    timestamp=base_point.timestamp,
                )
            )
        strokes.append(StrokeTrajectory(stroke_id=base.stroke_id, points=points))
    return CharacterTrajectory(
        character=template.character, strokes=strokes, meta=dict(template.meta)
    )


def _score(rendered: np.ndarray, target: np.ndarray) -> float:
    foreground = target > 0.1
    global_mae = float(np.abs(rendered - target).mean())
    foreground_mae = (
        float(np.abs(rendered[foreground] - target[foreground]).mean())
        if np.any(foreground)
        else global_mae
    )
    ink_gap = abs(float(rendered.mean()) - float(target.mean()))
    return 0.25 * global_mae + foreground_mae + 0.5 * ink_gap


class TrajectoryOptimizer:
    def __init__(
        self,
        renderer: FusionRenderer,
        render_samples: int = 128,
        jacobian_epsilon: float = 1e-5,
        xy_margin_ratio: float = 0.03,
        z_margin: float = 0.25,
        angle_margin_radians: float = 0.35,
        xyz_reg_weight: float = 1e-4,
        z_reg_weight: float = 1e-2,
        angle_reg_weight: float = 1e-2,
    ):
        self.renderer = renderer
        self.render_samples = render_samples
        self.jacobian_epsilon = jacobian_epsilon
        self.xy_margin_ratio = xy_margin_ratio
        self.z_margin = z_margin
        self.angle_margin_radians = angle_margin_radians
        self.xyz_reg_weight = xyz_reg_weight
        self.z_reg_weight = z_reg_weight
        self.angle_reg_weight = angle_reg_weight

    def _bounds(
        self, sample: CharacterTrajectory
    ) -> Tuple[Tuple[float, float, float, float], Dict[str, Tuple[float, float]]]:
        fixed = trajectory_bounds(sample)
        min_x, max_x, min_y, max_y = fixed
        width = max(max_x - min_x, 1e-6)
        height = max(max_y - min_y, 1e-6)
        points = sample.all_points()
        z_values = np.asarray([point.z for point in points])
        angle_values = np.asarray(
            [[point.alpha, point.beta, point.gamma] for point in points]
        )
        bounds = {
            "x": (
                min_x - self.xy_margin_ratio * width,
                max_x + self.xy_margin_ratio * width,
            ),
            "y": (
                min_y - self.xy_margin_ratio * height,
                max_y + self.xy_margin_ratio * height,
            ),
            "z": (
                float(z_values.min()) - self.z_margin,
                float(z_values.max()) + self.z_margin,
            ),
        }
        for index in range(3):
            bounds[f"angle_{index}"] = (
                float(angle_values[:, index].min()) - self.angle_margin_radians,
                float(angle_values[:, index].max()) + self.angle_margin_radians,
            )
        return fixed, bounds

    def optimize(
        self,
        template: CharacterTrajectory,
        target_image: np.ndarray,
        order: int = 4,
        damping: float = 5e-2,
        max_steps: int = 20,
        optimize_angles: bool = False,
    ) -> TrajectoryOptimizationResult:
        if optimize_angles and self.renderer.feature_schema != STROKE10_POSE_V2:
            raise ValueError(
                "Angle optimization requires feature_schema=stroke10_pose_v2; "
                "legacy stroke10_v1 does not consume trajectory alpha/beta."
            )
        target = np.asarray(target_image, dtype=np.float64)
        fixed_bounds, bounds = self._bounds(template)
        decision0, specifications = build_decision(
            template, order, optimize_angles
        )
        initial_sample = decode_decision(
            template, decision0, specifications, self.render_samples,
            optimize_angles, bounds,
        )
        initial_image = np.asarray(
            self.renderer.render_character(
                initial_sample, fixed_bounds=fixed_bounds
            )["character_image"],
            dtype=np.float64,
        )
        foreground = target > 0.1
        pixel_weights = np.sqrt(1.0 + 4.0 * target)

        component_count = 6 if optimize_angles else 3
        nodes_per_stroke = order + 1

        def residual(decision: np.ndarray) -> np.ndarray:
            sample = decode_decision(
                template, decision, specifications, self.render_samples,
                optimize_angles, bounds,
            )
            rendered = np.asarray(
                self.renderer.render_character(
                    sample, fixed_bounds=fixed_bounds
                )["character_image"],
                dtype=np.float64,
            )
            pixel = ((rendered - target) * pixel_weights).reshape(-1)
            under = (
                0.5 * np.maximum(target[foreground] - rendered[foreground], 0.0)
                if np.any(foreground)
                else np.zeros((0,), dtype=np.float64)
            )
            regularization: List[np.ndarray] = []
            for specification in specifications:
                start = specification["start"]
                for component in range(component_count):
                    left = start + component * nodes_per_stroke
                    right = left + nodes_per_stroke
                    weight = (
                        self.z_reg_weight
                        if component == 2
                        else self.angle_reg_weight
                        if component >= 3
                        else self.xyz_reg_weight
                    )
                    regularization.append(
                        np.sqrt(weight) * (decision[left:right] - decision0[left:right])
                    )
            ink = np.asarray(
                [2.0 * (float(rendered.mean()) - float(target.mean()))],
                dtype=np.float64,
            )
            return np.concatenate([pixel, under, ink, *regularization])

        lm_result = lm_solve(
            residual,
            decision0,
            damping=damping,
            max_steps=max_steps,
            eps_jac=self.jacobian_epsilon,
        )
        candidate_sample = decode_decision(
            template, lm_result.x, specifications, self.render_samples,
            optimize_angles, bounds,
        )
        candidate_image = np.asarray(
            self.renderer.render_character(
                candidate_sample, fixed_bounds=fixed_bounds
            )["character_image"],
            dtype=np.float64,
        )
        initial_score = _score(initial_image, target)
        candidate_score = _score(candidate_image, target)
        if candidate_score < initial_score:
            optimized_sample = candidate_sample
            rendered_image = candidate_image
            final_score = candidate_score
        else:
            optimized_sample = initial_sample
            rendered_image = initial_image
            final_score = initial_score
        return TrajectoryOptimizationResult(
            order=order,
            lm_result=lm_result,
            optimized_sample=optimized_sample,
            target_image=target.astype(np.float32),
            rendered_image=rendered_image.astype(np.float32),
            initial_image=initial_image.astype(np.float32),
            initial_score=initial_score,
            final_score=final_score,
            optimize_angles=optimize_angles,
        )
