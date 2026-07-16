from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from utils.types import DynamicBrushState, TrajectoryPoint


LEGACY_5D = "stroke5_v0"
STROKE10_V1 = "stroke10_v1"
STROKE10_POSE_V2 = "stroke10_pose_v2"


@dataclass(frozen=True)
class FeatureSchema:
    name: str
    fields: Tuple[str, ...]
    description: str

    @property
    def input_dim(self) -> int:
        return len(self.fields)


SCHEMAS: Dict[str, FeatureSchema] = {
    LEGACY_5D: FeatureSchema(
        LEGACY_5D,
        ("h", "alpha", "beta", "x0", "y0"),
        "Legacy first-state conditioning.",
    ),
    STROKE10_V1: FeatureSchema(
        STROKE10_V1,
        ("h", "heading", "heading_copy", "x0", "y0", "x1", "y1", "dx", "dy", "length"),
        "Compatible schema used by the existing 10D NPZ and checkpoints.",
    ),
    STROKE10_POSE_V2: FeatureSchema(
        STROKE10_POSE_V2,
        ("z", "alpha", "beta", "x0", "y0", "x1", "y1", "dx", "dy", "length"),
        "Pose-aware schema using the trajectory's alpha and beta angles.",
    ),
}


def get_feature_schema(name: str) -> FeatureSchema:
    if name not in SCHEMAS:
        raise ValueError(f"Unknown feature schema '{name}'. Available: {sorted(SCHEMAS)}")
    return SCHEMAS[name]


def infer_legacy_schema(input_dim: int) -> str:
    if input_dim == 5:
        return LEGACY_5D
    if input_dim == 10:
        return STROKE10_V1
    raise ValueError(f"Cannot infer a legacy schema for input_dim={input_dim}")


def read_npz_schema(npz: Any, input_dim: int) -> str:
    if "feature_schema" not in npz.files:
        return infer_legacy_schema(input_dim)
    value = npz["feature_schema"]
    if isinstance(value, np.ndarray):
        value = value.item() if value.ndim == 0 else value.reshape(-1)[0]
    return str(value)


def normalization_for_inputs(
    inputs: np.ndarray,
    schema_name: str,
    coordinate_scale: float,
) -> Dict[str, Any]:
    schema = get_feature_schema(schema_name)
    if inputs.ndim != 2 or inputs.shape[1] != schema.input_dim:
        raise ValueError(
            f"Input shape {inputs.shape} does not match schema {schema_name} "
            f"with dimension {schema.input_dim}"
        )
    scales = np.ones((schema.input_dim,), dtype=np.float32)
    scales[0] = max(float(np.nanmax(np.abs(inputs[:, 0]))), 1.0)
    if schema.input_dim > 3:
        scales[3:] = float(coordinate_scale)
    return {
        "version": 2,
        "feature_schema": schema_name,
        "input_dim": schema.input_dim,
        "scales": scales.tolist(),
    }


def polyline_length(points: Sequence[Tuple[float, float]]) -> float:
    return float(
        sum(
            ((points[i][0] - points[i - 1][0]) ** 2 + (points[i][1] - points[i - 1][1]) ** 2) ** 0.5
            for i in range(1, len(points))
        )
    )


def build_stroke_features(
    schema_name: str,
    first_point: TrajectoryPoint,
    first_state: DynamicBrushState,
    normalized_points: Sequence[Tuple[float, float]],
) -> List[float]:
    schema = get_feature_schema(schema_name)
    if not normalized_points:
        raise ValueError("normalized_points must not be empty")
    x0, y0 = normalized_points[0]
    if schema.name == LEGACY_5D:
        return [first_state.z, first_state.theta, first_state.theta, x0, y0]

    x1, y1 = normalized_points[-1]
    geometry = [x0, y0, x1, y1, x1 - x0, y1 - y0, polyline_length(normalized_points)]
    if schema.name == STROKE10_V1:
        return [first_state.z, first_state.theta, first_state.theta, *geometry]
    if schema.name == STROKE10_POSE_V2:
        return [first_point.z, first_point.alpha, first_point.beta, *geometry]
    raise AssertionError(f"Unhandled feature schema: {schema.name}")


def checkpoint_schema(checkpoint: Any, input_dim: int) -> str:
    if isinstance(checkpoint, dict):
        schema = checkpoint.get("feature_schema")
        if schema:
            return str(schema)
        normalization = checkpoint.get("input_normalization") or {}
        if normalization.get("feature_schema"):
            return str(normalization["feature_schema"])
    return infer_legacy_schema(input_dim)
