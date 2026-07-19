from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from models.dynamic_brush import DynamicBrushModel
from models.geometry import dynamic_state_to_bbsmg_input, normalize_trajectory_xy
from utils.types import CharacterTrajectory


CHARACTER_FEATURE_NAMES = (
    "h",
    "alpha",
    "beta",
    "x0",
    "y0",
    "x1",
    "y1",
    "dx",
    "dy",
    "length",
)


def polyline_length(points: List[Tuple[float, float]]) -> float:
    total = 0.0
    for start, end in zip(points, points[1:]):
        total += float(((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5)
    return total


def extract_character_features(
    sample: CharacterTrajectory,
    max_strokes: int,
    canvas_size: int = 128,
    padding: int = 4,
    brush: Optional[DynamicBrushModel] = None,
) -> Tuple[np.ndarray, np.ndarray, List[List[Tuple[float, float]]]]:
    """Encode every stroke, in order, without rendering intermediate images."""
    strokes = sample.sorted_strokes()
    if not strokes:
        raise ValueError("Character trajectory contains no strokes")
    if len(strokes) > max_strokes:
        raise ValueError(
            f"Character {sample.character!r} has {len(strokes)} strokes, "
            f"which exceeds max_strokes={max_strokes}"
        )

    norm_strokes = normalize_trajectory_xy(
        sample,
        canvas_size=canvas_size,
        padding=padding,
    )
    brush = brush or DynamicBrushModel()
    states_by_stroke = brush.simulate_character(sample, reset_each_stroke=True)

    features = np.zeros((max_strokes, len(CHARACTER_FEATURE_NAMES)), dtype=np.float32)
    stroke_mask = np.zeros((max_strokes,), dtype=np.bool_)
    for order, stroke in enumerate(strokes):
        points = norm_strokes[order]
        states = states_by_stroke.get(stroke.stroke_id, [])
        if not points or not states:
            raise ValueError(
                f"Character {sample.character!r}, stroke {stroke.stroke_id} has no usable points"
            )
        x0, y0 = points[0]
        x1, y1 = points[-1]
        bb_input = dynamic_state_to_bbsmg_input(states[0], x0=x0, y0=y0)
        features[order] = np.asarray(
            [
                bb_input.h,
                bb_input.alpha,
                bb_input.beta,
                x0,
                y0,
                x1,
                y1,
                x1 - x0,
                y1 - y0,
                polyline_length(points),
            ],
            dtype=np.float32,
        )
        stroke_mask[order] = True
    return features, stroke_mask, norm_strokes


def compute_character_normalization(
    inputs: np.ndarray,
    stroke_masks: np.ndarray,
    coordinate_scale: float,
) -> Dict[str, Any]:
    if inputs.ndim != 3:
        raise ValueError(f"Expected inputs [N,S,D], got {inputs.shape}")
    if stroke_masks.shape != inputs.shape[:2]:
        raise ValueError("stroke_masks shape must match the first two input dimensions")
    valid = inputs[stroke_masks.astype(bool)]
    if valid.size == 0:
        raise ValueError("No valid strokes are present")
    scales = np.ones((inputs.shape[-1],), dtype=np.float32)
    scales[0] = max(float(np.nanmax(valid[:, 0])), 1.0)
    if inputs.shape[-1] > 3:
        scales[3:] = float(coordinate_scale)
    return {
        "version": 1,
        "input_dim": int(inputs.shape[-1]),
        "scales": scales.tolist(),
        "feature_names": list(CHARACTER_FEATURE_NAMES),
    }


def normalize_character_features(
    inputs: np.ndarray,
    stroke_masks: np.ndarray,
    normalization: Dict[str, Any],
) -> np.ndarray:
    scales = np.asarray(normalization.get("scales"), dtype=np.float32)
    if scales.shape != (inputs.shape[-1],):
        raise ValueError(
            f"Normalization has {scales.size} scales for input_dim={inputs.shape[-1]}"
        )
    normalized = inputs.astype(np.float32, copy=True) / scales.reshape(1, 1, -1)
    normalized *= stroke_masks[..., None].astype(np.float32)
    return normalized
