from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from utils.types import (
    BBSMGInput,
    CharacterTrajectory,
    DynamicBrushState,
    StrokeTrajectory,
    TrajectoryPoint,
)


@dataclass
class CanvasTransform:
    src_min_x: float
    src_max_x: float
    src_min_y: float
    src_max_y: float
    dst_size: int = 128
    padding: int = 4

    def map_point(self, x: float, y: float) -> Tuple[float, float]:
        width = max(self.src_max_x - self.src_min_x, 1e-6)
        height = max(self.src_max_y - self.src_min_y, 1e-6)
        available = max(self.dst_size - 2 * self.padding, 1)
        scale = min(available / width, available / height)
        offset_x = self.padding + (available - width * scale) / 2.0
        offset_y = self.padding + (available - height * scale) / 2.0
        return (
            (x - self.src_min_x) * scale + offset_x,
            (self.src_max_y - y) * scale + offset_y,
        )


def makehanzi_to_display(x: float, y: float) -> Tuple[float, float]:
    return x, 900.0 - y


def makehanzi_to_normalized(
    x: float, y: float, canvas_size: int = 128
) -> Tuple[float, float]:
    display_x, display_y = makehanzi_to_display(x, y)
    return (
        display_x / 1024.0 * (canvas_size - 1),
        display_y / 1024.0 * (canvas_size - 1),
    )


def normalize_points(
    points: Sequence[Tuple[float, float]],
    canvas_size: int = 128,
    padding: int = 4,
) -> List[Tuple[float, float]]:
    if not points:
        return []
    xs, ys = zip(*points)
    transform = CanvasTransform(
        min(xs), max(xs), min(ys), max(ys), canvas_size, padding
    )
    return [transform.map_point(x, y) for x, y in points]


def normalize_makehanzi_median(
    median: Sequence[Tuple[int, int]], canvas_size: int = 128
) -> List[Tuple[float, float]]:
    return [
        makehanzi_to_normalized(float(x), float(y), canvas_size)
        for x, y in median
    ]


def stroke_start_point_from_median(
    median: Sequence[Tuple[int, int]], canvas_size: int = 128
) -> Tuple[float, float]:
    return (
        normalize_makehanzi_median(median, canvas_size)[0]
        if median
        else (0.0, 0.0)
    )


def estimate_initial_theta_from_median(
    median: Sequence[Tuple[int, int]],
) -> float:
    if len(median) < 2:
        return 0.0
    (x0, y0), (x1, y1) = median[:2]
    return math.atan2(float(y1 - y0), float(x1 - x0))


def compute_heading(points: Sequence[Tuple[float, float]]) -> List[float]:
    if not points:
        return []
    if len(points) == 1:
        return [0.0]
    result = []
    for index in range(len(points)):
        left = max(0, index - 1)
        right = min(len(points) - 1, index + 1)
        dx = points[right][0] - points[left][0]
        dy = points[right][1] - points[left][1]
        result.append(math.atan2(dy, dx))
    return result


def _arc_positions(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.zeros((len(points),), dtype=np.float64)
    lengths = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(lengths)])


def resample_polyline(
    points: Sequence[Tuple[float, float]], num_samples: int
) -> List[Tuple[float, float]]:
    if num_samples < 1:
        raise ValueError("num_samples must be >= 1")
    if not points:
        return []
    values = np.asarray(points, dtype=np.float64)
    arc = _arc_positions(values)
    if arc[-1] <= 1e-12:
        return [tuple(values[0])] * num_samples
    target = np.linspace(0.0, arc[-1], num_samples)
    return list(zip(np.interp(target, arc, values[:, 0]), np.interp(target, arc, values[:, 1])))


def _interp_discrete(values: np.ndarray, arc: np.ndarray, target: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(arc, target, side="left").clip(0, len(values) - 1)
    previous = (indices - 1).clip(0, len(values) - 1)
    choose_previous = np.abs(target - arc[previous]) <= np.abs(arc[indices] - target)
    indices[choose_previous] = previous[choose_previous]
    return values[indices]


def resample_stroke(
    stroke: StrokeTrajectory, num_samples: int
) -> StrokeTrajectory:
    points = stroke.sorted_points()
    if not points:
        return StrokeTrajectory(stroke_id=stroke.stroke_id, points=[])
    values = np.asarray(
        [[p.x, p.y, p.z, p.alpha, p.beta, p.gamma] for p in points],
        dtype=np.float64,
    )
    arc = _arc_positions(values)
    target = np.linspace(0.0, arc[-1], num_samples)
    if arc[-1] <= 1e-12:
        continuous = np.repeat(values[:1], num_samples, axis=0)
    else:
        continuous = np.empty((num_samples, 6), dtype=np.float64)
        for column in range(3):
            continuous[:, column] = np.interp(target, arc, values[:, column])
        for column in range(3, 6):
            unwrapped = np.unwrap(values[:, column])
            continuous[:, column] = np.interp(target, arc, unwrapped)
    states = _interp_discrete(
        np.asarray([int(p.state) for p in points]), arc, target
    )
    timestamps = None
    if all(p.timestamp is not None for p in points):
        timestamps = np.interp(
            target, arc, np.asarray([float(p.timestamp) for p in points])
        )
    output = []
    for index, row in enumerate(continuous):
        output.append(
            TrajectoryPoint(
                stroke_id=stroke.stroke_id,
                point_id=index,
                x=float(row[0]),
                y=float(row[1]),
                z=float(row[2]),
                alpha=float(row[3]),
                beta=float(row[4]),
                gamma=float(row[5]),
                state=type(points[0].state)(int(states[index])),
                timestamp=None if timestamps is None else float(timestamps[index]),
            )
        )
    return StrokeTrajectory(stroke_id=stroke.stroke_id, points=output)


def resample_character_trajectory(
    sample: CharacterTrajectory, points_per_stroke: int
) -> CharacterTrajectory:
    return CharacterTrajectory(
        character=sample.character,
        strokes=[
            resample_stroke(stroke, points_per_stroke)
            for stroke in sample.sorted_strokes()
        ],
        meta=dict(sample.meta),
    )


def trajectory_bounds(
    sample: CharacterTrajectory,
) -> Tuple[float, float, float, float]:
    points = sample.all_points()
    if not points:
        return 0.0, 1.0, 0.0, 1.0
    xs = [point.x for point in points]
    ys = [point.y for point in points]
    return min(xs), max(xs), min(ys), max(ys)


def normalize_trajectory_xy_with_bounds(
    sample: CharacterTrajectory,
    bounds: Tuple[float, float, float, float],
    canvas_size: int = 128,
    padding: int = 4,
) -> List[List[Tuple[float, float]]]:
    transform = CanvasTransform(*bounds, dst_size=canvas_size, padding=padding)
    return [
        [transform.map_point(point.x, point.y) for point in stroke.sorted_points()]
        for stroke in sample.sorted_strokes()
    ]


def normalize_trajectory_xy(
    sample: CharacterTrajectory,
    canvas_size: int = 128,
    padding: int = 4,
) -> List[List[Tuple[float, float]]]:
    return normalize_trajectory_xy_with_bounds(
        sample, trajectory_bounds(sample), canvas_size, padding
    )


def stroke_to_headings(stroke: StrokeTrajectory) -> List[float]:
    return compute_heading([(point.x, point.y) for point in stroke.sorted_points()])


def dynamic_state_to_bezier(
    state: DynamicBrushState,
) -> Tuple[float, float, float]:
    return max(state.d * 0.7, 0.0), max(state.d * 0.3, 0.0), max(state.w, 0.0)


def dynamic_state_to_bbsmg_input(
    state: DynamicBrushState,
    x0: Optional[float] = None,
    y0: Optional[float] = None,
) -> BBSMGInput:
    return BBSMGInput(
        h=float(state.z),
        alpha=float(state.theta),
        beta=float(state.theta),
        x0=float(state.x if x0 is None else x0),
        y0=float(state.y if y0 is None else y0),
    )


def pair_trajectory_strokes_with_medians(
    sample: CharacterTrajectory,
    medians: List[List[Tuple[int, int]]],
) -> List[Dict[str, Any]]:
    result = []
    for stroke, median in zip(sample.sorted_strokes(), medians):
        result.append(
            {
                "stroke": stroke,
                "median_raw": median,
                "median_norm": normalize_makehanzi_median(median),
                "start_point": stroke_start_point_from_median(median),
                "theta0": estimate_initial_theta_from_median(median),
            }
        )
    return result
