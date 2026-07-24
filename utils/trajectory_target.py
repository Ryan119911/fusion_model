from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from utils.types import CharacterTrajectory


TRAJECTORY_TARGET_MODE = "trajectory_faithful_mask"
TRAJECTORY_RENDER_VERSION = "trajectory_pressure_render_v1"


def _draw_disk(
    draw: ImageDraw.ImageDraw,
    point: Tuple[float, float],
    width: float,
) -> None:
    radius = max(float(width) / 2.0, 0.5)
    x, y = point
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=255,
    )


def _normalized_pressures(
    trajectory: CharacterTrajectory,
    pressure_invert: bool,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    strokes = trajectory.sorted_strokes()
    raw_values = np.asarray(
        [
            max(float(point.z), 0.0)
            for stroke in strokes
            for point in stroke.sorted_points()
        ],
        dtype=np.float32,
    )
    if raw_values.size == 0:
        raise ValueError("Trajectory contains no pressure samples")
    pressure_min = float(raw_values.min())
    pressure_max = float(raw_values.max())
    pressure_range = pressure_max - pressure_min
    constant_pressure = pressure_range < 1e-6

    result: List[np.ndarray] = []
    for stroke in strokes:
        values = np.asarray(
            [max(float(point.z), 0.0) for point in stroke.sorted_points()],
            dtype=np.float32,
        )
        if constant_pressure:
            normalized = np.full_like(values, 0.5)
        else:
            normalized = (values - pressure_min) / pressure_range
        if pressure_invert:
            normalized = 1.0 - normalized
        result.append(np.clip(normalized, 0.0, 1.0))
    return result, {
        "pressure_min": pressure_min,
        "pressure_max": pressure_max,
        "pressure_constant": bool(constant_pressure),
        "pressure_invert": bool(pressure_invert),
    }


def render_trajectory_target(
    trajectory: CharacterTrajectory,
    normalized_strokes: Sequence[Sequence[Tuple[float, float]]],
    canvas_size: int,
    min_width: float = 4.0,
    max_width: float = 8.0,
    pressure_gamma: float = 1.0,
    pressure_invert: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Render a binary glyph whose geometry comes only from its own trajectory."""
    if canvas_size < 1:
        raise ValueError("canvas_size must be positive")
    if min_width < 1.0 or max_width < min_width:
        raise ValueError("Require 1 <= min_width <= max_width")
    if pressure_gamma <= 0.0:
        raise ValueError("pressure_gamma must be positive")

    strokes = trajectory.sorted_strokes()
    if len(strokes) != len(normalized_strokes):
        raise ValueError("Raw and normalized stroke counts differ")
    normalized_pressures, pressure_info = _normalized_pressures(
        trajectory,
        pressure_invert=pressure_invert,
    )
    image = Image.new("L", (canvas_size, canvas_size), 0)
    draw = ImageDraw.Draw(image)
    rendered_widths: List[float] = []

    for stroke, points, pressures in zip(
        strokes,
        normalized_strokes,
        normalized_pressures,
    ):
        raw_points = stroke.sorted_points()
        if len(points) != len(raw_points) or len(points) != len(pressures):
            raise ValueError(
                f"Stroke {stroke.stroke_id} raw/normalized point counts differ"
            )
        if not points:
            continue
        widths = min_width + (max_width - min_width) * np.power(
            pressures,
            pressure_gamma,
        )
        rendered_widths.extend(float(value) for value in widths)
        if len(points) == 1:
            _draw_disk(draw, tuple(points[0]), float(widths[0]))
            continue
        for index in range(len(points) - 1):
            start = tuple(points[index])
            end = tuple(points[index + 1])
            segment_width = float((widths[index] + widths[index + 1]) / 2.0)
            integer_width = max(1, int(round(segment_width)))
            draw.line(
                [start, end],
                fill=255,
                width=integer_width,
                joint="curve",
            )
            _draw_disk(draw, start, segment_width)
            _draw_disk(draw, end, segment_width)

    target = (np.asarray(image, dtype=np.uint8) > 0).astype(np.float32)
    if not np.any(target):
        raise ValueError("Trajectory renderer produced an empty target")
    width_values = np.asarray(rendered_widths, dtype=np.float32)
    info: Dict[str, Any] = {
        "mode": TRAJECTORY_TARGET_MODE,
        "render_version": TRAJECTORY_RENDER_VERSION,
        "min_width": float(min_width),
        "max_width": float(max_width),
        "pressure_gamma": float(pressure_gamma),
        "rendered_width_min": float(width_values.min()),
        "rendered_width_max": float(width_values.max()),
        "rendered_width_mean": float(width_values.mean()),
        "foreground_pixels": int(target.sum()),
        "foreground_fraction": float(target.mean()),
        **pressure_info,
    }
    return target, info
