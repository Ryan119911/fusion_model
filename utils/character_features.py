import math
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from models.geometry import normalize_trajectory_xy
from utils.types import CharacterTrajectory


SPATIAL_CHANNEL_NAMES = (
    "centerline",
    "proximity",
    "pressure",
    "stroke_order",
    "direction_cos",
    "direction_sin",
)


def _new_float_canvas(canvas_size: int) -> Image.Image:
    return Image.new("F", (canvas_size, canvas_size), 0.0)


def _draw_point(draw: ImageDraw.ImageDraw, point, value: float, width: int) -> None:
    x, y = point
    radius = max(width / 2.0, 1.0)
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=float(value))


def extract_character_spatial_maps(
    sample: CharacterTrajectory,
    canvas_size: int = 128,
    padding: int = 16,
    line_width: int = 3,
) -> Tuple[np.ndarray, List[List[Tuple[float, float]]]]:
    """Rasterize the complete trajectory into six aligned U-Net input maps."""
    strokes = sample.sorted_strokes()
    if not strokes:
        raise ValueError("Character trajectory contains no strokes")
    if line_width < 1:
        raise ValueError("line_width must be positive")

    normalized_strokes = normalize_trajectory_xy(
        sample,
        canvas_size=canvas_size,
        padding=padding,
    )
    centerline = Image.new("L", (canvas_size, canvas_size), 0)
    pressure = _new_float_canvas(canvas_size)
    stroke_order = _new_float_canvas(canvas_size)
    direction_cos = _new_float_canvas(canvas_size)
    direction_sin = _new_float_canvas(canvas_size)
    centerline_draw = ImageDraw.Draw(centerline)
    pressure_draw = ImageDraw.Draw(pressure)
    order_draw = ImageDraw.Draw(stroke_order)
    cos_draw = ImageDraw.Draw(direction_cos)
    sin_draw = ImageDraw.Draw(direction_sin)

    all_z = [max(float(point.z), 0.0) for stroke in strokes for point in stroke.sorted_points()]
    pressure_scale = max(max(all_z, default=0.0), 1e-6)
    stroke_count = len(strokes)

    for order, (stroke, normalized_points) in enumerate(zip(strokes, normalized_strokes)):
        raw_points = stroke.sorted_points()
        if not normalized_points or len(raw_points) != len(normalized_points):
            raise ValueError(
                f"Character {sample.character!r}, stroke {stroke.stroke_id} has invalid points"
            )
        order_value = float(order + 1) / float(stroke_count)

        if len(normalized_points) == 1:
            pressure_value = max(float(raw_points[0].z), 0.0) / pressure_scale
            _draw_point(centerline_draw, normalized_points[0], 255.0, line_width)
            _draw_point(pressure_draw, normalized_points[0], pressure_value, line_width)
            _draw_point(order_draw, normalized_points[0], order_value, line_width)
            continue

        for index in range(len(normalized_points) - 1):
            start = normalized_points[index]
            end = normalized_points[index + 1]
            dx = float(end[0] - start[0])
            dy = float(end[1] - start[1])
            length = math.hypot(dx, dy)
            cos_value = dx / length if length > 1e-8 else 0.0
            sin_value = dy / length if length > 1e-8 else 0.0
            pressure_value = (
                max(float(raw_points[index].z), 0.0)
                + max(float(raw_points[index + 1].z), 0.0)
            ) / (2.0 * pressure_scale)
            segment = [start, end]
            centerline_draw.line(segment, fill=255, width=line_width)
            pressure_draw.line(segment, fill=float(pressure_value), width=line_width)
            order_draw.line(segment, fill=order_value, width=line_width)
            cos_draw.line(segment, fill=float(cos_value), width=line_width)
            sin_draw.line(segment, fill=float(sin_value), width=line_width)

    centerline_array = np.asarray(centerline, dtype=np.float32) / 255.0
    proximity_image = centerline.filter(
        ImageFilter.GaussianBlur(radius=max(2.0, float(line_width) * 1.5))
    )
    proximity = np.asarray(proximity_image, dtype=np.float32) / 255.0
    proximity_max = float(proximity.max())
    if proximity_max > 1e-6:
        proximity = proximity / proximity_max

    maps = np.stack(
        [
            centerline_array,
            proximity,
            np.asarray(pressure, dtype=np.float32),
            np.asarray(stroke_order, dtype=np.float32),
            np.asarray(direction_cos, dtype=np.float32),
            np.asarray(direction_sin, dtype=np.float32),
        ],
        axis=0,
    )
    if maps.shape != (len(SPATIAL_CHANNEL_NAMES), canvas_size, canvas_size):
        raise RuntimeError(f"Unexpected spatial feature shape: {maps.shape}")
    if not np.isfinite(maps).all():
        raise RuntimeError("Spatial trajectory maps contain NaN or Inf")
    return maps.astype(np.float32), normalized_strokes
