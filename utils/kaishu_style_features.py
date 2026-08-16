"""Shared v16 Kaishu style-refiner features.

The target crop supplies real local footprint evidence (binary support,
distance/width and antialiased ink).  The matched trajectory supplies spatial
direction, stroke order, z-derived pressure proxy and per-sample speed proxy.
The latter two are explicitly proxies because the current CSV has no force
sensor or timestamps; they must not be reported as calibrated physical force
or velocity.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, gaussian_filter

from utils.character_features import extract_character_spatial_maps
from utils.structure_mask import build_structure_mask, skeletonize_binary
from utils.types import CharacterTrajectory


STYLE_FEATURE_CHANNEL_NAMES = (
    "geometry_mask",
    "skeleton",
    "interior_distance",
    "soft_geometry",
    "real_footprint_width",
    "trajectory_centerline",
    "trajectory_proximity",
    "trajectory_pressure_proxy",
    "trajectory_stroke_order",
    "trajectory_direction_cos",
    "trajectory_direction_sin",
    "trajectory_speed_proxy",
)


def geometry_features(gray: np.ndarray, threshold: float = 0.35) -> np.ndarray:
    """Return four bounded geometry channels from a real target crop."""
    mask, _ = build_structure_mask(
        gray, threshold=threshold, min_component_pixels=8, opening_iterations=1
    )
    binary = mask >= 0.5
    skeleton = skeletonize_binary(binary).astype(np.float32)
    inside = distance_transform_edt(binary).astype(np.float32)
    if inside.max() > 0:
        inside /= inside.max()
    soft = gaussian_filter(mask.astype(np.float32), sigma=1.2)
    return np.stack([mask, skeleton, inside, soft], axis=0).astype(np.float32)


def _speed_map(
    sample: CharacterTrajectory,
    normalized_strokes: List[List[Tuple[float, float]]],
    canvas_size: int,
    line_width: int,
) -> np.ndarray:
    """Rasterize displacement-per-sample as a bounded speed proxy."""
    segments: List[Tuple[List[Tuple[float, float]], float]] = []
    lengths: List[float] = []
    for stroke, points in zip(sample.sorted_strokes(), normalized_strokes):
        raw_points = stroke.sorted_points()
        for index in range(max(0, len(points) - 1)):
            dx = float(points[index + 1][0] - points[index][0])
            dy = float(points[index + 1][1] - points[index][1])
            length = math.hypot(dx, dy)
            segments.append(([points[index], points[index + 1]], length))
            lengths.append(length)
    scale = float(np.percentile(lengths, 95)) if lengths else 1.0
    scale = max(scale, 1e-6)
    canvas = Image.new("F", (canvas_size, canvas_size), 0.0)
    draw = ImageDraw.Draw(canvas)
    for segment, length in segments:
        draw.line(segment, fill=float(np.clip(length / scale, 0.0, 1.0)), width=line_width)
    return np.asarray(canvas, dtype=np.float32)


def build_style_features(
    gray: np.ndarray,
    trajectory: CharacterTrajectory,
    *,
    canvas_size: int = 128,
    trajectory_padding: int = 4,
    trajectory_width: int = 3,
    structure_threshold: float = 0.35,
    footprint_width_scale_px: float = 16.0,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Build v16 features and real-target/trajectory alignment metrics."""
    if gray.shape != (canvas_size, canvas_size):
        raise ValueError(f"Expected gray canvas {(canvas_size, canvas_size)}, got {gray.shape}")
    geometry = geometry_features(gray, threshold=structure_threshold)
    binary = geometry[0] >= 0.5
    width = np.clip(
        distance_transform_edt(binary).astype(np.float32)
        / max(float(footprint_width_scale_px), 1e-6),
        0.0,
        1.0,
    )
    trajectory_maps, normalized_strokes = extract_character_spatial_maps(
        trajectory,
        canvas_size=canvas_size,
        padding=trajectory_padding,
        line_width=trajectory_width,
    )
    speed = _speed_map(
        trajectory,
        normalized_strokes,
        canvas_size=canvas_size,
        line_width=trajectory_width,
    )
    features = np.concatenate(
        [geometry, width[None], trajectory_maps, speed[None]], axis=0
    ).astype(np.float32)
    if features.shape[0] != len(STYLE_FEATURE_CHANNEL_NAMES):
        raise RuntimeError(f"Unexpected v16 feature shape: {features.shape}")
    centerline = trajectory_maps[0] > 0.1
    target = binary
    intersection = float(np.logical_and(centerline, target).sum())
    centerline_area = float(centerline.sum())
    target_area = float(target.sum())
    metrics = {
        "trajectory_target_coverage": intersection / max(centerline_area, 1.0),
        "support_dice": (2.0 * intersection) / max(centerline_area + target_area, 1.0),
        "trajectory_target_area_ratio": centerline_area / max(target_area, 1.0),
    }
    if not np.isfinite(features).all():
        raise RuntimeError("v16 style features contain NaN or Inf")
    return features, metrics
