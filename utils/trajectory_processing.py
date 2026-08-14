"""Robot-independent trajectory repair, smoothing, validation, and previews.

The renderers operate on one stroke at a time, but exported trajectories are
also consumed by robot adapters.  This module makes that contract explicit:
stroke boundaries are never connected, every stroke has a deterministic
down/move/up state sequence, and safety checks are emitted as JSON-friendly
diagnostics before a trajectory is sent to hardware.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from utils.types import (
    CharacterTrajectory,
    PointState,
    StrokeTrajectory,
    TrajectoryPoint,
)


@dataclass(frozen=True)
class TrajectorySafetyLimits:
    """Limits used by :func:`validate_trajectory`.

    ``None`` disables a limit.  The defaults are deliberately conservative for
    paper-space trajectories and do not assume a particular robot brand.
    """

    x_min: Optional[float] = None
    x_max: Optional[float] = None
    y_min: Optional[float] = None
    y_max: Optional[float] = None
    z_min: Optional[float] = None
    z_max: Optional[float] = None
    max_step_xy: Optional[float] = None
    max_step_z: Optional[float] = None
    max_angle_step_rad: Optional[float] = math.pi
    max_zero_length_fraction: float = 0.25


def _copy_point(point: TrajectoryPoint, **updates: Any) -> TrajectoryPoint:
    """Copy a point without mutating a trajectory held by another stage."""

    return replace(point, **updates)


def _sorted_strokes(sample: CharacterTrajectory) -> List[StrokeTrajectory]:
    return [
        StrokeTrajectory(
            stroke_id=int(stroke.stroke_id),
            points=stroke.sorted_points(),
        )
        for stroke in sample.sorted_strokes()
        if len(stroke.points) > 0
    ]


def repair_sample_states(
    sample: CharacterTrajectory,
    *,
    lift_z: Optional[float] = None,
) -> CharacterTrajectory:
    """Return a copy with explicit, deterministic pen states.

    The first point of each stroke is ``DOWN``, interior points are ``MOVE``
    and the last point is ``UP``.  Coordinates are unchanged unless
    ``lift_z`` is provided, in which case only the final point is assigned the
    requested clearance value.  Keeping point count and IDs unchanged makes
    this safe to apply before loading a pose CSV keyed by stroke/point IDs.
    """

    repaired: List[StrokeTrajectory] = []
    for stroke in _sorted_strokes(sample):
        points = stroke.sorted_points()
        out: List[TrajectoryPoint] = []
        for index, point in enumerate(points):
            if len(points) == 1:
                state = PointState.DOWN
            elif index == 0:
                state = PointState.DOWN
            elif index == len(points) - 1:
                state = PointState.UP
            else:
                state = PointState.MOVE
            updates: Dict[str, Any] = {
                "stroke_id": int(stroke.stroke_id),
                "state": state,
            }
            if lift_z is not None and index == len(points) - 1:
                updates["z"] = float(lift_z)
            out.append(_copy_point(point, **updates))
        repaired.append(StrokeTrajectory(stroke_id=stroke.stroke_id, points=out))
    return CharacterTrajectory(
        character=sample.character,
        strokes=repaired,
        meta={**sample.meta, "trajectory_processing": "states_repaired_v1"},
    )


def _smooth_vector(values: np.ndarray, passes: int, strength: float) -> np.ndarray:
    """Laplacian smooth a vector while preserving its endpoints."""

    result = np.asarray(values, dtype=np.float64).copy()
    if len(result) < 3 or passes <= 0 or strength <= 0.0:
        return result
    strength = float(np.clip(strength, 0.0, 1.0))
    for _ in range(int(passes)):
        previous = result.copy()
        result[1:-1] = (1.0 - strength) * previous[1:-1] + strength * 0.5 * (
            previous[:-2] + previous[2:]
        )
    return result


def smooth_sample(
    sample: CharacterTrajectory,
    *,
    passes: int = 2,
    strength: float = 0.25,
    smooth_z: bool = True,
    smooth_angles: bool = True,
) -> CharacterTrajectory:
    """Smooth each stroke independently, never across a pen-up boundary.

    Endpoints remain fixed so stroke placement and the pen-up/down event are
    preserved.  Angles are unwrapped before smoothing and wrapped back to
    ``[-pi, pi]`` afterwards.
    """

    repaired = repair_sample_states(sample)
    new_strokes: List[StrokeTrajectory] = []
    for stroke in repaired.sorted_strokes():
        points = stroke.sorted_points()
        if len(points) < 3:
            new_strokes.append(stroke)
            continue
        x = _smooth_vector(np.asarray([p.x for p in points]), passes, strength)
        y = _smooth_vector(np.asarray([p.y for p in points]), passes, strength)
        z = _smooth_vector(np.asarray([p.z for p in points]), passes, strength)
        angle_arrays: List[np.ndarray] = []
        for name in ("alpha", "beta", "gamma"):
            values = np.asarray([getattr(p, name) for p in points], dtype=np.float64)
            angle_arrays.append(
                _smooth_vector(np.unwrap(values), passes, strength)
                if smooth_angles
                else values
            )
        out: List[TrajectoryPoint] = []
        for index, point in enumerate(points):
            out.append(
                _copy_point(
                    point,
                    x=float(x[index]),
                    y=float(y[index]),
                    z=float(z[index]) if smooth_z else point.z,
                    alpha=float(angle_arrays[0][index]),
                    beta=float(angle_arrays[1][index]),
                    gamma=float(
                        (angle_arrays[2][index] + math.pi) % (2.0 * math.pi)
                        - math.pi
                    ),
                )
            )
        new_strokes.append(StrokeTrajectory(stroke_id=stroke.stroke_id, points=out))
    return CharacterTrajectory(
        character=sample.character,
        strokes=new_strokes,
        meta={
            **sample.meta,
            "trajectory_processing": "states_repaired_laplacian_smooth_v1",
            "smooth_passes": int(passes),
            "smooth_strength": float(strength),
        },
    )


def _interp_point(a: TrajectoryPoint, b: TrajectoryPoint, t: float, point_id: int) -> TrajectoryPoint:
    """Interpolate a point for optional execution-time densification."""

    def lerp(name: str) -> float:
        return float(getattr(a, name) + t * (getattr(b, name) - getattr(a, name)))

    return TrajectoryPoint(
        stroke_id=a.stroke_id,
        point_id=point_id,
        x=lerp("x"),
        y=lerp("y"),
        z=lerp("z"),
        alpha=lerp("alpha"),
        beta=lerp("beta"),
        gamma=float(
            (np.unwrap(np.asarray([a.gamma, b.gamma]))[0] + t *
             (np.unwrap(np.asarray([a.gamma, b.gamma]))[1] -
              np.unwrap(np.asarray([a.gamma, b.gamma]))[0]) + math.pi)
            % (2.0 * math.pi) - math.pi
        ),
        state=PointState.MOVE,
    )


def densify_sample(
    sample: CharacterTrajectory,
    *,
    max_step_xy: float,
) -> CharacterTrajectory:
    """Insert points inside strokes so no XY segment exceeds ``max_step_xy``."""

    if max_step_xy <= 0.0:
        return repair_sample_states(sample)
    repaired = repair_sample_states(sample)
    new_strokes: List[StrokeTrajectory] = []
    for stroke in repaired.sorted_strokes():
        source = stroke.sorted_points()
        if len(source) < 2:
            new_strokes.append(stroke)
            continue
        out: List[TrajectoryPoint] = []
        for index, (a, b) in enumerate(zip(source[:-1], source[1:])):
            if index == 0:
                out.append(_copy_point(a, point_id=0, state=PointState.DOWN))
            distance = math.hypot(b.x - a.x, b.y - a.y)
            steps = max(1, int(math.ceil(distance / float(max_step_xy))))
            for step in range(1, steps + 1):
                point = _interp_point(a, b, step / float(steps), len(out))
                if index == len(source) - 2 and step == steps:
                    point = _copy_point(point, state=PointState.UP)
                out.append(point)
        new_strokes.append(StrokeTrajectory(stroke_id=stroke.stroke_id, points=out))
    return CharacterTrajectory(
        character=sample.character,
        strokes=new_strokes,
        meta={**sample.meta, "trajectory_processing": "densified_v1", "max_step_xy": float(max_step_xy)},
    )


def expand_pen_up_transitions(
    sample: CharacterTrajectory,
    *,
    clearance_z: float,
) -> CharacterTrajectory:
    """Add explicit vertical clearance and inter-stroke travel points.

    This representation is intended for a robot adapter, not for B-BSMG
    rendering: transition points carry ``UP``/``TRANSITION`` states and must
    never contribute ink.  Each stroke ends at a same-XY clearance point; the
    next stroke begins at a clearance travel point and then descends vertically
    to its original first contact point.  Point IDs are re-numbered within each
    expanded stroke because new points are inserted.
    """

    if not np.isfinite(clearance_z):
        raise ValueError("clearance_z must be finite")
    repaired = repair_sample_states(sample)
    strokes = repaired.sorted_strokes()
    expanded: List[StrokeTrajectory] = []
    for stroke_index, stroke in enumerate(strokes):
        source = stroke.sorted_points()
        if not source:
            continue
        points: List[TrajectoryPoint] = []
        for point in source:
            points.append(_copy_point(point, point_id=len(points)))
        last = source[-1]
        points.append(
            _copy_point(
                last,
                point_id=len(points),
                z=float(clearance_z),
                state=PointState.UP,
            )
        )
        if stroke_index > 0:
            first = source[0]
            # The travel point is placed before the contact point in the same
            # logical stroke.  Its state tells the adapter to remain lifted.
            points.insert(
                0,
                _copy_point(
                    first,
                    point_id=0,
                    z=float(clearance_z),
                    state=PointState.TRANSITION,
                ),
            )
            points = [
                _copy_point(point, point_id=index)
                for index, point in enumerate(points)
            ]
        expanded.append(StrokeTrajectory(stroke_id=stroke.stroke_id, points=points))
    return CharacterTrajectory(
        character=sample.character,
        strokes=expanded,
        meta={
            **sample.meta,
            "trajectory_processing": "execution_pen_up_expanded_v1",
            "execution_expanded": True,
            "clearance_z": float(clearance_z),
        },
    )


def _angle_delta(a: float, b: float) -> float:
    return abs((float(b) - float(a) + math.pi) % (2.0 * math.pi) - math.pi)


def validate_trajectory(
    sample: CharacterTrajectory,
    limits: TrajectorySafetyLimits | None = None,
) -> Dict[str, Any]:
    """Audit a trajectory and return JSON-serializable safety diagnostics."""

    limits = limits or TrajectorySafetyLimits()
    errors: List[str] = []
    warnings: List[str] = []
    points = sample.all_points()
    if not points:
        errors.append("empty_trajectory")
    stroke_ids = [stroke.stroke_id for stroke in sample.sorted_strokes()]
    if len(stroke_ids) != len(set(stroke_ids)):
        errors.append("duplicate_stroke_id")
    if stroke_ids != sorted(stroke_ids):
        errors.append("stroke_order_not_monotonic")
    finite_fields = ("x", "y", "z", "alpha", "beta", "gamma")
    nonfinite = []
    for point in points:
        if not all(np.isfinite(float(getattr(point, field))) for field in finite_fields):
            nonfinite.append([point.stroke_id, point.point_id])
    if nonfinite:
        errors.append("nonfinite_values")

    within_steps: List[float] = []
    z_steps: List[float] = []
    angle_steps: List[float] = []
    zero_segments = 0
    total_segments = 0
    state_errors: List[str] = []
    for stroke in sample.sorted_strokes():
        ordered = stroke.sorted_points()
        if len(ordered) == 1:
            warnings.append(f"single_point_stroke_{stroke.stroke_id}")
        point_ids = [p.point_id for p in ordered]
        if point_ids != sorted(point_ids) or len(point_ids) != len(set(point_ids)):
            errors.append(f"point_id_not_monotonic_stroke_{stroke.stroke_id}")
        if len(ordered) >= 2:
            execution_expanded = bool(sample.meta.get("execution_expanded"))
            first_contact = next(
                (point for point in ordered if point.state != PointState.TRANSITION),
                ordered[0],
            )
            if (not execution_expanded and ordered[0].state != PointState.DOWN) or (
                execution_expanded and first_contact.state != PointState.DOWN
            ):
                state_errors.append(f"stroke_{stroke.stroke_id}_first_not_down")
            if ordered[-1].state != PointState.UP:
                state_errors.append(f"stroke_{stroke.stroke_id}_last_not_up")
        for a, b in zip(ordered[:-1], ordered[1:]):
            total_segments += 1
            xy_step = math.hypot(b.x - a.x, b.y - a.y)
            within_steps.append(xy_step)
            z_steps.append(abs(b.z - a.z))
            angle_steps.append(max(_angle_delta(a.alpha, b.alpha), _angle_delta(a.beta, b.beta), _angle_delta(a.gamma, b.gamma)))
            if xy_step <= 1e-9:
                zero_segments += 1
    if state_errors:
        errors.extend(state_errors)
    if total_segments and zero_segments / total_segments > limits.max_zero_length_fraction:
        warnings.append("too_many_zero_length_segments")

    def bounds_violation(field: str, low: Optional[float], high: Optional[float]) -> int:
        values = [float(getattr(point, field)) for point in points]
        count = 0
        if low is not None:
            count += sum(value < low for value in values)
        if high is not None:
            count += sum(value > high for value in values)
        return int(count)

    violations = {
        "x": bounds_violation("x", limits.x_min, limits.x_max),
        "y": bounds_violation("y", limits.y_min, limits.y_max),
        "z": bounds_violation("z", limits.z_min, limits.z_max),
    }
    for field, count in violations.items():
        if count:
            errors.append(f"{field}_out_of_bounds")
    if limits.max_step_xy is not None and any(step > limits.max_step_xy for step in within_steps):
        errors.append("xy_step_exceeds_limit")
    if limits.max_step_z is not None and any(step > limits.max_step_z for step in z_steps):
        errors.append("z_step_exceeds_limit")
    if limits.max_angle_step_rad is not None and any(step > limits.max_angle_step_rad for step in angle_steps):
        errors.append("angle_step_exceeds_limit")

    interstroke_gaps = []
    ordered_strokes = sample.sorted_strokes()
    for previous, current in zip(ordered_strokes[:-1], ordered_strokes[1:]):
        if previous.points and current.points:
            a = previous.sorted_points()[-1]
            b = current.sorted_points()[0]
            interstroke_gaps.append(math.hypot(b.x - a.x, b.y - a.y))
    return {
        "safe": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "point_count": len(points),
        "stroke_count": len(ordered_strokes),
        "within_stroke": {
            "max_step_xy": float(max(within_steps, default=0.0)),
            "max_step_z": float(max(z_steps, default=0.0)),
            "max_angle_step_rad": float(max(angle_steps, default=0.0)),
            "zero_length_fraction": float(zero_segments / total_segments) if total_segments else 0.0,
        },
        "interstroke": {
            "gap_count": len(interstroke_gaps),
            "min_gap_xy": float(min(interstroke_gaps)) if interstroke_gaps else None,
            "cross_stroke_segments_rendered": 0,
        },
        "bounds_violations": violations,
        "state_errors": state_errors,
    }


def _canvas_transform(sample: CharacterTrajectory, size: int, padding: int) -> Tuple[float, float, float, float, float]:
    points = sample.all_points()
    xs = [p.x for p in points] or [0.0]
    ys = [p.y for p in points] or [0.0]
    min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
    span = max(max_x - min_x, max_y - min_y, 1e-6)
    scale = (size - 2 * padding) / span
    return min_x, min_y, scale, float(size), float(padding)


def render_trajectory_preview(
    sample: CharacterTrajectory,
    output_path: str | Path,
    *,
    size: int = 512,
    padding: int = 32,
) -> None:
    """Render a stroke-separated diagnostic preview with endpoints and IDs."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    min_x, min_y, scale, _, pad = _canvas_transform(sample, size, padding)
    palette = [
        (32, 102, 181), (220, 80, 60), (45, 150, 90),
        (150, 75, 170), (220, 145, 30), (20, 150, 160),
    ]

    def to_pixel(point: TrajectoryPoint) -> Tuple[int, int]:
        return int(round(pad + (point.x - min_x) * scale)), int(round(pad + (point.y - min_y) * scale))

    for order, stroke in enumerate(sample.sorted_strokes()):
        points = stroke.sorted_points()
        if not points:
            continue
        color = palette[order % len(palette)]
        pixels = [to_pixel(point) for point in points]
        if len(pixels) > 1:
            draw.line(pixels, fill=color, width=max(2, size // 128), joint="curve")
        start, end = pixels[0], pixels[-1]
        radius = max(3, size // 64)
        draw.ellipse((start[0] - radius, start[1] - radius, start[0] + radius, start[1] + radius), fill=(30, 170, 70))
        draw.ellipse((end[0] - radius, end[1] - radius, end[0] + radius, end[1] + radius), fill=(210, 40, 40))
        draw.text((start[0] + radius + 2, start[1] - radius - 2), f"S{stroke.stroke_id}", fill="black")
    image.save(output)


def render_trajectory_overlay(
    reference: CharacterTrajectory,
    processed: CharacterTrajectory,
    output_path: str | Path,
    *,
    size: int = 512,
    padding: int = 32,
) -> None:
    """Draw raw trajectory in gray and processed trajectory in color."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    min_x, min_y, scale, _, pad = _canvas_transform(reference, size, padding)

    def to_pixel(point: TrajectoryPoint) -> Tuple[int, int]:
        return int(round(pad + (point.x - min_x) * scale)), int(round(pad + (point.y - min_y) * scale))

    for stroke in reference.sorted_strokes():
        pixels = [to_pixel(point) for point in stroke.sorted_points()]
        if len(pixels) > 1:
            draw.line(pixels, fill=(185, 185, 185), width=max(1, size // 256))
    palette = [(32, 102, 181), (220, 80, 60), (45, 150, 90), (150, 75, 170), (220, 145, 30)]
    for order, stroke in enumerate(processed.sorted_strokes()):
        pixels = [to_pixel(point) for point in stroke.sorted_points()]
        if len(pixels) > 1:
            draw.line(pixels, fill=palette[order % len(palette)], width=max(2, size // 128), joint="curve")
    image.save(output)


def write_trajectory_csv(sample: CharacterTrajectory, output_path: str | Path) -> None:
    """Write a processed trajectory with explicit ``contact_state`` metadata."""

    import csv

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "character", "sample_id", "stroke_id", "point_id", "x", "y", "z",
        "alpha", "beta", "gamma", "state", "contact_state",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for point in sample.all_points():
            writer.writerow({
                "character": sample.character or "",
                "sample_id": sample.meta.get("sample_id", ""),
                **point.as_dict(),
                "state": int(point.state),
                "contact_state": point.state.to_name(),
            })
