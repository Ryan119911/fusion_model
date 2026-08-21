"""Replay a paper trajectory with CoppeliaSim's official UR5 model.

Live mode loads ``UR5.ttm`` from the CoppeliaSim installation, creates a
simIK pose task for its six real joints, and moves the standard model's
end-effector to each mapped paper point.  A rigid 48 mm by 6 mm-radius virtual
Langhao brush bundle is fixed to the flange, with its local +Z axis constrained
toward the paper.  Only contact writing leaves a visible trace; lift, travel,
and descend motions are animated without drawing auxiliary paths.  One draw
object is created per stroke, and lift states never connect separate strokes.  This is still a
simulation/visualization experiment: dynamics and real robot calibration are
not inferred from it.  The UR5 keeps its normal floor-mounted pose while the
default writing plane is 0.65 m above the world floor; the
``--ur5_paper_z_m`` option can override it after a reachability check.

The script has an offline mode for validating CSV selection, coordinate
mapping, lift logic, and cross-stroke safety without a running simulator.
For live mode start CoppeliaSim with the ZMQ remote API add-on enabled, then
run this file with the same trajectory CSV.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


REQUIRED_COLUMNS = {
    "character",
    "sample_id",
    "stroke_id",
    "point_id",
    "x",
    "y",
    "z",
    "alpha",
    "beta",
    "gamma",
    "state",
}


@dataclass(frozen=True)
class PoseRow:
    character: str
    sample_id: str
    stroke_id: int
    point_id: int
    x: float
    y: float
    z: float
    alpha: float
    beta: float
    gamma: float
    state: int


@dataclass(frozen=True)
class WorldPoint:
    position: Tuple[float, float, float]
    orientation: Tuple[float, float, float]
    signed_pressure_z: float
    state: int
    stroke_id: int
    point_id: int


@dataclass(frozen=True)
class CoordinateMapper:
    """Map image/paper coordinates to a compact CoppeliaSim paper frame."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    paper_width: float = 0.24
    paper_height: float = 0.24
    paper_z: float = 0.015
    margin: float = 0.018
    # Guo & Yan (2024) define H=0 at first paper contact and increase H as the
    # robot descends.  Their Langhao brush bundle is L=48 mm, R=6 mm, and the
    # collected writing range is H=11..20 mm.  Robot paper coordinates use the
    # opposite sign, hence signed pressure z = -H/1000 metres.
    posture_h_contact_mm: float = 0.0
    posture_h_max_mm: float = 20.0
    max_brush_compression: float = 0.020
    lift_height: float = 0.075
    # The CSV uses image/paper coordinates (y grows downwards).  CoppeliaSim's
    # standard top view is the reference view for writing, so the default
    # mapping keeps the paper orientation in that view.  ``flip_y=True`` is
    # retained only for legacy trajectories that were authored for a bottom
    # (up-looking) camera.
    flip_y: bool = False

    def map_row(self, row: PoseRow) -> WorldPoint:
        x_span = max(self.x_max - self.x_min, 1e-9)
        y_span = max(self.y_max - self.y_min, 1e-9)
        usable_x = self.paper_width - 2.0 * self.margin
        usable_y = self.paper_height - 2.0 * self.margin
        x_norm = (row.x - self.x_min) / x_span
        y_norm = (row.y - self.y_min) / y_span
        if self.flip_y:
            y_norm = 1.0 - y_norm
        world_x = -0.5 * self.paper_width + self.margin + x_norm * usable_x
        world_y = -0.5 * self.paper_height + self.margin + y_norm * usable_y
        descending_h_m = np.clip(
            (row.z - self.posture_h_contact_mm) / 1000.0,
            0.0,
            self.max_brush_compression,
        )
        signed_pressure_z = -float(descending_h_m)
        if row.state in (2, 3):  # UP/TRANSITION: visibly clear the paper.
            world_z = self.paper_z + self.lift_height
        else:
            # This is the virtual brush contact coordinate.  A separate TCP
            # offset is added for UR5 IK so its rigid flange never goes below
            # the paper when pressure_z is negative.
            world_z = self.paper_z + signed_pressure_z
        return WorldPoint(
            position=(float(world_x), float(world_y), float(world_z)),
            orientation=(float(row.alpha), float(row.beta), float(row.gamma)),
            signed_pressure_z=float(signed_pressure_z),
            state=int(row.state),
            stroke_id=row.stroke_id,
            point_id=row.point_id,
        )


def load_rows(
    csv_path: str | Path,
    *,
    character: str = "武",
    sample_id: Optional[str] = None,
) -> List[PoseRow]:
    """Load one character/sample while preserving stroke and point order."""
    path = Path(csv_path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"Trajectory CSV missing columns: {sorted(missing)}")
        raw = [row for row in reader if row.get("character", "") == character]
    if not raw:
        raise ValueError(f"No trajectory rows found for character {character!r}")
    if sample_id is None:
        counts: Dict[str, int] = {}
        for row in raw:
            key = row.get("sample_id", "")
            counts[key] = counts.get(key, 0) + 1
        sample_id = max(counts, key=counts.get)
    selected = [row for row in raw if row.get("sample_id", "") == sample_id]
    if not selected:
        raise ValueError(f"No trajectory rows found for {character!r}/{sample_id!r}")
    rows = [
        PoseRow(
            character=character,
            sample_id=str(row["sample_id"]),
            stroke_id=int(float(row["stroke_id"])),
            point_id=int(float(row["point_id"])),
            x=float(row["x"]),
            y=float(row["y"]),
            z=float(row["z"]),
            alpha=float(row["alpha"]),
            beta=float(row["beta"]),
            gamma=float(row["gamma"]),
            state=int(float(row["state"])),
        )
        for row in selected
    ]
    rows.sort(key=lambda item: (item.stroke_id, item.point_id))
    if not np.isfinite(np.asarray([[r.x, r.y, r.z, r.alpha, r.beta, r.gamma] for r in rows])).all():
        raise ValueError("Trajectory contains NaN or Inf pose values")
    return rows


def trajectory_provenance(
    csv_path: str | Path,
    *,
    character: str,
    sample_id: str,
) -> dict:
    """Record the exact trajectory file and selected sample used for replay."""
    path = Path(csv_path).expanduser().resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        selected = [
            row
            for row in reader
            if row.get("character", "") == character
            and row.get("sample_id", "") == sample_id
        ]
    if not selected:
        raise ValueError(
            f"No provenance rows found for {character!r}/{sample_id!r} in {path}"
        )

    def distinct(column: str) -> List[str]:
        if column not in fieldnames:
            return []
        return sorted({str(row.get(column, "")).strip() for row in selected})

    pose_ranges = {}
    nonzero_counts = {}
    for column in ("z", "alpha", "beta", "gamma"):
        values = [float(row[column]) for row in selected]
        pose_ranges[column] = [min(values), max(values)]
        nonzero_counts[column] = sum(abs(value) > 1e-8 for value in values)

    stat = path.stat()
    return {
        "requested_path": str(csv_path),
        "resolved_path": str(path),
        "sha256": digest.hexdigest(),
        "size_bytes": stat.st_size,
        "mtime_unix_s": stat.st_mtime,
        "selected_character": character,
        "selected_sample_id": sample_id,
        "selected_rows": len(selected),
        "fieldnames": fieldnames,
        "prototype": distinct("prototype"),
        "pose_frame": distinct("pose_frame"),
        "z_unit": distinct("z_unit"),
        "angle_unit": distinct("angle_unit"),
        "gamma_semantics": distinct("gamma_semantics"),
        "pose_ranges": pose_ranges,
        "nonzero_pose_counts": nonzero_counts,
    }


def validate_trajectory_identity(
    provenance: dict,
    *,
    required_prototype: Optional[str] = None,
    required_sha256: Optional[str] = None,
) -> None:
    """Abort before replay if the requested trajectory identity does not match."""
    if required_prototype:
        prototypes = provenance.get("prototype", [])
        if required_prototype not in prototypes:
            raise ValueError(
                "Trajectory prototype mismatch: "
                f"required {required_prototype!r}, found {prototypes!r} in "
                f"{provenance['resolved_path']}"
            )
    if required_sha256:
        expected = required_sha256.strip().lower()
        actual = str(provenance["sha256"]).lower()
        if actual != expected:
            raise ValueError(
                f"Trajectory SHA256 mismatch: required {expected}, found {actual}"
            )


def make_mapper(rows: Sequence[PoseRow], **kwargs: object) -> CoordinateMapper:
    """Create a mapper from the selected sample's paper-space bounds."""
    return CoordinateMapper(
        x_min=min(item.x for item in rows),
        x_max=max(item.x for item in rows),
        y_min=min(item.y for item in rows),
        y_max=max(item.y for item in rows),
        z_min=min(item.z for item in rows),
        z_max=max(item.z for item in rows),
        **kwargs,
    )


def mapped_strokes(rows: Sequence[PoseRow], mapper: CoordinateMapper) -> Dict[int, List[WorldPoint]]:
    strokes: Dict[int, List[WorldPoint]] = {}
    for row in rows:
        strokes.setdefault(row.stroke_id, []).append(mapper.map_row(row))
    return strokes


def interpolate_stroke(points: Sequence[WorldPoint], max_step: float) -> List[WorldPoint]:
    """Densify a single stroke without ever joining two different strokes."""
    if not points:
        return []
    output = [points[0]]
    step = max(float(max_step), 1e-5)
    for left, right in zip(points, points[1:]):
        a = np.asarray(left.position, dtype=np.float64)
        b = np.asarray(right.position, dtype=np.float64)
        distance = float(np.linalg.norm(b - a))
        count = max(1, int(math.ceil(distance / step)))
        for index in range(1, count + 1):
            ratio = index / count
            position = tuple((a + ratio * (b - a)).tolist())
            orientation = tuple(
                (1.0 - ratio) * np.asarray(left.orientation)
                + ratio * np.asarray(right.orientation)
            )
            output.append(
                WorldPoint(
                    position=position,
                    orientation=orientation,
                    signed_pressure_z=(
                        (1.0 - ratio) * left.signed_pressure_z
                        + ratio * right.signed_pressure_z
                    ),
                    state=right.state,
                    stroke_id=right.stroke_id,
                    point_id=right.point_id,
                )
            )
    return output


def build_report(
    rows: Sequence[PoseRow],
    strokes: Dict[int, List[WorldPoint]],
    mapper: CoordinateMapper,
    *,
    provenance: Optional[dict] = None,
) -> dict:
    state_counts: Dict[str, int] = {}
    for row in rows:
        state_counts[str(row.state)] = state_counts.get(str(row.state), 0) + 1
    active_segments = sum(
        sum(1 for left, right in zip(points, points[1:]) if left.state not in (2, 3) and right.state not in (2, 3))
        for points in strokes.values()
    )
    mapped_pressure_values = [
        mapper.map_row(row).signed_pressure_z
        for row in rows
        if row.state not in (2, 3)
    ]
    report = {
        "format": "coppeliasim_virtual_arm_replay_v1",
        "simulation_only": True,
        "character": rows[0].character,
        "sample_id": rows[0].sample_id,
        "rows": len(rows),
        "strokes": len(strokes),
        "state_counts": state_counts,
        "active_segments": active_segments,
        "cross_stroke_segments": 0,
        "coordinate_mapping": {
            "paper_width_m": mapper.paper_width,
            "paper_height_m": mapper.paper_height,
            "paper_z_m": mapper.paper_z,
            "margin_m": mapper.margin,
            "flip_y": mapper.flip_y,
            "view_reference": "bottom_view_legacy" if mapper.flip_y else "top_view",
            "source_x_bounds": [mapper.x_min, mapper.x_max],
            "source_y_bounds": [mapper.y_min, mapper.y_max],
            "source_z_bounds": [mapper.z_min, mapper.z_max],
            "z_semantics": "signed_descent_from_H0_paper_contact",
            "posture_h_contact_mm": mapper.posture_h_contact_mm,
            "posture_h_max_mm": mapper.posture_h_max_mm,
            "signed_pressure_z_range_m": [
                -mapper.max_brush_compression,
                0.0,
            ],
            "active_mapped_pressure_z_range_m": [
                min(mapped_pressure_values, default=0.0),
                max(mapped_pressure_values, default=0.0),
            ],
        },
        "safety": {
            "no_cross_stroke_drawing": True,
            "lift_states": [2, 3],
            "actual_robot_ik": False,
            "brush_physics": False,
            "rigid_virtual_pen": True,
            "rigid_pen_bundle_length_m": 0.048,
            "rigid_pen_bundle_radius_m": 0.006,
            "air_motion_trails_visible": False,
        },
    }
    if provenance is not None:
        report["trajectory_source"] = provenance
    return report


def save_offline_preview(
    strokes: Dict[int, List[WorldPoint]],
    output_dir: Path,
    report: dict,
    mapper: CoordinateMapper,
    image_size: int = 768,
) -> None:
    """Save a simple paper-frame preview without requiring a GUI or simulator."""
    from PIL import Image, ImageDraw

    output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (image_size, image_size), "white")
    draw = ImageDraw.Draw(image)
    palette = [(30, 90, 190), (190, 55, 45), (25, 130, 75), (150, 80, 180)]
    for index, points in enumerate(strokes.values()):
        color = palette[index % len(palette)]
        previous = None
        for point in points:
            x, y, _ = point.position
            px = int(round((x / 0.24 + 0.5) * image_size))
            # Keep the offline preview in the same convention as the live
            # CoppeliaSim view.  The standard top-view convention used by the
            # project has +Y downward on screen; the legacy flipped mapping is
            # retained for trajectories authored for an up-looking camera.
            if mapper.flip_y:
                py = int(round((0.5 - y / 0.24) * image_size))
            else:
                py = int(round((0.5 + y / 0.24) * image_size))
            if previous is not None and point.state not in (2, 3) and previous[2] not in (2, 3):
                draw.line([previous[0], previous[1], px, py], fill=color, width=3)
            previous = (px, py, point.state)
    image.save(output_dir / "trajectory_preview.png")
    (output_dir / "trajectory_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def save_pressure_trajectory(
    rows: Sequence[PoseRow], mapper: CoordinateMapper, output_dir: Path
) -> Path:
    """Export the pressure-space trajectory without overwriting source H."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "mapped_pressure_trajectory.csv"
    fieldnames = [
        "character", "sample_id", "stroke_id", "point_id", "state",
        "x_paper_m", "y_paper_m", "z", "z_unit", "z_semantics",
        "original_h_mm", "alpha", "beta", "gamma", "angle_unit",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            point = mapper.map_row(row)
            writer.writerow(
                {
                    "character": row.character,
                    "sample_id": row.sample_id,
                    "stroke_id": row.stroke_id,
                    "point_id": row.point_id,
                    "state": row.state,
                    "x_paper_m": point.position[0],
                    "y_paper_m": point.position[1],
                    "z": point.signed_pressure_z,
                    "z_unit": "m",
                    "z_semantics": "negative_brush_compression",
                    "original_h_mm": row.z,
                    "alpha": row.alpha,
                    "beta": row.beta,
                    "gamma": row.gamma,
                    "angle_unit": "rad",
                }
            )
    return path


def _add_drawing(sim, drawing_type: int, size: float, max_items: int, color: Sequence[float]):
    try:
        return sim.addDrawingObject(drawing_type, size, 0, -1, max_items, list(color))
    except Exception:
        return sim.addDrawingObject(sim.drawing_spherepoints, size * 0.002, 0, -1, max_items, list(color))


def _multiply_quaternions(left: Sequence[float], right: Sequence[float]) -> List[float]:
    """Multiply CoppeliaSim quaternions in ``[x, y, z, w]`` order."""
    lx, ly, lz, lw = (float(value) for value in left)
    rx, ry, rz, rw = (float(value) for value in right)
    return [
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ]


def _euler_xyz_to_quaternion(orientation: Sequence[float]) -> List[float]:
    """Convert paper-model alpha/beta/gamma radians to an XYZW quaternion."""
    alpha, beta, gamma = (float(value) for value in orientation)
    qx = [math.sin(alpha * 0.5), 0.0, 0.0, math.cos(alpha * 0.5)]
    qy = [0.0, math.sin(beta * 0.5), 0.0, math.cos(beta * 0.5)]
    qz = [0.0, 0.0, math.sin(gamma * 0.5), math.cos(gamma * 0.5)]
    return _multiply_quaternions(_multiply_quaternions(qx, qy), qz)


def _interpolate_euler_shortest(
    left: Sequence[float], right: Sequence[float], ratio: float
) -> Tuple[float, float, float]:
    """Interpolate Euler fields through the shortest wrapped angular delta."""
    result = []
    for start, end in zip(left, right):
        delta = (float(end) - float(start) + math.pi) % (2.0 * math.pi) - math.pi
        result.append(float(start) + float(ratio) * delta)
    return tuple(result)


def _quaternion_angle_error(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the shortest angular distance between two XYZW quaternions."""
    left_vec = np.asarray(list(left), dtype=np.float64)
    right_vec = np.asarray(list(right), dtype=np.float64)
    left_vec /= max(float(np.linalg.norm(left_vec)), 1e-12)
    right_vec /= max(float(np.linalg.norm(right_vec)), 1e-12)
    dot = float(np.clip(abs(np.dot(left_vec, right_vec)), -1.0, 1.0))
    return float(2.0 * math.acos(dot))


def _rotate_vector_by_quaternion(quaternion: Sequence[float], vector: Sequence[float]) -> np.ndarray:
    """Rotate a 3D vector by an XYZW quaternion."""
    x, y, z, w = (float(value) for value in quaternion)
    q_vec = np.asarray([x, y, z], dtype=np.float64)
    v = np.asarray(vector, dtype=np.float64)
    return v + 2.0 * (w * np.cross(q_vec, v) + np.cross(q_vec, np.cross(q_vec, v)))


def _tool_axis_angle_error(left: Sequence[float], right: Sequence[float]) -> float:
    """Return angular error of the tool local +Z axes, ignoring free gamma."""
    left_axis = _rotate_vector_by_quaternion(left, [0.0, 0.0, 1.0])
    right_axis = _rotate_vector_by_quaternion(right, [0.0, 0.0, 1.0])
    left_axis /= max(float(np.linalg.norm(left_axis)), 1e-12)
    right_axis /= max(float(np.linalg.norm(right_axis)), 1e-12)
    dot = float(np.clip(np.dot(left_axis, right_axis), -1.0, 1.0))
    return float(math.acos(dot))


def _world_bbox_min_z(matrix: Sequence[float], bounds: Sequence[float]) -> float:
    """Return the world-space minimum Z of an oriented local bounding box."""
    if len(matrix) != 12 or len(bounds) != 6:
        raise ValueError("Expected a 3x4 transform and six bounding-box limits")
    min_x, min_y, min_z, max_x, max_y, max_z = (float(v) for v in bounds)
    world_z = []
    for x in (min_x, max_x):
        for y in (min_y, max_y):
            for z in (min_z, max_z):
                world_z.append(
                    float(matrix[8]) * x
                    + float(matrix[9]) * y
                    + float(matrix[10]) * z
                    + float(matrix[11])
                )
    return min(world_z)


def _shape_local_bbox_bounds(sim, shape: int) -> Tuple[float, ...]:
    """Read a shape's static local bounding-box limits once."""
    names = (
        "objfloatparam_objbbox_min_x",
        "objfloatparam_objbbox_min_y",
        "objfloatparam_objbbox_min_z",
        "objfloatparam_objbbox_max_x",
        "objfloatparam_objbbox_max_y",
        "objfloatparam_objbbox_max_z",
    )
    return tuple(
        float(sim.getObjectFloatParam(shape, getattr(sim, name))) for name in names
    )


def _shape_world_bbox_min_z(
    sim,
    shape: int,
    bounds: Optional[Sequence[float]] = None,
) -> float:
    """Measure a shape's actual lowest geometry point in world coordinates."""
    try:
        local_bounds = tuple(bounds) if bounds is not None else _shape_local_bbox_bounds(sim, shape)
        matrix = [float(v) for v in sim.getObjectMatrix(shape, -1)]
        return _world_bbox_min_z(matrix, local_bounds)
    except Exception:
        # A few older remote-API builds do not expose object bounding-box
        # parameters.  Keep a conservative fallback so replay still works.
        return float(sim.getObjectPosition(shape, -1)[2])


def _remove_existing_ur5_models(sim) -> int:
    """Remove UR5 model roots left by an earlier ``--keep_scene`` replay.

    Live replays deliberately keep the scene open for visual inspection.  If
    another replay then loads a fresh ``UR5.ttm`` without removing the prior
    one, CoppeliaSim displays multiple arms (often with an older arm still at
    the ground plane).  Only exact model-root aliases are removed; links,
    drawing objects, and unrelated user models are left untouched.
    """
    roots = []
    try:
        handles = sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 0)
    except Exception:
        handles = []
    for handle in handles:
        try:
            alias = str(sim.getObjectAlias(handle, 1))
        except Exception:
            continue
        leaf = alias.rsplit("/", 1)[-1]
        if leaf == "UR5" or (leaf.startswith("UR5[") and leaf.endswith("]")):
            # A model root is the only object with an alias consisting solely
            # of UR5/UR5[n]; child links have additional path components.
            if alias.count("/") == 1:
                roots.append(handle)
    removed = 0
    for handle in roots:
        try:
            sim.removeModel(handle)
            removed += 1
        except Exception:
            # A stale handle may disappear if CoppeliaSim removes a model as
            # part of another cleanup; the new load can still proceed.
            continue
    return removed


def _load_ur5_ik(
    sim,
    ik,
    model_path: str,
    base_x: float,
    base_z: float,
    orientation_mode: str,
):
    """Load CoppeliaSim's official UR5 model and configure simIK.

    The model is not recreated by this project: all six joints and visible
    links come from ``UR5.ttm``.  IK is position-constrained so the standard
    UR5 wrist orientation is retained while the pen tip follows the paper.
    """
    _remove_existing_ur5_models(sim)
    root = sim.loadModel(model_path)
    sim.setObjectPosition(root, -1, [float(base_x), 0.0, float(base_z)])
    disabled_model_scripts = 0
    for script in sim.getObjectsInTree(root, sim.object_script_type, 0):
        try:
            sim.setScriptInt32Param(script, sim.scriptintparam_enabled, 0)
            disabled_model_scripts += 1
        except Exception as exc:
            raise RuntimeError(
                "Could not disable the stock UR5 controller script"
            ) from exc
    joints = sim.getObjectsInTree(root, sim.object_joint_type, 0)
    shapes = sim.getObjectsInTree(root, sim.object_shape_type, 0)
    if len(joints) != 6:
        raise RuntimeError(f"Expected 6 UR5 joints, found {len(joints)}")
    # The stock model's zero pose is elbow-down and can place the long middle
    # links below the ground plane.  Seed simIK from a compact elbow-up pose so
    # both the initial scene and every rollback state are physically safe.
    initial_joint_seed = [
        0.0,
        -0.6,
        1.4,
        -1.5,
        -math.pi / 2.0,
        0.0,
    ]
    for joint, value in zip(joints, initial_joint_seed):
        sim.setJointPosition(joint, value)
    last_link = None
    for handle in shapes:
        try:
            if "link7_visible" in sim.getObjectAlias(handle, 1):
                last_link = handle
                break
        except Exception:
            continue
    if last_link is None:
        last_link = shapes[-1]
    # The official model exposes a dedicated attachment frame named
    # ``connection`` at the black tool-mount point.  link7_visible is only a
    # cosmetic shape whose origin lies inside the wrist housing.
    tool_mount = None
    for handle in sim.getObjectsInTree(root, sim.handle_all, 0):
        try:
            alias = sim.getObjectAlias(handle, 1).lower().rstrip("/")
        except Exception:
            continue
        if alias.endswith("/connection") or alias == "connection":
            tool_mount = handle
            break
    if tool_mount is None:
        raise RuntimeError("Official UR5 model has no tool connection frame")
    # CoppeliaSim quaternions are [x, y, z, w].  The virtual pen and IK tip use
    # local +Z as the pen axis.  A pi rotation about world X maps local +Z to
    # world -Z, so the pen tip always points toward the horizontal paper.
    initial_tip_quaternion = [
        float(value) for value in sim.getObjectQuaternion(tool_mount, -1)
    ]
    pen_down_quaternion = [1.0, 0.0, 0.0, 0.0]
    tip = sim.createDummy(0.010, 12 * [0.0])
    sim.setObjectAlias(tip, "penMountIKTip")
    sim.setObjectParent(tip, tool_mount, False)
    tip_position = sim.getObjectPosition(tool_mount, -1)
    sim.setObjectPosition(tip, -1, tip_position)
    # Keep the IK tip aligned with the actual terminal link.  The previous
    # implementation put the 90-degree rotation on the child tip itself, which
    # hid the wrist rotation from simIK and could leave the visible UR5 flange
    # perpendicular even though the dummy reported a horizontal pose.
    sim.setObjectQuaternion(tip, -1, initial_tip_quaternion)
    target = sim.createDummy(0.012, 12 * [0.0])
    sim.setObjectPosition(target, -1, tip_position)
    sim.setObjectQuaternion(target, -1, pen_down_quaternion)
    environment = ik.createEnvironment()
    group = ik.createGroup(environment)
    # Alpha/beta constrain the downward pen axis; csv_pose additionally uses
    # gamma around that axis.  Neither mode permits an upward-pointing pen.
    if orientation_mode == "csv_pose":
        orientation_constraints = ik.constraint_pose
        orientation_constraint_name = "csv_pose_full_xyz"
    else:
        orientation_constraints = ik.constraint_position | ik.constraint_alpha_beta
        orientation_constraint_name = "pen_down_alpha_beta"
    add_result = ik.addElementFromScene(
        environment, group, root, tip, target, orientation_constraints
    )
    if isinstance(add_result, tuple) and int(add_result[0]) != 0:
        raise RuntimeError(f"simIK.addElementFromScene failed: {add_result}")
    if not isinstance(add_result, tuple) or len(add_result) < 2:
        raise RuntimeError(f"simIK.addElementFromScene returned no object mapping: {add_result}")
    sim_to_ik_mapping = add_result[1]
    ik_tip = sim_to_ik_mapping[tip]
    ik.setGroupCalculation(
        environment, group, ik.method_damped_least_squares, 0.03, 500
    )
    body_shapes = []
    for handle in shapes:
        try:
            alias = sim.getObjectAlias(handle, 1).lower()
        except Exception:
            alias = ""
        if handle != last_link and "link" in alias and "link1_visible" not in alias:
            body_shapes.append(handle)
    body_shape_bounds = {
        handle: _shape_local_bbox_bounds(sim, handle) for handle in body_shapes
    }
    initial_body_geometry_min_z = min(
        (
            _shape_world_bbox_min_z(sim, shape, body_shape_bounds[shape])
            for shape in body_shapes
        ),
        default=float("inf"),
    )
    return {
        "root": root,
        "joints": joints,
        "body_shapes": body_shapes,
        "body_shape_bounds": body_shape_bounds,
        "terminal_shape": last_link,
        "tool_mount": tool_mount,
        "tool_mount_alias": sim.getObjectAlias(tool_mount, 1),
        "tip": tip,
        "target": target,
        "tool_quaternion_xyzw": pen_down_quaternion,
        "tool_orientation_constraint": orientation_constraint_name,
        "orientation_mode": orientation_mode,
        "initial_tip_quaternion_xyzw": initial_tip_quaternion,
        "initial_joint_seed_rad": initial_joint_seed,
        "initial_body_geometry_min_z_m": initial_body_geometry_min_z,
        "disabled_stock_model_scripts": disabled_model_scripts,
        "environment": environment,
        "group": group,
        "ik_tip": ik_tip,
        "tip_position": tip_position,
    }


def _signed_xy_clearance_to_rect(
    point: Sequence[float], center_x: float, center_y: float, width: float, height: float
) -> float:
    """Return signed XY distance from a point to a rectangle.

    Positive values are outside the rectangle; negative values mean that the
    point lies inside it.  The live replay uses this with a conservative link
    radius to reject body links whose projection could cover the writing
    plane.  The terminal wrist link is intentionally excluded because it is
    the writing end-effector itself.
    """
    dx = abs(float(point[0]) - float(center_x))
    dy = abs(float(point[1]) - float(center_y))
    half_w = 0.5 * float(width)
    half_h = 0.5 * float(height)
    outside_dx = max(dx - half_w, 0.0)
    outside_dy = max(dy - half_h, 0.0)
    if outside_dx > 0.0 or outside_dy > 0.0:
        return float(math.hypot(outside_dx, outside_dy))
    return -float(min(half_w - dx, half_h - dy))


def _add_paper(
    sim,
    width: float,
    height: float,
    top_z: float,
    center_x: float = 0.0,
    center_y: float = 0.0,
):
    """Add a thin static paper plane only for visual contact reference."""
    paper = sim.createPrimitiveShape(
        sim.primitiveshape_cuboid, [width, height, 0.004], 0
    )
    sim.setObjectAlias(paper, "writingPaper")
    try:
        sim.setObjectInt32Param(paper, sim.shapeintparam_static, 1)
    except Exception:
        pass
    sim.setObjectPosition(paper, -1, [float(center_x), float(center_y), float(top_z) - 0.002])
    try:
        sim.setShapeColor(paper, None, sim.colorcomponent_ambient_diffuse, [0.92, 0.92, 0.86])
    except Exception:
        pass
    return paper


def _add_paper_reference(
    sim,
    width: float,
    height: float,
    top_z: float,
    center_x: float = 0.0,
    center_y: float = 0.0,
):
    """Draw a thin outline at the actual writing height.

    Some CoppeliaSim 4.7 remote API builds reject ``createPureShape`` calls
    issued after connecting.  The outline is therefore kept as a drawing
    object as well as the optional cosmetic cuboid, so the low writing plane
    remains visible even when the cuboid cannot be created.
    """
    drawing = _add_drawing(sim, sim.drawing_lines, 2, 32, [0.78, 0.78, 0.70])
    half_w = float(width) * 0.5
    half_h = float(height) * 0.5
    corners = [
        (float(center_x) - half_w, float(center_y) - half_h, float(top_z)),
        (float(center_x) + half_w, float(center_y) - half_h, float(top_z)),
        (float(center_x) + half_w, float(center_y) + half_h, float(top_z)),
        (float(center_x) - half_w, float(center_y) + half_h, float(top_z)),
    ]
    for start, end in zip(corners, corners[1:] + corners[:1]):
        try:
            sim.addDrawingObjectItem(drawing, list(start) + list(end))
        except Exception:
            break
    return drawing


def _set_tool_visible(sim, tools: Sequence[int], visible: bool) -> None:
    for tool in tools:
        try:
            sim.setObjectInt32Param(
                tool,
                sim.objintparam_visibility_layer,
                1 if visible else 0,
            )
        except Exception:
            pass


def _add_ur5_tool(
    sim,
    tip: int,
    brush_length: float,
    handle_length: float,
    brush_radius: float = 0.006,
):
    """Attach a visible rigid holder, brush body and distal tip marker."""
    diameter = 2.0 * float(brush_radius)
    holder = sim.createPrimitiveShape(
        sim.primitiveshape_cuboid, [0.050, 0.036, 0.008], 0
    )
    handle_diameter = 0.010
    handle_shape = sim.createPrimitiveShape(
        sim.primitiveshape_cylinder,
        [handle_diameter, handle_diameter, float(handle_length)],
        0,
    )
    brush = sim.createPrimitiveShape(
        sim.primitiveshape_cone,
        [diameter, diameter, float(brush_length)],
        0,
    )
    pen_tip = sim.createDummy(0.006, 12 * [0.0])
    objects = [holder, handle_shape, brush, pen_tip]
    aliases = [
        "virtualPenHolder",
        "virtualPenHandle",
        "virtualPenBrushBundle",
        "virtualPenTip",
    ]
    for obj, alias in zip(objects, aliases):
        sim.setObjectAlias(obj, alias)
        sim.setObjectParent(obj, tip, False)
        sim.setObjectQuaternion(obj, tip, [0.0, 0.0, 0.0, 1.0])
    # The holder's broad XY face is parallel to paper.  All pen components
    # extend along the flange dummy's local +Z, constrained to world -Z.
    sim.setObjectPosition(holder, tip, [0.0, 0.0, 0.004])
    sim.setObjectPosition(
        handle_shape, tip, [0.0, 0.0, 0.5 * float(handle_length)]
    )
    sim.setObjectPosition(
        brush,
        tip,
        [0.0, 0.0, float(handle_length) + 0.5 * float(brush_length)],
    )
    total_length = float(handle_length) + float(brush_length)
    sim.setObjectPosition(pen_tip, tip, [0.0, 0.0, total_length])
    for shape, color in (
        (holder, [0.18, 0.18, 0.20]),
        (handle_shape, [0.42, 0.18, 0.06]),
        (brush, [0.04, 0.04, 0.04]),
    ):
        sim.setShapeColor(shape, None, sim.colorcomponent_ambient_diffuse, color)
        try:
            sim.setObjectInt32Param(shape, sim.shapeintparam_static, 1)
        except Exception:
            pass
    # Do not briefly show the stock model's arbitrary initial wrist pose.  The
    # rigid pen becomes visible only after the first accepted downward IK pose.
    _set_tool_visible(sim, objects, False)
    return {
        "objects": objects,
        "pen_tip": pen_tip,
        "handle_length_m": float(handle_length),
        "brush_bundle_length_m": float(brush_length),
        "tcp_length_m": total_length,
    }


def run_live(
    rows: Sequence[PoseRow],
    mapper: CoordinateMapper,
    *,
    max_step: float,
    interval: float,
    transition_step: float,
    transition_interval: float,
    pen_lift_clearance: float,
    virtual_brush_length: float,
    virtual_pen_handle_length: float,
    arm_base_z: float,
    ground_clearance_m: float,
    tip_paper_clearance_m: float,
    keep_scene: bool,
    client_port: int,
    scene_output: Optional[str],
    arm_model: str,
    ur5_model_path: str,
    ur5_base_x: float,
    ur5_paper_z: float,
    paper_offset_x: float,
    paper_offset_y: float,
    body_clearance_radius: float,
    paper_width: float,
    paper_height: float,
    orientation_mode: str,
    strict_ik: bool,
    start_simulation: bool,
) -> dict:
    """Create the scene objects and replay the mapped trajectory."""
    zmq_root = os.environ.get(
        "COPPELIASIM_ZMQ_CLIENT",
        "/home/robot/CoppeliaSim_Edu_V4_7_0_rev4_Ubuntu22_04/programming/zmqRemoteApi/clients/python",
    )
    if zmq_root not in sys.path:
        sys.path.insert(0, zmq_root)
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient

    client = RemoteAPIClient(host="localhost", port=client_port)
    sim = client.require("sim")
    # A running CoppeliaSim clock is useful for an authentic GUI replay, but
    # the stock UR5 dynamic controller must not compete with remote simIK.
    # Rebuild from a stopped state and disable physics handling while keeping
    # the simulation/play state active for deterministic kinematic animation.
    if sim.getSimulationState() != sim.simulation_stopped:
        sim.stopSimulation()
        deadline = time.time() + 8.0
        while (
            sim.getSimulationState() != sim.simulation_stopped
            and time.time() < deadline
        ):
            time.sleep(0.05)
        if sim.getSimulationState() != sim.simulation_stopped:
            raise RuntimeError("CoppeliaSim did not reach stopped state before replay")
    try:
        sim.setBoolParam(sim.boolparam_dynamics_handling_enabled, False)
    except Exception as exc:
        raise RuntimeError(
            "Could not disable CoppeliaSim dynamics for deterministic remote IK"
        ) from exc
    if arm_model != "ur5":
        raise ValueError("Only the official CoppeliaSim UR5 model is supported in live mode")
    ik = client.require("simIK")
    strokes = mapped_strokes(rows, mapper)
    colors = ([0.05, 0.35, 0.95], [0.95, 0.18, 0.08], [0.08, 0.65, 0.22], [0.65, 0.2, 0.8])
    trajectory_drawings = []
    for index, _ in enumerate(strokes):
        trajectory_drawings.append(
            _add_drawing(sim, sim.drawing_lines, 3, 200000, colors[index % len(colors)])
        )
    ur5 = _load_ur5_ik(
        sim,
        ik,
        ur5_model_path,
        ur5_base_x,
        arm_base_z,
        orientation_mode,
    )
    if ur5["initial_body_geometry_min_z_m"] < float(ground_clearance_m):
        raise RuntimeError(
            "UR5 initial elbow-up seed violates ground clearance: "
            f"{ur5['initial_body_geometry_min_z_m']:.6f} m < "
            f"{float(ground_clearance_m):.6f} m"
        )
    initial_tip_paper_clearance = float(ur5["tip_position"][2] - ur5_paper_z)
    if initial_tip_paper_clearance < float(tip_paper_clearance_m):
        raise RuntimeError(
            "UR5 initial tip is below the required paper clearance: "
            f"{initial_tip_paper_clearance:.6f} m < "
            f"{float(tip_paper_clearance_m):.6f} m"
        )
    _add_paper(
        sim,
        paper_width,
        paper_height,
        ur5_paper_z,
        center_x=paper_offset_x,
        center_y=paper_offset_y,
    )
    _add_paper_reference(
        sim,
        paper_width,
        paper_height,
        ur5_paper_z,
        center_x=paper_offset_x,
        center_y=paper_offset_y,
    )
    # The CSV pressure coordinate may be negative.  The rigid UR5 flange is
    # therefore separated from the virtual paper contact point by a positive
    # brush/TCP length instead of being commanded to the pressure coordinate.
    rigid_pen = _add_ur5_tool(
        sim,
        ur5["tip"],
        virtual_brush_length,
        virtual_pen_handle_length,
    )
    virtual_pen_tcp_length = rigid_pen["tcp_length_m"]
    rigid_pen_visible = False
    tip_drawing = _add_drawing(sim, sim.drawing_spherepoints, 0.008, 200000, [0.9, 0.1, 0.05])
    previous_by_stroke: Dict[int, Tuple[float, float, float]] = {}
    offset = np.asarray(
        [paper_offset_x, paper_offset_y, float(ur5_paper_z - mapper.paper_z)],
        dtype=np.float64,
    )
    ik_success = 0
    ik_failures = 0
    max_residual = 0.0
    ik_success_by_stroke: Dict[str, int] = {}
    ik_failure_by_stroke: Dict[str, int] = {}
    previous_target: Optional[np.ndarray] = None
    previous_orientation: Optional[Tuple[float, float, float]] = None
    approach_success = 0
    approach_failures = 0
    body_overlap_steps = 0
    min_body_clearance = float("inf")
    planned_fallback_strokes = 0
    max_orientation_residual = 0.0
    max_tool_axis_residual = 0.0
    below_ground_steps = 0
    ground_rejected_steps = 0
    min_body_link_center_z = float("inf")
    min_attempted_body_link_center_z = float("inf")
    min_body_geometry_z = float("inf")
    min_attempted_body_geometry_z = float("inf")
    motion_phase_steps = {"lift": 0, "travel": 0, "descend": 0}
    motion_phase_failures = {"lift": 0, "travel": 0, "descend": 0}
    paper_target_clamp_steps = 0
    paper_rejected_steps = 0
    min_attempted_tip_paper_clearance = float("inf")
    min_tip_paper_clearance = float("inf")
    min_pen_endpoint_paper_clearance = float("inf")
    max_pen_endpoint_paper_clearance = -float("inf")
    upward_pen_rejected_steps = 0
    minimum_pen_downward_component = float("inf")

    def flange_target(point: WorldPoint) -> np.ndarray:
        target = np.asarray(point.position, dtype=np.float64) + offset
        target[2] += float(virtual_pen_tcp_length)
        return target

    def paper_trace(position: Sequence[float]) -> List[float]:
        return [float(position[0]), float(position[1]), float(ur5_paper_z) + 0.001]

    def solve_target(
        target_position: np.ndarray,
        target_orientation: Sequence[float],
    ) -> Tuple[np.ndarray, int, float]:
        nonlocal ik_success, ik_failures, max_residual, max_orientation_residual
        nonlocal max_tool_axis_residual
        nonlocal body_overlap_steps, min_body_clearance
        nonlocal below_ground_steps, ground_rejected_steps
        nonlocal min_body_link_center_z, min_attempted_body_link_center_z
        nonlocal min_body_geometry_z, min_attempted_body_geometry_z
        nonlocal paper_target_clamp_steps, paper_rejected_steps
        nonlocal min_attempted_tip_paper_clearance, min_tip_paper_clearance
        nonlocal upward_pen_rejected_steps, minimum_pen_downward_component
        nonlocal rigid_pen_visible
        nonlocal min_pen_endpoint_paper_clearance, max_pen_endpoint_paper_clearance
        target_position = np.asarray(target_position, dtype=np.float64).copy()
        minimum_flange_clearance = max(
            float(tip_paper_clearance_m),
            float(virtual_pen_tcp_length) - float(mapper.max_brush_compression),
        )
        minimum_tip_z = float(ur5_paper_z) + minimum_flange_clearance
        if float(target_position[2]) < minimum_tip_z:
            target_position[2] = minimum_tip_z
            paper_target_clamp_steps += 1
        joint_snapshot = [float(sim.getJointPosition(joint)) for joint in ur5["joints"]]
        sim.setObjectPosition(ur5["target"], -1, target_position.tolist())
        desired_quaternion = ur5["tool_quaternion_xyzw"]
        if orientation_mode == "csv_pose":
            desired_quaternion = _multiply_quaternions(
                ur5["tool_quaternion_xyzw"],
                _euler_xyz_to_quaternion(target_orientation),
            )
        sim.setObjectQuaternion(ur5["target"], -1, desired_quaternion)
        # Solve in the private IK world first.  The visible UR5 is only updated
        # after the candidate tip has passed the paper-height guard, preventing
        # rejected below-paper configurations from flashing in the GUI.
        ik.syncFromSim(ur5["environment"], [ur5["group"]])
        result = ik.handleGroup(
            ur5["environment"],
            ur5["group"],
            {"syncWorlds": False, "allowError": not strict_ik},
        )
        status = int(result[0]) if isinstance(result, tuple) else int(result)
        attempted_tip_pose = ik.getObjectPose(
            ur5["environment"], ur5["ik_tip"], ik.handle_world
        )
        attempted_tip_position = np.asarray(attempted_tip_pose[:3], dtype=np.float64)
        attempted_axis = _rotate_vector_by_quaternion(
            attempted_tip_pose[3:7], [0.0, 0.0, 1.0]
        )
        attempted_axis /= max(float(np.linalg.norm(attempted_axis)), 1e-12)
        # The fitted alpha/beta range is at most about 11.2 degrees jointly.
        # Twenty degrees leaves numerical IK tolerance but forbids a sideways
        # or upward pen under every circumstance.
        pen_points_down = float(-attempted_axis[2]) >= math.cos(math.radians(20.0))
        if not pen_points_down:
            status = 0
            upward_pen_rejected_steps += 1
        attempted_tip_clearance = float(attempted_tip_position[2] - ur5_paper_z)
        min_attempted_tip_paper_clearance = min(
            min_attempted_tip_paper_clearance, attempted_tip_clearance
        )
        paper_rejected = attempted_tip_clearance < float(tip_paper_clearance_m)
        if paper_rejected:
            status = 0
            paper_rejected_steps += 1
        if status == 1:
            ik.syncToSim(ur5["environment"], [ur5["group"]])
            if not rigid_pen_visible:
                _set_tool_visible(sim, rigid_pen["objects"], True)
                rigid_pen_visible = True
        attempted_link_z = [
            float(sim.getObjectPosition(shape, -1)[2])
            for shape in ur5["body_shapes"]
        ]
        attempted_geometry_z = [
            _shape_world_bbox_min_z(
                sim, shape, ur5["body_shape_bounds"][shape]
            )
            for shape in ur5["body_shapes"]
        ]
        attempted_min_link_z = min(attempted_link_z, default=float("inf"))
        attempted_min_geometry_z = min(attempted_geometry_z, default=float("inf"))
        min_attempted_body_link_center_z = min(
            min_attempted_body_link_center_z, attempted_min_link_z
        )
        min_attempted_body_geometry_z = min(
            min_attempted_body_geometry_z, attempted_min_geometry_z
        )
        if attempted_min_geometry_z < 0.0:
            below_ground_steps += 1
        if attempted_min_geometry_z < float(ground_clearance_m):
            for joint, position in zip(ur5["joints"], joint_snapshot):
                sim.setJointPosition(joint, position)
            status = 0
            ground_rejected_steps += 1
        final_link_z = [
            float(sim.getObjectPosition(shape, -1)[2])
            for shape in ur5["body_shapes"]
        ]
        min_body_link_center_z = min(
            min_body_link_center_z, min(final_link_z, default=float("inf"))
        )
        final_geometry_z = [
            _shape_world_bbox_min_z(
                sim, shape, ur5["body_shape_bounds"][shape]
            )
            for shape in ur5["body_shapes"]
        ]
        min_body_geometry_z = min(
            min_body_geometry_z, min(final_geometry_z, default=float("inf"))
        )
        actual_position = np.asarray(sim.getObjectPosition(ur5["tip"], -1), dtype=np.float64)
        actual_pen_endpoint = np.asarray(
            sim.getObjectPosition(rigid_pen["pen_tip"], -1), dtype=np.float64
        )
        pen_endpoint_clearance = float(actual_pen_endpoint[2] - ur5_paper_z)
        min_pen_endpoint_paper_clearance = min(
            min_pen_endpoint_paper_clearance, pen_endpoint_clearance
        )
        max_pen_endpoint_paper_clearance = max(
            max_pen_endpoint_paper_clearance, pen_endpoint_clearance
        )
        min_tip_paper_clearance = min(
            min_tip_paper_clearance,
            float(actual_position[2] - ur5_paper_z),
        )
        residual = float(np.linalg.norm(actual_position - target_position))
        max_residual = max(max_residual, residual)
        actual_quaternion = sim.getObjectQuaternion(ur5["tip"], -1)
        actual_axis = _rotate_vector_by_quaternion(
            actual_quaternion, [0.0, 0.0, 1.0]
        )
        actual_axis /= max(float(np.linalg.norm(actual_axis)), 1e-12)
        if rigid_pen_visible:
            minimum_pen_downward_component = min(
                minimum_pen_downward_component, float(-actual_axis[2])
            )
        max_orientation_residual = max(
            max_orientation_residual,
            _quaternion_angle_error(actual_quaternion, desired_quaternion),
        )
        max_tool_axis_residual = max(
            max_tool_axis_residual,
            _tool_axis_angle_error(actual_quaternion, desired_quaternion),
        )
        body_overlaps = 0
        for shape in ur5["body_shapes"]:
            shape_position = sim.getObjectPosition(shape, -1)
            signed_clearance = _signed_xy_clearance_to_rect(
                shape_position,
                paper_offset_x,
                paper_offset_y,
                paper_width,
                paper_height,
            )
            effective_clearance = signed_clearance - float(body_clearance_radius)
            min_body_clearance = min(min_body_clearance, effective_clearance)
            if effective_clearance < 0.0:
                body_overlaps += 1
        if body_overlaps:
            body_overlap_steps += 1
        if status == 1:
            ik_success += 1
        else:
            ik_failures += 1
        return actual_position, status, residual

    def animate_motion_phase(
        start_position: np.ndarray,
        end_position: np.ndarray,
        start_orientation: Sequence[float],
        end_orientation: Sequence[float],
        phase: str,
    ) -> np.ndarray:
        """Animate one explicit lift/travel/descend phase without teleporting."""
        distance = float(np.linalg.norm(end_position - start_position))
        count = max(
            1,
            int(math.ceil(distance / max(float(transition_step), 1e-5))),
        )
        previous_actual = np.asarray(
            sim.getObjectPosition(ur5["tip"], -1), dtype=np.float64
        )
        for index in range(1, count + 1):
            ratio = index / count
            position = start_position + ratio * (end_position - start_position)
            orientation = _interpolate_euler_shortest(
                start_orientation, end_orientation, ratio
            )
            actual, status, _ = solve_target(position, orientation)
            motion_phase_steps[phase] += 1
            if status != 1:
                motion_phase_failures[phase] += 1
            # Lift/travel/descend motion is executed and timed, but no ink or
            # auxiliary path is drawn while the rigid pen is off the paper.
            previous_actual = actual
            time.sleep(max(float(transition_interval), 0.0))
        return previous_actual

    simulation_started_here = False
    try:
        if start_simulation and sim.getSimulationState() == sim.simulation_stopped:
            sim.startSimulation()
            simulation_started_here = True
        for stroke_index, (stroke_id, raw_points) in enumerate(strokes.items()):
            previous_by_stroke.pop(stroke_id, None)
            stroke_success = 0
            stroke_failures = 0
            dense_points = interpolate_stroke(raw_points, max_step)
            first_target = flange_target(dense_points[0])
            first_orientation = dense_points[0].orientation
            lift_plane_z = max(
                float(ur5_paper_z) + float(pen_lift_clearance),
                float(first_target[2]) + float(pen_lift_clearance),
            )
            if previous_target is None:
                # Start from the model's home pose, visibly travel to a point
                # above the first stroke, then descend vertically to contact.
                current_tip = np.asarray(
                    sim.getObjectPosition(ur5["tip"], -1), dtype=np.float64
                )
                hover_target = first_target.copy()
                hover_target[2] = lift_plane_z
                before_steps = motion_phase_steps["travel"]
                before_failures = motion_phase_failures["travel"]
                hover_actual = animate_motion_phase(
                    current_tip,
                    hover_target,
                    first_orientation,
                    first_orientation,
                    "travel",
                )
                added_steps = motion_phase_steps["travel"] - before_steps
                added_failures = motion_phase_failures["travel"] - before_failures
                approach_failures += added_failures
                approach_success += added_steps - added_failures
                before_failures = motion_phase_failures["descend"]
                before_steps = motion_phase_steps["descend"]
                animate_motion_phase(
                    hover_actual,
                    first_target,
                    first_orientation,
                    first_orientation,
                    "descend",
                )
                added_steps = motion_phase_steps["descend"] - before_steps
                added_failures = motion_phase_failures["descend"] - before_failures
                approach_failures += added_failures
                approach_success += added_steps - added_failures
            if previous_target is not None:
                # Three explicit phases make pen-up motion legible: vertical
                # lift, high horizontal travel, then vertical descent.
                previous_pose = previous_orientation or first_orientation
                lift_start = np.asarray(
                    sim.getObjectPosition(ur5["tip"], -1), dtype=np.float64
                )
                lift_end = lift_start.copy()
                lift_end[2] = max(lift_plane_z, float(lift_start[2]))
                lift_actual = animate_motion_phase(
                    lift_start,
                    lift_end,
                    previous_pose,
                    previous_pose,
                    "lift",
                )
                travel_end = first_target.copy()
                travel_end[2] = lift_end[2]
                travel_actual = animate_motion_phase(
                    lift_actual,
                    travel_end,
                    previous_pose,
                    first_orientation,
                    "travel",
                )
                animate_motion_phase(
                    travel_actual,
                    first_target,
                    first_orientation,
                    first_orientation,
                    "descend",
                )
            for point in dense_points:
                target_position = flange_target(point)
                actual_position, status, residual = solve_target(
                    target_position, point.orientation
                )
                if status == 1:
                    stroke_success += 1
                else:
                    stroke_failures += 1
                if point.state not in (2, 3):
                    # Ink follows the actual distal virtual-pen marker, not
                    # the UR5 flange or IK dummy.  Z is projected onto paper
                    # only for displaying the deposited trace.
                    actual_pen_endpoint = sim.getObjectPosition(
                        rigid_pen["pen_tip"], -1
                    )
                    actual_trace = paper_trace(actual_pen_endpoint)
                    previous = previous_by_stroke.get(stroke_id)
                    if previous is not None:
                        sim.addDrawingObjectItem(
                            trajectory_drawings[stroke_index], list(previous) + actual_trace
                        )
                    sim.addDrawingObjectItem(tip_drawing, actual_trace)
                    previous_by_stroke[stroke_id] = tuple(actual_trace)
                else:
                    previous_by_stroke.pop(stroke_id, None)
                if point.state in (2, 3):
                    motion_phase_steps["lift"] += 1
                    if status != 1:
                        motion_phase_failures["lift"] += 1
                    time.sleep(max(float(transition_interval), 0.0))
                else:
                    time.sleep(max(interval, 0.0))
            previous_target = flange_target(dense_points[-1])
            previous_orientation = dense_points[-1].orientation
            ik_success_by_stroke[str(stroke_id)] = stroke_success
            ik_failure_by_stroke[str(stroke_id)] = stroke_failures
        if scene_output:
            scene_path = str(Path(scene_output).expanduser().resolve())
            if not sim.saveScene(scene_path):
                raise RuntimeError(f"CoppeliaSim failed to save scene: {scene_path}")
            print(f"[DONE] Saved CoppeliaSim scene: {scene_path}")
    finally:
        if simulation_started_here and not keep_scene:
            try:
                sim.stopSimulation()
            except Exception:
                pass
    print("[DONE] CoppeliaSim virtual-arm trajectory replay finished")
    return {
        "arm_model": "CoppeliaSim_official_UR5.ttm",
        "actual_robot_ik": True,
        "ik_success_steps": ik_success,
        "ik_failure_steps": ik_failures,
        "ik_max_residual_m": max_residual,
        "ik_max_orientation_residual_rad": max_orientation_residual,
        "ik_max_tool_axis_residual_rad": max_tool_axis_residual,
        "ik_success_by_stroke": ik_success_by_stroke,
        "ik_failure_by_stroke": ik_failure_by_stroke,
        "ik_approach_success_steps": approach_success,
        "ik_approach_failure_steps": approach_failures,
        "dynamics_enabled": False,
        "kinematic_replay_with_running_simulation": True,
        "coppeliasim_simulation_started": start_simulation,
        "coppeliasim_simulation_left_running": bool(start_simulation and keep_scene),
        "arm_base_z_m": arm_base_z,
        "paper_top_z_m": ur5_paper_z,
        "paper_height_above_ground_m": ur5_paper_z,
        "replay_max_step_m": max_step,
        "replay_interval_s": interval,
        "transition_step_m": transition_step,
        "transition_interval_s": transition_interval,
        "pen_lift_clearance_m": pen_lift_clearance,
        "virtual_brush_bundle_length_m": virtual_brush_length,
        "virtual_pen_handle_length_m": virtual_pen_handle_length,
        "virtual_pen_tcp_length_m": virtual_pen_tcp_length,
        "virtual_brush_max_compression_m": mapper.max_brush_compression,
        "signed_pressure_z_range_m": [
            -mapper.max_brush_compression,
            0.0,
        ],
        "mechanical_target_semantics": (
            "paper_contact_z + signed_pressure_z + fixed_pen_tcp_length"
        ),
        "motion_phase_steps": motion_phase_steps,
        "motion_phase_failures": motion_phase_failures,
        "paper_offset_x_m": paper_offset_x,
        "paper_offset_y_m": paper_offset_y,
        "tool_orientation_constraint": ur5["tool_orientation_constraint"],
        "orientation_mode": orientation_mode,
        "orientation_mapping": (
            "local_xyz_relative_to_pen_down_base"
            if orientation_mode == "csv_pose"
            else "fixed_pen_down_with_free_gamma"
        ),
        "rigid_pen_attachment": "fixed_to_official_ur5_connection_frame",
        "rigid_pen_mount_source": ur5["tool_mount_alias"],
        "air_motion_trails_visible": False,
        "ik_allow_error": not strict_ik,
        "tool_quaternion_xyzw": ur5["tool_quaternion_xyzw"],
        "initial_tip_quaternion_xyzw": ur5["initial_tip_quaternion_xyzw"],
        "initial_joint_seed_rad": ur5["initial_joint_seed_rad"],
        "disabled_stock_model_scripts": ur5["disabled_stock_model_scripts"],
        "initial_body_geometry_min_z_m": ur5["initial_body_geometry_min_z_m"],
        "initial_tip_paper_clearance_m": initial_tip_paper_clearance,
        "nonterminal_link_overlap_steps": body_overlap_steps,
        "minimum_nonterminal_link_clearance_m": min_body_clearance,
        "nonterminal_link_clearance_passed": body_overlap_steps == 0,
        "planned_fallback_strokes": planned_fallback_strokes,
        "below_ground_steps": below_ground_steps,
        "ground_rejected_steps": ground_rejected_steps,
        "minimum_body_link_center_z_m": min_body_link_center_z,
        "minimum_attempted_body_link_center_z_m": min_attempted_body_link_center_z,
        "minimum_body_geometry_z_m": min_body_geometry_z,
        "minimum_attempted_body_geometry_z_m": min_attempted_body_geometry_z,
        "ground_clearance_m": ground_clearance_m,
        "ground_clearance_passed": (
            min_body_geometry_z >= float(ground_clearance_m)
        ),
        "tip_paper_clearance_m": tip_paper_clearance_m,
        "minimum_required_flange_paper_clearance_m": max(
            float(tip_paper_clearance_m),
            float(virtual_pen_tcp_length) - float(mapper.max_brush_compression),
        ),
        "paper_target_clamp_steps": paper_target_clamp_steps,
        "paper_rejected_steps": paper_rejected_steps,
        "upward_pen_rejected_steps": upward_pen_rejected_steps,
        "minimum_pen_axis_downward_component": minimum_pen_downward_component,
        "pen_tip_points_down": minimum_pen_downward_component > 0.0,
        "rigid_pen_visible_after_first_valid_ik": rigid_pen_visible,
        "minimum_attempted_tip_paper_clearance_m": min_attempted_tip_paper_clearance,
        "minimum_tip_paper_clearance_m": min_tip_paper_clearance,
        "minimum_pen_endpoint_paper_clearance_m": min_pen_endpoint_paper_clearance,
        "maximum_pen_endpoint_paper_clearance_m": max_pen_endpoint_paper_clearance,
        "ink_trace_source": "actual_virtual_pen_tip_xy_projected_to_paper",
        "tip_stays_above_paper": (
            min_tip_paper_clearance >= float(tip_paper_clearance_m)
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory_csv", required=True)
    parser.add_argument(
        "--require_trajectory_prototype",
        default=None,
        help="Abort unless the selected CSV declares this exact prototype.",
    )
    parser.add_argument(
        "--require_trajectory_sha256",
        default=None,
        help="Abort unless the source CSV has this SHA256 digest.",
    )
    parser.add_argument("--character", default="武")
    parser.add_argument("--sample_id", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--orientation_mode",
        choices=("paper_parallel", "csv_pose"),
        default="csv_pose",
        help=(
            "paper_parallel keeps a fixed downward pen axis with free gamma; "
            "csv_pose applies CSV alpha/beta/gamma relative to the downward base."
        ),
    )
    parser.add_argument(
        "--strict_ik",
        action="store_true",
        help="Do not let simIK apply a partial solution when pose tolerances fail.",
    )
    parser.add_argument(
        "--max_step_m",
        type=float,
        default=0.008,
        help="Maximum replay segment length in metres; smaller is smoother but slower.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.04,
        help="Delay between replay samples in seconds; larger is slower to watch.",
    )
    parser.add_argument(
        "--transition_step_m",
        type=float,
        default=0.004,
        help="Maximum step for lift/travel/descend phases (m).",
    )
    parser.add_argument(
        "--transition_interval",
        type=float,
        default=0.048,
        help="Delay between lift/travel/descend samples (s).",
    )
    parser.add_argument(
        "--pen_lift_clearance_m",
        type=float,
        default=0.07,
        help="Explicit pen-up clearance above each stroke start/end (m).",
    )
    parser.add_argument(
        "--virtual_brush_length_m",
        type=float,
        default=0.048,
        help=(
            "Fixed Langhao brush-bundle length (m). Guo & Yan (2024) use "
            "L=48 mm and R=6 mm. The rigid pen is fixed to the official "
            "UR5 connection frame."
        ),
    )
    parser.add_argument(
        "--virtual_pen_handle_length_m",
        type=float,
        default=0.060,
        help=(
            "Visible rigid pen-handle length between the UR5 connection mount "
            "and the 48 mm brush bundle (m)."
        ),
    )
    parser.add_argument(
        "--max_brush_compression_m",
        type=float,
        default=0.020,
        help=(
            "Maximum B-BSMG descending depth (m). H=0 is first paper contact; "
            "the paper experiment uses H=11..20 mm."
        ),
    )
    parser.add_argument(
        "--arm_base_z_m",
        type=float,
        default=0.04,
        help="UR5 model base height above the world floor (m); default keeps the normal installation.",
    )
    parser.add_argument(
        "--ground_clearance_m",
        type=float,
        default=0.02,
        help="Reject IK poses whose nonterminal link center is below this height (m).",
    )
    parser.add_argument(
        "--tip_paper_clearance_m",
        type=float,
        default=0.001,
        help="Minimum allowed end-effector height above the paper surface (m).",
    )
    parser.add_argument(
        "--arm_model",
        choices=("ur5",),
        default="ur5",
        help="Use the official CoppeliaSim UR5.ttm model for live replay.",
    )
    parser.add_argument(
        "--ur5_model_path",
        default="/home/robot/CoppeliaSim_Edu_V4_7_0_rev4_Ubuntu22_04/models/robots/non-mobile/UR5.ttm",
    )
    parser.add_argument("--ur5_base_x_m", type=float, default=0.176)
    parser.add_argument(
        "--ur5_paper_z_m",
        type=float,
        default=0.60,
        help="Top surface of the writing plane above the world floor (m); default is 60 cm.",
    )
    parser.add_argument(
        "--paper_offset_x_m",
        type=float,
        default=-0.28,
        help="World x offset of the paper/trajectory from the UR5 base frame (m).",
    )
    parser.add_argument(
        "--paper_offset_y_m",
        type=float,
        default=0.0,
        help="World y offset of the paper/trajectory from the UR5 base frame (m).",
    )
    parser.add_argument(
        "--body_clearance_radius_m",
        type=float,
        default=0.06,
        help="Conservative XY radius for nonterminal UR5 links in the paper overlap check (m).",
    )
    parser.add_argument("--paper_z_m", type=float, default=0.015)
    parser.add_argument("--paper_width_m", type=float, default=0.24)
    parser.add_argument("--paper_height_m", type=float, default=0.24)
    parser.add_argument("--margin_m", type=float, default=0.018)
    parser.add_argument(
        "--z_height_m",
        type=float,
        default=None,
        help="Deprecated compatibility option; use --max_brush_compression_m.",
    )
    parser.add_argument("--lift_height_m", type=float, default=0.075)
    view_group = parser.add_mutually_exclusive_group()
    view_group.add_argument(
        "--flip_y",
        action="store_true",
        help="Legacy bottom/up-looking view mapping. Do not use for the standard top view.",
    )
    view_group.add_argument(
        "--no_flip_y",
        action="store_true",
        help="Explicit standard top-view mapping (kept as a backwards-compatible alias).",
    )
    parser.add_argument("--client_port", type=int, default=23000)
    parser.add_argument("--keep_scene", action="store_true")
    simulation_group = parser.add_mutually_exclusive_group()
    simulation_group.add_argument(
        "--start_simulation",
        dest="start_simulation",
        action="store_true",
        default=True,
        help="Start CoppeliaSim so the GUI play control enters the running state (default).",
    )
    simulation_group.add_argument(
        "--no_start_simulation",
        dest="start_simulation",
        action="store_false",
        help="Run kinematic remote IK without starting CoppeliaSim simulation.",
    )
    parser.add_argument(
        "--scene_output",
        default=None,
        help="Optional server-side .ttt path to save after replay",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    total_pen_length = (
        args.virtual_pen_handle_length_m + args.virtual_brush_length_m
    )
    if total_pen_length <= args.max_brush_compression_m:
        raise ValueError(
            "total rigid pen length must exceed max_brush_compression_m so "
            "the rigid UR5 flange remains above the paper"
        )
    rows = load_rows(args.trajectory_csv, character=args.character, sample_id=args.sample_id)
    provenance = trajectory_provenance(
        args.trajectory_csv,
        character=rows[0].character,
        sample_id=rows[0].sample_id,
    )
    validate_trajectory_identity(
        provenance,
        required_prototype=args.require_trajectory_prototype,
        required_sha256=args.require_trajectory_sha256,
    )
    mapper = make_mapper(
        rows,
        paper_width=args.paper_width_m,
        paper_height=args.paper_height_m,
        paper_z=args.paper_z_m,
        margin=args.margin_m,
        max_brush_compression=args.max_brush_compression_m,
        lift_height=args.lift_height_m,
        flip_y=bool(args.flip_y),
    )
    strokes = mapped_strokes(rows, mapper)
    report = build_report(rows, strokes, mapper, provenance=provenance)
    output_dir = Path(args.output_dir)
    save_offline_preview(strokes, output_dir, report, mapper)
    pressure_path = save_pressure_trajectory(rows, mapper, output_dir)
    report["mapped_pressure_trajectory"] = str(pressure_path.resolve())
    (output_dir / "trajectory_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.offline:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    live_report = run_live(
        rows,
        mapper,
        max_step=args.max_step_m,
        interval=args.interval,
        transition_step=args.transition_step_m,
        transition_interval=args.transition_interval,
        pen_lift_clearance=args.pen_lift_clearance_m,
        virtual_brush_length=args.virtual_brush_length_m,
        virtual_pen_handle_length=args.virtual_pen_handle_length_m,
        arm_base_z=args.arm_base_z_m,
        ground_clearance_m=args.ground_clearance_m,
        tip_paper_clearance_m=args.tip_paper_clearance_m,
        keep_scene=args.keep_scene,
        client_port=args.client_port,
        scene_output=args.scene_output,
        arm_model=args.arm_model,
        ur5_model_path=args.ur5_model_path,
        ur5_base_x=args.ur5_base_x_m,
        ur5_paper_z=args.ur5_paper_z_m,
        paper_offset_x=args.paper_offset_x_m,
        paper_offset_y=args.paper_offset_y_m,
        body_clearance_radius=args.body_clearance_radius_m,
        paper_width=args.paper_width_m,
        paper_height=args.paper_height_m,
        orientation_mode=args.orientation_mode,
        strict_ik=bool(args.strict_ik),
        start_simulation=args.start_simulation,
    )
    report["live_replay"] = True
    report["live"] = live_report
    (output_dir / "trajectory_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main(parse_args())
