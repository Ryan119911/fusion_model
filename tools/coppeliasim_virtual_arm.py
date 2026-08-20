"""Replay a paper trajectory with CoppeliaSim's official UR5 model.

Live mode loads ``UR5.ttm`` from the CoppeliaSim installation, creates a
simIK pose task for its six real joints, and moves the standard model's
end-effector to each mapped paper point.  The visual tool's local Z axis is
rotated into the horizontal paper plane (90 degrees about the initial tool's
local X axis), while the tip position follows the trajectory.  One draw object is created per stroke,
and lift states never connect separate strokes.  This is still a
simulation/visualization experiment: dynamics and real robot calibration are
not inferred from it.  The default writing plane is 0.06 m above the UR5
base/ground (a few centimetres, rather than the old 0.96 m test plane); the
``--ur5_paper_z_m`` option can override it after a reachability check.

The script has an offline mode for validating CSV selection, coordinate
mapping, lift logic, and cross-stroke safety without a running simulator.
For live mode start CoppeliaSim with the ZMQ remote API add-on enabled, then
run this file with the same trajectory CSV.
"""
from __future__ import annotations

import argparse
import csv
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
    contact_clearance: float = 0.004
    z_height: float = 0.045
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
        z_span = max(self.z_max - self.z_min, 1e-9)
        usable_x = self.paper_width - 2.0 * self.margin
        usable_y = self.paper_height - 2.0 * self.margin
        x_norm = (row.x - self.x_min) / x_span
        y_norm = (row.y - self.y_min) / y_span
        if self.flip_y:
            y_norm = 1.0 - y_norm
        world_x = -0.5 * self.paper_width + self.margin + x_norm * usable_x
        world_y = -0.5 * self.paper_height + self.margin + y_norm * usable_y
        z_norm = (row.z - self.z_min) / z_span
        if row.state in (2, 3):  # UP/TRANSITION: visibly clear the paper.
            world_z = self.paper_z + self.lift_height
        else:
            world_z = self.paper_z + self.contact_clearance + z_norm * self.z_height
        return WorldPoint(
            position=(float(world_x), float(world_y), float(world_z)),
            orientation=(float(row.alpha), float(row.beta), float(row.gamma)),
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
                    state=right.state,
                    stroke_id=right.stroke_id,
                    point_id=right.point_id,
                )
            )
    return output


def build_report(rows: Sequence[PoseRow], strokes: Dict[int, List[WorldPoint]], mapper: CoordinateMapper) -> dict:
    state_counts: Dict[str, int] = {}
    for row in rows:
        state_counts[str(row.state)] = state_counts.get(str(row.state), 0) + 1
    active_segments = sum(
        sum(1 for left, right in zip(points, points[1:]) if left.state not in (2, 3) and right.state not in (2, 3))
        for points in strokes.values()
    )
    return {
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
        },
        "safety": {
            "no_cross_stroke_drawing": True,
            "lift_states": [2, 3],
            "actual_robot_ik": False,
            "brush_physics": False,
        },
    }


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


def _quaternion_angle_error(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the shortest angular distance between two XYZW quaternions."""
    left_vec = np.asarray(list(left), dtype=np.float64)
    right_vec = np.asarray(list(right), dtype=np.float64)
    left_vec /= max(float(np.linalg.norm(left_vec)), 1e-12)
    right_vec /= max(float(np.linalg.norm(right_vec)), 1e-12)
    dot = float(np.clip(abs(np.dot(left_vec, right_vec)), -1.0, 1.0))
    return float(2.0 * math.acos(dot))


def _load_ur5_ik(sim, ik, model_path: str, base_x: float):
    """Load CoppeliaSim's official UR5 model and configure simIK.

    The model is not recreated by this project: all six joints and visible
    links come from ``UR5.ttm``.  IK is position-constrained so the standard
    UR5 wrist orientation is retained while the pen tip follows the paper.
    """
    root = sim.loadModel(model_path)
    sim.setObjectPosition(root, -1, [float(base_x), 0.0, 0.0])
    joints = sim.getObjectsInTree(root, sim.object_joint_type, 0)
    shapes = sim.getObjectsInTree(root, sim.object_shape_type, 0)
    if len(joints) != 6:
        raise RuntimeError(f"Expected 6 UR5 joints, found {len(joints)}")
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
    # CoppeliaSim quaternions are [x, y, z, w].  The visual cylinder uses its
    # local +Z axis as its long/tool axis.  Rotate the current UR5 terminal
    # pose by 90 degrees in its *local* X direction.  Relative rotation is
    # important here: a fixed world quaternion can be outside the reachable
    # wrist configuration at the low writing plane, causing simIK to restore
    # the old perpendicular pose after a failed solve.
    initial_tip_quaternion = [
        float(value) for value in sim.getObjectQuaternion(last_link, -1)
    ]
    half = math.radians(90.0) * 0.5
    local_quarter_turn = [math.sin(half), 0.0, 0.0, math.cos(half)]
    paper_parallel_quaternion = _multiply_quaternions(
        initial_tip_quaternion, local_quarter_turn
    )
    tip = sim.createDummy(0.010, 12 * [0.0])
    sim.setObjectParent(tip, last_link, False)
    tip_position = sim.getObjectPosition(last_link, -1)
    sim.setObjectPosition(tip, -1, tip_position)
    # Keep the IK tip aligned with the actual terminal link.  The previous
    # implementation put the 90-degree rotation on the child tip itself, which
    # hid the wrist rotation from simIK and could leave the visible UR5 flange
    # perpendicular even though the dummy reported a horizontal pose.
    sim.setObjectQuaternion(tip, -1, initial_tip_quaternion)
    target = sim.createDummy(0.012, 12 * [0.0])
    sim.setObjectPosition(target, -1, tip_position)
    sim.setObjectQuaternion(target, -1, paper_parallel_quaternion)
    environment = ik.createEnvironment()
    group = ik.createGroup(environment)
    add_result = ik.addElementFromScene(
        environment, group, root, tip, target, ik.constraint_pose
    )
    if isinstance(add_result, tuple) and int(add_result[0]) != 0:
        raise RuntimeError(f"simIK.addElementFromScene failed: {add_result}")
    ik.setGroupCalculation(
        environment, group, ik.method_damped_least_squares, 0.03, 500
    )
    return {
        "root": root,
        "joints": joints,
        "body_shapes": [handle for handle in shapes if handle != last_link],
        "terminal_shape": last_link,
        "tip": tip,
        "target": target,
        "tool_quaternion_xyzw": paper_parallel_quaternion,
        "tool_orientation_constraint": "paper_parallel_world_xy",
        "initial_tip_quaternion_xyzw": initial_tip_quaternion,
        "environment": environment,
        "group": group,
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
    try:
        paper = sim.createPureShape(sim.primitiveshape_cuboid, 0, [width, height, 0.004], 0, [])
    except Exception:
        # Some 4.7 builds reject run-time pure-shape creation from the remote
        # API.  The official UR5 model remains fully visible; the paper is
        # cosmetic, so omit it rather than aborting the replay.
        return None
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


def _add_ur5_tool(sim, tip: int):
    try:
        tool = sim.createPureShape(sim.primitiveshape_cylinder, 0, [0.012, 0.012, 0.06], 0, [])
    except Exception:
        tool = sim.createDummy(0.014, 12 * [0.0])
    sim.setObjectParent(tool, tip, False)
    sim.setObjectPosition(tool, -1, sim.getObjectPosition(tip, -1))
    sim.setObjectQuaternion(tool, -1, sim.getObjectQuaternion(tip, -1))
    try:
        sim.setShapeColor(tool, None, sim.colorcomponent_ambient_diffuse, [0.12, 0.12, 0.12])
    except Exception:
        pass
    return tool


def run_live(
    rows: Sequence[PoseRow],
    mapper: CoordinateMapper,
    *,
    max_step: float,
    interval: float,
    arm_base_z: float,
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
    if arm_model != "ur5":
        raise ValueError("Only the official CoppeliaSim UR5 model is supported in live mode")
    ik = client.require("simIK")
    strokes = mapped_strokes(rows, mapper)
    colors = ([0.05, 0.35, 0.95], [0.95, 0.18, 0.08], [0.08, 0.65, 0.22], [0.65, 0.2, 0.8])
    trajectory_drawings = []
    planned_drawings = []
    for index, _ in enumerate(strokes):
        trajectory_drawings.append(
            _add_drawing(sim, sim.drawing_lines, 3, 200000, colors[index % len(colors)])
        )
        planned_drawings.append(
            _add_drawing(sim, sim.drawing_lines, 1, 200000, [0.75, 0.75, 0.75])
        )
    ur5 = _load_ur5_ik(sim, ik, ur5_model_path, ur5_base_x)
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
    _add_ur5_tool(sim, ur5["tip"])
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
    approach_success = 0
    approach_failures = 0
    body_overlap_steps = 0
    min_body_clearance = float("inf")
    planned_fallback_strokes = 0
    max_orientation_residual = 0.0

    # Draw the requested path before IK starts.  It is deliberately separate
    # from the measured tip path: a failed IK step must not make a short stroke
    # (for example the top horizontal of ``止``) disappear from the scene.
    for stroke_index, (_, raw_points) in enumerate(strokes.items()):
        dense_points = interpolate_stroke(raw_points, max_step)
        draw_points = [point for point in dense_points if point.state not in (2, 3)]
        if len(draw_points) < 2:
            draw_points = dense_points
            planned_fallback_strokes += 1
        for left, right in zip(draw_points, draw_points[1:]):
            left_position = np.asarray(left.position, dtype=np.float64) + offset
            right_position = np.asarray(right.position, dtype=np.float64) + offset
            sim.addDrawingObjectItem(
                planned_drawings[stroke_index], list(left_position) + list(right_position)
            )

    def solve_target(target_position: np.ndarray) -> Tuple[np.ndarray, int, float]:
        nonlocal ik_success, ik_failures, max_residual, max_orientation_residual
        nonlocal body_overlap_steps, min_body_clearance
        sim.setObjectPosition(ur5["target"], -1, target_position.tolist())
        result = ik.handleGroup(ur5["environment"], ur5["group"], {"syncWorlds": True})
        status = int(result[0]) if isinstance(result, tuple) else int(result)
        actual_position = np.asarray(sim.getObjectPosition(ur5["tip"], -1), dtype=np.float64)
        residual = float(np.linalg.norm(actual_position - target_position))
        max_residual = max(max_residual, residual)
        actual_quaternion = sim.getObjectQuaternion(ur5["tip"], -1)
        max_orientation_residual = max(
            max_orientation_residual,
            _quaternion_angle_error(actual_quaternion, ur5["tool_quaternion_xyzw"]),
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

    try:
        if start_simulation:
            sim.startSimulation()
        for stroke_index, (stroke_id, raw_points) in enumerate(strokes.items()):
            previous_by_stroke.pop(stroke_id, None)
            stroke_success = 0
            stroke_failures = 0
            dense_points = interpolate_stroke(raw_points, max_step)
            first_target = np.asarray(dense_points[0].position, dtype=np.float64) + offset
            if previous_target is None:
                # Start from the UR5 model's home pose without teleporting the
                # end effector to a few-centimetre-high paper plane.  The
                # approach is lifted above the paper, then the normal stroke
                # loop makes the final vertical descent while drawing starts.
                current_tip = np.asarray(
                    sim.getObjectPosition(ur5["tip"], -1), dtype=np.float64
                )
                approach_target = first_target.copy()
                approach_target[2] = max(approach_target[2], ur5_paper_z + 0.055)
                approach_distance = float(np.linalg.norm(approach_target - current_tip))
                approach_count = max(1, int(math.ceil(approach_distance / max(max_step, 1e-5))))
                for approach_index in range(1, approach_count + 1):
                    ratio = approach_index / approach_count
                    approach_position = current_tip + ratio * (approach_target - current_tip)
                    _, approach_status, _ = solve_target(approach_position)
                    if approach_status == 1:
                        approach_success += 1
                    else:
                        approach_failures += 1
                    time.sleep(max(interval, 0.0))
            if previous_target is not None:
                # Do not teleport the UR5 between strokes.  Move to the next
                # stroke start through the lifted plane, without drawing.
                transit_distance = float(np.linalg.norm(first_target - previous_target))
                transit_count = max(1, int(math.ceil(transit_distance / max(max_step, 1e-5))))
                for transit_index in range(1, transit_count + 1):
                    ratio = transit_index / transit_count
                    transit_target = previous_target + ratio * (first_target - previous_target)
                    transit_target[2] = max(transit_target[2], ur5_paper_z + 0.055)
                    solve_target(transit_target)
                    time.sleep(max(interval, 0.0))
            for point in dense_points:
                target_position = np.asarray(point.position, dtype=np.float64) + offset
                actual_position, status, residual = solve_target(target_position)
                if status == 1:
                    stroke_success += 1
                else:
                    stroke_failures += 1
                if point.state not in (2, 3):
                    previous = previous_by_stroke.get(stroke_id)
                    if previous is not None:
                        sim.addDrawingObjectItem(
                            trajectory_drawings[stroke_index], list(previous) + list(actual_position)
                        )
                    sim.addDrawingObjectItem(tip_drawing, list(actual_position))
                    previous_by_stroke[stroke_id] = tuple(actual_position.tolist())
                else:
                    previous_by_stroke.pop(stroke_id, None)
                time.sleep(max(interval, 0.0))
            previous_target = np.asarray(dense_points[-1].position, dtype=np.float64) + offset
            ik_success_by_stroke[str(stroke_id)] = stroke_success
            ik_failure_by_stroke[str(stroke_id)] = stroke_failures
        if scene_output:
            scene_path = str(Path(scene_output).expanduser().resolve())
            if not sim.saveScene(scene_path):
                raise RuntimeError(f"CoppeliaSim failed to save scene: {scene_path}")
            print(f"[DONE] Saved CoppeliaSim scene: {scene_path}")
    finally:
        if start_simulation and not keep_scene:
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
        "ik_success_by_stroke": ik_success_by_stroke,
        "ik_failure_by_stroke": ik_failure_by_stroke,
        "ik_approach_success_steps": approach_success,
        "ik_approach_failure_steps": approach_failures,
        "dynamics_enabled": False,
        "paper_top_z_m": ur5_paper_z,
        "paper_height_above_ground_m": ur5_paper_z,
        "paper_offset_x_m": paper_offset_x,
        "paper_offset_y_m": paper_offset_y,
        "tool_orientation_constraint": ur5["tool_orientation_constraint"],
        "tool_quaternion_xyzw": ur5["tool_quaternion_xyzw"],
        "nonterminal_link_overlap_steps": body_overlap_steps,
        "minimum_nonterminal_link_clearance_m": min_body_clearance,
        "nonterminal_link_clearance_passed": body_overlap_steps == 0,
        "planned_fallback_strokes": planned_fallback_strokes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory_csv", required=True)
    parser.add_argument("--character", default="武")
    parser.add_argument("--sample_id", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--max_step_m", type=float, default=0.002)
    parser.add_argument("--interval", type=float, default=0.015)
    parser.add_argument("--arm_base_z_m", type=float, default=0.18)
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
        default=0.06,
        help="Top surface of the writing plane above the UR5 base/ground (m); default is 6 cm.",
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
    parser.add_argument("--z_height_m", type=float, default=0.045)
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
    parser.add_argument(
        "--start_simulation",
        action="store_true",
        help="Also start CoppeliaSim physics. Omit for stable kinematic IK visualization.",
    )
    parser.add_argument(
        "--scene_output",
        default=None,
        help="Optional server-side .ttt path to save after replay",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    rows = load_rows(args.trajectory_csv, character=args.character, sample_id=args.sample_id)
    mapper = make_mapper(
        rows,
        paper_width=args.paper_width_m,
        paper_height=args.paper_height_m,
        paper_z=args.paper_z_m,
        margin=args.margin_m,
        z_height=args.z_height_m,
        lift_height=args.lift_height_m,
        flip_y=bool(args.flip_y),
    )
    strokes = mapped_strokes(rows, mapper)
    report = build_report(rows, strokes, mapper)
    output_dir = Path(args.output_dir)
    save_offline_preview(strokes, output_dir, report, mapper)
    if args.offline:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    live_report = run_live(
        rows,
        mapper,
        max_step=args.max_step_m,
        interval=args.interval,
        arm_base_z=args.arm_base_z_m,
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
        start_simulation=args.start_simulation,
    )
    report["live_replay"] = True
    report["live"] = live_report
    (output_dir / "trajectory_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main(parse_args())
