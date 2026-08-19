"""Replay a paper trajectory with CoppeliaSim's official UR5 model.

Live mode loads ``UR5.ttm`` from the CoppeliaSim installation, creates a
simIK position task for its six real joints, and moves the standard model's
end-effector to each mapped paper point.  One draw object is created per
stroke, and lift states never connect separate strokes.  This is still a
simulation/visualization experiment: dynamics and real robot calibration are
not inferred from it.

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
    tip = sim.createDummy(0.010, 12 * [0.0])
    sim.setObjectParent(tip, last_link, False)
    tip_position = sim.getObjectPosition(last_link, -1)
    tip_quaternion = sim.getObjectQuaternion(last_link, -1)
    sim.setObjectPosition(tip, -1, tip_position)
    sim.setObjectQuaternion(tip, -1, tip_quaternion)
    target = sim.createDummy(0.012, 12 * [0.0])
    sim.setObjectPosition(target, -1, tip_position)
    sim.setObjectQuaternion(target, -1, tip_quaternion)
    environment = ik.createEnvironment()
    group = ik.createGroup(environment)
    add_result = ik.addElementFromScene(
        environment, group, root, tip, target, ik.constraint_position
    )
    if isinstance(add_result, tuple) and int(add_result[0]) != 0:
        raise RuntimeError(f"simIK.addElementFromScene failed: {add_result}")
    ik.setGroupCalculation(
        environment, group, ik.method_damped_least_squares, 0.05, 200
    )
    return {
        "root": root,
        "joints": joints,
        "tip": tip,
        "target": target,
        "environment": environment,
        "group": group,
        "tip_position": tip_position,
    }


def _add_paper(sim, width: float, height: float, top_z: float):
    """Add a thin static paper plane only for visual contact reference."""
    try:
        paper = sim.createPureShape(sim.primitiveshape_cuboid, 0, [width, height, 0.004], 0, [])
    except Exception:
        # Some 4.7 builds reject run-time pure-shape creation from the remote
        # API.  The official UR5 model remains fully visible; the paper is
        # cosmetic, so omit it rather than aborting the replay.
        return None
    sim.setObjectPosition(paper, -1, [0.0, 0.0, float(top_z) - 0.002])
    try:
        sim.setShapeColor(paper, None, sim.colorcomponent_ambient_diffuse, [0.92, 0.92, 0.86])
    except Exception:
        pass
    return paper


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
    for index, _ in enumerate(strokes):
        trajectory_drawings.append(
            _add_drawing(sim, sim.drawing_lines, 3, 200000, colors[index % len(colors)])
        )
    ur5 = _load_ur5_ik(sim, ik, ur5_model_path, ur5_base_x)
    _add_paper(sim, paper_width, paper_height, ur5_paper_z)
    _add_ur5_tool(sim, ur5["tip"])
    tip_drawing = _add_drawing(sim, sim.drawing_spherepoints, 0.008, 200000, [0.9, 0.1, 0.05])
    previous_by_stroke: Dict[int, Tuple[float, float, float]] = {}
    offset = np.asarray([0.0, 0.0, float(ur5_paper_z - mapper.paper_z)], dtype=np.float64)
    ik_success = 0
    ik_failures = 0
    max_residual = 0.0
    ik_success_by_stroke: Dict[str, int] = {}
    ik_failure_by_stroke: Dict[str, int] = {}
    previous_target: Optional[np.ndarray] = None

    def solve_target(target_position: np.ndarray) -> Tuple[np.ndarray, int, float]:
        nonlocal ik_success, ik_failures, max_residual
        sim.setObjectPosition(ur5["target"], -1, target_position.tolist())
        result = ik.handleGroup(ur5["environment"], ur5["group"], {"syncWorlds": True})
        status = int(result[0]) if isinstance(result, tuple) else int(result)
        actual_position = np.asarray(sim.getObjectPosition(ur5["tip"], -1), dtype=np.float64)
        residual = float(np.linalg.norm(actual_position - target_position))
        max_residual = max(max_residual, residual)
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
        "ik_success_by_stroke": ik_success_by_stroke,
        "ik_failure_by_stroke": ik_failure_by_stroke,
        "dynamics_enabled": False,
        "paper_top_z_m": ur5_paper_z,
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
    parser.add_argument("--ur5_paper_z_m", type=float, default=0.96)
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
