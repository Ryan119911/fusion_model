"""Replay a paper trajectory in a CoppeliaSim visualization prototype.

The prototype deliberately separates visualization from robot calibration.  It
creates a six-marker kinematic arm, a colored end-effector marker, and one
draw object per stroke.  The marker follows the mapped CSV pose directly; no
brush contact, force model, or robot IK claim is made in this stage.

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
    strokes: Dict[int, List[WorldPoint]], output_dir: Path, report: dict, image_size: int = 768
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
) -> None:
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
    strokes = mapped_strokes(rows, mapper)
    colors = ([0.05, 0.35, 0.95], [0.95, 0.18, 0.08], [0.08, 0.65, 0.22], [0.65, 0.2, 0.8])
    trajectory_drawings = []
    for index, _ in enumerate(strokes):
        trajectory_drawings.append(
            _add_drawing(sim, sim.drawing_lines, 3, 200000, colors[index % len(colors)])
        )
    arm_drawing = _add_drawing(sim, sim.drawing_lines, 4, 20, [0.12, 0.12, 0.12])
    joint_markers = [sim.createDummy(0.008, 12 * [0.0]) for _ in range(6)]
    ee_marker = sim.createDummy(0.014, 12 * [0.0])
    try:
        # A simple cylinder stands in for a writing tool.  It is visual only.
        tool_marker = sim.createPureShape(
            sim.primitiveshape_cylinder, 0, [0.012, 0.012, 0.04], 0, []
        )
    except Exception:
        tool_marker = sim.createDummy(0.014, 12 * [0.0])
    tip_drawing = _add_drawing(sim, sim.drawing_spherepoints, 0.008, 200000, [0.9, 0.1, 0.05])
    base = np.asarray([0.0, 0.0, float(arm_base_z)], dtype=np.float64)
    previous_by_stroke: Dict[int, Tuple[float, float, float]] = {}
    try:
        sim.startSimulation()
        for stroke_index, (stroke_id, raw_points) in enumerate(strokes.items()):
            previous_by_stroke.pop(stroke_id, None)
            for point in interpolate_stroke(raw_points, max_step):
                position = np.asarray(point.position, dtype=np.float64)
                # A six-marker visual chain. It is intentionally not advertised as IK.
                fractions = (0.18, 0.36, 0.54, 0.70, 0.84, 1.0)
                joint_positions = []
                for fraction in fractions:
                    offset = np.asarray([0.0, 0.0, 0.025 * math.sin(math.pi * fraction)], dtype=np.float64)
                    joint_positions.append(tuple((base + fraction * (position - base) + offset).tolist()))
                for handle, joint_position in zip(joint_markers, joint_positions):
                    sim.setObjectPosition(handle, -1, list(joint_position))
                sim.setObjectPosition(ee_marker, -1, list(position))
                sim.setObjectOrientation(ee_marker, -1, list(point.orientation))
                sim.setObjectPosition(tool_marker, -1, list(position))
                sim.setObjectOrientation(tool_marker, -1, list(point.orientation))
                sim.addDrawingObjectItem(arm_drawing, None)
                for left, right in zip([base] + [np.asarray(p) for p in joint_positions], joint_positions):
                    right_array = np.asarray(right, dtype=np.float64)
                    sim.addDrawingObjectItem(arm_drawing, list(left) + list(right_array))
                if point.state not in (2, 3):
                    previous = previous_by_stroke.get(stroke_id)
                    if previous is not None:
                        sim.addDrawingObjectItem(
                            trajectory_drawings[stroke_index], list(previous) + list(position)
                        )
                    sim.addDrawingObjectItem(tip_drawing, list(position))
                    previous_by_stroke[stroke_id] = tuple(position.tolist())
                else:
                    previous_by_stroke.pop(stroke_id, None)
                time.sleep(max(interval, 0.0))
        if scene_output:
            scene_path = str(Path(scene_output).expanduser().resolve())
            if not sim.saveScene(scene_path):
                raise RuntimeError(f"CoppeliaSim failed to save scene: {scene_path}")
            print(f"[DONE] Saved CoppeliaSim scene: {scene_path}")
    finally:
        if not keep_scene:
            try:
                sim.stopSimulation()
            except Exception:
                pass
    print("[DONE] CoppeliaSim virtual-arm trajectory replay finished")


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
    save_offline_preview(strokes, output_dir, report)
    if args.offline:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    run_live(
        rows,
        mapper,
        max_step=args.max_step_m,
        interval=args.interval,
        arm_base_z=args.arm_base_z_m,
        keep_scene=args.keep_scene,
        client_port=args.client_port,
        scene_output=args.scene_output,
    )
    report["live_replay"] = True
    (output_dir / "trajectory_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main(parse_args())
