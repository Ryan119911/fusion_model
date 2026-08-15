"""Select simulation posture candidates using footprint and continuity checks.

This is the next stage after ``generate_width_equivalent_poses.py``.  It does
not calibrate a real brush.  It combines the image-derived local footprint
table with B-BSMG geometry, posture bounds, within-stroke continuity, XY step
limits, and pen-up/stroke-boundary checks.  A candidate is recommended only if
all configured constraints pass; otherwise the report keeps the best
diagnostic candidate and lists the exact blockers.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.paper_bbsm import (  # noqa: E402
    PAPER_ANGLE_BASIS_RADIAN,
    PAPER_POSTURE_MAX,
    PAPER_POSTURE_MIN,
    posture_to_geometry_numpy,
    regression_matrix_numpy,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def _float(row: dict[str, str], name: str, default: float = 0.0) -> float:
    value = row.get(name, "")
    if value in (None, ""):
        return float(default)
    return float(value)


def _wrap_delta(values: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(values), np.cos(values))


def _basis(rows: list[dict[str, str]]) -> str:
    value = next(
        (row.get("regression_angle_basis", "") for row in rows if row.get("regression_angle_basis", "")),
        PAPER_ANGLE_BASIS_RADIAN,
    )
    if value not in (PAPER_ANGLE_BASIS_RADIAN, "degree_fitted"):
        raise ValueError(f"Unsupported regression angle basis: {value}")
    return value


def _group_steps(values: np.ndarray, stroke_ids: np.ndarray, wrap: bool = False) -> dict[str, float]:
    steps: list[float] = []
    for stroke in np.unique(stroke_ids):
        local = values[stroke_ids == stroke]
        if len(local) < 2:
            continue
        delta = np.diff(local, axis=0)
        if wrap:
            delta = _wrap_delta(delta)
        if delta.ndim == 1:
            steps.extend(np.abs(delta).tolist())
        else:
            steps.extend(np.linalg.norm(delta, axis=1).tolist())
    if not steps:
        return {"max": 0.0, "rms": 0.0, "p95": 0.0}
    array = np.asarray(steps, dtype=np.float64)
    return {"max": float(array.max()), "rms": float(np.sqrt(np.mean(array**2))), "p95": float(np.percentile(array, 95))}


def _load_candidate(path: Path) -> tuple[list[dict[str, str]], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = _read_csv(path)
    required = {"stroke_id", "point_id", "x", "y", "z", "alpha", "beta", "gamma"}
    missing = sorted(required.difference(rows[0]))
    if missing:
        raise ValueError(f"Candidate {path} is missing fields: {missing}")
    order = np.arange(len(rows))
    stroke_ids = np.asarray([int(_float(row, "stroke_id")) for row in rows], dtype=np.int64)
    point_ids = np.asarray([int(_float(row, "point_id")) for row in rows], dtype=np.int64)
    posture = np.asarray([[_float(row, field) for field in ("z", "alpha", "beta")] for row in rows], dtype=np.float64)
    gamma = np.asarray([_float(row, "gamma") for row in rows], dtype=np.float64)
    xy = np.asarray([[_float(row, "x"), _float(row, "y")] for row in rows], dtype=np.float64)
    if np.any(np.diff(stroke_ids) < 0):
        raise ValueError(f"Candidate stroke order is not monotonic: {path}")
    for stroke in np.unique(stroke_ids):
        local_points = point_ids[stroke_ids == stroke]
        if np.any(np.diff(local_points) < 0):
            raise ValueError(f"Candidate point order is not monotonic in stroke {stroke}: {path}")
    return rows, posture, gamma, xy, stroke_ids, point_ids, order


def _footprint_reference(path: Path | None, candidate_rows: list[dict[str, str]]) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray]:
    if path is None:
        target = np.asarray([_float(row, "target_width_Lr", np.nan) for row in candidate_rows], dtype=np.float64)
        valid = np.isfinite(target)
        return target, None, valid
    rows = _read_csv(path)
    keys = {(int(_float(row, "stroke_id")), int(_float(row, "point_id"))): row for row in rows}
    target_width, target_drag, confidence = [], [], []
    for row in candidate_rows:
        key = (int(_float(row, "stroke_id")), int(_float(row, "point_id")))
        if key not in keys:
            target_width.append(np.nan)
            target_drag.append(np.nan)
            confidence.append(0.0)
            continue
        ref = keys[key]
        target_width.append(_float(ref, "target_half_width", np.nan))
        target_drag.append(_float(ref, "target_drag", np.nan))
        confidence.append(_float(ref, "confidence", 0.0))
    width = np.asarray(target_width, dtype=np.float64)
    drag = np.asarray(target_drag, dtype=np.float64)
    conf = np.asarray(confidence, dtype=np.float64)
    valid = np.isfinite(width) & (width > 0.0) & (conf > 0.0)
    return width, drag, valid


def _trajectory_report_metrics(path: Path | None, args: argparse.Namespace) -> dict[str, Any] | None:
    if path is None:
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    processed = report.get("processed", report)
    within = processed.get("within_stroke", {})
    interstroke = processed.get("interstroke", {})
    safe = bool(processed.get("safe", False))
    max_step = float(within.get("max_step_xy", float("inf")))
    return {
        "source": str(path),
        "safe": safe,
        "xy_step_max": max_step,
        "xy_ok": bool(safe and max_step <= args.max_xy_step),
        "z_step_max": float(within.get("max_step_z", float("inf"))),
        "angle_step_max": float(within.get("max_angle_step_rad", float("inf"))),
        "cross_stroke_segments_rendered": int(interstroke.get("cross_stroke_segments_rendered", -1)),
        "state_errors": list(processed.get("state_errors", [])),
        "bounds_violations": dict(processed.get("bounds_violations", {})),
    }


def _candidate_report(
    candidate_path: Path,
    footprint_path: Path | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    rows, posture, gamma, xy, stroke_ids, point_ids, _ = _load_candidate(candidate_path)
    external_trajectory = _trajectory_report_metrics(
        Path(args.trajectory_report) if args.trajectory_report else None,
        args,
    )
    basis = _basis(rows)
    geometry = posture_to_geometry_numpy(posture, angle_basis=basis)
    target_width, target_drag, footprint_valid = _footprint_reference(footprint_path, rows)
    if target_width is None:
        target_width = geometry[:, 2].copy()
        footprint_valid = np.ones(len(rows), dtype=bool)
    width_valid = footprint_valid & np.isfinite(target_width) & (target_width > 1e-8)
    width_rel = np.abs(geometry[:, 2] - target_width) / np.maximum(np.abs(target_width), 1e-8)
    width_rel_valid = width_rel[width_valid]
    width_rel_rmse = float(np.sqrt(np.mean(width_rel_valid**2))) if len(width_rel_valid) else float("inf")
    width_rel_max = float(width_rel_valid.max()) if len(width_rel_valid) else float("inf")

    drag_valid = width_valid & (target_drag is not None) & np.isfinite(target_drag) & (target_drag > 1e-8)
    predicted_drag = geometry[:, 0] + geometry[:, 1]
    drag_rel = np.abs(predicted_drag - target_drag) / np.maximum(np.abs(target_drag), 1e-8) if target_drag is not None else np.full(len(rows), np.nan)
    drag_rel_valid = drag_rel[drag_valid]
    drag_rel_rmse = float(np.sqrt(np.mean(drag_rel_valid**2))) if len(drag_rel_valid) else None

    lower = PAPER_POSTURE_MIN[None, :]
    upper = PAPER_POSTURE_MAX[None, :]
    boundary = np.any((posture <= lower + args.boundary_tolerance) | (posture >= upper - args.boundary_tolerance), axis=1)
    boundary_fraction = float(np.mean(boundary))
    finite_angles = np.isfinite(gamma) & (np.abs(gamma) <= np.pi + args.angle_tolerance)
    pose_steps = {
        "H_mm": _group_steps(posture[:, 0], stroke_ids),
        "alpha_rad": _group_steps(posture[:, 1], stroke_ids),
        "beta_rad": _group_steps(posture[:, 2], stroke_ids),
        "gamma_rad": _group_steps(gamma, stroke_ids, wrap=True),
    }
    step_ok = (
        pose_steps["H_mm"]["max"] <= args.max_h_step_mm
        and pose_steps["alpha_rad"]["max"] <= args.max_alpha_step_rad
        and pose_steps["beta_rad"]["max"] <= args.max_beta_step_rad
        and pose_steps["gamma_rad"]["max"] <= args.max_gamma_step_rad
    )
    xy_steps = _group_steps(xy, stroke_ids)
    xy_ok = xy_steps["max"] <= args.max_xy_step
    state_values = [int(_float(row, "state", 1.0)) for row in rows]
    state_ok = all(state_values[index] >= 0 for index in range(len(state_values)))
    stroke_changes = np.flatnonzero(np.diff(stroke_ids) != 0)
    pen_up_boundary_ok = bool(len(stroke_changes) == len(np.unique(stroke_ids)) - 1)
    if external_trajectory is not None:
        xy_ok = bool(external_trajectory["xy_ok"])
        state_ok = bool(not external_trajectory["state_errors"] and not any(external_trajectory["bounds_violations"].values()))
        pen_up_boundary_ok = bool(external_trajectory["cross_stroke_segments_rendered"] == 0)
    trajectory_ok = bool(xy_ok and state_ok and pen_up_boundary_ok)
    footprint_ok = bool(
        len(width_rel_valid) >= args.minimum_footprint_points
        and width_rel_rmse <= args.max_width_relative_rmse
        and width_rel_max <= args.max_width_relative_error
        and (drag_rel_rmse is None or drag_rel_rmse <= args.max_drag_relative_rmse)
    )
    boundary_ok = boundary_fraction <= args.max_boundary_fraction
    all_ok = bool(footprint_ok and boundary_ok and step_ok and trajectory_ok and finite_angles.all())
    footprint_score = math.exp(-min(width_rel_rmse, 10.0) / max(args.max_width_relative_rmse, 1e-8)) if np.isfinite(width_rel_rmse) else 0.0
    drag_score = 1.0 if drag_rel_rmse is None else math.exp(-min(drag_rel_rmse, 10.0) / max(args.max_drag_relative_rmse, 1e-8))
    continuity_score = float(np.mean([
        math.exp(-pose_steps["H_mm"]["max"] / max(args.max_h_step_mm, 1e-8)),
        math.exp(-pose_steps["alpha_rad"]["max"] / max(args.max_alpha_step_rad, 1e-8)),
        math.exp(-pose_steps["beta_rad"]["max"] / max(args.max_beta_step_rad, 1e-8)),
        math.exp(-pose_steps["gamma_rad"]["max"] / max(args.max_gamma_step_rad, 1e-8)),
    ]))
    safety_score = float(max(0.0, 1.0 - boundary_fraction))
    trajectory_score = float(0.5 * xy_ok + 0.25 * state_ok + 0.25 * pen_up_boundary_ok)
    selection_score = float(0.35 * footprint_score + 0.15 * drag_score + 0.25 * continuity_score + 0.15 * safety_score + 0.10 * trajectory_score)
    blockers = []
    if not footprint_ok:
        blockers.append("local_footprint_error")
    if not boundary_ok:
        blockers.append("posture_boundary_saturation")
    if not step_ok:
        blockers.append("within_stroke_pose_or_gamma_jump")
    if not xy_ok:
        blockers.append("xy_step_too_large")
    if not pen_up_boundary_ok:
        blockers.append("stroke_boundary_sequence_invalid")
    if not finite_angles.all():
        blockers.append("gamma_nonfinite_or_out_of_range")
    return {
        "candidate_id": candidate_path.parent.name,
        "pose_csv": str(candidate_path),
        "simulation_only": True,
        "regression_angle_basis": basis,
        "footprint_reference": str(footprint_path) if footprint_path else "candidate_target_width_Lr",
        "footprint_points": int(width_rel_valid.size),
        "footprint_valid_fraction": float(np.mean(footprint_valid)),
        "width_relative_rmse": width_rel_rmse,
        "width_max_relative_error": width_rel_max,
        "drag_relative_rmse": drag_rel_rmse,
        "boundary_fraction": boundary_fraction,
        "continuity": pose_steps,
        "trajectory": {
            "xy_step": xy_steps,
            "external_report": external_trajectory,
            "xy_ok": xy_ok,
            "state_ok": state_ok,
            "pen_up_boundary_ok": pen_up_boundary_ok,
            "stroke_count": int(len(np.unique(stroke_ids))),
            "point_count": int(len(rows)),
        },
        "checks": {
            "footprint_ok": footprint_ok,
            "boundary_ok": boundary_ok,
            "within_stroke_steps_ok": step_ok,
            "trajectory_ok": trajectory_ok,
            "finite_angles_ok": bool(finite_angles.all()),
        },
        "selection_score": selection_score,
        "accepted": all_ok,
        "robot_test_ready": False,
        "blockers": blockers,
        "warning": "Simulation screening only; no real brush/TCP calibration was used.",
    }


def main(args: argparse.Namespace) -> None:
    root = Path(args.candidate_root)
    paths = sorted(root.glob("candidate_*/pose.csv"))
    if not paths:
        raise FileNotFoundError(f"No candidate_*/pose.csv under {root}")
    footprint = Path(args.footprint_csv) if args.footprint_csv else None
    if footprint is not None and not footprint.exists():
        raise FileNotFoundError(footprint)
    reports = [_candidate_report(path, footprint, args) for path in paths]
    reports.sort(key=lambda item: float(item["selection_score"]), reverse=True)
    for rank, report in enumerate(reports, start=1):
        report["rank"] = rank
    accepted = [report for report in reports if report["accepted"]]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "format": "paper_width_pose_screening_v1",
        "candidate_root": str(root),
        "footprint_reference": str(footprint) if footprint else None,
        "simulation_only": True,
        "real_brush_calibration_used": False,
        "candidate_count": len(reports),
        "accepted_count": len(accepted),
        "recommended_candidate": accepted[0]["candidate_id"] if accepted else None,
        "best_diagnostic_candidate": reports[0]["candidate_id"],
        "constraints": {
            "max_width_relative_rmse": args.max_width_relative_rmse,
            "max_width_relative_error": args.max_width_relative_error,
            "max_drag_relative_rmse": args.max_drag_relative_rmse,
            "minimum_footprint_points": args.minimum_footprint_points,
            "max_boundary_fraction": args.max_boundary_fraction,
            "max_h_step_mm": args.max_h_step_mm,
            "max_alpha_step_rad": args.max_alpha_step_rad,
            "max_beta_step_rad": args.max_beta_step_rad,
            "max_gamma_step_rad": args.max_gamma_step_rad,
            "max_xy_step": args.max_xy_step,
        },
        "interpretation": "No candidate is robot-ready until all checks pass; diagnostics remain usable for simulation comparison.",
        "candidates": reports,
    }
    (output / "screening_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = sorted({key for report in reports for key in report if not isinstance(report[key], (dict, list))})
    with (output / "screening_summary.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: report.get(field) for field in fields} for report in reports])
    for report in reports:
        (output / f"{report['candidate_id']}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate_root", required=True)
    parser.add_argument("--footprint_csv", default=None)
    parser.add_argument("--trajectory_report", default=None)
    parser.add_argument("--output_dir", default="outputs/wu_width_equivalent_screening_v1")
    parser.add_argument("--max_width_relative_rmse", type=float, default=0.05)
    parser.add_argument("--max_width_relative_error", type=float, default=0.15)
    parser.add_argument("--max_drag_relative_rmse", type=float, default=0.20)
    parser.add_argument("--minimum_footprint_points", type=int, default=4)
    parser.add_argument("--max_boundary_fraction", type=float, default=0.25)
    parser.add_argument("--boundary_tolerance", type=float, default=1e-4)
    parser.add_argument("--angle_tolerance", type=float, default=1e-6)
    parser.add_argument("--max_h_step_mm", type=float, default=1.0)
    parser.add_argument("--max_alpha_step_rad", type=float, default=float(np.deg2rad(5.0)))
    parser.add_argument("--max_beta_step_rad", type=float, default=float(np.deg2rad(3.0)))
    parser.add_argument("--max_gamma_step_rad", type=float, default=float(np.deg2rad(45.0)))
    parser.add_argument("--max_xy_step", type=float, default=2.0)
    main(parser.parse_args())
