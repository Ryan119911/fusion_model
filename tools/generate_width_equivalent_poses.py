"""Generate bounded posture alternatives with the same B-BSMG footprint width.

The B-BSMG regression exposes the transverse footprint radius ``Lr`` as a
linear function of ``H, alpha, beta``.  This script keeps the width of an
existing pose CSV as the target and constructs several bounded solutions with
different H/alpha/beta preferences.  Gamma is varied only as a heading
candidate because it is not present in the width equation.

Outputs are simulation candidates, not physical brush/TCP calibration.  A
robot experiment must select a candidate only after independent safety and
real-brush calibration checks.
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


def _read_pose_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"Pose CSV has no header: {path}")
        rows = list(reader)
        fields = list(reader.fieldnames)
    required = {"x", "y", "z", "alpha", "beta", "gamma", "stroke_id", "point_id"}
    missing = sorted(required.difference(fields))
    if missing:
        raise ValueError(f"Pose CSV is missing required fields: {missing}")
    if not rows:
        raise ValueError(f"Pose CSV is empty: {path}")
    return rows, fields


def _angle_basis(rows: list[dict[str, str]], requested: str) -> str:
    if requested != "auto":
        return requested
    for row in rows:
        value = row.get("regression_angle_basis", "")
        if value:
            return value
    return PAPER_ANGLE_BASIS_RADIAN


def _wrap_angle(value: np.ndarray) -> np.ndarray:
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def _solve_width_line(
    target_l_r: np.ndarray,
    height: np.ndarray,
    alpha_preference: np.ndarray,
    beta_preference: np.ndarray,
    matrix: np.ndarray,
    bias: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve c_a*alpha+c_b*beta = target for bounded alpha/beta.

    The unconstrained projection is the closest point to the requested angle
    preference.  If it leaves the prototype box, a dense one-dimensional scan
    chooses the closest feasible point.  Thus width error remains numerical
    round-off while the result is still inside H=11..20 mm, alpha=0..10 deg,
    beta=0..5 deg.
    """
    ca, cb = float(matrix[2, 1]), float(matrix[2, 2])
    rhs = target_l_r - float(bias[2]) - float(matrix[2, 0]) * height
    pref = np.stack((alpha_preference, beta_preference), axis=1)
    coeff = np.asarray([ca, cb], dtype=np.float64)
    residual = rhs - pref @ coeff
    denom = float(np.dot(coeff, coeff))
    projected = pref + residual[:, None] * coeff[None, :] / max(denom, 1e-12)
    lower = PAPER_POSTURE_MIN[1:].astype(np.float64)
    upper = PAPER_POSTURE_MAX[1:].astype(np.float64)
    inside = np.all((projected >= lower[None, :]) & (projected <= upper[None, :]), axis=1)
    solution = projected.copy()

    if not np.all(inside):
        # Scanning alpha is cheap and robust for this 2-variable equality.
        alpha_grid = np.linspace(lower[0], upper[0], 2049, dtype=np.float64)
        for index in np.flatnonzero(~inside):
            beta_grid = (rhs[index] - ca * alpha_grid) / max(cb, 1e-12)
            feasible = (beta_grid >= lower[1]) & (beta_grid <= upper[1])
            if not np.any(feasible):
                # The requested height may make this width impossible inside
                # the prototype box.  Return the closest bounded point and
                # expose the residual in the report instead of hiding it.
                clipped = np.clip(projected[index], lower, upper)
                solution[index] = clipped
                continue
            alpha_values = alpha_grid[feasible]
            beta_values = beta_grid[feasible]
            distance = (alpha_values - alpha_preference[index]) ** 2 + (
                beta_values - beta_preference[index]
            ) ** 2
            best = int(np.argmin(distance))
            solution[index] = [alpha_values[best], beta_values[best]]

    width = posture_to_geometry_numpy(
        np.column_stack((height, solution)),
        angle_basis=matrix_to_basis(matrix),
    )[:, 2]
    return solution[:, 0], solution[:, 1], width


def matrix_to_basis(matrix: np.ndarray) -> str:
    """Infer the supported basis from the exact matrix scaling."""
    rad = regression_matrix_numpy(PAPER_ANGLE_BASIS_RADIAN)
    deg = regression_matrix_numpy("degree_fitted")
    if np.allclose(matrix, deg, rtol=1e-5, atol=1e-7):
        return "degree_fitted"
    if np.allclose(matrix, rad, rtol=1e-5, atol=1e-7):
        return PAPER_ANGLE_BASIS_RADIAN
    raise ValueError("Unsupported regression angle basis")


def _candidate_specs() -> list[dict[str, Any]]:
    # Deliberately include independent H, alpha, beta and gamma alternatives.
    # The final ranking uses measured width residual and safety metrics.
    return [
        {"name": "base", "h_shift": 0.0, "alpha_shift": 0.0, "beta_shift": 0.0, "gamma_deg": 0.0},
        {"name": "height_low", "h_shift": -2.0, "alpha_shift": 0.0, "beta_shift": 0.0, "gamma_deg": 0.0},
        {"name": "height_high", "h_shift": 2.0, "alpha_shift": 0.0, "beta_shift": 0.0, "gamma_deg": 0.0},
        {"name": "alpha_preferred_low", "h_shift": 0.0, "alpha_shift": -0.045, "beta_shift": 0.0, "gamma_deg": 0.0},
        {"name": "alpha_preferred_high", "h_shift": 0.0, "alpha_shift": 0.045, "beta_shift": 0.0, "gamma_deg": 0.0},
        {"name": "beta_preferred_low", "h_shift": 0.0, "alpha_shift": 0.0, "beta_shift": -0.025, "gamma_deg": 0.0},
        {"name": "beta_preferred_high", "h_shift": 0.0, "alpha_shift": 0.0, "beta_shift": 0.025, "gamma_deg": 0.0},
        {"name": "height_low_gamma_plus8", "h_shift": -2.0, "alpha_shift": 0.0, "beta_shift": 0.0, "gamma_deg": 8.0},
        {"name": "height_high_gamma_minus8", "h_shift": 2.0, "alpha_shift": 0.0, "beta_shift": 0.0, "gamma_deg": -8.0},
        {"name": "alpha_beta_tradeoff", "h_shift": 0.0, "alpha_shift": 0.045, "beta_shift": -0.025, "gamma_deg": 0.0},
        {"name": "gamma_plus8", "h_shift": 0.0, "alpha_shift": 0.0, "beta_shift": 0.0, "gamma_deg": 8.0},
        {"name": "gamma_minus8", "h_shift": 0.0, "alpha_shift": 0.0, "beta_shift": 0.0, "gamma_deg": -8.0},
    ]


def _smoothness(values: np.ndarray, stroke_ids: np.ndarray) -> float:
    jumps: list[float] = []
    for stroke in np.unique(stroke_ids):
        local = values[stroke_ids == stroke]
        if len(local) > 1:
            jumps.extend(np.linalg.norm(np.diff(local, axis=0), axis=1).tolist())
    return float(np.max(jumps)) if jumps else 0.0


def _write_candidate(
    path: Path,
    rows: list[dict[str, str]],
    fields: list[str],
    posture: np.ndarray,
    gamma: np.ndarray,
    candidate_id: str,
    target_width: np.ndarray,
    predicted_width: np.ndarray,
) -> None:
    extra = ["candidate_id", "candidate_group", "target_width_Lr", "predicted_width_Lr", "width_relative_error"]
    output_fields = fields + [field for field in extra if field not in fields]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=output_fields)
        writer.writeheader()
        for index, source in enumerate(rows):
            row = dict(source)
            row.update(
                {
                    "z": repr(float(posture[index, 0])),
                    "alpha": repr(float(posture[index, 1])),
                    "beta": repr(float(posture[index, 2])),
                    "gamma": repr(float(gamma[index])),
                    "candidate_id": candidate_id,
                    "candidate_group": "same_width_simulation_candidates_v1",
                    "target_width_Lr": repr(float(target_width[index])),
                    "predicted_width_Lr": repr(float(predicted_width[index])),
                    "width_relative_error": repr(float(abs(predicted_width[index] - target_width[index]) / max(abs(target_width[index]), 1e-8))),
                }
            )
            for field, source_name in (
                ("prototype", "paper_width_equivalent_v1"),
                ("z_source", "width_equivalent_candidate"),
                ("alpha_source", "width_equivalent_candidate"),
                ("beta_source", "width_equivalent_candidate"),
                ("gamma_source", "heading_candidate_not_width_observed"),
            ):
                if field in output_fields:
                    row[field] = source_name
            writer.writerow(row)


def _build_candidate(
    base: np.ndarray,
    target_width: np.ndarray,
    matrix: np.ndarray,
    bias: np.ndarray,
    spec: dict[str, Any],
    stroke_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    height = np.clip(base[:, 0] + float(spec["h_shift"]), PAPER_POSTURE_MIN[0], PAPER_POSTURE_MAX[0])
    alpha_pref = np.clip(base[:, 1] + float(spec["alpha_shift"]), PAPER_POSTURE_MIN[1], PAPER_POSTURE_MAX[1])
    beta_pref = np.clip(base[:, 2] + float(spec["beta_shift"]), PAPER_POSTURE_MIN[2], PAPER_POSTURE_MAX[2])
    alpha, beta, width = _solve_width_line(target_width, height, alpha_pref, beta_pref, matrix, bias)
    # Keep float64 through export.  Casting the exact equality solution to
    # float32 makes the reported width residual needlessly look larger than
    # the actual regression error.
    posture = np.column_stack((height, alpha, beta)).astype(np.float64)
    gamma = _wrap_angle(base[:, 3] + np.deg2rad(float(spec["gamma_deg"]))).astype(np.float64)
    rel_error = np.abs(width - target_width) / np.maximum(np.abs(target_width), 1e-8)
    lower = PAPER_POSTURE_MIN[None, :]
    upper = PAPER_POSTURE_MAX[None, :]
    boundary = np.any((posture <= lower + 1e-5) | (posture >= upper - 1e-5), axis=1)
    delta = posture - base[:, :3]
    normalized_delta = delta / np.maximum(upper - lower, 1e-8)
    width_rmse = float(np.sqrt(np.mean((width - target_width) ** 2)))
    rel_rmse = float(np.sqrt(np.mean(rel_error ** 2)))
    max_rel = float(np.max(rel_error))
    drag = posture_to_geometry_numpy(posture, angle_basis=matrix_to_basis(matrix))[:, 0:2].sum(axis=1)
    base_drag = posture_to_geometry_numpy(base[:, :3], angle_basis=matrix_to_basis(matrix))[:, 0:2].sum(axis=1)
    drag_rel_rmse = float(np.sqrt(np.mean(((drag - base_drag) / np.maximum(np.abs(base_drag), 1e-8)) ** 2)))
    pose_step = _smoothness(posture[:, 1:], stroke_ids)
    gamma_step = _smoothness(gamma[:, None], stroke_ids)
    smooth = max(pose_step, gamma_step)
    boundary_fraction = float(np.mean(boundary))
    # This score is deliberately only confidence in mathematical width
    # equivalence.  Boundary saturation and heading discontinuity are
    # reported separately and may make a candidate unsuitable for a robot.
    confidence = float(np.exp(-rel_rmse / 0.005))
    report = {
        "name": spec["name"],
        "h_shift_requested_mm": float(spec["h_shift"]),
        "alpha_preference_shift_rad": float(spec["alpha_shift"]),
        "beta_preference_shift_rad": float(spec["beta_shift"]),
        "gamma_offset_deg": float(spec["gamma_deg"]),
        "width_target_Lr_mean": float(np.mean(target_width)),
        "width_predicted_Lr_mean": float(np.mean(width)),
        "width_rmse_Lr": width_rmse,
        "width_relative_rmse": rel_rmse,
        "width_max_relative_error": max_rel,
        "drag_relative_rmse": drag_rel_rmse,
        "boundary_fraction": boundary_fraction,
        "max_within_stroke_pose_step_rad": float(pose_step),
        "max_within_stroke_gamma_step_rad": float(gamma_step),
        "max_within_stroke_angle_or_gamma_step": float(smooth),
        "pose_delta_normalized_rms": float(np.sqrt(np.mean(normalized_delta ** 2))),
        "width_equivalence_confidence": confidence,
        "high_confidence_width": bool(rel_rmse <= 0.005 and max_rel <= 0.02),
        "gamma_is_width_unobserved": True,
    }
    return posture, gamma, report


def main(args: argparse.Namespace) -> None:
    rows, fields = _read_pose_csv(Path(args.pose_csv))
    matrix = regression_matrix_numpy(_angle_basis(rows, args.angle_basis))
    bias = np.asarray([0.0267, 0.0372, 0.1137], dtype=np.float32)
    base = np.asarray(
        [[float(row["z"]), float(row["alpha"]), float(row["beta"]), float(row.get("gamma", 0.0) or 0.0)] for row in rows],
        dtype=np.float32,
    )
    if np.any(base[:, :3] < PAPER_POSTURE_MIN[None, :] - 1e-5) or np.any(base[:, :3] > PAPER_POSTURE_MAX[None, :] + 1e-5):
        raise ValueError("Input pose exceeds H=11..20 mm, alpha=0..10 deg, beta=0..5 deg")
    stroke_ids = np.asarray([int(row["stroke_id"]) for row in rows], dtype=np.int64)
    target_width = posture_to_geometry_numpy(base[:, :3], angle_basis=_angle_basis(rows, args.angle_basis))[:, 2]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for spec in _candidate_specs():
        posture, gamma, report = _build_candidate(base, target_width, matrix, bias, spec, stroke_ids)
        accepted = bool(
            report["width_relative_rmse"] <= args.max_width_relative_rmse
            and report["width_max_relative_error"] <= args.max_width_relative_error
        )
        report["accepted"] = accepted
        report["robot_test_ready"] = bool(
            accepted
            and report["boundary_fraction"] <= args.max_boundary_fraction
            and report["max_within_stroke_angle_or_gamma_step"] <= args.max_step_rad
        )
        if not report["robot_test_ready"]:
            report["robot_test_warning"] = (
                "Width-equivalent simulation candidate only: inspect boundary "
                "saturation and within-stroke angle/gamma continuity before robot use."
            )
        if accepted or args.keep_rejected:
            candidate_id = f"candidate_{len(candidates):02d}_{spec['name']}"
            candidate_dir = output / candidate_id
            _write_candidate(
                candidate_dir / "pose.csv",
                rows,
                fields,
                posture,
                gamma,
                candidate_id,
                target_width,
                posture_to_geometry_numpy(posture, angle_basis=_angle_basis(rows, args.angle_basis))[:, 2],
            )
            report["candidate_id"] = candidate_id
            report["pose_csv"] = str(candidate_dir / "pose.csv")
            (candidate_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            candidates.append(report)
            seen.add(spec["name"])
        if len(candidates) >= args.candidate_count:
            break
    candidates.sort(key=lambda item: (-float(item["width_equivalence_confidence"]), float(item["width_relative_rmse"])))
    for index, item in enumerate(candidates):
        item["rank"] = index + 1
    summary = {
        "format": "paper_width_equivalent_pose_candidates_v1",
        "source_pose_csv": str(Path(args.pose_csv)),
        "candidate_group": "same_width_simulation_candidates_v1",
        "angle_basis": _angle_basis(rows, args.angle_basis),
        "width_definition": "B-BSMG Lr from posture_to_geometry; transverse radius in model units",
        "candidate_count_requested": args.candidate_count,
        "candidate_count_saved": len(candidates),
        "accepted_count": int(sum(bool(item["accepted"]) for item in candidates)),
        "simulation_only": True,
        "gamma_note": "Gamma does not enter Lr; gamma offsets are heading alternatives and are not width observations.",
        "not_physical_calibration": "Requires real brush/TCP/paper calibration before robot execution.",
        "constraints": {
            "max_width_relative_rmse": args.max_width_relative_rmse,
            "max_width_relative_error": args.max_width_relative_error,
            "max_boundary_fraction": args.max_boundary_fraction,
            "max_step_rad": args.max_step_rad,
        },
        "candidates": candidates,
    }
    (output / "candidate_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "candidate_summary.csv").open("w", encoding="utf-8-sig", newline="") as file:
        if candidates:
            writer = csv.DictWriter(file, fieldnames=sorted({key for row in candidates for key in row}))
            writer.writeheader()
            writer.writerows(candidates)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose_csv", required=True)
    parser.add_argument("--output_dir", default="outputs/width_equivalent_pose_candidates")
    parser.add_argument("--candidate_count", type=int, default=8)
    parser.add_argument("--angle_basis", choices=("auto", "paper_declared_radian", "degree_fitted"), default="auto")
    parser.add_argument("--max_width_relative_rmse", type=float, default=0.005)
    parser.add_argument("--max_width_relative_error", type=float, default=0.02)
    parser.add_argument("--max_boundary_fraction", type=float, default=0.25)
    parser.add_argument("--max_step_rad", type=float, default=float(np.deg2rad(35.0)))
    parser.add_argument("--keep_rejected", action="store_true")
    main(parser.parse_args())
