"""Generate posture candidates directly from image local-footprint targets.

For each trajectory point the target pair is ``(Lt+Lh, Lr)`` from
``calibrate_target_local_footprints.py``.  Since this pair gives two equations
for ``H, alpha, beta``, the script scans bounded H values and solves alpha/beta
exactly where possible.  Several H preferences and gamma heading offsets are
exported for later simulation comparison.  This is still a paper/image
simulation and deliberately does not claim real brush calibration.
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


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"Missing CSV header: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return rows


def f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    return float(default) if value in (None, "") else float(value)


def wrapped_delta(values: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(values), np.cos(values))


def grouped_max(values: np.ndarray, stroke_ids: np.ndarray, wrap: bool = False) -> float:
    result: list[float] = []
    for stroke in np.unique(stroke_ids):
        local = values[stroke_ids == stroke]
        if len(local) > 1:
            delta = np.diff(local, axis=0)
            if wrap:
                delta = wrapped_delta(delta)
            result.extend(np.abs(delta).reshape(len(delta), -1).max(axis=1).tolist())
    return float(max(result)) if result else 0.0


def load_targets(path: Path, pose_rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference = {(int(f(row, "stroke_id")), int(f(row, "point_id"))): row for row in read_rows(path)}
    drag, width, confidence = [], [], []
    for row in pose_rows:
        key = (int(f(row, "stroke_id")), int(f(row, "point_id")))
        target = reference.get(key)
        if target is None:
            drag.append(np.nan)
            width.append(np.nan)
            confidence.append(0.0)
        else:
            drag.append(f(target, "target_drag", np.nan))
            width.append(f(target, "target_half_width", np.nan))
            confidence.append(f(target, "confidence", 0.0))
    return np.asarray(drag), np.asarray(width), np.asarray(confidence)


def solve_point(
    target_drag: float,
    target_width: float,
    confidence: float,
    requested_h: float,
    base_angles: np.ndarray,
    matrix: np.ndarray,
    bias: np.ndarray,
    h_grid: np.ndarray,
) -> tuple[np.ndarray, float, float, bool]:
    """Return the bounded posture closest to requested H and target footprint."""
    if not (np.isfinite(target_drag) and np.isfinite(target_width) and confidence > 0.0):
        return np.asarray([requested_h, base_angles[0], base_angles[1]], dtype=np.float64), float("inf"), float("inf"), False
    reduced = np.asarray([matrix[0] + matrix[1], matrix[2]], dtype=np.float64)
    reduced_bias = np.asarray([bias[0] + bias[1], bias[2]], dtype=np.float64)
    angle_matrix = reduced[:, 1:]
    rhs = np.asarray([target_drag, target_width], dtype=np.float64) - reduced_bias
    solutions: list[tuple[float, np.ndarray, float]] = []
    for height in h_grid:
        try:
            angles = np.linalg.solve(angle_matrix, rhs - reduced[:, 0] * height)
        except np.linalg.LinAlgError:
            continue
        if np.all(angles >= PAPER_POSTURE_MIN[1:] - 1e-9) and np.all(angles <= PAPER_POSTURE_MAX[1:] + 1e-9):
            solutions.append((abs(float(height) - requested_h), angles, float(height)))
    if solutions:
        _, angles, height = min(solutions, key=lambda item: item[0])
        return np.asarray([height, angles[0], angles[1]], dtype=np.float64), 0.0, 0.0, True

    # If the target pair is outside the posture box, minimize normalized
    # footprint error while remaining bounded, with a weak H preference.
    candidates: list[tuple[float, np.ndarray, float, float, float]] = []
    for height in h_grid:
        angles = np.asarray(base_angles, dtype=np.float64)
        for free in (0, 1):
            other = 1 - free
            numerator = rhs[0] - reduced[0, 0] * height - angle_matrix[0, other] * angles[other]
            denominator = angle_matrix[0, free]
            angles[free] = numerator / max(denominator, 1e-12)
            angles = np.clip(angles, PAPER_POSTURE_MIN[1:], PAPER_POSTURE_MAX[1:])
            geometry = np.asarray([height, angles[0], angles[1]]) @ matrix.T + bias
            drag_error = abs(float(geometry[0] + geometry[1] - target_drag)) / max(abs(target_drag), 1e-6)
            width_error = abs(float(geometry[2] - target_width)) / max(abs(target_width), 1e-6)
            score = drag_error + width_error + 0.02 * abs(float(height) - requested_h)
            candidates.append((score, np.asarray([height, angles[0], angles[1]]), drag_error, width_error, float(height)))
    score, posture, drag_error, width_error, height = min(candidates, key=lambda item: item[0])
    return posture, drag_error, width_error, False


def candidate_specs() -> list[dict[str, float | str]]:
    return [
        {"name": "target_h_base", "h_shift": 0.0, "gamma_deg": 0.0},
        {"name": "target_h_low", "h_shift": -2.0, "gamma_deg": 0.0},
        {"name": "target_h_high", "h_shift": 2.0, "gamma_deg": 0.0},
        {"name": "target_h_low_gamma_plus8", "h_shift": -2.0, "gamma_deg": 8.0},
        {"name": "target_h_high_gamma_minus8", "h_shift": 2.0, "gamma_deg": -8.0},
        {"name": "target_h_mid_gamma_plus5", "h_shift": 0.5, "gamma_deg": 5.0},
        {"name": "target_h_mid_gamma_minus5", "h_shift": -0.5, "gamma_deg": -5.0},
        {"name": "target_h_base_gamma_plus8", "h_shift": 0.0, "gamma_deg": 8.0},
    ]


def write_candidate(path: Path, rows: list[dict[str, str]], posture: np.ndarray, gamma: np.ndarray, target_drag: np.ndarray, target_width: np.ndarray, confidence: np.ndarray, errors: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    extras = ["candidate_id", "target_drag", "target_half_width", "footprint_confidence", "predicted_drag", "predicted_width_Lr", "drag_relative_error", "width_relative_error", "solve_exact"]
    output_fields = fields + [item for item in extras if item not in fields]
    predicted_drag, predicted_width, exact = errors
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=output_fields)
        writer.writeheader()
        for index, source in enumerate(rows):
            row = dict(source)
            row.update({
                "z": repr(float(posture[index, 0])),
                "alpha": repr(float(posture[index, 1])),
                "beta": repr(float(posture[index, 2])),
                "gamma": repr(float(gamma[index])),
                "candidate_id": path.parent.name,
                "target_drag": repr(float(target_drag[index])),
                "target_half_width": repr(float(target_width[index])),
                "footprint_confidence": repr(float(confidence[index])),
                "predicted_drag": repr(float(predicted_drag[index])),
                "predicted_width_Lr": repr(float(predicted_width[index])),
                "drag_relative_error": repr(float(abs(predicted_drag[index] - target_drag[index]) / max(abs(target_drag[index]), 1e-8))) if np.isfinite(target_drag[index]) else "nan",
                "width_relative_error": repr(float(abs(predicted_width[index] - target_width[index]) / max(abs(target_width[index]), 1e-8))) if np.isfinite(target_width[index]) else "nan",
                "solve_exact": int(exact[index]),
            })
            writer.writerow(row)


def main(args: argparse.Namespace) -> None:
    pose_rows = read_rows(Path(args.pose_csv))
    target_drag, target_width, confidence = load_targets(Path(args.footprint_csv), pose_rows)
    basis = next((row.get("regression_angle_basis", "") for row in pose_rows if row.get("regression_angle_basis", "")), PAPER_ANGLE_BASIS_RADIAN)
    matrix = regression_matrix_numpy(basis)
    bias = np.asarray([0.0267, 0.0372, 0.1137], dtype=np.float64)
    base = np.asarray([[f(row, "z"), f(row, "alpha"), f(row, "beta"), f(row, "gamma")] for row in pose_rows], dtype=np.float64)
    stroke_ids = np.asarray([int(f(row, "stroke_id")) for row in pose_rows], dtype=np.int64)
    h_grid = np.linspace(PAPER_POSTURE_MIN[0], PAPER_POSTURE_MAX[0], args.h_grid_points)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    for spec in candidate_specs()[: args.candidate_count]:
        posture = np.zeros((len(pose_rows), 3), dtype=np.float64)
        exact = np.zeros(len(pose_rows), dtype=bool)
        drag_error = np.full(len(pose_rows), np.nan)
        width_error = np.full(len(pose_rows), np.nan)
        for index in range(len(pose_rows)):
            requested_h = float(np.clip(base[index, 0] + float(spec["h_shift"]), PAPER_POSTURE_MIN[0], PAPER_POSTURE_MAX[0]))
            # ``solve_point`` solves only the two linear angle variables
            # (alpha, beta); gamma is deliberately kept separate as the
            # heading/next-stroke direction variable.
            posture[index], drag_error[index], width_error[index], exact[index] = solve_point(target_drag[index], target_width[index], confidence[index], requested_h, base[index, 1:3], matrix, bias, h_grid)
        gamma = wrapped_delta(base[:, 3] + np.deg2rad(float(spec["gamma_deg"])))
        geometry = posture_to_geometry_numpy(posture, angle_basis=basis)
        predicted_drag = geometry[:, 0] + geometry[:, 1]
        predicted_width = geometry[:, 2]
        valid = np.isfinite(target_drag) & np.isfinite(target_width) & (confidence > args.minimum_confidence)
        d = np.abs(predicted_drag[valid] - target_drag[valid]) / np.maximum(np.abs(target_drag[valid]), 1e-8)
        w = np.abs(predicted_width[valid] - target_width[valid]) / np.maximum(np.abs(target_width[valid]), 1e-8)
        lower, upper = PAPER_POSTURE_MIN[None, :], PAPER_POSTURE_MAX[None, :]
        boundary_fraction = float(np.mean(np.any((posture <= lower + 1e-4) | (posture >= upper - 1e-4), axis=1)))
        report = {
            "candidate_id": f"candidate_{len(reports):02d}_{spec['name']}",
            "requested_h_shift_mm": float(spec["h_shift"]),
            "gamma_offset_deg": float(spec["gamma_deg"]),
            "footprint_valid_points": int(valid.sum()),
            "drag_relative_rmse": float(np.sqrt(np.mean(d**2))) if len(d) else float("inf"),
            "width_relative_rmse": float(np.sqrt(np.mean(w**2))) if len(w) else float("inf"),
            "drag_max_relative_error": float(d.max()) if len(d) else float("inf"),
            "width_max_relative_error": float(w.max()) if len(w) else float("inf"),
            "exact_solve_fraction": float(np.mean(exact[valid])) if np.any(valid) else 0.0,
            "boundary_fraction": boundary_fraction,
            "h_mean_mm": float(posture[:, 0].mean()),
            "h_min_mm": float(posture[:, 0].min()),
            "h_max_mm": float(posture[:, 0].max()),
            "max_pose_step": max(grouped_max(posture[:, 0], stroke_ids), grouped_max(posture[:, 1], stroke_ids), grouped_max(posture[:, 2], stroke_ids)),
            "max_gamma_step_rad": grouped_max(gamma, stroke_ids, wrap=True),
            "simulation_only": True,
            "real_brush_calibration_used": False,
        }
        report["footprint_ok"] = bool(report["drag_relative_rmse"] <= args.max_drag_relative_rmse and report["width_relative_rmse"] <= args.max_width_relative_rmse and report["exact_solve_fraction"] >= args.minimum_exact_fraction)
        report["pose_csv"] = str(output / report["candidate_id"] / "pose.csv")
        write_candidate(Path(report["pose_csv"]), pose_rows, posture, gamma, target_drag, target_width, confidence, (predicted_drag, predicted_width, exact))
        (output / report["candidate_id"] / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        reports.append(report)
    summary = {
        "format": "paper_target_footprint_pose_candidates_v1",
        "source_pose_csv": args.pose_csv,
        "footprint_csv": args.footprint_csv,
        "candidate_count": len(reports),
        "simulation_only": True,
        "real_brush_calibration_used": False,
        "candidates": reports,
    }
    (output / "candidate_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose_csv", required=True)
    parser.add_argument("--footprint_csv", required=True)
    parser.add_argument("--output_dir", default="outputs/wu_target_footprint_pose_candidates_v1")
    parser.add_argument("--candidate_count", type=int, default=8)
    parser.add_argument("--h_grid_points", type=int, default=181)
    parser.add_argument("--minimum_confidence", type=float, default=0.1)
    parser.add_argument("--minimum_exact_fraction", type=float, default=0.75)
    parser.add_argument("--max_drag_relative_rmse", type=float, default=0.20)
    parser.add_argument("--max_width_relative_rmse", type=float, default=0.20)
    main(parser.parse_args())
