"""Audit pose-recovery restart stability when real pose truth is unavailable."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np


FORMAT = "pose_refinement_restart_stability_v1"
DEFAULT_RANGES = {
    "z": 9.0,
    "alpha": float(np.deg2rad(10.0)),
    "beta": float(np.deg2rad(5.0)),
    "gamma": float(np.deg2rad(60.0)),
}


def load_csv(path: str) -> tuple[list[tuple[int, int]], dict[str, np.ndarray]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: (int(row["stroke_id"]), int(row["point_id"])))
    keys = [(int(row["stroke_id"]), int(row["point_id"])) for row in rows]
    fields = {
        field: np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        for field in ("x", "y", "z", "alpha", "beta", "gamma")
    }
    return keys, fields


def compare_pose_runs(
    csv_paths: list[str],
    report_paths: list[str],
    pose_fields: list[str],
    stability_limit: float = 0.02,
    max_boundary_fraction: float = 0.05,
    min_coverage: float = 0.99,
) -> dict[str, Any]:
    if len(csv_paths) < 2 or len(csv_paths) != len(report_paths):
        raise ValueError("Need at least two matching CSV/report pairs")
    loaded = [load_csv(path) for path in csv_paths]
    if any(keys != loaded[0][0] for keys, _ in loaded[1:]):
        raise ValueError("Trajectory keys differ across pose runs")
    field_stability = {}
    for field in pose_fields:
        stacked = np.stack([values[field] for _, values in loaded])
        point_std = stacked.std(axis=0)
        normalized = float(
            np.sqrt(np.mean(point_std**2)) / DEFAULT_RANGES[field]
        )
        pairwise = {}
        for left, right in itertools.combinations(range(len(csv_paths)), 2):
            pairwise[f"{left}:{right}"] = float(
                np.sqrt(np.mean((stacked[left] - stacked[right]) ** 2))
            )
        field_stability[field] = {
            "normalized_rms_std": normalized,
            "limit": stability_limit,
            "passed": normalized <= stability_limit,
            "pairwise_rmse": pairwise,
        }
    xy_max_difference = {}
    for field in ("x", "y"):
        stacked = np.stack([values[field] for _, values in loaded])
        xy_max_difference[field] = float(
            np.max(np.abs(stacked - stacked[0:1]))
        )
    summaries = []
    for csv_path, report_path in zip(csv_paths, report_paths):
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        decisions = report.get("field_decisions", {})
        boundary = max(
            (
                float(decisions[field]["boundary_fraction"])
                for field in pose_fields
                if field in decisions
                and decisions[field].get("boundary_fraction") is not None
            ),
            default=0.0,
        )
        audit = report.get("identifiability", {}).get("joint_jacobian_audit", {})
        coverage = report.get("trajectory_target_coverage_at_5px")
        summary = {
            "trajectory_csv": csv_path,
            "report": report_path,
            "metrics": report.get("metrics", {}),
            "maximum_optimized_boundary_fraction": boundary,
            "joint_jacobian_audit": audit,
            "trajectory_target_coverage_at_5px": coverage,
        }
        summary["eligible"] = bool(
            boundary <= max_boundary_fraction
            and audit.get("jointly_identifiable") is True
            and coverage is not None
            and coverage >= min_coverage
        )
        summaries.append(summary)
    eligible = [
        index for index, summary in enumerate(summaries) if summary["eligible"]
    ]
    selected = (
        max(
            eligible,
            key=lambda index: summaries[index]["metrics"].get(
                "iou_at_0.5", float("-inf")
            ),
        )
        if eligible
        else None
    )
    stable = bool(
        all(item["passed"] for item in field_stability.values())
        and max(xy_max_difference.values(), default=0.0) <= 1e-8
    )
    if not stable:
        selected = None
    return {
        "format": FORMAT,
        "run_count": len(csv_paths),
        "pose_fields": pose_fields,
        "fields": field_stability,
        "xy_max_abs_difference": xy_max_difference,
        "xy_frozen_exactly": max(xy_max_difference.values(), default=0.0) <= 1e-8,
        "stable": stable,
        "selection_constraints": {
            "max_boundary_fraction": max_boundary_fraction,
            "min_trajectory_target_coverage_at_5px": min_coverage,
            "joint_jacobian_required": True,
        },
        "selected_run_index": selected,
        "selected_trajectory_csv": (
            summaries[selected]["trajectory_csv"] if selected is not None else None
        ),
        "runs": summaries,
        "interpretation": (
            "Restart agreement supports simulation stability only; it does not "
            "provide real brush or robot pose ground truth."
        ),
    }


def main(args: argparse.Namespace) -> None:
    report = compare_pose_runs(
        args.trajectory_csvs,
        args.report_jsons,
        args.pose_fields,
        stability_limit=args.max_normalized_rms_std,
        max_boundary_fraction=args.max_boundary_fraction,
        min_coverage=args.min_coverage,
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory_csvs", nargs="+", required=True)
    parser.add_argument("--report_jsons", nargs="+", required=True)
    parser.add_argument(
        "--pose_fields",
        nargs="+",
        choices=["z", "alpha", "beta", "gamma"],
        default=["gamma"],
    )
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--max_normalized_rms_std", type=float, default=0.02)
    parser.add_argument("--max_boundary_fraction", type=float, default=0.05)
    parser.add_argument("--min_coverage", type=float, default=0.99)
    main(parser.parse_args())
