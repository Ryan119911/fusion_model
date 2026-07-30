"""Compare bounded x/y inversion restarts without conflating pose fields."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np


FORMAT = "xy_refinement_stability_v1"
POSE_FIELDS = ("z", "alpha", "beta", "gamma")


def load_pose_csv(path: str) -> tuple[list[tuple[str, str]], dict[str, np.ndarray]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: (int(row["stroke_id"]), int(row["point_id"])))
    keys = [(row["stroke_id"], row["point_id"]) for row in rows]
    values = {
        field: np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        for field in ("x", "y", *POSE_FIELDS)
    }
    return keys, values


def compare_runs(
    csv_paths: list[str],
    report_paths: list[str],
    drawable_pixels: float = 96.0,
) -> dict[str, Any]:
    if len(csv_paths) < 2 or len(csv_paths) != len(report_paths):
        raise ValueError("Need at least two matching CSV/report pairs")
    loaded = [load_pose_csv(path) for path in csv_paths]
    reference_keys = loaded[0][0]
    if any(keys != reference_keys for keys, _ in loaded[1:]):
        raise ValueError("Trajectory keys differ across refinements")
    coordinates = np.stack(
        [np.stack([values["x"], values["y"]], axis=-1) for _, values in loaded]
    )
    reference_range = np.ptp(coordinates[0], axis=0)
    reference_range = np.maximum(reference_range, 1e-8)
    normalized = coordinates / reference_range[None, None, :]
    point_std = normalized.std(axis=0)
    rms_std = np.sqrt(np.mean(np.square(point_std), axis=0))
    pairwise = {}
    for left, right in itertools.combinations(range(len(csv_paths)), 2):
        delta_px = (normalized[left] - normalized[right]) * drawable_pixels
        pairwise[f"{left}:{right}"] = {
            "rms_canvas_px": float(np.sqrt(np.mean(np.sum(delta_px**2, axis=1)))),
            "max_canvas_px": float(np.sqrt(np.sum(delta_px**2, axis=1)).max()),
        }
    posture_max_difference = {}
    for field in POSE_FIELDS:
        values = np.stack([item[1][field] for item in loaded])
        posture_max_difference[field] = float(
            np.max(np.abs(values - values[0:1]))
        )
    reports = [
        json.loads(Path(path).read_text(encoding="utf-8"))
        for path in report_paths
    ]
    summaries = []
    for csv_path, report_path, report in zip(csv_paths, report_paths, reports):
        summaries.append(
            {
                "trajectory_csv": csv_path,
                "report": report_path,
                "metrics": report.get("metrics", {}),
                "xy_optimization": report.get("xy_optimization", {}),
                "trajectory_target_coverage_at_5px": report.get(
                    "trajectory_target_coverage_at_5px"
                ),
            }
        )
    return {
        "format": FORMAT,
        "run_count": len(csv_paths),
        "points": len(reference_keys),
        "normalized_rms_std": {
            "x": float(rms_std[0]),
            "y": float(rms_std[1]),
            "xy": float(np.sqrt(np.mean(np.square(point_std)))),
        },
        "pairwise_canvas_distance": pairwise,
        "posture_max_abs_difference": posture_max_difference,
        "posture_frozen_exactly": all(
            difference <= 1e-8
            for difference in posture_max_difference.values()
        ),
        "runs": summaries,
    }


def main(args: argparse.Namespace) -> None:
    report = compare_runs(
        args.trajectory_csvs,
        args.report_jsons,
        drawable_pixels=args.drawable_pixels,
    )
    report["stability_threshold"] = args.max_normalized_rms_std
    report["stable"] = bool(
        report["normalized_rms_std"]["xy"] <= args.max_normalized_rms_std
        and report["posture_frozen_exactly"]
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
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--drawable_pixels", type=float, default=96.0)
    parser.add_argument("--max_normalized_rms_std", type=float, default=0.02)
    main(parser.parse_args())
