"""Compare an inverse paper-pose CSV with known simulated ground truth."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


FIELDS = {
    "z": 9.0,
    "alpha": float(np.deg2rad(10.0)),
    "beta": float(np.deg2rad(5.0)),
    "gamma": float(np.deg2rad(60.0)),
}


def load_rows(path: str) -> dict[tuple[int, int], dict]:
    with open(path, "r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = {}
    for row in rows:
        key = (int(row["stroke_id"]), int(row["point_id"]))
        if key in result:
            raise ValueError(f"Duplicate stroke/point key {key} in {path}")
        result[key] = row
    return result


def evaluate(reference_csv: str, estimate_csv: str) -> dict:
    reference = load_rows(reference_csv)
    estimate = load_rows(estimate_csv)
    if reference.keys() != estimate.keys():
        raise ValueError("Reference and estimate stroke/point keys differ")
    keys = sorted(reference)
    metrics = {}
    for field, physical_range in FIELDS.items():
        truth = np.asarray([float(reference[key][field]) for key in keys])
        predicted = np.asarray([float(estimate[key][field]) for key in keys])
        error = predicted - truth
        metrics[field] = {
            "rmse": float(np.sqrt(np.mean(error**2))),
            "mae": float(np.mean(np.abs(error))),
            "max_abs_error": float(np.max(np.abs(error))),
            "normalized_rmse": float(
                np.sqrt(np.mean(error**2)) / physical_range
            ),
            "truth_range": [float(truth.min()), float(truth.max())],
            "estimate_range": [
                float(predicted.min()),
                float(predicted.max()),
            ],
        }
    return {
        "format": "paper_pose_recovery_evaluation_v1",
        "simulation_only": True,
        "reference_csv": reference_csv,
        "estimate_csv": estimate_csv,
        "point_count": len(keys),
        "angle_unit": "rad",
        "metrics": metrics,
        "interpretation": (
            "Pose recovery is only identifiable for fields/nodes whose image "
            "Jacobian exceeds the configured observability noise gate."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_csv", required=True)
    parser.add_argument("--estimate_csv", required=True)
    parser.add_argument(
        "--output_json",
        default="outputs/paper_pose_recovery.json",
    )
    args = parser.parse_args()
    report = evaluate(args.reference_csv, args.estimate_csv)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for field, metrics in report["metrics"].items():
        print(
            f"{field}: rmse={metrics['rmse']:.6f}, "
            f"normalized_rmse={metrics['normalized_rmse']:.6f}"
        )
    print(f"[DONE] Pose recovery report: {output}")


if __name__ == "__main__":
    main()
