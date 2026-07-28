"""Build a deterministic simulated pose trajectory for inverse round-trip tests.

The generated CSV is not robot calibration data. It supplies known, smooth
H/alpha/beta/gamma ground truth so the forward renderer and inverse solver can
be tested without conflating optimizer errors with a real calligraphy target.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def build_probe(
    input_csv: str, output_csv: str, profile: str = "truth"
) -> dict:
    with open(input_csv, "r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    required = {"stroke_id", "point_id", "x", "y"}
    if not rows or not required.issubset(fieldnames):
        raise ValueError(
            f"Pose CSV must contain rows and fields {sorted(required)}"
        )
    for field in (
        "z",
        "alpha",
        "beta",
        "gamma",
        "z_unit",
        "angle_unit",
        "pose_frame",
        "prototype",
        "z_source",
        "alpha_source",
        "beta_source",
        "gamma_source",
    ):
        if field not in fieldnames:
            fieldnames.append(field)

    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row["stroke_id"])].append(row)
    if profile not in {"truth", "perturbed_initial"}:
        raise ValueError(f"Unknown probe profile: {profile}")
    for stroke_id, stroke_rows in grouped.items():
        stroke_rows.sort(key=lambda row: int(row["point_id"]))
        denominator = max(len(stroke_rows) - 1, 1)
        for index, row in enumerate(stroke_rows):
            if profile == "truth":
                t = index / denominator
                reverse = bool(stroke_id % 2)
                u = 1.0 - t if reverse else t
                # Linear profiles are exactly representable by low-order CGL
                # interpolation and deliberately avoid physical boundaries.
                row["z"] = repr(13.0 + 4.0 * u)
                row["alpha"] = repr(float(np.deg2rad(2.0 + 6.0 * u)))
                row["beta"] = repr(
                    float(np.deg2rad(1.0 + 3.0 * (1.0 - u)))
                )
                row["gamma"] = repr(
                    float(np.deg2rad(-15.0 + 30.0 * u))
                )
                source = "synthetic_known_ground_truth"
            else:
                row["z"] = repr(
                    float(np.clip(float(row["z"]) + 0.4, 11.0, 20.0))
                )
                row["alpha"] = repr(
                    float(
                        np.clip(
                            float(row["alpha"]) + np.deg2rad(0.5),
                            0.0,
                            np.deg2rad(10.0),
                        )
                    )
                )
                row["beta"] = repr(
                    float(
                        np.clip(
                            float(row["beta"]) - np.deg2rad(0.25),
                            0.0,
                            np.deg2rad(5.0),
                        )
                    )
                )
                row["gamma"] = repr(
                    float(
                        np.clip(
                            float(row["gamma"]) + np.deg2rad(2.0),
                            -np.deg2rad(30.0),
                            np.deg2rad(30.0),
                        )
                    )
                )
                source = "synthetic_perturbed_initial_guess"
            row["z_unit"] = "mm"
            row["angle_unit"] = "rad"
            row["pose_frame"] = "paper_model"
            row["prototype"] = "paper_pose_roundtrip_probe_v1"
            for field in (
                "z_source",
                "alpha_source",
                "beta_source",
                "gamma_source",
            ):
                row[field] = source

    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fieldnames, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
    arrays = {
        field: np.asarray([float(row[field]) for row in rows])
        for field in ("z", "alpha", "beta", "gamma")
    }
    report = {
        "format": "paper_pose_roundtrip_probe_v1",
        "simulation_only": True,
        "profile": profile,
        "source_pose_csv": input_csv,
        "output_pose_csv": str(output),
        "point_count": len(rows),
        "stroke_count": len(grouped),
        "angle_unit": "rad",
        "pose_frame": "paper_model",
        "ranges": {
            field: [float(values.min()), float(values.max())]
            for field, values in arrays.items()
        },
        "warning": (
            "Synthetic observability probe only; values are not calibrated "
            "robot commands."
        ),
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pose_csv", required=True)
    parser.add_argument("--output_pose_csv", required=True)
    parser.add_argument(
        "--profile",
        choices=("truth", "perturbed_initial"),
        default="truth",
    )
    args = parser.parse_args()
    report = build_probe(args.input_pose_csv, args.output_pose_csv, args.profile)
    print(
        f"[DONE] Synthetic pose probe: points={report['point_count']}, "
        f"strokes={report['stroke_count']}, output={args.output_pose_csv}"
    )
    for field, limits in report["ranges"].items():
        print(f"{field}: {limits[0]:.6f} .. {limits[1]:.6f}")


if __name__ == "__main__":
    main()
