"""Validate the real-brush/robot calibration interchange format.

The inverse renderer must not treat angles as physical labels until the
calibration experiment varies them independently. This tool validates units,
keys, ranges and excitation coverage before a CSV is used for model fitting.
It does not claim image-based observability; that remains a renderer/Jacobian
test performed on held-out calibration trials.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = (
    "trial_id", "point_id", "timestamp_s", "x_mm", "y_mm", "z_mm",
    "alpha_rad", "beta_rad", "gamma_rad", "contact",
)
OPTIONAL_MEASUREMENTS = (
    "footprint_width_mm", "footprint_length_mm", "footprint_angle_rad",
    "image_path",
)
PROTOTYPE_RANGES = {
    "z_mm": (11.0, 20.0),
    "alpha_rad": (0.0, math.radians(10.0)),
    "beta_rad": (0.0, math.radians(5.0)),
}
MIN_EXCITATION_SPAN = {
    "z_mm": 4.0,
    "alpha_rad": math.radians(5.0),
    "beta_rad": math.radians(2.5),
    "gamma_rad": math.radians(20.0),
}


def _finite(row: dict[str, str], field: str, row_number: int) -> float:
    raw = (row.get(field) or "").strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"row {row_number}: {field} must be a finite number, got {raw!r}"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(f"row {row_number}: {field} must be finite")
    return value


def _contact(row: dict[str, str], row_number: int) -> int:
    raw = (row.get("contact") or "").strip().lower()
    mapping = {"0": 0, "1": 1, "false": 0, "true": 1}
    if raw not in mapping:
        raise ValueError(
            f"row {row_number}: contact must be 0/1/false/true, got {raw!r}"
        )
    return mapping[raw]


def validate_calibration_csv(
    csv_path: str | Path,
    *,
    enforce_prototype_ranges: bool = True,
    require_measurement: bool = True,
) -> dict[str, Any]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Calibration CSV not found: {path}")

    values: dict[str, list[float]] = defaultdict(list)
    trial_counts: Counter[str] = Counter()
    measurement_counts: Counter[str] = Counter()
    keys: set[tuple[str, int]] = set()
    previous_time: dict[str, float] = {}
    contact_count = 0

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames or []
        missing = [name for name in REQUIRED_COLUMNS if name not in fieldnames]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")

        for row_number, row in enumerate(reader, start=2):
            trial_id = (row.get("trial_id") or "").strip()
            if not trial_id:
                raise ValueError(f"row {row_number}: trial_id must not be empty")
            point_value = _finite(row, "point_id", row_number)
            if not point_value.is_integer() or point_value < 0:
                raise ValueError(
                    f"row {row_number}: point_id must be a non-negative integer"
                )
            point_id = int(point_value)
            key = (trial_id, point_id)
            if key in keys:
                raise ValueError(
                    f"row {row_number}: duplicate trial_id/point_id key {key}"
                )
            keys.add(key)

            timestamp = _finite(row, "timestamp_s", row_number)
            if timestamp < 0:
                raise ValueError(f"row {row_number}: timestamp_s must be >= 0")
            if trial_id in previous_time and timestamp <= previous_time[trial_id]:
                raise ValueError(
                    f"row {row_number}: timestamps must strictly increase within "
                    f"trial {trial_id!r}"
                )
            previous_time[trial_id] = timestamp

            for field in (
                "x_mm", "y_mm", "z_mm", "alpha_rad", "beta_rad", "gamma_rad"
            ):
                values[field].append(_finite(row, field, row_number))

            if enforce_prototype_ranges:
                for field, (lower, upper) in PROTOTYPE_RANGES.items():
                    value = values[field][-1]
                    tolerance = 1e-6
                    if value < lower - tolerance or value > upper + tolerance:
                        raise ValueError(
                            f"row {row_number}: {field}={value:.9g} exceeds "
                            f"prototype range [{lower:.9g}, {upper:.9g}]"
                        )

            contact_count += _contact(row, row_number)
            trial_counts[trial_id] += 1
            for field in OPTIONAL_MEASUREMENTS:
                raw = (row.get(field) or "").strip()
                if not raw:
                    continue
                if field == "image_path":
                    measurement_counts[field] += 1
                    continue
                value = _finite(row, field, row_number)
                if field in ("footprint_width_mm", "footprint_length_mm") and value <= 0:
                    raise ValueError(f"row {row_number}: {field} must be positive")
                measurement_counts[field] += 1

    if not keys:
        raise ValueError("Calibration CSV contains no samples")
    if contact_count == 0:
        raise ValueError("Calibration CSV contains no contact samples")
    if require_measurement and not any(measurement_counts.values()):
        raise ValueError(
            "Calibration CSV needs at least one measured footprint field or image_path"
        )

    ranges = {
        field: {
            "min": min(series),
            "max": max(series),
            "span": max(series) - min(series),
        }
        for field, series in values.items()
    }
    excitation = {}
    for field in ("z_mm", "alpha_rad", "beta_rad", "gamma_rad"):
        required = MIN_EXCITATION_SPAN[field]
        span = ranges[field]["span"]
        excitation[field] = {
            "span": span,
            "minimum_recommended_span": required,
            "excitation_ready": span >= required,
        }

    return {
        "format": "robot_brush_calibration_v1",
        "source_csv": str(path),
        "units": {"position": "mm", "time": "s", "angles": "rad"},
        "samples": len(keys),
        "contact_samples": contact_count,
        "trials": len(trial_counts),
        "points_per_trial": dict(sorted(trial_counts.items())),
        "measurement_counts": dict(measurement_counts),
        "ranges": ranges,
        "excitation_audit": excitation,
        "observability_status": {
            "state": "not_tested",
            "note": (
                "Excitation readiness is necessary but not sufficient. Run held-out "
                "image/Jacobian observability tests before optimizing z or angles."
            ),
        },
    }


def main(args: argparse.Namespace) -> None:
    report = validate_calibration_csv(
        args.calibration_csv,
        enforce_prototype_ranges=not args.allow_outside_prototype,
        require_measurement=not args.allow_pose_only,
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[DONE] Validated {report['samples']} samples in {report['trials']} trials")
    for field, audit in report["excitation_audit"].items():
        print(
            f"[EXCITATION] {field}: span={audit['span']:.6g}, "
            f"ready={audit['excitation_ready']}"
        )
    print(f"[DONE] Report saved to: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration_csv", required=True)
    parser.add_argument(
        "--output_json",
        default="data/processed/robot_brush_calibration.summary.json",
    )
    parser.add_argument(
        "--allow_outside_prototype",
        action="store_true",
        help="Allow z/alpha/beta outside the current simulation bounds",
    )
    parser.add_argument(
        "--allow_pose_only",
        action="store_true",
        help="Allow CSVs without footprint measurements or image paths",
    )
    main(parser.parse_args())
