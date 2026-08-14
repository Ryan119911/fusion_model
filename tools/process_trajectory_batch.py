"""Batch-repair every character trajectory in a CSV database.

The batch artifact is deliberately one combined CSV plus JSONL reports.  This
keeps 9k-character databases manageable while retaining a machine-readable
record for every character/sample and a small configurable visual audit set.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable

from datasets.trajectory_dataset import load_trajectory_csv
from utils.trajectory_processing import (
    TrajectorySafetyLimits,
    densify_sample,
    expand_pen_up_transitions,
    render_trajectory_overlay,
    render_trajectory_preview,
    repair_sample_states,
    smooth_sample,
    validate_trajectory,
)


CSV_FIELDS = [
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
    "contact_state",
]


def _safe_stem(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z_-]+", "_", str(value))
    return value.strip("_") or "sample"


def _process_sample(sample, args):
    safety_max_step_xy = args.safety_max_step_xy
    if safety_max_step_xy is None and args.max_step_xy > 0.0:
        safety_max_step_xy = args.max_step_xy
    limits = TrajectorySafetyLimits(
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
        z_min=args.z_min,
        z_max=args.z_max,
        max_step_xy=safety_max_step_xy,
        max_step_z=args.safety_max_step_z,
        max_angle_step_rad=args.safety_max_angle_step_rad,
    )
    raw_report = validate_trajectory(sample, limits)
    processed = repair_sample_states(sample)
    if args.smooth_passes > 0:
        processed = smooth_sample(
            processed,
            passes=args.smooth_passes,
            strength=args.smooth_strength,
        )
    if args.max_step_xy > 0.0:
        processed = densify_sample(processed, max_step_xy=args.max_step_xy)
    if args.expand_pen_up:
        if args.lift_z is None:
            raise ValueError("--expand_pen_up requires --lift_z")
        processed = expand_pen_up_transitions(processed, clearance_z=args.lift_z)

    processed_report = validate_trajectory(processed, limits)
    if not processed_report["safe"] and args.fail_on_unsafe:
        raise RuntimeError(
            f"Unsafe sample {sample.meta.get('sample_id')}: "
            + ", ".join(processed_report["errors"])
        )
    return processed, raw_report, processed_report


def _write_sample_rows(writer: csv.DictWriter, sample) -> int:
    count = 0
    for point in sample.all_points():
        writer.writerow(
            {
                "character": sample.character or "",
                "sample_id": sample.meta.get("sample_id", ""),
                **point.as_dict(),
                "state": int(point.state),
                "contact_state": point.state.to_name(),
            }
        )
        count += 1
    return count


def _should_preview(index: int, args) -> bool:
    if index < args.preview_count:
        return True
    return args.preview_every > 0 and index % args.preview_every == 0


def run_batch(args: argparse.Namespace) -> Dict[str, Any]:
    samples = load_trajectory_csv(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / "previews"
    if args.preview_count > 0 or args.preview_every > 0:
        preview_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "trajectories_processed.csv"
    report_path = output_dir / "trajectory_reports.jsonl"
    summary: Dict[str, Any] = {
        "format": "robot_independent_trajectory_batch_v1",
        "input_csv": str(args.input_csv),
        "output_csv": str(csv_path),
        "sample_count": len(samples),
        "processed_samples": 0,
        "safe_samples": 0,
        "unsafe_samples": 0,
        "total_raw_points": 0,
        "total_processed_points": 0,
        "stroke_count": 0,
        "cross_stroke_segments_rendered": 0,
        "warnings": Counter(),
        "errors": Counter(),
        "parameters": {
            "smooth_passes": args.smooth_passes,
            "smooth_strength": args.smooth_strength,
            "max_step_xy": args.max_step_xy,
            "safety_max_step_xy": (
                args.safety_max_step_xy
                if args.safety_max_step_xy is not None
                else args.max_step_xy
            ),
            "preview_flip_y": not args.preview_y_up,
            "expand_pen_up": args.expand_pen_up,
        },
    }

    with csv_path.open("w", encoding="utf-8-sig", newline="") as csv_stream, report_path.open(
        "w", encoding="utf-8"
    ) as report_stream:
        writer = csv.DictWriter(csv_stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for index, sample in enumerate(samples):
            processed, raw_report, processed_report = _process_sample(sample, args)
            processed_points = _write_sample_rows(writer, processed)
            raw_points = int(raw_report["point_count"])
            report = {
                "index": index,
                "character": sample.character,
                "sample_id": sample.meta.get("sample_id"),
                "raw": raw_report,
                "processed": processed_report,
                "processed_point_count": processed_points,
            }
            report_stream.write(json.dumps(report, ensure_ascii=False) + "\n")

            summary["processed_samples"] += 1
            summary["safe_samples"] += int(processed_report["safe"])
            summary["unsafe_samples"] += int(not processed_report["safe"])
            summary["total_raw_points"] += raw_points
            summary["total_processed_points"] += processed_points
            summary["stroke_count"] += int(processed_report["stroke_count"])
            summary["cross_stroke_segments_rendered"] += int(
                processed_report["interstroke"]["cross_stroke_segments_rendered"]
            )
            summary["warnings"].update(processed_report["warnings"])
            summary["errors"].update(processed_report["errors"])

            if _should_preview(index, args):
                stem = f"{index:05d}_{_safe_stem(sample.character)}"
                flip_y = not args.preview_y_up
                render_trajectory_preview(
                    sample, preview_dir / f"{stem}_raw.png", flip_y=flip_y
                )
                render_trajectory_preview(
                    processed, preview_dir / f"{stem}_processed.png", flip_y=flip_y
                )
                render_trajectory_overlay(
                    sample,
                    processed,
                    preview_dir / f"{stem}_overlay.png",
                    flip_y=flip_y,
                )

            if args.report_every > 0 and (index + 1) % args.report_every == 0:
                print(
                    f"[BATCH] {index + 1}/{len(samples)} "
                    f"safe={summary['safe_samples']} "
                    f"unsafe={summary['unsafe_samples']}"
                )

    summary["warnings"] = dict(summary["warnings"])
    summary["errors"] = dict(summary["errors"])
    summary["safe_fraction"] = (
        summary["safe_samples"] / summary["processed_samples"]
        if summary["processed_samples"]
        else 0.0
    )
    (output_dir / "trajectory_batch_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_dir", default="outputs/trajectory_batch_processed")
    parser.add_argument("--smooth_passes", type=int, default=2)
    parser.add_argument("--smooth_strength", type=float, default=0.25)
    parser.add_argument("--max_step_xy", type=float, default=2.0)
    parser.add_argument("--lift_z", type=float, default=None)
    parser.add_argument("--expand_pen_up", action="store_true")
    parser.add_argument("--preview_count", type=int, default=24)
    parser.add_argument("--preview_every", type=int, default=0)
    parser.add_argument("--report_every", type=int, default=250)
    parser.add_argument("--preview_y_up", action="store_true")
    parser.add_argument("--x_min", type=float, default=None)
    parser.add_argument("--x_max", type=float, default=None)
    parser.add_argument("--y_min", type=float, default=None)
    parser.add_argument("--y_max", type=float, default=None)
    parser.add_argument("--z_min", type=float, default=None)
    parser.add_argument("--z_max", type=float, default=None)
    parser.add_argument("--safety_max_step_xy", type=float, default=None)
    parser.add_argument("--safety_max_step_z", type=float, default=None)
    parser.add_argument("--safety_max_angle_step_rad", type=float, default=3.141592653589793)
    parser.add_argument("--fail_on_unsafe", action="store_true")
    run_batch(parser.parse_args())


if __name__ == "__main__":
    main()
