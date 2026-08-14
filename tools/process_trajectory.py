"""Repair, smooth, validate and visualize one character trajectory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    write_trajectory_csv,
)


def pick_sample(samples, sample_id=None, character=None, index=0):
    if sample_id is not None:
        for sample in samples:
            if str(sample.meta.get("sample_id")) == str(sample_id):
                return sample
    if character is not None:
        candidates = [sample for sample in samples if sample.character == character]
        if candidates:
            return candidates[min(index, len(candidates) - 1)]
    if not samples:
        raise ValueError("No trajectory samples found")
    return samples[min(index, len(samples) - 1)]


def main(args: argparse.Namespace) -> None:
    sample = pick_sample(
        load_trajectory_csv(args.input_csv),
        sample_id=args.sample_id,
        character=args.character,
        index=args.index,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_report = validate_trajectory(sample)
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
        processed = expand_pen_up_transitions(
            processed,
            clearance_z=args.lift_z,
        )
    limits = TrajectorySafetyLimits(
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
        z_min=args.z_min,
        z_max=args.z_max,
        max_step_xy=args.safety_max_step_xy,
        max_step_z=args.safety_max_step_z,
        max_angle_step_rad=args.safety_max_angle_step_rad,
    )
    processed_report = validate_trajectory(processed, limits)
    if not processed_report["safe"] and args.fail_on_unsafe:
        raise RuntimeError(
            "Processed trajectory failed safety checks: "
            + ", ".join(processed_report["errors"])
        )
    write_trajectory_csv(processed, output_dir / "trajectory_processed.csv")
    render_trajectory_preview(sample, output_dir / "trajectory_raw_preview.png")
    render_trajectory_preview(processed, output_dir / "trajectory_processed_preview.png")
    render_trajectory_overlay(sample, processed, output_dir / "trajectory_overlay.png")
    report = {
        "format": "robot_independent_trajectory_processing_v1",
        "input_csv": str(args.input_csv),
        "character": processed.character,
        "sample_id": processed.meta.get("sample_id"),
        "raw": raw_report,
        "processed": processed_report,
        "processing": {
            "state_repair": True,
            "smooth_passes": args.smooth_passes,
            "smooth_strength": args.smooth_strength,
            "max_step_xy": args.max_step_xy,
            "lift_z": args.lift_z,
            "expand_pen_up": args.expand_pen_up,
        },
        "rendering_contract": {
            "cross_stroke_segments_rendered": 0,
            "stroke_boundary": "hard_break",
            "contact_states": {"0": "down", "1": "move", "2": "up", "3": "transition"},
        },
    }
    (output_dir / "trajectory_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(processed_report, ensure_ascii=False))
    print(f"[DONE] processed trajectory: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_dir", default="outputs/trajectory_processed")
    parser.add_argument("--sample_id", default=None)
    parser.add_argument("--character", default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--smooth_passes", type=int, default=2)
    parser.add_argument("--smooth_strength", type=float, default=0.25)
    parser.add_argument("--max_step_xy", type=float, default=0.0)
    parser.add_argument("--lift_z", type=float, default=None)
    parser.add_argument(
        "--expand_pen_up",
        action="store_true",
        help="add explicit clearance/travel points for a robot adapter",
    )
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
    main(parser.parse_args())
