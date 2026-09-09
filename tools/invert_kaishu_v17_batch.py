"""Run v17 multi-start pose inversion for a strictly 楷书 character library.

The v1 batch coordinator produces one local solution per character.  This
coordinator keeps the same audited target-selection/compatibility gate, then
calls :mod:`run_paper_v17_multisolution` in ``--real_target`` mode.  Each
compatible character therefore retains several feasible candidates, a Top-K
ranking, footprint/continuity/boundary metrics, and empirical cross-start
intervals.  Database images are supervision targets only: the initial pose is
never treated as physical ground truth.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.trajectory_dataset import load_trajectory_csv
from tools.invert_kaishu_trajectory_batch import (
    _write_json,
    build_kaishu_dataset,
    build_skeleton_snap_initial_pose,
    character_stem,
    choose_target,
    filter_target_items,
    group_target_items,
    load_override_target,
    load_target_exclusions,
    parse_target_overrides,
    save_target_image,
    select_trajectory_samples,
    target_compatibility_failures,
)
from tools.invert_paper_trajectory import save_pose_csv
from utils.trajectory_processing import repair_sample_states, write_trajectory_csv


def _make_initial_pose(sample: Any, output: Path, h_mm: float) -> None:
    sample = repair_sample_states(sample)
    points = sample.all_points()
    posture = np.tile(np.asarray([h_mm, 0.0, 0.0], dtype=np.float32), (len(points), 1))
    xy = np.asarray([[point.x, point.y] for point in points], dtype=np.float32)
    gamma = np.zeros(len(points), dtype=np.float32)
    prior = {"source": "command_line_default_prior", "confidence": "low_simulation"}
    save_pose_csv(
        sample,
        posture,
        output,
        "paper_declared_radian",
        {"H": dict(prior), "alpha": dict(prior), "beta": dict(prior), "gamma": dict(prior),
         "x": {"source": "trajectory_csv", "confidence": "prior"},
         "y": {"source": "trajectory_csv", "confidence": "prior"}},
        xy_source=xy,
        gamma=gamma,
        prototype="paper_v17_real_target_prior_v1",
    )


def _v17_command(args: argparse.Namespace, sample_id: str, target: Path,
                 input_csv: Path, prior_csv: Path, output: Path,
                 character: str) -> list[str]:
    command = [
        args.python, "-u", str(ROOT / "tools" / "run_paper_v17_multisolution.py"),
        "--real_target", "--trajectory_csv", str(input_csv),
        "--initial_pose_csv", str(prior_csv), "--target_image", str(target),
        "--bbsmg_ckpt", str(args.bbsmg_ckpt), "--character", character,
        "--sample_id", sample_id, "--output_dir", str(output),
        "--output_stem", "candidate", "--device", args.device,
        "--image_size", str(args.image_size), "--padding", str(args.padding),
        "--order", str(args.order), "--optimization_size", str(args.optimization_size),
        "--max_steps", str(args.max_steps), "--damping", str(args.damping),
        "--finite_difference_eps", str(args.finite_difference_eps),
        "--gamma_max_abs_deg", str(args.gamma_max_abs_deg),
        "--pixel_weight", str(args.pixel_weight),
        "--h_smoothness_weight", str(args.h_smoothness_weight),
        "--h_point_velocity_weight", str(args.h_point_velocity_weight),
        "--h_point_acceleration_weight", str(args.h_point_acceleration_weight),
        "--alpha_smoothness_weight", str(args.alpha_smoothness_weight),
        "--beta_smoothness_weight", str(args.beta_smoothness_weight),
        "--gamma_smoothness_weight", str(args.gamma_smoothness_weight),
        "--h_prior_weight", str(args.h_prior_weight),
        "--alpha_prior_weight", str(args.alpha_prior_weight),
        "--beta_prior_weight", str(args.beta_prior_weight),
        "--gamma_prior_weight", str(args.gamma_prior_weight),
        "--footprint_longitudinal_scale", str(args.footprint_longitudinal_scale),
        "--footprint_transverse_scale", str(args.footprint_transverse_scale),
        "--pixels_per_model_unit", str(args.pixels_per_model_unit),
        "--render_max_step_px", str(args.render_max_step_px),
        "--point_batch_size", str(args.point_batch_size),
        "--footprint_radius_px", str(args.footprint_radius_px),
        "--footprint_samples", str(args.footprint_samples),
        "--footprint_threshold", str(args.footprint_threshold),
        "--footprint_temperature", str(args.footprint_temperature),
        "--top_k", str(args.top_k), "--min_iou", str(args.min_iou),
        "--max_boundary_fraction", str(args.max_boundary_fraction),
        "--max_continuity_jump", str(args.max_continuity_jump),
        "--image_weight", str(args.image_weight),
        "--footprint_weight", str(args.footprint_weight),
        "--continuity_weight", str(args.continuity_weight),
        "--boundary_weight", str(args.boundary_weight),
        "--stability_weight", str(args.stability_weight),
        "--perturbation_scales", *[str(x) for x in args.perturbation_scales],
        "--copy_top_k",
    ]
    if args.optimize_xy:
        command.extend(
            [
                "--optimize_xy",
                "--xy_max_offset_px", str(args.xy_max_offset_px),
                "--xy_smoothness_weight", str(args.xy_smoothness_weight),
                "--xy_prior_weight", str(args.xy_prior_weight),
                "--xy_segment_length_weight", str(args.xy_segment_length_weight),
                "--xy_segment_direction_weight", str(args.xy_segment_direction_weight),
                "--xy_target_skeleton_weight", str(args.xy_target_skeleton_weight),
                "--xy_target_skeleton_max_distance_px", str(args.xy_target_skeleton_max_distance_px),
                "--xy_target_skeleton_threshold", str(args.xy_target_skeleton_threshold),
            ]
        )
    if args.resume_completed:
        command.append("--resume_completed")
    return command


def _copy_top1(summary_path: Path, destination: Path) -> str | None:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    candidates = data.get("candidates", [])
    if not candidates:
        return None
    source = Path(candidates[0].get("estimate_csv", ""))
    if not source.exists():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return str(destination)


def run(args: argparse.Namespace) -> dict[str, Any]:
    for name in ("trajectory_csv", "bbsmg_ckpt", "image_dir", "json_dir", "data_csv", "output_dir"):
        setattr(args, name, str(Path(getattr(args, name)).expanduser().resolve()))
    dataset = build_kaishu_dataset(args.image_dir, args.json_dir, args.data_csv, args.image_ext, args.chirography)
    target_items = group_target_items(dataset)
    target_items, removed = filter_target_items(target_items, load_target_exclusions(args.exclude_targets_json))
    overrides = {k: str(Path(v).expanduser().resolve()) for k, v in parse_target_overrides(args.target_override).items()}
    samples = load_trajectory_csv(args.trajectory_csv)
    selected, duplicate_counts = select_trajectory_samples(samples, args.trajectory_selection)
    characters = sorted(target_items)
    if args.include_characters:
        allowed = set(args.include_characters)
        characters = [c for c in characters if c in allowed]
    if args.shard_count > 1:
        characters = [c for i, c in enumerate(characters) if i % args.shard_count == args.shard_index]
    if args.max_characters > 0:
        characters = [c for c in characters if c in selected][:args.max_characters]
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    suffix = f"_shard_{args.shard_index}_of_{args.shard_count}" if args.shard_count > 1 else ""
    summary: dict[str, Any] = {
        "format": "kaishu_pose_trajectory_batch_v17",
        "simulation_only": True,
        "real_target_mode": True,
        "trajectory_csv": args.trajectory_csv,
        "bbsmg_checkpoint": args.bbsmg_ckpt,
        "style_filter": {"chirography": args.chirography, "data_csv": args.data_csv,
                          "image_dir": args.image_dir, "json_dir": args.json_dir,
                          "database_character_count": len(target_items),
                          "excluded_target_candidates": len(removed)},
        "selection": {"trajectory_selection": args.trajectory_selection, "target_selection": args.target_selection,
                       "model_support_width_px": args.model_support_width_px,
                       "min_target_coverage": args.min_target_coverage,
                       "min_model_support_dice": args.min_model_support_dice,
                       "target_support_ink_ratio_range": [args.min_target_support_ink_ratio, args.max_target_support_ink_ratio],
                       "target_overrides": overrides},
        "parameters": {"perturbation_scales": args.perturbation_scales, "top_k": args.top_k,
                       "pixel_weight": args.pixel_weight, "h_smoothness_weight": args.h_smoothness_weight,
                       "h_point_velocity_weight": args.h_point_velocity_weight,
                       "h_point_acceleration_weight": args.h_point_acceleration_weight},
        "shard": {"index": args.shard_index, "count": args.shard_count},
        "requested_characters": len(characters), "processed": 0, "completed": 0,
        "accepted": 0, "low_quality": 0, "failed": 0, "target_compatible": 0,
        "target_incompatible": 0, "skipped_existing": 0,
        "missing_trajectory": sorted(set(target_items) - set(selected)),
        "trajectory_without_kaishu_target": sorted(set(selected) - set(target_items)),
        "duplicate_trajectory_counts": duplicate_counts, "characters": [],
    }
    manifest_path = root / f"manifest{suffix}.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index, character in enumerate(characters):
            summary["processed"] += 1
            char_dir = root / character_stem(character)
            char_dir.mkdir(parents=True, exist_ok=True)
            sample = selected.get(character)
            record: dict[str, Any] = {"index": index, "character": character, "output_dir": str(char_dir), "status": "failed"}
            if sample is None:
                record.update({"status": "missing_trajectory", "error": "no trajectory sample"})
                summary["failed"] += 1
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                summary["characters"].append(record)
                continue
            v17_dir = char_dir / "v17"
            candidate_summary = v17_dir / "candidate_summary.json"
            if args.resume_completed and candidate_summary.exists():
                record.update({"status": "skipped_existing", "candidate_summary": str(candidate_summary)})
                summary["skipped_existing"] += 1
                summary["completed"] += 1
                summary["characters"].append(record)
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue
            try:
                if character in overrides:
                    target_canvas, target_meta = load_override_target(overrides[character], args.image_size, args.padding)
                else:
                    target_canvas, target_meta = choose_target(dataset, target_items[character], sample,
                                                               args.image_size, args.padding, args.target_selection,
                                                               args.target_match_tolerance_px, args.target_match_threshold,
                                                               args.model_support_width_px)
                target_path = char_dir / "target.png"
                save_target_image(target_canvas, target_path)
                _write_json(char_dir / "target_source.json", target_meta)
                failures = target_compatibility_failures(target_meta, args)
                record["target_compatibility"] = {"accepted": not failures, "failures": failures}
                record["target_image"] = str(target_path)
                if failures:
                    record.update({"status": "target_incompatible", "error": ",".join(failures)})
                    summary["target_incompatible"] += 1
                    summary["failed"] += 1
                    manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                    summary["characters"].append(record)
                    continue
                summary["target_compatible"] += 1
                input_csv = char_dir / "input_trajectory.csv"
                if not input_csv.exists():
                    write_trajectory_csv(sample, input_csv)
                prior_csv = char_dir / "initial_pose.csv"
                if not prior_csv.exists():
                    if args.optimize_xy:
                        snap_report = build_skeleton_snap_initial_pose(
                            sample, target_canvas, prior_csv, args.image_size,
                            args.padding, args.snap_threshold, args.snap_max_px,
                            args.snap_blend, args.snap_smooth_sigma,
                            args.initial_h_mm, 0.0, 0.0, 0.0,
                        )
                        _write_json(char_dir / "skeleton_snap_report.json", snap_report)
                    else:
                        _make_initial_pose(sample, prior_csv, args.initial_h_mm)
                command = _v17_command(args, str(sample.meta.get("sample_id", "")), target_path,
                                       input_csv, prior_csv, v17_dir, character)
                record["command"] = " ".join(command)
                log_path = char_dir / "v17_batch.log"
                if not candidate_summary.exists() or not args.resume_completed:
                    with log_path.open("w", encoding="utf-8") as log:
                        completed = subprocess.run(command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                                                   check=False, timeout=args.timeout_seconds if args.timeout_seconds > 0 else None)
                    if completed.returncode != 0 or not candidate_summary.exists():
                        raise RuntimeError(f"v17 failed returncode={completed.returncode}; see {log_path}")
                top1 = _copy_top1(candidate_summary, char_dir / "pose_top1.csv")
                record.update({"status": "completed", "candidate_summary": str(candidate_summary),
                               "pose_top1": top1, "trajectory_input": str(input_csv), "prior_pose": str(prior_csv)})
                data = json.loads(candidate_summary.read_text(encoding="utf-8"))
                candidates = data.get("candidates", [])
                record["top_candidate"] = candidates[0] if candidates else None
                summary["completed"] += 1
                if candidates and bool(candidates[0].get("export_eligible")):
                    summary["accepted"] += 1
                else:
                    summary["low_quality"] += 1
            except Exception as exc:
                record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}", "log": str(char_dir / "v17_batch.log")})
                summary["failed"] += 1
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            summary["characters"].append(record)
            print(f"[V17-BATCH] {index + 1}/{len(characters)} {character} status={record['status']}", flush=True)
    summary["completed_fraction"] = summary["completed"] / summary["requested_characters"] if summary["requested_characters"] else 0.0
    _write_json(root / f"batch_summary{suffix}.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory_csv", required=True)
    parser.add_argument("--bbsmg_ckpt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_dir", default="data/raw/images")
    parser.add_argument("--json_dir", default="data/raw/json_files")
    parser.add_argument("--data_csv", default="data/raw/data.csv")
    parser.add_argument("--image_ext", default=".jpg")
    parser.add_argument("--chirography", default="楷")
    parser.add_argument("--exclude_targets_json", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--padding", type=int, default=16)
    parser.add_argument("--trajectory_selection", choices=("longest", "first"), default="longest")
    parser.add_argument("--target_selection", choices=("first", "best_trajectory_coverage", "best_model_support"), default="best_model_support")
    parser.add_argument("--target_override", action="append", default=[])
    parser.add_argument("--target_match_tolerance_px", type=int, default=5)
    parser.add_argument("--target_match_threshold", type=float, default=0.35)
    parser.add_argument("--model_support_width_px", type=float, default=7.0)
    parser.add_argument("--min_target_coverage", type=float, default=0.75)
    parser.add_argument("--min_model_support_dice", type=float, default=0.30)
    parser.add_argument("--min_target_support_ink_ratio", type=float, default=0.45)
    parser.add_argument("--max_target_support_ink_ratio", type=float, default=1.65)
    parser.add_argument("--include_characters", nargs="*", default=None)
    parser.add_argument("--max_characters", type=int, default=0)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--resume_completed", action="store_true")
    parser.add_argument("--timeout_seconds", type=int, default=0)
    parser.add_argument("--perturbation_scales", type=float, nargs="+", default=[-1.0, 0.0, 1.0])
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--optimization_size", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=6)
    parser.add_argument("--damping", type=float, default=0.1)
    parser.add_argument("--finite_difference_eps", type=float, default=0.01)
    parser.add_argument("--gamma_max_abs_deg", type=float, default=180.0)
    parser.add_argument("--pixel_weight", type=float, default=12.0)
    parser.add_argument("--h_smoothness_weight", type=float, default=0.2)
    parser.add_argument("--h_point_velocity_weight", type=float, default=5.0)
    parser.add_argument("--h_point_acceleration_weight", type=float, default=10.0)
    parser.add_argument("--initial_h_mm", type=float, default=15.5)
    parser.add_argument("--alpha_smoothness_weight", type=float, default=0.2)
    parser.add_argument("--beta_smoothness_weight", type=float, default=0.2)
    parser.add_argument("--gamma_smoothness_weight", type=float, default=0.2)
    parser.add_argument("--h_prior_weight", type=float, default=0.001)
    parser.add_argument("--alpha_prior_weight", type=float, default=0.05)
    parser.add_argument("--beta_prior_weight", type=float, default=0.05)
    parser.add_argument("--gamma_prior_weight", type=float, default=0.05)
    parser.add_argument("--footprint_longitudinal_scale", type=float, default=0.22)
    parser.add_argument("--footprint_transverse_scale", type=float, default=0.258)
    parser.add_argument("--pixels_per_model_unit", type=float, default=20.0)
    parser.add_argument("--render_max_step_px", type=float, default=2.0)
    parser.add_argument("--point_batch_size", type=int, default=64)
    parser.add_argument("--footprint_radius_px", type=float, default=10.0)
    parser.add_argument("--footprint_samples", type=int, default=33)
    parser.add_argument("--footprint_threshold", type=float, default=0.35)
    parser.add_argument("--footprint_temperature", type=float, default=0.08)
    parser.add_argument("--image_weight", type=float, default=0.45)
    parser.add_argument("--footprint_weight", type=float, default=0.25)
    parser.add_argument("--continuity_weight", type=float, default=0.15)
    parser.add_argument("--boundary_weight", type=float, default=0.10)
    parser.add_argument("--stability_weight", type=float, default=0.05)
    parser.add_argument("--min_iou", type=float, default=0.95)
    parser.add_argument("--max_boundary_fraction", type=float, default=0.05)
    parser.add_argument("--max_continuity_jump", type=float, default=0.25)
    parser.add_argument("--optimize_xy", action="store_true")
    parser.add_argument("--xy_max_offset_px", type=float, default=3.0)
    parser.add_argument("--xy_smoothness_weight", type=float, default=1.0)
    parser.add_argument("--xy_prior_weight", type=float, default=0.5)
    parser.add_argument("--xy_segment_length_weight", type=float, default=0.10)
    parser.add_argument("--xy_segment_direction_weight", type=float, default=0.10)
    parser.add_argument("--xy_target_skeleton_weight", type=float, default=0.20)
    parser.add_argument("--xy_target_skeleton_max_distance_px", type=float, default=8.0)
    parser.add_argument("--xy_target_skeleton_threshold", type=float, default=0.35)
    parser.add_argument("--snap_threshold", type=float, default=0.35)
    parser.add_argument("--snap_max_px", type=float, default=8.0)
    parser.add_argument("--snap_blend", type=float, default=0.75)
    parser.add_argument("--snap_smooth_sigma", type=float, default=0.75)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
