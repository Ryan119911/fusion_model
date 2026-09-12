"""Screen a model pose library and register only robot-validated trajectories.

The source directory is immutable model output.  This tool creates a separate,
versioned ROS registry with three machine-readable views:

* ``screening_manifest.jsonl`` records every discovered character and all
  rejection reasons;
* ``robot_validation_queue.jsonl`` contains model-qualified trajectories that
  still need IK, collision, singularity, and joint-limit validation;
* ``manifest.jsonl`` contains only trajectories that passed both gates.  Each
  registered character is copied to ``char_uXXXX/pose_refined.csv`` so existing
  ROS consumers do not have to understand a model-specific filename.

The default policy is intentionally strict.  A v17 candidate must pass its own
export gate, the explicit thresholds below, cross-start stability, and per-field
confidence.  A robot report is bound to the exact pose by SHA-256, preventing a
report from an older model iteration from approving a newer trajectory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REQUIRED_POSE_COLUMNS = {
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
    "z_unit",
    "angle_unit",
    "pose_frame",
}
NUMERIC_POSE_COLUMNS = ("x", "y", "z", "alpha", "beta", "gamma")
CONFIDENCE_FIELDS = ("z", "alpha", "beta", "gamma")
SUMMARY_CONFIDENCE_KEYS = {"z": "H", "alpha": "alpha", "beta": "beta", "gamma": "gamma"}
CONFIDENCE_ORDER = {"low": 0, "medium_simulation": 1, "high_simulation": 2}


class SourceIncompleteError(RuntimeError):
    """Raised when formal registration is requested before every shard ends."""


@dataclass(frozen=True)
class SourceCompletion:
    complete: bool
    expected_shards: int
    found_shards: tuple[int, ...]
    reasons: tuple[str, ...]
    summaries: tuple[str, ...]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def character_from_directory(path: Path) -> str | None:
    stem = path.name
    if not stem.startswith("char_u"):
        return None
    try:
        return chr(int(stem[6:], 16))
    except (ValueError, OverflowError):
        return None


def inspect_source_completion(root: Path, expected_shards: int) -> SourceCompletion:
    shard_paths = sorted(root.glob("batch_summary_shard_*_of_*.json"))
    single = root / "batch_summary.json"
    reasons: list[str] = []
    found: dict[int, dict[str, Any]] = {}
    declared_counts: set[int] = set()

    for path in shard_paths:
        try:
            data = _read_json(path)
            shard = data.get("shard", {})
            index = int(shard.get("index"))
            count = int(shard.get("count"))
            found[index] = data
            declared_counts.add(count)
            requested = int(data.get("requested_characters", -1))
            processed = int(data.get("processed", -2))
            if requested < 0 or processed != requested:
                reasons.append(f"shard_{index}_not_fully_processed:{processed}/{requested}")
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            reasons.append(f"invalid_shard_summary:{path.name}:{type(exc).__name__}")

    if shard_paths:
        inferred = max(declared_counts) if len(declared_counts) == 1 else 0
        required = expected_shards or inferred
        if len(declared_counts) != 1:
            reasons.append("inconsistent_declared_shard_count")
        if required <= 0:
            reasons.append("expected_shard_count_unknown")
        else:
            missing = sorted(set(range(required)) - set(found))
            if missing:
                reasons.append("missing_shards:" + ",".join(map(str, missing)))
            extras = sorted(set(found) - set(range(required)))
            if extras:
                reasons.append("unexpected_shards:" + ",".join(map(str, extras)))
        return SourceCompletion(
            complete=not reasons,
            expected_shards=required,
            found_shards=tuple(sorted(found)),
            reasons=tuple(reasons),
            summaries=tuple(str(path.resolve()) for path in shard_paths),
        )

    if single.exists():
        try:
            data = _read_json(single)
            requested = int(data.get("requested_characters", -1))
            processed = int(data.get("processed", -2))
            if requested < 0 or processed != requested:
                reasons.append(f"single_batch_not_fully_processed:{processed}/{requested}")
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            reasons.append(f"invalid_batch_summary:{type(exc).__name__}")
        if expected_shards not in (0, 1):
            reasons.append(f"expected_{expected_shards}_shards_but_found_single_batch")
        return SourceCompletion(
            complete=not reasons,
            expected_shards=1,
            found_shards=(0,),
            reasons=tuple(reasons),
            summaries=(str(single.resolve()),),
        )

    reasons.append("batch_summary_missing")
    return SourceCompletion(False, expected_shards, (), tuple(reasons), ())


def _first_candidate(summary: dict[str, Any]) -> dict[str, Any] | None:
    candidates = summary.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return None
    ranked = sorted(
        (item for item in candidates if isinstance(item, dict)),
        key=lambda item: int(item.get("rank", 10**9)),
    )
    return ranked[0] if ranked else None


def validate_pose_csv(path: Path, expected_character: str) -> tuple[dict[str, Any], list[str]]:
    reasons: list[str] = []
    if not path.exists():
        return {}, ["pose_top1_missing"]
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            columns = set(reader.fieldnames or [])
            missing = sorted(REQUIRED_POSE_COLUMNS - columns)
            if missing:
                return {}, ["pose_columns_missing:" + ",".join(missing)]
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        return {}, [f"pose_csv_unreadable:{type(exc).__name__}"]

    if not rows:
        return {}, ["pose_csv_empty"]
    keys: set[tuple[int, int]] = set()
    z_values: list[float] = []
    angle_values: dict[str, list[float]] = {name: [] for name in ("alpha", "beta", "gamma")}
    sample_ids: set[str] = set()
    prototypes: set[str] = set()
    for row_index, row in enumerate(rows, start=2):
        if row.get("character") != expected_character:
            reasons.append(f"pose_character_mismatch:row_{row_index}")
        sample_ids.add(row.get("sample_id", ""))
        if row.get("prototype"):
            prototypes.add(str(row["prototype"]))
        try:
            key = (int(row["stroke_id"]), int(row["point_id"]))
            if key in keys:
                reasons.append(f"duplicate_stroke_point:{key[0]}:{key[1]}")
            keys.add(key)
            values = {name: float(row[name]) for name in NUMERIC_POSE_COLUMNS}
            if not all(math.isfinite(value) for value in values.values()):
                reasons.append(f"pose_nonfinite:row_{row_index}")
            z_values.append(values["z"])
            for name in angle_values:
                angle_values[name].append(values[name])
        except (KeyError, TypeError, ValueError):
            reasons.append(f"pose_numeric_parse_failed:row_{row_index}")
        if row.get("z_unit") != "mm":
            reasons.append(f"unexpected_z_unit:{row.get('z_unit')}")
        if row.get("angle_unit") != "rad":
            reasons.append(f"unexpected_angle_unit:{row.get('angle_unit')}")
        if row.get("pose_frame") != "paper_model":
            reasons.append(f"unexpected_pose_frame:{row.get('pose_frame')}")

    metadata = {
        "row_count": len(rows),
        "stroke_count": len({key[0] for key in keys}),
        "sample_ids": sorted(sample_ids),
        "prototypes": sorted(prototypes),
        "z_range_mm": [min(z_values), max(z_values)] if z_values else None,
        "alpha_range_rad": [min(angle_values["alpha"]), max(angle_values["alpha"])] if angle_values["alpha"] else None,
        "beta_range_rad": [min(angle_values["beta"]), max(angle_values["beta"])] if angle_values["beta"] else None,
        "gamma_range_rad": [min(angle_values["gamma"]), max(angle_values["gamma"])] if angle_values["gamma"] else None,
    }
    return metadata, sorted(set(reasons))


def screen_model_candidate(
    summary: dict[str, Any],
    pose_path: Path,
    expected_character: str,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[str]]:
    reasons: list[str] = []
    if summary.get("format") != args.summary_format:
        reasons.append(f"unsupported_summary_format:{summary.get('format')}")
    if summary.get("real_target") is not True:
        reasons.append("not_real_target_inversion")
    if summary.get("simulation_only") is not True:
        reasons.append("simulation_provenance_missing")
    if int(summary.get("candidate_count", 0)) < args.min_candidate_count:
        reasons.append("candidate_count_below_threshold")

    candidate = _first_candidate(summary)
    if candidate is None:
        return {"candidate": None}, reasons + ["top_candidate_missing"]
    if int(candidate.get("rank", -1)) != 1:
        reasons.append("top_candidate_rank_is_not_one")
    if candidate.get("export_eligible") is not True:
        reasons.append("candidate_export_ineligible")
        reasons.extend(f"candidate:{reason}" for reason in candidate.get("withheld_reasons", []))

    image_metrics = candidate.get("image_metrics", {})
    continuity = candidate.get("continuity_metrics", {})
    iou = float(image_metrics.get("iou_at_0.5", -math.inf))
    boundary = float(candidate.get("maximum_boundary_fraction", math.inf))
    continuity_jump = float(continuity.get("second_difference_normalized_max", math.inf))
    if not math.isfinite(iou) or iou < args.min_iou:
        reasons.append("image_iou_below_registry_threshold")
    if not math.isfinite(boundary) or boundary > args.max_boundary_fraction:
        reasons.append("boundary_fraction_above_registry_threshold")
    if not math.isfinite(continuity_jump) or continuity_jump > args.max_continuity_jump:
        reasons.append("continuity_jump_above_registry_threshold")

    stability = summary.get("field_stability", {})
    confidence = candidate.get("field_confidence", {})
    confidence_details: dict[str, Any] = {}
    for field in args.required_confidence_fields:
        stability_key = "z" if field == "z" else field
        stability_item = stability.get(stability_key, {})
        stability_value = float(stability_item.get("normalized_cross_start_std_rmse", math.inf))
        if not math.isfinite(stability_value) or stability_value > args.max_cross_start_std:
            reasons.append(f"field_unstable:{field}")
        summary_key = SUMMARY_CONFIDENCE_KEYS[field]
        confidence_item = confidence.get(summary_key, {})
        level = str(confidence_item.get("level", "missing"))
        confidence_details[field] = confidence_item
        if CONFIDENCE_ORDER.get(level, -1) < CONFIDENCE_ORDER[args.min_field_confidence]:
            reasons.append(f"field_confidence_below_threshold:{field}:{level}")

    pose_metadata, pose_reasons = validate_pose_csv(pose_path, expected_character)
    reasons.extend(pose_reasons)
    if pose_metadata.get("z_range_mm"):
        z_min, z_max = pose_metadata["z_range_mm"]
        if z_min < args.min_z_mm - 1e-6 or z_max > args.max_z_mm + 1e-6:
            reasons.append("z_outside_physical_range")

    estimate_value = str(candidate.get("estimate_csv", "")).strip()
    candidate_path = Path(estimate_value) if estimate_value else None
    if candidate_path is None or not candidate_path.is_file():
        reasons.append("rank1_estimate_missing")
    elif pose_path.exists():
        if sha256_file(candidate_path) != sha256_file(pose_path):
            reasons.append("pose_top1_does_not_match_rank1_candidate")

    details = {
        "candidate": {
            "rank": candidate.get("rank"),
            "label": candidate.get("label"),
            "score": candidate.get("score", {}).get("total"),
            "export_eligible": candidate.get("export_eligible"),
            "estimate_csv": candidate.get("estimate_csv"),
        },
        "metrics": {
            "iou_at_0.5": iou,
            "maximum_boundary_fraction": boundary,
            "second_difference_normalized_max": continuity_jump,
            "footprint_combined_normalized_rmse": candidate.get("footprint_metrics", {}).get("combined_normalized_rmse"),
            "cross_start_stability": stability,
            "field_confidence": confidence_details,
        },
        "pose": pose_metadata,
    }
    return details, sorted(set(reasons))


def validate_robot_report(
    report_path: Path | None,
    pose_sha256: str,
    robot_model: str,
) -> tuple[str, dict[str, Any], list[str]]:
    if report_path is None or not report_path.exists():
        return "awaiting", {}, ["robot_validation_report_missing"]
    try:
        report = _read_json(report_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return "rejected", {}, [f"robot_validation_report_invalid:{type(exc).__name__}"]

    reasons: list[str] = []
    if str(report.get("robot_model", "")).lower() != robot_model.lower():
        reasons.append("robot_model_mismatch")
    bound_hash = report.get("trajectory_sha256")
    if bound_hash != pose_sha256:
        reasons.append("robot_report_trajectory_hash_mismatch")
    checks = {
        "ik_passed": report.get("ik_passed"),
        "collision_free": report.get("collision_free"),
        "singularity_free": report.get("singularity_free"),
        "joint_limits_passed": report.get("joint_limits_passed"),
        "trajectory_continuity_passed": report.get("trajectory_continuity_passed"),
    }
    for name, value in checks.items():
        if value is not True:
            reasons.append(f"robot_check_failed_or_missing:{name}")
    status = "passed" if not reasons else "rejected"
    return status, {"path": str(report_path.resolve()), "checks": checks, "report": report}, sorted(set(reasons))


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if source_root == output_root:
        raise ValueError("output_root must differ from immutable model input_root")
    if not source_root.exists():
        raise FileNotFoundError(source_root)
    completion = inspect_source_completion(source_root, args.expected_shards)
    if args.mode == "register" and not completion.complete:
        output_root.mkdir(parents=True, exist_ok=True)
        summary = {
            "format": "ros_pose_registry_summary_v1",
            "registry_ready": False,
            "source_root": str(source_root),
            "output_root": str(output_root),
            "source_completion": completion.__dict__,
            "status": "source_incomplete",
        }
        _write_json(output_root / "registry_summary.json", summary)
        raise SourceIncompleteError("source model run is incomplete: " + ";".join(completion.reasons))

    output_root.mkdir(parents=True, exist_ok=True)
    screening_rows: list[dict[str, Any]] = []
    registered_rows: list[dict[str, Any]] = []
    queue_rows: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    character_dirs = sorted(path for path in source_root.glob("char_u*") if path.is_dir())

    for index, char_dir in enumerate(character_dirs):
        character = character_from_directory(char_dir)
        if character is None:
            continue
        summary_path = char_dir / args.candidate_summary_relpath
        pose_path = char_dir / args.pose_top1_name
        target_path = char_dir / args.target_name
        record: dict[str, Any] = {
            "index": index,
            "character": character,
            "source_character_dir": str(char_dir.resolve()),
            "source_candidate_summary": str(summary_path.resolve()),
            "source_pose_top1": str(pose_path.resolve()),
            "source_target_image": str(target_path.resolve()),
            "status": "pending",
        }
        if not summary_path.exists():
            record["reasons"] = ["candidate_summary_missing"]
            reason_counts.update(record["reasons"])
            screening_rows.append(record)
            continue
        try:
            source_summary = _read_json(summary_path)
            details, model_reasons = screen_model_candidate(source_summary, pose_path, character, args)
            if not target_path.is_file():
                model_reasons.append("target_image_missing")
            model_reasons = sorted(set(model_reasons))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            details = {}
            model_reasons = [f"candidate_summary_invalid:{type(exc).__name__}"]
        record.update(details)
        record["model_screen"] = {"passed": not model_reasons, "reasons": model_reasons}
        if model_reasons:
            record["status"] = "model_rejected"
            record["reasons"] = model_reasons
            reason_counts.update(model_reasons)
            screening_rows.append(record)
            continue

        pose_hash = sha256_file(pose_path)
        robot_report_path = None
        if args.robot_validation_root:
            robot_report_path = (
                Path(args.robot_validation_root).expanduser().resolve()
                / char_dir.name
                / args.robot_report_name
            )
        robot_status, robot_details, robot_reasons = validate_robot_report(
            robot_report_path, pose_hash, args.robot_model
        )
        record["pose_sha256"] = pose_hash
        record["robot_validation"] = {
            "status": robot_status,
            "robot_model": args.robot_model,
            **robot_details,
            "reasons": robot_reasons,
        }
        if robot_status != "passed":
            record["status"] = "awaiting_robot_validation" if robot_status == "awaiting" else "robot_rejected"
            record["reasons"] = robot_reasons
            reason_counts.update(robot_reasons)
            screening_rows.append(record)
            if robot_status == "awaiting":
                queue_rows.append(
                    {
                        "character": character,
                        "source_pose": str(pose_path.resolve()),
                        "trajectory_sha256": pose_hash,
                        "target_image": str(target_path.resolve()),
                        "robot_model": args.robot_model,
                        "required_report": str(robot_report_path.resolve()) if robot_report_path else None,
                        "required_checks": [
                            "ik_passed",
                            "collision_free",
                            "singularity_free",
                            "joint_limits_passed",
                            "trajectory_continuity_passed",
                        ],
                    }
                )
            continue

        if args.mode == "audit":
            record.update(
                {
                    "status": "robot_validated",
                    "registration_status": "audit_only",
                    "reasons": [],
                }
            )
            screening_rows.append(record)
            continue

        destination = output_root / char_dir.name
        destination.mkdir(parents=True, exist_ok=True)
        registered_pose = destination / "pose_refined.csv"
        shutil.copy2(pose_path, registered_pose)
        if target_path.exists():
            shutil.copy2(target_path, destination / "target.png")
        shutil.copy2(summary_path, destination / "candidate_summary.json")
        _write_json(
            destination / "registration.json",
            {
                "format": "ros_pose_registration_v1",
                "character": character,
                "source_pose": str(pose_path.resolve()),
                "source_pose_sha256": pose_hash,
                "registered_pose": str(registered_pose.resolve()),
                "robot_validation": record["robot_validation"],
                "model_screen": record["model_screen"],
                "metrics": record.get("metrics", {}),
            },
        )
        record.update(
            {
                "status": "registered",
                "registration_status": "registered",
                "pose_csv": str(registered_pose.resolve()),
                "registered_pose": str(registered_pose.resolve()),
                "output_dir": str(destination.resolve()),
                "target_image": str((destination / "target.png").resolve()) if target_path.exists() else None,
                "reasons": [],
            }
        )
        screening_rows.append(record)
        # Existing ROS consumers use status=completed and pose_refined.csv.
        registered_rows.append({**record, "status": "completed"})

    status_counts = Counter(row["status"] for row in screening_rows)
    formal_ready = completion.complete and args.mode == "register"
    summary = {
        "format": "ros_pose_registry_summary_v1",
        "registry_ready": formal_ready,
        "simulation_only": True,
        "source_root": str(source_root),
        "output_root": str(output_root),
        "robot_model": args.robot_model,
        "source_completion": completion.__dict__,
        "policy": {
            "candidate": "rank_1_only",
            "summary_format": args.summary_format,
            "min_candidate_count": args.min_candidate_count,
            "min_iou": args.min_iou,
            "max_boundary_fraction": args.max_boundary_fraction,
            "max_continuity_jump": args.max_continuity_jump,
            "max_cross_start_std": args.max_cross_start_std,
            "required_confidence_fields": args.required_confidence_fields,
            "min_field_confidence": args.min_field_confidence,
            "z_range_mm": [args.min_z_mm, args.max_z_mm],
            "robot_report_required": True,
        },
        "discovered_characters": len(character_dirs),
        "screened_characters": len(screening_rows),
        "registered_characters": len(registered_rows),
        "status_counts": dict(sorted(status_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "files": {
            "ros_manifest": str((output_root / "manifest.jsonl").resolve()),
            "screening_manifest": str((output_root / "screening_manifest.jsonl").resolve()),
            "robot_validation_queue": str((output_root / "robot_validation_queue.jsonl").resolve()),
        },
    }
    _write_jsonl(output_root / "screening_manifest.jsonl", screening_rows)
    _write_jsonl(output_root / "robot_validation_queue.jsonl", queue_rows)
    _write_jsonl(output_root / "manifest.jsonl", registered_rows if formal_ready else [])
    _write_json(output_root / "registry_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--mode", choices=("audit", "register"), default="register")
    parser.add_argument("--expected_shards", type=int, default=0)
    parser.add_argument("--candidate_summary_relpath", default="v17/candidate_summary.json")
    parser.add_argument("--summary_format", default="paper_pose_multisolution_v17")
    parser.add_argument("--pose_top1_name", default="pose_top1.csv")
    parser.add_argument("--target_name", default="target.png")
    parser.add_argument("--min_candidate_count", type=int, default=3)
    parser.add_argument("--min_iou", type=float, default=0.95)
    parser.add_argument("--max_boundary_fraction", type=float, default=0.05)
    parser.add_argument("--max_continuity_jump", type=float, default=0.25)
    parser.add_argument("--max_cross_start_std", type=float, default=0.02)
    parser.add_argument(
        "--required_confidence_fields",
        nargs="+",
        choices=CONFIDENCE_FIELDS,
        default=list(CONFIDENCE_FIELDS),
    )
    parser.add_argument(
        "--min_field_confidence",
        choices=tuple(CONFIDENCE_ORDER),
        default="medium_simulation",
    )
    parser.add_argument("--min_z_mm", type=float, default=11.0)
    parser.add_argument("--max_z_mm", type=float, default=20.0)
    parser.add_argument("--robot_model", default="ur10")
    parser.add_argument("--robot_validation_root", default=None)
    parser.add_argument("--robot_report_name", default="robot_validation_report.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.expected_shards < 0:
        raise ValueError("expected_shards must be non-negative")
    if args.min_z_mm >= args.max_z_mm:
        raise ValueError("min_z_mm must be smaller than max_z_mm")
    run(args)


if __name__ == "__main__":
    main()
