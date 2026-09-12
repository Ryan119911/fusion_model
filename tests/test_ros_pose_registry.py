import argparse
import csv
import hashlib
import json
from pathlib import Path

import pytest

from tools.register_pose_library_for_ros import SourceIncompleteError, run


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _pose(path: Path, character: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "character", "sample_id", "stroke_id", "point_id", "x", "y",
        "z", "alpha", "beta", "gamma", "state", "z_unit",
        "angle_unit", "pose_frame", "prototype",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            [
                {
                    "character": character, "sample_id": f"{character}_test",
                    "stroke_id": 0, "point_id": 0, "x": 10, "y": 20,
                    "z": 15, "alpha": 0.05, "beta": 0.02, "gamma": 0.1,
                    "state": 0, "z_unit": "mm", "angle_unit": "rad",
                    "pose_frame": "paper_model", "prototype": "test_v17",
                },
                {
                    "character": character, "sample_id": f"{character}_test",
                    "stroke_id": 0, "point_id": 1, "x": 30, "y": 40,
                    "z": 15.2, "alpha": 0.05, "beta": 0.02, "gamma": 0.1,
                    "state": 1, "z_unit": "mm", "angle_unit": "rad",
                    "pose_frame": "paper_model", "prototype": "test_v17",
                },
            ]
        )


def _summary(estimate: Path, *, iou: float = 0.97, alpha_level: str = "medium_simulation") -> dict:
    stability = {
        field: {
            "normalized_cross_start_std_rmse": 0.01,
            "passed_v17_stability": True,
        }
        for field in ("z", "alpha", "beta", "gamma")
    }
    confidence = {
        field: {"level": alpha_level if field == "alpha" else "medium_simulation", "reasons": []}
        for field in ("H", "alpha", "beta", "gamma")
    }
    return {
        "format": "paper_pose_multisolution_v17",
        "simulation_only": True,
        "real_target": True,
        "candidate_count": 3,
        "field_stability": stability,
        "candidates": [
            {
                "rank": 1,
                "label": "p0",
                "estimate_csv": str(estimate),
                "export_eligible": iou >= 0.95,
                "withheld_reasons": [] if iou >= 0.95 else ["image_iou_below_threshold"],
                "image_metrics": {"iou_at_0.5": iou},
                "maximum_boundary_fraction": 0.0,
                "continuity_metrics": {"second_difference_normalized_max": 0.01},
                "footprint_metrics": {"combined_normalized_rmse": 0.03},
                "field_confidence": confidence,
                "score": {"total": 0.9},
            }
        ],
    }


def _args(source: Path, output: Path, robot: Path | None, mode: str = "register") -> argparse.Namespace:
    return argparse.Namespace(
        input_root=str(source), output_root=str(output), mode=mode,
        expected_shards=1, candidate_summary_relpath="v17/candidate_summary.json",
        summary_format="paper_pose_multisolution_v17",
        pose_top1_name="pose_top1.csv", target_name="target.png",
        min_candidate_count=3, min_iou=0.95, max_boundary_fraction=0.05,
        max_continuity_jump=0.25, max_cross_start_std=0.02,
        required_confidence_fields=["z", "alpha", "beta", "gamma"],
        min_field_confidence="medium_simulation", min_z_mm=11.0,
        max_z_mm=20.0, robot_model="ur10",
        robot_validation_root=None if robot is None else str(robot),
        robot_report_name="robot_validation_report.json",
    )


def _complete_source(source: Path) -> None:
    _write_json(
        source / "batch_summary_shard_0_of_1.json",
        {
            "requested_characters": 1,
            "processed": 1,
            "shard": {"index": 0, "count": 1},
        },
    )


def test_registers_only_hash_bound_robot_validated_pose(tmp_path):
    source = tmp_path / "model_run"
    output = tmp_path / "registry"
    robot = tmp_path / "robot_reports"
    char_dir = source / "char_u6d4b"
    pose = char_dir / "pose_top1.csv"
    estimate = char_dir / "v17" / "p0" / "candidate_trajectory.csv"
    _pose(pose, "测")
    estimate.parent.mkdir(parents=True)
    estimate.write_bytes(pose.read_bytes())
    (char_dir / "target.png").write_bytes(b"target")
    _write_json(char_dir / "v17" / "candidate_summary.json", _summary(estimate))
    _complete_source(source)
    pose_hash = hashlib.sha256(pose.read_bytes()).hexdigest()
    _write_json(
        robot / "char_u6d4b" / "robot_validation_report.json",
        {
            "format": "ros_pose_validation_v1", "robot_model": "ur10",
            "trajectory_sha256": pose_hash, "ik_passed": True,
            "collision_free": True, "singularity_free": True,
            "joint_limits_passed": True, "trajectory_continuity_passed": True,
        },
    )

    summary = run(_args(source, output, robot))

    assert summary["registry_ready"] is True
    assert summary["registered_characters"] == 1
    assert (output / "char_u6d4b" / "pose_refined.csv").read_bytes() == pose.read_bytes()
    manifest = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
    assert manifest[0]["character"] == "测"
    assert manifest[0]["status"] == "completed"
    assert (output / "robot_validation_queue.jsonl").read_text() == ""


def test_model_qualified_pose_waits_for_robot_report(tmp_path):
    source = tmp_path / "model_run"
    output = tmp_path / "registry"
    char_dir = source / "char_u6d4b"
    pose = char_dir / "pose_top1.csv"
    estimate = char_dir / "v17" / "p0" / "candidate_trajectory.csv"
    _pose(pose, "测")
    estimate.parent.mkdir(parents=True)
    estimate.write_bytes(pose.read_bytes())
    (char_dir / "target.png").write_bytes(b"target")
    _write_json(char_dir / "v17" / "candidate_summary.json", _summary(estimate))
    _complete_source(source)

    summary = run(_args(source, output, None))

    assert summary["registered_characters"] == 0
    assert summary["status_counts"] == {"awaiting_robot_validation": 1}
    assert (output / "manifest.jsonl").read_text() == ""
    queued = [json.loads(line) for line in (output / "robot_validation_queue.jsonl").read_text().splitlines()]
    assert queued[0]["trajectory_sha256"] == hashlib.sha256(pose.read_bytes()).hexdigest()


def test_low_confidence_candidate_is_rejected_before_robot_validation(tmp_path):
    source = tmp_path / "model_run"
    output = tmp_path / "registry"
    char_dir = source / "char_u6d4b"
    pose = char_dir / "pose_top1.csv"
    estimate = char_dir / "v17" / "p0" / "candidate_trajectory.csv"
    _pose(pose, "测")
    estimate.parent.mkdir(parents=True)
    estimate.write_bytes(pose.read_bytes())
    (char_dir / "target.png").write_bytes(b"target")
    _write_json(
        char_dir / "v17" / "candidate_summary.json",
        _summary(estimate, alpha_level="low"),
    )
    _complete_source(source)

    summary = run(_args(source, output, None))

    assert summary["status_counts"] == {"model_rejected": 1}
    assert summary["registered_characters"] == 0
    assert "field_confidence_below_threshold:alpha:low" in summary["reason_counts"]


def test_formal_registration_refuses_incomplete_shards(tmp_path):
    source = tmp_path / "model_run"
    source.mkdir()
    output = tmp_path / "registry"

    with pytest.raises(SourceIncompleteError):
        run(_args(source, output, None))

    status = json.loads((output / "registry_summary.json").read_text())
    assert status["registry_ready"] is False
    assert status["status"] == "source_incomplete"
