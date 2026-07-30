import csv
import json

import numpy as np
from PIL import Image

from tools.evaluate_style_refined_render import (
    exact_canvas,
    image_metrics,
    pose_continuity,
    pose_safety,
    solve_clipped_ink_gain,
)


def test_image_metrics_identical_masks():
    image = np.zeros((32, 32), dtype=np.float32)
    image[8:24, 14:18] = 1
    metrics = image_metrics(image, image, 0.35)
    assert metrics["mse"] == 0
    assert metrics["iou"] == 1
    assert metrics["symmetric_skeleton_distance_px"] == 0


def test_exact_canvas_preserves_blank_margins(tmp_path):
    image = np.full((64, 64), 255, dtype=np.uint8)
    image[24:40, 28:36] = 0
    path = tmp_path / "glyph.png"
    Image.fromarray(image).save(path)
    canvas = exact_canvas(str(path), 32)
    assert canvas.shape == (32, 32)
    assert np.count_nonzero(canvas[:10]) == 0


def test_clipped_ink_gain_matches_reachable_target_mean():
    prediction = np.asarray([[0.2, 0.8], [0.9, 1.0]], dtype=np.float32)
    target_mean = 0.8
    gain = solve_clipped_ink_gain(prediction, target_mean, 0.8, 2.0)
    calibrated = np.clip(prediction * gain, 0, 1)
    assert abs(float(calibrated.mean()) - target_mean) < 1e-6
    assert gain > target_mean / float(prediction.mean())


def test_clipped_ink_gain_respects_unreachable_bound():
    prediction = np.full((4, 4), 0.1, dtype=np.float32)
    gain = solve_clipped_ink_gain(prediction, 0.9, 0.8, 1.25)
    assert gain == 1.25


def test_pose_safety_keeps_geometry_diagnostics(tmp_path):
    trajectory = tmp_path / "pose.csv"
    with trajectory.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["stroke_id", "point_id", "z", "alpha", "beta", "gamma"],
        )
        writer.writeheader()
        writer.writerow(
            {"stroke_id": 0, "point_id": 0, "z": 12, "alpha": 0, "beta": 0, "gamma": 0}
        )
        writer.writerow(
            {"stroke_id": 0, "point_id": 1, "z": 13, "alpha": 0.1, "beta": 0, "gamma": 0.2}
        )
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "simulation_only": True,
                "trajectory_target_coverage_at_5px": 0.9,
                "field_decisions": {"gamma": {"boundary_fraction": 0.1}},
                "nested": {"joint_jacobian_audit": {"passed": True}},
            }
        ),
        encoding="utf-8",
    )
    result = pose_safety(str(report), str(trajectory))
    assert result["trajectory_target_coverage_at_5px"] == 0.9
    assert result["field_boundary_fractions"]["gamma"] == 0.1
    assert result["joint_jacobian_audit"]["passed"]
    assert result["continuity"]["max_adjacent_step"]["z"] == 1


def test_xy_stage_inherits_parent_posture_jacobian(tmp_path):
    trajectory = tmp_path / "pose.csv"
    trajectory.write_text(
        "stroke_id,point_id,z,alpha,beta,gamma\n0,0,12,0,0,0\n",
        encoding="utf-8",
    )
    xy_report = tmp_path / "xy.json"
    xy_report.write_text(
        json.dumps({"simulation_only": True, "xy_optimization": {"enabled": True}}),
        encoding="utf-8",
    )
    posture_report = tmp_path / "posture.json"
    posture_report.write_text(
        json.dumps({"joint_jacobian_audit": {"jointly_identifiable": True}}),
        encoding="utf-8",
    )
    result = pose_safety(
        str(xy_report), str(trajectory), posture_report_path=str(posture_report)
    )
    assert result["joint_jacobian_audit"]["jointly_identifiable"]
    assert result["xy_optimization"]["enabled"]
