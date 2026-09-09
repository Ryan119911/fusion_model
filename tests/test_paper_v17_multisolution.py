import csv
import json

import numpy as np

from tools.build_paper_v17_truth import build_random_truth
from tools.run_paper_v17_multisolution import (
    boundary_fractions,
    continuity_metrics,
    gamma_heading_metrics,
    score_candidate,
)


def _write_pose_csv(path):
    fields = ["character", "sample_id", "stroke_id", "point_id", "x", "y"]
    rows = []
    for stroke_id in (0, 1):
        for point_id in range(3):
            rows.append(
                {
                    "character": "测",
                    "sample_id": "测_test",
                    "stroke_id": stroke_id,
                    "point_id": point_id,
                    "x": 10 + 20 * point_id,
                    "y": 10 + 5 * stroke_id,
                }
            )
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_random_truth_is_seeded_and_in_bounds(tmp_path):
    source = tmp_path / "trajectory.csv"
    first = tmp_path / "truth_a.csv"
    second = tmp_path / "truth_b.csv"
    _write_pose_csv(source)
    report_a = build_random_truth(
        str(source), str(first), character="测", seed=7, gamma_mode="heading"
    )
    report_b = build_random_truth(
        str(source), str(second), character="测", seed=7, gamma_mode="heading"
    )
    assert first.read_bytes() == second.read_bytes()
    assert report_a["ranges"] == report_b["ranges"]
    assert report_a["gamma_mode"] == "heading"
    rows = list(csv.DictReader(first.open(encoding="utf-8-sig")))
    assert all(11.0 < float(row["z"]) < 20.0 for row in rows)
    assert all(0.0 < float(row["alpha"]) < np.deg2rad(10.0) for row in rows)
    assert all(0.0 < float(row["beta"]) < np.deg2rad(5.0) for row in rows)
    assert json.loads(first.with_suffix(".json").read_text())["simulation_only"]


def test_continuity_does_not_bridge_strokes():
    posture = np.asarray(
        [[13.0, 0.02, 0.04], [14.0, 0.03, 0.03], [15.0, 0.04, 0.02], [12.0, 0.02, 0.04]],
        dtype=np.float64,
    )
    gamma = np.zeros(4, dtype=np.float64)
    metrics = continuity_metrics(posture, gamma, np.asarray([0, 0, 0, 1]))
    # The large transition from 19 to 12 belongs to a pen-up boundary and is
    # therefore absent from the reported within-stroke first differences.
    assert metrics["first_difference_normalized_max"] < 0.2


def test_score_penalizes_boundary_and_footprint():
    good = score_candidate(
        {"iou_at_0.5": 0.98},
        {"combined_normalized_rmse": 0.05},
        {"first_difference_normalized_rms": 0.01},
        0.0,
        0.0,
        {
            "image_iou": 0.45,
            "local_footprint": 0.25,
            "continuity": 0.15,
            "boundary_safety": 0.10,
            "cross_start_stability": 0.05,
        },
    )
    bad = score_candidate(
        {"iou_at_0.5": 0.98},
        {"combined_normalized_rmse": 2.0},
        {"first_difference_normalized_rms": 0.5},
        1.0,
        0.2,
        good["weights"],
    )
    assert good["total"] > bad["total"]


def test_boundary_fraction_uses_physical_limits():
    posture = np.asarray([[11.0, 0.0, 0.0], [15.0, 0.1, 0.04]], dtype=np.float64)
    gamma = np.asarray([-np.pi, 0.0], dtype=np.float64)
    fractions = boundary_fractions(posture, gamma)
    assert fractions["z"] == 0.5
    assert fractions["alpha"] == 0.5
    assert fractions["beta"] == 0.5
    assert fractions["gamma"] == 0.5


def test_gamma_heading_is_forward_and_stroke_local():
    # The terminal point inherits the last segment and no heading is formed
    # across the pen-up boundary between the two strokes.
    xy = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 0.0], [2.0, -1.0]],
        dtype=np.float32,
    )
    stroke_ids = np.asarray([0, 0, 0, 1, 1], dtype=np.int64)
    gamma = np.asarray([0.0, 0.0, 0.0, -np.pi / 2.0, -np.pi / 2.0])
    metrics = gamma_heading_metrics(gamma, xy, stroke_ids)
    assert metrics["rmse_rad"] < 1e-6
