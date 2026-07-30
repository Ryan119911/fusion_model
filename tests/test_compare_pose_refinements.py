import csv
import json

from tools.compare_pose_refinements import compare_pose_runs


def write_run(tmp_path, name, gamma, iou):
    csv_path = tmp_path / f"{name}.csv"
    fields = ["stroke_id", "point_id", "x", "y", "z", "alpha", "beta", "gamma"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for point in range(3):
            writer.writerow(
                {
                    "stroke_id": 0,
                    "point_id": point,
                    "x": point,
                    "y": point,
                    "z": 12,
                    "alpha": 0,
                    "beta": 0,
                    "gamma": gamma,
                }
            )
    report_path = tmp_path / f"{name}.json"
    report_path.write_text(
        json.dumps(
            {
                "metrics": {"iou_at_0.5": iou},
                "field_decisions": {"gamma": {"boundary_fraction": 0}},
                "identifiability": {
                    "joint_jacobian_audit": {"jointly_identifiable": True}
                },
                "trajectory_target_coverage_at_5px": 1.0,
            }
        ),
        encoding="utf-8",
    )
    return str(csv_path), str(report_path)


def test_pose_restart_stability_selects_best_eligible_run(tmp_path):
    first = write_run(tmp_path, "a", 0.10, 0.8)
    second = write_run(tmp_path, "b", 0.11, 0.82)
    result = compare_pose_runs(
        [first[0], second[0]],
        [first[1], second[1]],
        ["gamma"],
    )
    assert result["stable"]
    assert result["xy_frozen_exactly"]
    assert result["fields"]["gamma"]["normalized_rms_std"] < 0.02
    assert result["selected_run_index"] == 1
