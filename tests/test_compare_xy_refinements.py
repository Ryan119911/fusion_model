import csv
import json

from tools.compare_xy_refinements import compare_runs


def write_run(tmp_path, name, offset):
    csv_path = tmp_path / f"{name}.csv"
    fields = ["stroke_id", "point_id", "x", "y", "z", "alpha", "beta", "gamma"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for point, x in enumerate((0.0, 10.0, 20.0)):
            writer.writerow(
                {
                    "stroke_id": 0,
                    "point_id": point,
                    "x": x + offset,
                    "y": x,
                    "z": 12,
                    "alpha": 0,
                    "beta": 0,
                    "gamma": 0.1,
                }
            )
    report_path = tmp_path / f"{name}.json"
    report_path.write_text(
        json.dumps(
            {
                "metrics": {"iou_at_0.5": 0.8},
                "xy_optimization": {"max_abs_change_px": 1},
            }
        ),
        encoding="utf-8",
    )
    return str(csv_path), str(report_path)


def test_compare_xy_refinements_detects_stable_frozen_pose(tmp_path):
    first = write_run(tmp_path, "a", 0)
    second = write_run(tmp_path, "b", 0.1)
    result = compare_runs(
        [first[0], second[0]], [first[1], second[1]], drawable_pixels=96
    )
    assert result["posture_frozen_exactly"]
    assert result["normalized_rms_std"]["xy"] < 0.01
    assert result["pairwise_canvas_distance"]["0:1"]["rms_canvas_px"] > 0
