import json
import tempfile
import unittest
from pathlib import Path

from utils.trajectory_processing import (
    TrajectorySafetyLimits,
    densify_sample,
    expand_pen_up_transitions,
    render_trajectory_preview,
    render_trajectory_overlay,
    repair_sample_states,
    smooth_sample,
    validate_trajectory,
    write_trajectory_csv,
)
from utils.types import CharacterTrajectory, PointState, StrokeTrajectory, TrajectoryPoint


def p(stroke, index, x, y, z=15.5, gamma=0.0, state=PointState.MOVE):
    return TrajectoryPoint(stroke, index, x, y, z, 0.0, 0.0, gamma, state)


class TrajectoryProcessingTests(unittest.TestCase):
    def setUp(self):
        self.sample = CharacterTrajectory(
            character="武",
            meta={"sample_id": "test"},
            strokes=[
                StrokeTrajectory(0, [p(0, 0, 0, 0), p(0, 1, 5, 0), p(0, 2, 10, 0)]),
                StrokeTrajectory(1, [p(1, 0, 50, 0), p(1, 1, 50, 5), p(1, 2, 50, 10)]),
            ],
        )

    def test_repair_states_is_per_stroke(self):
        repaired = repair_sample_states(self.sample)
        for stroke in repaired.sorted_strokes():
            self.assertEqual(stroke.points[0].state, PointState.DOWN)
            self.assertEqual(stroke.points[-1].state, PointState.UP)
            self.assertTrue(all(p.state == PointState.MOVE for p in stroke.points[1:-1]))

    def test_smoothing_does_not_mix_strokes_or_endpoints(self):
        sample = CharacterTrajectory(
            strokes=[StrokeTrajectory(0, [p(0, 0, 0, 0), p(0, 1, 2, 10), p(0, 2, 4, 0)]),
                     StrokeTrajectory(1, [p(1, 0, 100, 100), p(1, 1, 102, 110), p(1, 2, 104, 100)])]
        )
        smoothed = smooth_sample(sample, passes=4, strength=0.5)
        self.assertEqual((smoothed.strokes[0].points[0].x, smoothed.strokes[0].points[-1].x), (0, 4))
        self.assertEqual(smoothed.strokes[1].points[0].x, 100)
        self.assertGreater(smoothed.strokes[1].points[0].x, smoothed.strokes[0].points[-1].x)

    def test_densify_keeps_hard_stroke_boundaries(self):
        dense = densify_sample(self.sample, max_step_xy=2.0)
        self.assertGreater(len(dense.strokes[0].points), 3)
        self.assertEqual(len(dense.strokes), 2)
        self.assertEqual(dense.strokes[0].points[-1].state, PointState.UP)
        self.assertEqual(dense.strokes[1].points[0].state, PointState.DOWN)

    def test_execution_expansion_adds_clearance_and_travel_states(self):
        expanded = expand_pen_up_transitions(self.sample, clearance_z=25.0)
        self.assertEqual(expanded.strokes[0].points[-2].z, 15.5)
        self.assertEqual(expanded.strokes[0].points[-1].state, PointState.UP)
        self.assertEqual(expanded.strokes[0].points[-1].z, 25.0)
        self.assertEqual(expanded.strokes[1].points[0].state, PointState.TRANSITION)
        self.assertEqual(expanded.strokes[1].points[1].state, PointState.DOWN)
        self.assertTrue(validate_trajectory(expanded)["safe"])

    def test_safety_report_and_preview_are_machine_readable(self):
        repaired = repair_sample_states(self.sample)
        report = validate_trajectory(repaired, TrajectorySafetyLimits(max_step_xy=6.0))
        self.assertTrue(report["safe"])
        path = Path("tmp") / "trajectory_processing_test"
        path.mkdir(parents=True, exist_ok=True)
        render_trajectory_preview(repaired, path / "preview.png")
        render_trajectory_overlay(self.sample, repaired, path / "overlay.png")
        write_trajectory_csv(repaired, path / "trajectory.csv")
        self.assertTrue((path / "preview.png").exists())
        self.assertTrue((path / "overlay.png").exists())
        self.assertTrue((path / "trajectory.csv").exists())
        json.dumps(report)

    def test_safety_rejects_large_segment(self):
        report = validate_trajectory(self.sample, TrajectorySafetyLimits(max_step_xy=1.0))
        self.assertFalse(report["safe"])
        self.assertIn("xy_step_exceeds_limit", report["errors"])
