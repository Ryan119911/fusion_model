import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.diagnose_trajectory_alignment import (
    TRANSFORM_NAMES,
    apply_coordinate_transform,
    diagnose_transforms,
    load_alignment_trajectory_csv,
    place_strokes,
    rasterize_strokes,
    search_alignment,
    transform_strokes,
)


def _asymmetric_strokes():
    return [
        np.asarray(
            [
                [0.18, 0.18],
                [0.18, 0.82],
                [0.48, 0.82],
                [0.48, 0.62],
                [0.31, 0.62],
            ],
            dtype=np.float64,
        ),
        np.asarray(
            [[0.63, 0.23], [0.84, 0.36], [0.71, 0.51]], dtype=np.float64
        ),
    ]


class TrajectoryAlignmentTests(unittest.TestCase):
    def test_lightweight_csv_loader_preserves_unicode_sample_id(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory) / "trajectory.csv"
            with open(path, "w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "character",
                        "sample_id",
                        "stroke_id",
                        "point_id",
                        "x",
                        "y",
                        "z",
                        "state",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "character": "武",
                        "sample_id": "武_fake_sim",
                        "stroke_id": 0,
                        "point_id": 0,
                        "x": 10.0,
                        "y": 20.0,
                        "z": 1.5,
                        "state": 0,
                    }
                )
            samples = load_alignment_trajectory_csv(str(path))
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].meta["sample_id"], "武_fake_sim")
            self.assertEqual(samples[0].character, "武")
            self.assertAlmostEqual(samples[0].all_points()[0].z, 1.5)

    def test_all_d4_transforms_stay_in_unit_square(self):
        points = np.asarray([[0.1, 0.2], [0.8, 0.3], [0.4, 0.9]])
        self.assertEqual(len(TRANSFORM_NAMES), 8)
        self.assertEqual(len(set(TRANSFORM_NAMES)), 8)
        for name in TRANSFORM_NAMES:
            transformed = apply_coordinate_transform(points, name)
            self.assertEqual(transformed.shape, points.shape)
            self.assertTrue(np.all(transformed >= 0.0))
            self.assertTrue(np.all(transformed <= 1.0))

    def test_known_rotation_ranks_first(self):
        strokes = _asymmetric_strokes()
        expected = "rotate_90"
        transformed = transform_strokes(strokes, expected)
        placed = place_strokes(transformed, 64, 0.9, 3.0, -4.0)
        target = rasterize_strokes(placed, 64, 4)
        rows = diagnose_transforms(
            strokes,
            target,
            min_scale=0.8,
            max_scale=1.0,
            max_offset_ratio=0.12,
            coarse_steps=5,
            offset_steps=7,
            fine_steps=5,
        )
        self.assertEqual(rows[0]["transform"], expected)
        self.assertLess(rows[0]["symmetric_chamfer_px"], 1.0)

    def test_scale_and_translation_are_recovered(self):
        strokes = _asymmetric_strokes()
        expected_scale = 0.85
        expected_x = 5.0
        expected_y = -3.0
        placed = place_strokes(
            strokes, 64, expected_scale, expected_x, expected_y
        )
        target = rasterize_strokes(placed, 64, 4)
        result = search_alignment(
            strokes,
            target,
            min_scale=0.75,
            max_scale=0.95,
            max_offset_ratio=0.12,
            coarse_steps=5,
            offset_steps=7,
            fine_steps=5,
        )
        self.assertAlmostEqual(result["scale"], expected_scale, delta=0.06)
        self.assertAlmostEqual(result["offset_x_px"], expected_x, delta=1.5)
        self.assertAlmostEqual(result["offset_y_px"], expected_y, delta=1.5)
        self.assertLess(result["symmetric_chamfer_px"], 1.0)


if __name__ == "__main__":
    unittest.main()
