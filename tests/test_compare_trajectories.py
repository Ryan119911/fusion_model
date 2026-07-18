import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from utils.comparison_metrics import (
    image_metrics,
    read_comparison_manifest,
    signed_difference_image,
    trajectory_metrics,
)
from utils.types import (
    CharacterTrajectory,
    PointState,
    StrokeTrajectory,
    TrajectoryPoint,
)


def _trajectory(x_offset: float = 0.0) -> CharacterTrajectory:
    points = [
        TrajectoryPoint(0, index, x + x_offset, 0.0, 1.0, 0.0, 0.0, 0.0, PointState.MOVE)
        for index, x in enumerate((0.0, 1.0, 2.0))
    ]
    return CharacterTrajectory("test", [StrokeTrajectory(0, points)], {"sample_id": "sample-1"})


class TrajectoryComparisonTests(unittest.TestCase):
    def test_identical_images_have_perfect_overlap(self):
        image = np.zeros((16, 16), dtype=np.float32)
        image[4:12, 6:10] = 1.0
        metrics = image_metrics(image, image)
        self.assertAlmostEqual(metrics["mse"], 0.0)
        self.assertAlmostEqual(metrics["dice_score"], 1.0)
        self.assertAlmostEqual(metrics["dice_at_0.5"], 1.0)
        self.assertAlmostEqual(metrics["iou_at_0.5"], 1.0)
        self.assertAlmostEqual(metrics["global_ssim"], 1.0)
        self.assertGreater(metrics["ssim_score"], 0.6)
        self.assertAlmostEqual(metrics["ink_delta"], 0.0)

    def test_signed_difference_color_semantics(self):
        generated = np.asarray([[1.0, 0.5, 0.0]], dtype=np.float32)
        target = np.asarray([[0.0, 0.5, 1.0]], dtype=np.float32)
        result = signed_difference_image(generated, target)
        np.testing.assert_allclose(result[0, 0], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(result[0, 1], [0.0, 0.5, 0.0])
        np.testing.assert_allclose(result[0, 2], [0.0, 0.0, 1.0])

    def test_trajectory_metrics_report_known_offset(self):
        metrics = trajectory_metrics(_trajectory(), _trajectory(2.0), 8)
        self.assertAlmostEqual(metrics["x_mae"], 2.0)
        self.assertAlmostEqual(metrics["y_mae"], 0.0)
        self.assertAlmostEqual(metrics["xyz_mean_distance"], 2.0)
        self.assertAlmostEqual(metrics["angle_rmse_radians"], 0.0)

    def test_manifest_prefers_working_directory_paths(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            image_path = root / "target.png"
            Image.new("L", (4, 4)).save(image_path)
            manifest_dir = root / "configs"
            manifest_dir.mkdir()
            manifest_path = manifest_dir / "manifest.csv"
            relative_image = image_path.relative_to(Path.cwd())
            with open(manifest_path, "w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["sample_id", "target_image"])
                writer.writeheader()
                writer.writerow({"sample_id": "sample-1", "target_image": relative_image})
            rows = read_comparison_manifest(str(manifest_path))
            self.assertEqual(Path(rows[0]["target_image"]), image_path.resolve())


if __name__ == "__main__":
    unittest.main()
