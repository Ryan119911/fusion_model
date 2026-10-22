import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from fusion_model_ros2_beta.offline_fontsize_inversion import (
    OfflineFontSizeGenerator,
    export_physical_trajectory,
    scale_complete_target_image,
    scale_initial_pose_csv,
)
from fusion_model_ros2_beta.trajectory_catalog import TrajectoryEntry


FIELDS = [
    "character", "sample_id", "stroke_id", "point_id", "x", "y", "z",
    "alpha", "beta", "gamma", "state", "prototype",
]


def write_pose(path: Path) -> None:
    rows = [
        ["一", "sample", 0, 0, 0, 10, 15, 0.1, -0.2, 0.3, 1, "paper_psoc_lm_v42_fused_pose"],
        ["一", "sample", 0, 1, 100, 10, 16, 0.2, -0.1, 0.4, 2, "paper_psoc_lm_v42_fused_pose"],
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        writer.writerows(rows)


class OfflineFontSizeInversionTest(unittest.TestCase):
    def test_source_contract_distinguishes_pipeline_from_model_version(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pose = root / "pose.csv"
            target = root / "target.png"
            write_pose(pose)
            target.touch()
            entry = TrajectoryEntry(
                character="一", status="ready", trajectory_csv=pose,
                target_image=target, sample_id="sample", output_dir=root,
            )
            OfflineFontSizeGenerator._validate_source_contract(entry)
            text = pose.read_text(encoding="utf-8").replace(
                "paper_psoc_lm_v42_fused_pose", "paper_psoc_lm_v12"
            )
            pose.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "基础轨迹协议不是"):
                OfflineFontSizeGenerator._validate_source_contract(entry)

    def test_matching_v42_render_replaces_mislabeled_catalog_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target.png"
            rendered = root / "inversion_rendered.png"
            target.touch()
            rendered.touch()
            entry = TrajectoryEntry(
                character="汉",
                status="ready",
                trajectory_csv=root / "pose_refined.csv",
                target_image=target,
                sample_id="汉_fake_sim",
                output_dir=root,
            )
            self.assertEqual(
                OfflineFontSizeGenerator._shape_target(entry), rendered
            )

    def test_quality_gate_requires_shape_preserving_v42_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            report_path = Path(temporary) / "report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "format": "paper_psoc_lm_v42_fused_pose",
                        "fixed_xy": True,
                        "optimized_fields": ["H"],
                        "derived_fields": ["alpha", "beta", "gamma"],
                        "trajectory_safety": {"safe": True},
                        "metrics": {
                            "plain_mse": 0.1,
                            "dice_at_0.5": 0.5,
                            "ink_ratio_at_0.5": 1.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            accepted = OfflineFontSizeGenerator._validate_fused_report(
                report_path, "汉"
            )
            self.assertTrue(accepted["fixed_xy"])
            accepted["fixed_xy"] = False
            report_path.write_text(json.dumps(accepted), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "改变了字号缩放后的中心线"):
                OfflineFontSizeGenerator._validate_fused_report(report_path, "汉")

    def test_initial_pose_scales_xy_only_about_glyph_centre(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, scaled = root / "source.csv", root / "scaled.csv"
            write_pose(source)
            center_x, center_y, span = scale_initial_pose_csv(source, scaled, 0.5)
            with scaled.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual((center_x, center_y, span), (50.0, 10.0, 100.0))
            self.assertEqual([float(row["x"]) for row in rows], [25.0, 75.0])
            self.assertEqual([float(row["y"]) for row in rows], [10.0, 10.0])
            self.assertEqual([float(row["z"]) for row in rows], [15.0, 16.0])
            self.assertEqual([float(row["gamma"]) for row in rows], [0.3, 0.4])

    def test_export_converts_xy_once_and_keeps_full_pose(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.csv"
            inverted, output = root / "inverted.csv", root / "physical.csv"
            write_pose(source)
            scale_initial_pose_csv(source, inverted, 0.5)
            count = export_physical_trajectory(
                inverted, output, source_center_x=50.0, source_center_y=10.0,
                source_span=100.0, reference_font_size_m=0.29,
                font_size_m=0.145,
            )
            with output.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(count, 2)
            self.assertAlmostEqual(float(rows[0]["x"]), -0.0725)
            self.assertAlmostEqual(float(rows[1]["x"]), 0.0725)
            self.assertAlmostEqual(float(rows[0]["z"]), 15.0)
            self.assertAlmostEqual(float(rows[0]["alpha"]), 0.1)
            self.assertEqual(rows[0]["xy_unit"], "m")
            self.assertEqual(rows[0]["offline_inversion"], "v16_v42_fused_pose")

    def test_complete_target_canvas_is_scaled_about_centre(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "target.png", root / "scaled.png"
            image = Image.new("L", (128, 128), 255)
            ImageDraw.Draw(image).rectangle((24, 24, 103, 103), fill=0)
            image.save(source)
            scale_complete_target_image(source, output, 0.5, 128)
            scaled = Image.open(output).convert("L")
            self.assertGreater(scaled.getpixel((0, 0)), 240)
            self.assertLess(scaled.getpixel((64, 64)), 20)
            self.assertGreater(scaled.getpixel((30, 64)), 240)


if __name__ == "__main__":
    unittest.main()
