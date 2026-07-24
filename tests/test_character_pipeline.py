import unittest

import numpy as np
import torch

from models.character_generator import CharacterUNet
from datasets.character_dataset import deterministic_character_split_indices
from tools.train_character import StructureMaskLoss
from tools.build_character_pairs import candidate_exclusion, target_quality_failures
from utils.character_alignment import (
    align_target_to_trajectory,
    alignment_metrics,
    transform_target,
)
from utils.character_features import SPATIAL_CHANNEL_NAMES, extract_character_spatial_maps
from utils.image_preprocessing import letterbox_character_image
from utils.character_script import CharacterScriptMapper
from utils.structure_mask import (
    STRUCTURE_TARGET_MODE,
    build_structure_mask,
    symmetric_structure_metrics,
)
from utils.types import (
    CharacterTrajectory,
    PointState,
    StrokeTrajectory,
    TrajectoryPoint,
)


def point(stroke_id, point_id, x, y, z=1.0):
    return TrajectoryPoint(
        stroke_id=stroke_id,
        point_id=point_id,
        x=x,
        y=y,
        z=z,
        alpha=0.0,
        beta=0.0,
        gamma=0.0,
        state=PointState.MOVE,
    )


class CharacterPipelineTest(unittest.TestCase):
    def setUp(self):
        self.sample = CharacterTrajectory(
            character="武",
            strokes=[
                StrokeTrajectory(0, [point(0, 0, 0, 0), point(0, 1, 10, 0)]),
                StrokeTrajectory(1, [point(1, 0, 5, -5), point(1, 1, 5, 5)]),
            ],
            meta={"sample_id": "wu_test"},
        )

    def test_complete_spatial_trajectory_maps(self):
        inputs, normalized_strokes = extract_character_spatial_maps(
            self.sample,
            canvas_size=32,
            padding=2,
            line_width=2,
        )
        self.assertEqual(inputs.shape, (len(SPATIAL_CHANNEL_NAMES), 32, 32))
        self.assertEqual(len(normalized_strokes), 2)
        self.assertGreater(float(inputs[0].sum()), 0.0)
        self.assertGreater(float(inputs[1].sum()), 0.0)
        self.assertGreater(float(inputs[1].sum()), float(inputs[0].sum()))
        self.assertTrue(np.isfinite(inputs).all())

    def test_unet_emits_one_complete_image_without_transformer(self):
        model = CharacterUNet(
            input_channels=6,
            base_channels=8,
            image_size=32,
            depth=3,
            dropout=0.0,
        )
        output = model(torch.randn(2, 6, 32, 32))
        self.assertEqual(tuple(output.shape), (2, 1, 32, 32))
        self.assertTrue(torch.isfinite(output).all())
        self.assertFalse(any(isinstance(module, torch.nn.Transformer) for module in model.modules()))

    def test_v7_proximity_prior_is_narrow_and_learnable(self):
        model = CharacterUNet(
            input_channels=6,
            base_channels=8,
            image_size=32,
            depth=3,
            dropout=0.0,
            prior_strength=0.75,
            prior_threshold=0.70,
            prior_sharpness=10.0,
        )
        inputs = torch.zeros(1, 6, 32, 32)
        inputs[:, 1, 16, 16] = 1.0
        output = model(inputs)
        self.assertGreater(float(output[0, 0, 16, 16].detach()), 0.5)
        self.assertLess(float(output[0, 0, 0, 0].detach()), 0.1)
        output.mean().backward()
        self.assertIsNotNone(model.prior_gain_raw.grad)

    def test_image_preprocessing_uses_ink_positive_polarity(self):
        image = np.full((20, 30), 255, dtype=np.uint8)
        image[5:15, 12:18] = 0
        canvas, transform = letterbox_character_image(image, canvas_size=32, padding=2)
        self.assertEqual(canvas.shape, (32, 32))
        self.assertGreater(float(canvas.max()), 0.9)
        self.assertLess(float(canvas[0, 0]), 0.01)
        self.assertEqual(transform["crop_box"], [12, 5, 18, 15])

    def test_paper_background_is_removed_before_letterboxing(self):
        image = np.full((30, 30), 190, dtype=np.uint8)
        image[7:23, 13:17] = 25
        canvas, transform = letterbox_character_image(image, canvas_size=32, padding=2)
        self.assertLess(float(np.median(canvas[:2])), 0.01)
        self.assertGreater(float(canvas.max()), 0.95)
        self.assertEqual(
            transform["normalization"]["polarity"],
            "dark_ink_on_light_background",
        )

    def test_target_registration_improves_trajectory_coverage(self):
        inputs, _ = extract_character_spatial_maps(
            self.sample,
            canvas_size=32,
            padding=4,
            line_width=2,
        )
        shifted = transform_target(inputs[0], scale=0.72, shift_x=4, shift_y=-3)
        _, report = align_target_to_trajectory(
            shifted,
            centerline=inputs[0],
            proximity=inputs[1],
            local_shift=3,
        )
        self.assertGreater(report["after"]["coverage"], report["before"]["coverage"])
        self.assertGreater(report["after"]["score"], report["before"]["score"])

    def test_script_mapper_can_redirect_simplified_label_to_traditional_trajectory(self):
        mapping = {"丑": "醜", "儿": "兒"}
        mapper = CharacterScriptMapper(
            "traditional", converter=lambda value: mapping.get(value, value)
        )
        self.assertEqual(mapper.convert("丑"), "醜")
        self.assertEqual(mapper.convert("武"), "武")

    def test_v7_quality_filter_rejects_dense_target_outside_trajectory(self):
        centerline = np.zeros((32, 32), dtype=np.float32)
        centerline[15:17, 5:27] = 1.0
        proximity = np.zeros_like(centerline)
        proximity[11:21, 3:29] = 1.0
        target = np.zeros_like(centerline)
        target[4:28, 4:28] = 1.0
        metrics = alignment_metrics(target, centerline, proximity)
        metrics.update(
            symmetric_structure_metrics(target, centerline, proximity)
        )
        thresholds = {
            "min_alignment_coverage": 0.55,
            "min_support_dice": 0.45,
            "max_outside_support_fraction": 0.35,
            "min_target_support_area_ratio": 0.30,
            "max_target_support_area_ratio": 1.70,
            "max_target_ink_fraction": 0.45,
            "max_foreground_bbox_fill_fraction": 0.55,
            "max_border_ink_fraction": 0.02,
            "min_target_skeleton_in_support_fraction": 0.70,
            "min_trajectory_near_target_skeleton_fraction": 0.60,
        }
        failures = target_quality_failures(metrics, thresholds)
        self.assertIn("outside_support_above_threshold", failures)
        self.assertIn("target_ink_above_threshold", failures)

    def test_v7_structure_cleanup_is_binary_and_removes_speckles(self):
        target = np.zeros((32, 32), dtype=np.float32)
        target[12:20, 8:24] = 0.9
        target[2, 2] = 1.0
        target[29, 28] = 0.8
        mask, info = build_structure_mask(
            target,
            threshold=0.35,
            min_component_pixels=4,
        )
        self.assertEqual(info["mode"], STRUCTURE_TARGET_MODE)
        self.assertEqual(set(np.unique(mask)), {0.0, 1.0})
        self.assertEqual(float(mask[2, 2]), 0.0)
        self.assertEqual(float(mask[15, 15]), 1.0)
        self.assertEqual(
            info["foreground_pixels_before"] - info["foreground_pixels_after"],
            2,
        )

    def test_v7_symmetric_skeleton_filter_detects_structure_mismatch(self):
        centerline = np.zeros((32, 32), dtype=np.float32)
        centerline[15:17, 4:28] = 1.0
        proximity = np.zeros_like(centerline)
        proximity[11:21, 2:30] = 1.0
        aligned_target = np.zeros_like(centerline)
        aligned_target[12:20, 3:29] = 1.0
        mismatched_target = np.zeros_like(centerline)
        mismatched_target[3:11, 3:29] = 1.0
        aligned = symmetric_structure_metrics(
            aligned_target, centerline, proximity, skeleton_tolerance=3
        )
        mismatched = symmetric_structure_metrics(
            mismatched_target, centerline, proximity, skeleton_tolerance=3
        )
        self.assertGreater(
            aligned["symmetric_skeleton_score"],
            mismatched["symmetric_skeleton_score"],
        )

    def test_v7_structure_loss_is_finite_and_penalizes_gray(self):
        criterion = StructureMaskLoss()
        targets = torch.zeros(1, 1, 16, 16)
        targets[:, :, 6:10, 3:13] = 1.0
        inputs = torch.zeros(1, 6, 16, 16)
        inputs[:, 0:1] = targets
        gray = torch.full_like(targets, 0.5)
        accurate = targets * 0.98 + (1.0 - targets) * 0.02
        gray_loss = criterion(gray, targets, inputs)
        accurate_loss = criterion(accurate, targets, inputs)
        self.assertTrue(torch.isfinite(gray_loss))
        self.assertLess(float(accurate_loss), float(gray_loss))

    def test_exact_candidate_exclusion_preserves_other_boxes(self):
        exclusions = [{
            "character": "乘",
            "image_path": "data/raw/images/152/32.jpg",
            "bbox": [0.0, 2023.0, 391.0, 2408.0],
            "reason": "noise",
        }]
        bad = {
            "image_path": "data/raw/images/152/32.jpg",
            "bbox": [0.0, 2023.0, 391.0, 2408.0],
        }
        other_box = {
            "image_path": "data/raw/images/152/32.jpg",
            "bbox": [10.0, 20.0, 30.0, 40.0],
        }
        self.assertIsNotNone(candidate_exclusion(bad, "乘", exclusions))
        self.assertIsNone(candidate_exclusion(other_box, "乘", exclusions))
        self.assertIsNone(candidate_exclusion(bad, "武", exclusions))

    def test_character_split_has_no_identity_leakage(self):
        metadata = np.asarray([
            {"character": "武"},
            {"character": "武"},
            {"character": "永"},
            {"character": "大"},
        ], dtype=object)
        train_indices, val_indices = deterministic_character_split_indices(
            metadata, val_ratio=0.34, seed=42
        )
        train_characters = {metadata[index]["character"] for index in train_indices}
        val_characters = {metadata[index]["character"] for index in val_indices}
        self.assertFalse(train_characters & val_characters)
        self.assertEqual(sorted(train_indices + val_indices), list(range(len(metadata))))


if __name__ == "__main__":
    unittest.main()
