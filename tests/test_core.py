import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import torch
except ModuleNotFoundError:
    torch = None

from utils.character_groups import (
    CharacterBatchSampler,
    fuse_character_tensors,
    validate_character_target_mapping,
    validate_group_consistency,
)

from models.dynamic_brush import DynamicBrushModel, DynamicBrushParams, PolyFit1D
from models.geometry import resample_stroke
from utils.feature_schema import STROKE10_V1, get_feature_schema
from utils.image_preprocessing import letterbox_character_image
from utils.splits import build_split
from utils.types import StrokeTrajectory, TrajectoryPoint


class CoreTests(unittest.TestCase):
    def test_group_split_has_no_leakage(self):
        meta = [{"sample_id": index // 3} for index in range(30)]
        train, val, manifest = build_split(meta, len(meta), 0.2, 42)
        self.assertFalse(set(train) & set(val))
        self.assertFalse(set(manifest["train_groups"]) & set(manifest["val_groups"]))

    def test_schema_dimension(self):
        self.assertEqual(get_feature_schema(STROKE10_V1).input_dim, 10)

    def test_resample_preserves_endpoints(self):
        stroke = StrokeTrajectory(0, [
            TrajectoryPoint(0, 0, 0, 0, 0, 0, 0, 0, 1),
            TrajectoryPoint(0, 1, 10, 0, 4, 0, 0, 0, 2),
        ])
        result = resample_stroke(stroke, 5).sorted_points()
        self.assertEqual(len(result), 5)
        self.assertAlmostEqual(result[0].x, 0)
        self.assertAlmostEqual(result[-1].x, 10)
        self.assertAlmostEqual(result[-1].z, 4)

    def test_dynamic_model_json_roundtrip(self):
        model = DynamicBrushModel(DynamicBrushParams(
            mode="calibrated",
            width_fn=PolyFit1D([1, 2]),
            drag_fn=PolyFit1D([0, 3]),
            offset_fn=PolyFit1D([0.5]),
        ))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.save_json(str(path))
            loaded = DynamicBrushModel.load_json(str(path))
            self.assertAlmostEqual(loaded.width(2), 5)
            self.assertEqual(json.loads(path.read_text())["mode"], "calibrated")

    def test_character_batch_sampler_keeps_groups_complete(self):
        meta = [
            {"sample_id": "a", "stroke_order": 0},
            {"sample_id": "a", "stroke_order": 1},
            {"sample_id": "b", "stroke_order": 0},
            {"sample_id": "b", "stroke_order": 1},
            {"sample_id": "b", "stroke_order": 2},
        ]
        batches = list(CharacterBatchSampler(meta, range(5), max_batch_size=4))
        self.assertEqual(batches, [[0, 1], [2, 3, 4]])

    @unittest.skipUnless(torch is not None, "PyTorch is not installed")
    def test_character_tensor_fusion_uses_pixelwise_maximum(self):
        images = torch.tensor([[[[0.0, 1.0]]], [[[0.5, 0.0]]], [[[0.2, 0.3]]]])
        meta = [{"sample_id": "a"}, {"sample_id": "a"}, {"sample_id": "b"}]
        fused, keys = fuse_character_tensors(images, meta)
        self.assertEqual(keys, ["a", "b"])
        torch.testing.assert_close(
            fused,
            torch.tensor([[[[0.5, 1.0]]], [[[0.2, 0.3]]]]),
        )

    def test_consistency_rejects_multiple_sources_for_one_character(self):
        meta = [
            {
                "sample_id": "wu",
                "character": "武",
                "stroke_order": index,
                "num_strokes_in_traj": 2,
                "used_real_image": True,
                "image_path": f"source-{index}.jpg",
                "canvas_transform": {"padding": 4},
            }
            for index in range(2)
        ]
        errors = validate_group_consistency(meta)
        self.assertTrue(any("image_path" in error for error in errors))

    def test_character_target_mapping_rejects_shared_target(self):
        meta = [
            {
                "sample_id": sample_id,
                "stroke_order": 0,
                "character_target_index": 0,
            }
            for sample_id in ("a", "b")
        ]
        errors = validate_character_target_mapping(meta, [0, 0], target_count=1)
        self.assertTrue(any("shared by groups" in error for error in errors))

    def test_letterbox_character_preserves_aspect_ratio_and_padding(self):
        image = Image.new("L", (20, 10), 255)
        pixels = np.asarray(image).copy()
        pixels[2:8, 2:18] = 0
        canvas, transform = letterbox_character_image(
            Image.fromarray(pixels), canvas_size=32, padding=4
        )
        self.assertEqual(canvas.shape, (32, 32))
        self.assertEqual(transform["padding"], 4)
        self.assertLessEqual(max(transform["resized_size"]), 24)
        self.assertGreater(float(canvas.max()), 0.9)


if __name__ == "__main__":
    unittest.main()
