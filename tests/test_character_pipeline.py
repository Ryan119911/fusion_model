import unittest

import numpy as np
import torch

from models.character_generator import CharacterUNet
from utils.character_features import SPATIAL_CHANNEL_NAMES, extract_character_spatial_maps
from utils.image_preprocessing import letterbox_character_image
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
        self.assertTrue(np.isfinite(inputs).all())

    def test_unet_emits_one_complete_image_without_transformer(self):
        model = CharacterUNet(
            input_channels=5,
            base_channels=8,
            image_size=32,
            depth=3,
            dropout=0.0,
        )
        output = model(torch.randn(2, 5, 32, 32))
        self.assertEqual(tuple(output.shape), (2, 1, 32, 32))
        self.assertTrue(torch.isfinite(output).all())
        self.assertFalse(any(isinstance(module, torch.nn.Transformer) for module in model.modules()))

    def test_image_preprocessing_uses_ink_positive_polarity(self):
        image = np.full((20, 30), 255, dtype=np.uint8)
        image[5:15, 12:18] = 0
        canvas, transform = letterbox_character_image(image, canvas_size=32, padding=2)
        self.assertEqual(canvas.shape, (32, 32))
        self.assertGreater(float(canvas.max()), 0.9)
        self.assertLess(float(canvas[0, 0]), 0.01)
        self.assertEqual(transform["crop_box"], [12, 5, 18, 15])


if __name__ == "__main__":
    unittest.main()
