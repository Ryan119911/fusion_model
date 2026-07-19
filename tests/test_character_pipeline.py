import unittest

import numpy as np
import torch

from models.character_generator import CharacterGenerator
from utils.character_features import (
    compute_character_normalization,
    extract_character_features,
    normalize_character_features,
)
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

    def test_feature_sequence_and_normalization(self):
        inputs, mask, normalized_strokes = extract_character_features(
            self.sample,
            max_strokes=4,
            canvas_size=32,
            padding=2,
        )
        self.assertEqual(inputs.shape, (4, 10))
        self.assertEqual(mask.tolist(), [True, True, False, False])
        self.assertEqual(len(normalized_strokes), 2)
        normalization = compute_character_normalization(
            inputs[None, ...], mask[None, ...], coordinate_scale=32
        )
        normalized = normalize_character_features(
            inputs[None, ...], mask[None, ...], normalization
        )
        self.assertTrue(np.all(normalized[0, 2:] == 0.0))

    def test_model_emits_one_complete_image(self):
        model = CharacterGenerator(
            input_dim=10,
            latent_dim=32,
            base_channels=8,
            image_size=32,
            max_strokes=4,
            transformer_layers=1,
            attention_heads=4,
            dropout=0.0,
        )
        output = model(
            torch.randn(2, 4, 10),
            torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool),
        )
        self.assertEqual(tuple(output.shape), (2, 1, 32, 32))
        self.assertTrue(torch.isfinite(output).all())

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
