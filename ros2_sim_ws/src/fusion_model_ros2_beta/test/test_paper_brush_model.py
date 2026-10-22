import math
import unittest

import numpy as np

from fusion_model_ros2_beta.paper_brush_model import (
    FlexibleBrushModel,
    PaperBrushParameters,
    cosine_similarity,
    footprint_polygon,
    normalize_ink_mask,
    structural_similarity,
    triangle_fan,
)


class PaperBrushModelTest(unittest.TestCase):
    def setUp(self):
        self.parameters = PaperBrushParameters()

    def test_reported_regression_responds_to_press_depth(self):
        shallow = self.parameters.dimensions(11.0, 0.0, 0.0)
        deep = self.parameters.dimensions(
            20.0, math.radians(10.0), math.radians(5.0)
        )
        self.assertGreater(deep.tip_length_m, shallow.tip_length_m)
        self.assertGreater(deep.heel_length_m, shallow.heel_length_m)
        self.assertGreater(deep.half_width_m, shallow.half_width_m)
        self.assertAlmostEqual(deep.half_width_m, 0.005926, places=4)

    def test_bezier_footprint_is_closed_and_triangulated(self):
        polygon = footprint_polygon(
            0.0, 0.0, 0.0, 20.0, 0.0, 0.0, self.parameters
        )
        triangles = triangle_fan(polygon)
        self.assertEqual(polygon.shape, (26, 2))
        self.assertEqual(triangles.shape, (26, 3, 2))
        self.assertLess(float(np.min(polygon[:, 0])), 0.0)
        self.assertGreater(float(np.max(polygon[:, 0])), 0.0)
        self.assertAlmostEqual(
            abs(float(np.min(polygon[:, 1]))),
            float(np.max(polygon[:, 1])),
            places=10,
        )

    def test_dynamic_bridge_has_inertia_and_bounded_snap_offset(self):
        model = FlexibleBrushModel(self.parameters)
        state = model.initial_state(
            x_m=0.0,
            y_m=0.0,
            press_depth_mm=20.0,
            alpha_rad=0.0,
            beta_rad=0.0,
            heading_rad=0.0,
        )
        state = model.step(
            state,
            x_m=0.002,
            y_m=0.0,
            press_depth_mm=20.0,
            alpha_rad=0.0,
            beta_rad=0.0,
            heading_rad=0.0,
        )
        target = self.parameters.dimensions(20.0, 0.0, 0.0)
        self.assertGreater(state.half_width_m, 0.0)
        self.assertLess(state.half_width_m, target.half_width_m)
        self.assertLessEqual(state.offset_m, 0.5 * self.parameters.bundle_length_m)
        self.assertAlmostEqual(state.x_m, 0.002, places=12)
        self.assertAlmostEqual(state.root_x_m, 0.0, places=12)
        self.assertTrue(
            self.parameters.point_in_calibrated_domain(
                state.virtual_h_mm,
                state.virtual_alpha_rad,
                state.virtual_beta_rad,
            )
        )

    def test_image_metrics_equal_one_for_identical_images(self):
        image = np.zeros((128, 128), dtype=np.float32)
        image[32:96, 48:80] = 1.0
        self.assertAlmostEqual(cosine_similarity(image, image), 1.0, places=12)
        self.assertAlmostEqual(structural_similarity(image, image), 1.0, places=12)

    def test_mask_alignment_removes_translation_and_scale(self):
        first = np.zeros((128, 128), dtype=np.float32)
        second = np.zeros((128, 128), dtype=np.float32)
        first[20:60, 30:50] = 1.0
        second[40:100, 70:100] = 1.0
        aligned_first = normalize_ink_mask(first, image_size=128, margin_ratio=0.1)
        aligned_second = normalize_ink_mask(second, image_size=128, margin_ratio=0.1)
        self.assertAlmostEqual(
            cosine_similarity(aligned_first, aligned_second), 1.0, places=12
        )


if __name__ == "__main__":
    unittest.main()
