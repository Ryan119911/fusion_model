import unittest

import numpy as np

from models.paper_bbsm import (
    PAPER_POSTURE_MAX,
    PAPER_POSTURE_MIN,
    bbsm_boundary,
    posture_to_geometry_numpy,
    render_bbsm_mask,
)
from tools.build_paper_bbsmg_dataset import build_dataset


class PaperBBSMTests(unittest.TestCase):
    def test_regression_matches_paper_equations(self):
        posture = np.asarray([[11.0, 0.0, 0.0]], dtype=np.float32)
        geometry = posture_to_geometry_numpy(posture)[0]
        expected = np.asarray(
            [
                0.0672 * 11.0 + 0.0267,
                0.0196 * 11.0 + 0.0372,
                0.0239 * 11.0 + 0.1137,
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(geometry, expected, rtol=1e-6, atol=1e-6)

    def test_bezier_peak_and_anchor_geometry(self):
        boundary = bbsm_boundary(lt=1.2, lh=0.4, lr=0.5, samples_per_side=101)
        self.assertAlmostEqual(float(boundary[:, 1].max()), 0.5, places=5)
        self.assertAlmostEqual(float(boundary[0, 0]), -1.2, places=6)

    def test_rasterized_mask_is_finite_and_nonempty(self):
        posture = (PAPER_POSTURE_MIN + PAPER_POSTURE_MAX) / 2.0
        mask = render_bbsm_mask(posture, 64.0, 64.0)
        self.assertEqual(mask.shape, (128, 128))
        self.assertTrue(np.isfinite(mask).all())
        self.assertGreater(float(mask.sum()), 1.0)
        self.assertGreaterEqual(float(mask.min()), 0.0)
        self.assertLessEqual(float(mask.max()), 1.0)

    def test_dataset_records_units_and_angle_scales(self):
        inputs, targets, metadata = build_dataset(
            count=4,
            image_size=128,
            pixels_per_model_unit=20.0,
            supersample=1,
            seed=3,
        )
        self.assertEqual(inputs.shape, (4, 5))
        self.assertEqual(targets.shape, (4, 1, 128, 128))
        self.assertEqual(targets.dtype, np.uint8)
        self.assertEqual(metadata["format"], "paper_bbsmg_v1")
        self.assertEqual(metadata["units"]["alpha"], "rad")
        self.assertAlmostEqual(
            metadata["input_normalization"]["scales"][1],
            float(np.deg2rad(10.0)),
            places=6,
        )


try:
    import torch  # noqa: F401

    from optim.paper_psoc_lm import cgl_interpolation_matrix

    class PaperPSOCTests(unittest.TestCase):
        def test_cgl_interpolation_preserves_constant(self):
            matrix = cgl_interpolation_matrix(order=3, num_samples=17)
            np.testing.assert_allclose(
                matrix @ np.ones(4, dtype=np.float32),
                np.ones(17, dtype=np.float32),
                atol=1e-6,
            )

        def test_posture_is_bounded_between_cgl_nodes(self):
            from optim.paper_psoc_lm import PaperPSOCLM

            solver = object.__new__(PaperPSOCLM)
            solver.order = 3
            matrix = torch.as_tensor(
                cgl_interpolation_matrix(order=3, num_samples=41)
            )
            # Alternating large logits provoke polynomial overshoot before
            # sigmoid, but every decoded physical point must remain bounded.
            decision = torch.tensor(
                [
                    -8.0,
                    8.0,
                    -8.0,
                    8.0,
                    8.0,
                    -8.0,
                    8.0,
                    -8.0,
                    -6.0,
                    6.0,
                    -6.0,
                    6.0,
                ]
            )
            posture, _ = solver._decode(
                decision,
                [matrix],
                [np.arange(41)],
                41,
            )
            lower = torch.tensor(PAPER_POSTURE_MIN)
            upper = torch.tensor(PAPER_POSTURE_MAX)
            self.assertTrue(torch.all(posture >= lower))
            self.assertTrue(torch.all(posture <= upper))

        def test_render_densification_preserves_endpoints_and_pose_gradient(self):
            from types import SimpleNamespace

            from models.paper_fusion_renderer import (
                PaperDynamicConfig,
                PaperFusionRenderer,
            )

            fake_renderer = SimpleNamespace(
                dynamic=PaperDynamicConfig(render_max_step_px=1.5)
            )
            xy = torch.tensor([[0.0, 0.0], [6.0, 0.0]])
            posture = torch.tensor(
                [[11.0, 0.0, 0.0], [20.0, 0.1, 0.05]],
                requires_grad=True,
            )
            stroke_ids = torch.tensor([0, 0])
            dense_xy, dense_posture, dense_ids = (
                PaperFusionRenderer.densify_for_rendering(
                    fake_renderer, xy, posture, stroke_ids
                )
            )
            self.assertEqual(len(dense_xy), 5)
            torch.testing.assert_close(dense_xy[0], xy[0])
            torch.testing.assert_close(dense_xy[-1], xy[-1])
            self.assertTrue(torch.all(dense_ids == 0))
            dense_posture.sum().backward()
            self.assertTrue(torch.isfinite(posture.grad).all())

except ImportError:
    pass


if __name__ == "__main__":
    unittest.main()
