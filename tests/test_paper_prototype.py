import unittest

import numpy as np

from models.geometry import CanvasTransform
from models.paper_bbsm import (
    PAPER_ANGLE_BASIS_DEGREE_FITTED,
    PAPER_POSTURE_MAX,
    PAPER_POSTURE_MIN,
    PAPER_REGRESSION_MATRIX,
    bbsm_boundary,
    posture_to_geometry_numpy,
    render_bbsm_mask,
)
from models.paper_calibration import (
    WANG2020_DRAG_COEFFICIENTS,
    WANG2020_OFFSET_COEFFICIENTS,
    WANG2020_PROFILE,
    WANG2020_WIDTH_COEFFICIENTS,
    bbsm_h_to_wang_height_numpy,
    paper_calibration_metadata,
    wang2020_curves_numpy,
)
from tools.build_paper_bbsmg_dataset import build_dataset


class CanvasTransformTests(unittest.TestCase):
    def test_map_unmap_round_trip(self):
        transform = CanvasTransform(
            src_min_x=100.0,
            src_max_x=700.0,
            src_min_y=-50.0,
            src_max_y=850.0,
            dst_size=128,
            padding=16,
        )
        source = (321.5, 456.25)
        canvas = transform.map_point(*source)
        restored = transform.unmap_point(*canvas)
        np.testing.assert_allclose(restored, source, atol=1e-6)


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

    def test_degree_fitted_basis_keeps_external_angles_in_radians(self):
        posture_zero = np.asarray([[11.0, 0.0, 0.0]], dtype=np.float32)
        posture_ten_deg = np.asarray(
            [[11.0, np.deg2rad(10.0), 0.0]], dtype=np.float32
        )
        delta = (
            posture_to_geometry_numpy(
                posture_ten_deg,
                angle_basis=PAPER_ANGLE_BASIS_DEGREE_FITTED,
            )
            - posture_to_geometry_numpy(
                posture_zero,
                angle_basis=PAPER_ANGLE_BASIS_DEGREE_FITTED,
            )
        )[0]
        np.testing.assert_allclose(
            delta,
            PAPER_REGRESSION_MATRIX[:, 1] * 10.0,
            rtol=1e-6,
            atol=1e-6,
        )

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

    def test_degree_fitted_dataset_is_versioned_separately(self):
        _, _, metadata = build_dataset(
            count=2,
            image_size=128,
            pixels_per_model_unit=20.0,
            supersample=1,
            seed=7,
            regression_angle_basis=PAPER_ANGLE_BASIS_DEGREE_FITTED,
        )
        self.assertEqual(metadata["format"], "paper_bbsmg_degree_fitted_v2")
        self.assertEqual(
            metadata["input_normalization"]["regression_angle_basis"],
            PAPER_ANGLE_BASIS_DEGREE_FITTED,
        )
        self.assertEqual(metadata["units"]["alpha"], "rad")

    def test_wang_figure4_digitization_is_versioned_and_bounded(self):
        heights = bbsm_h_to_wang_height_numpy(
            np.asarray([11.0, 15.5, 20.0], dtype=np.float32)
        )
        np.testing.assert_allclose(heights, [0.0, 0.75, 1.5], atol=1e-6)
        curves = wang2020_curves_numpy(heights)
        self.assertTrue(np.all(curves["width_cm"] >= 0.0))
        self.assertTrue(np.all(curves["drag_cm"] > 0.0))
        self.assertTrue(np.all(curves["offset_cm"] >= 0.0))
        # The plotted offset fit has the local maximum described in the text.
        self.assertGreater(curves["offset_cm"][1], curves["offset_cm"][2])
        metadata = paper_calibration_metadata()
        self.assertEqual(metadata["profile"], WANG2020_PROFILE)
        self.assertIsNone(
            metadata["unreported_values"][
                "wang_polynomial_author_coefficients"
            ]
        )
        self.assertEqual(
            tuple(
                metadata["figure4_digitized_approximation"][
                    "width_coefficients"
                ]
            ),
            WANG2020_WIDTH_COEFFICIENTS,
        )
        self.assertEqual(
            tuple(
                metadata["figure4_digitized_approximation"][
                    "drag_coefficients"
                ]
            ),
            WANG2020_DRAG_COEFFICIENTS,
        )
        self.assertEqual(
            tuple(
                metadata["figure4_digitized_approximation"][
                    "offset_coefficients"
                ]
            ),
            WANG2020_OFFSET_COEFFICIENTS,
        )


try:
    import torch  # noqa: F401
    import torch.nn as nn

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

        def test_xy_offsets_are_bounded_between_cgl_nodes(self):
            from optim.paper_psoc_lm import PaperPSOCLM

            solver = object.__new__(PaperPSOCLM)
            solver.order = 3
            solver.xy_max_offset_px = 6.0
            matrix = torch.as_tensor(
                cgl_interpolation_matrix(order=3, num_samples=41)
            )
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
                ]
            )
            offsets, _ = solver._decode_xy_offsets(
                decision,
                [matrix],
                [np.arange(41)],
                41,
            )
            self.assertTrue(torch.all(offsets >= -6.0))
            self.assertTrue(torch.all(offsets <= 6.0))

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

        def test_anisotropic_footprint_scale_keeps_legacy_fallback(self):
            from models.paper_fusion_renderer import PaperDynamicConfig

            legacy = PaperDynamicConfig(footprint_scale=0.22)
            self.assertAlmostEqual(legacy.longitudinal_scale, 0.22)
            self.assertAlmostEqual(legacy.transverse_scale, 0.22)
            anisotropic = PaperDynamicConfig(
                footprint_scale=0.22,
                footprint_longitudinal_scale=0.21,
                footprint_transverse_scale=0.26,
            )
            self.assertAlmostEqual(anisotropic.longitudinal_scale, 0.21)
            self.assertAlmostEqual(anisotropic.transverse_scale, 0.26)

        def test_wang_root_sticks_then_snaps(self):
            from models.paper_fusion_renderer import (
                PaperDynamicConfig,
                PaperFusionRenderer,
            )

            renderer = object.__new__(PaperFusionRenderer)
            nn.Module.__init__(renderer)
            renderer.regression_angle_basis = "paper_declared_radian"
            renderer.dynamic = PaperDynamicConfig(
                calibration_profile=WANG2020_PROFILE,
                footprint_scale=0.22,
            )
            xy = torch.tensor(
                [[10.0, 10.0], [11.0, 10.0], [40.0, 10.0]]
            )
            posture = torch.tensor(
                [[15.5, 0.0, 0.0]] * 3, requires_grad=True
            )
            states = renderer.compute_dynamic_states(
                xy, posture, torch.zeros(3, dtype=torch.long)
            )
            # Friction holds the first root for the short movement.
            torch.testing.assert_close(
                states["contact_xy"][0], states["contact_xy"][1], atol=1e-5, rtol=0
            )
            # The long movement exceeds free offset and snaps the root.
            self.assertGreater(
                float(
                    torch.linalg.vector_norm(
                        states["contact_xy"][2] - states["contact_xy"][1]
                    )
                ),
                1.0,
            )
            states["contact_xy"].sum().backward()
            self.assertTrue(torch.isfinite(posture.grad).all())

        def test_observability_gate_fixes_unobservable_angles(self):
            from optim.paper_psoc_lm import PaperPSOCLM

            class HeightOnlyRenderer(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.bbsmg = nn.Linear(1, 1, bias=False)

                def forward(self, xy, posture, stroke_ids):
                    value = torch.sigmoid((posture[:, 0].mean() - 15.5) / 2.0)
                    return value.expand(1, 1, 8, 8)

            solver = PaperPSOCLM(
                HeightOnlyRenderer(),
                order=1,
                optimization_size=8,
                field_mode="auto",
                min_relative_median_sensitivity=0.35,
            )
            result = solver.optimize(
                xy_canvas=np.asarray(
                    [[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]],
                    dtype=np.float32,
                ),
                stroke_ids=np.zeros(3, dtype=np.int64),
                target_image=np.full((8, 8), 0.8, dtype=np.float32),
                max_steps=1,
            )
            gate = result.diagnostics["observability_gate"]
            self.assertEqual(gate["optimized_fields"], ["H"])
            self.assertEqual(gate["fixed_fields"], ["alpha", "beta"])
            np.testing.assert_allclose(result.posture[:, 1:], 0.0, atol=0.0)
            self.assertEqual(
                result.diagnostics["field_decisions"]["alpha"]["source"],
                "initial_default",
            )

        def test_bounded_xy_optimization_moves_toward_target(self):
            from optim.paper_psoc_lm import PaperPSOCLM

            class XYRenderer(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.bbsmg = nn.Linear(1, 1, bias=False)
                    yy, xx = torch.meshgrid(
                        torch.arange(8, dtype=torch.float32),
                        torch.arange(8, dtype=torch.float32),
                        indexing="ij",
                    )
                    self.register_buffer("xx", xx)
                    self.register_buffer("yy", yy)

                def forward(self, xy, posture, stroke_ids):
                    center = xy.mean(dim=0)
                    image = torch.exp(
                        -(
                            (self.xx - center[0]) ** 2
                            + (self.yy - center[1]) ** 2
                        )
                        / 2.0
                    )
                    return image.view(1, 1, 8, 8)

            yy, xx = np.mgrid[:8, :8].astype(np.float32)
            target = np.exp(-((xx - 4.0) ** 2 + (yy - 4.0) ** 2) / 2.0)
            solver = PaperPSOCLM(
                XYRenderer(),
                order=1,
                optimization_size=8,
                field_mode="xy_only",
                optimize_xy=True,
                xy_max_offset_px=3.0,
                xy_smoothness_weight=0.01,
                xy_prior_weight=0.001,
            )
            initial_xy = np.asarray(
                [[2.0, 4.0], [2.0, 4.0]], dtype=np.float32
            )
            result = solver.optimize(
                xy_canvas=initial_xy,
                stroke_ids=np.zeros(2, dtype=np.int64),
                target_image=target,
                max_steps=5,
                pixel_weight=0.0,
            )
            self.assertGreater(result.xy_canvas[:, 0].mean(), 2.0)
            self.assertLessEqual(
                float(np.abs(result.xy_canvas - initial_xy).max()),
                3.0 + 1e-6,
            )
            self.assertTrue(
                result.diagnostics["xy_optimization"]["enabled"]
            )
            np.testing.assert_allclose(
                result.posture[:, 0], 15.5, atol=0.0
            )
            self.assertEqual(
                result.diagnostics["observability_gate"]["optimized_fields"],
                ["x", "y"],
            )

except ImportError:
    pass


if __name__ == "__main__":
    unittest.main()
