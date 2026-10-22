import math
import unittest

from fusion_model_ros2_beta.brush_trajectory_driver import (
    SourcePoint,
    _brush_rotation,
    _flange_target,
    place_physical_source_points,
)


def source_point(
    point_id: int,
    x: float,
    y: float,
    *,
    z_mm: float = 15.0,
    alpha: float = 0.0,
    beta: float = 0.0,
    gamma: float = 0.0,
    state: int = 1,
) -> SourcePoint:
    return SourcePoint(
        character="一", sample_id="一_test", stroke_id=0, point_id=point_id,
        x=x, y=y, z_mm=z_mm, alpha=alpha, beta=beta, gamma=gamma, state=state,
    )


def placed(source):
    return place_physical_source_points(
        source,
        paper_z=0.0,
        paper_offset_x=0.10,
        paper_offset_y=0.65,
        max_compression=0.02,
        lift_height=0.025,
    )


class PhysicalTrajectoryPlacementTest(unittest.TestCase):
    def test_ros_only_translates_offline_xy(self):
        points = placed([
            source_point(0, -0.05, -0.01),
            source_point(1, 0.05, -0.009999),
        ])
        self.assertAlmostEqual(points[0].x, 0.05)
        self.assertAlmostEqual(points[-1].x, 0.15)
        self.assertAlmostEqual(points[-1].x - points[0].x, 0.10)
        self.assertAlmostEqual(points[0].y, 0.64)
        self.assertAlmostEqual(points[-1].y, 0.640001)
        self.assertTrue(all(point.footprint_scale == 1.0 for point in points))

    def test_full_offline_pose_is_not_recomputed(self):
        point = placed([source_point(
            0, 0.0, 0.0, z_mm=16.0, alpha=0.08, beta=-0.04, gamma=0.20,
        )])[0]
        self.assertAlmostEqual(point.press_depth_mm, 16.0)
        self.assertAlmostEqual(point.z, -0.016)
        self.assertAlmostEqual(point.alpha, 0.08)
        self.assertAlmostEqual(point.beta, -0.04)
        self.assertAlmostEqual(point.gamma, 0.20)
        tilted = _brush_rotation(point.alpha, point.beta, point.gamma)
        vertical = _brush_rotation(0.0, 0.0, point.gamma)
        self.assertGreater(abs(tilted - vertical).sum(), 1.0e-4)
        self.assertNotAlmostEqual(
            _flange_target(point, 0.108)[2, 3],
            _flange_target(placed([source_point(0, 0.0, 0.0)])[0], 0.108)[2, 3],
        )

    def test_air_point_uses_lift_without_compression(self):
        point = placed([source_point(0, 0.0, 0.0, z_mm=99.0, state=3)])[0]
        self.assertAlmostEqual(point.z, 0.025)
        self.assertAlmostEqual(point.press_depth_mm, 0.0)

    def test_unsafe_offline_compression_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside ROS safety range"):
            placed([source_point(0, 0.0, 0.0, z_mm=21.0)])

    def test_gamma_is_only_wrapped_not_rederived_from_path(self):
        point = placed([
            source_point(0, 0.0, 0.0, gamma=2.0 * math.pi + 0.3)
        ])[0]
        self.assertAlmostEqual(point.gamma, 0.3)


if __name__ == "__main__":
    unittest.main()
