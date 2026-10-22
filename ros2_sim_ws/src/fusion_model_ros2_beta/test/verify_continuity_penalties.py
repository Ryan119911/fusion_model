"""CPU checks for the penalties used by isolated A-reference experiments."""
import sys
from pathlib import Path
import unittest
import numpy as np
import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
               '/home/robot/coppeliasim/machine_learning/model']
from fusion_model_ros2_beta.joint_optimizer import (
    trajectory_difference_residuals, trajectory_shape_residuals, angle_point_residuals,
    spatial_depth_residuals,
)


def cost(parts):
    return sum(float((r*r).sum()) for r in parts)


class ContinuityTest(unittest.TestCase):
    def test_spatial_depth_linear_sampling_invariance(self):
        a = spatial_depth_residuals(torch.tensor([[0., 0.], [10., 0.]]),
                                   torch.tensor([0., 2.]), [np.arange(2)], 10., 1.)
        b = spatial_depth_residuals(torch.tensor([[0., 0.], [5., 0.], [10., 0.]]),
                                   torch.tensor([0., 1., 2.]), [np.arange(3)], 10., 1.)
        self.assertAlmostEqual(cost(a), cost(b), places=5)

    def test_short_depth_change_costs_more(self):
        xy = torch.tensor([[0., 0.], [1., 0.]])
        h = torch.tensor([0., 9.])
        self.assertGreater(cost(spatial_depth_residuals(xy, h, [np.arange(2)], 10., 1.)),
                           cost(spatial_depth_residuals(10*xy, h, [np.arange(2)], 10., 1.)))

    def test_spatial_depth_zero_length_and_stroke_boundary(self):
        xy = torch.zeros((2, 2), requires_grad=True)
        h = torch.tensor([0., 1.], requires_grad=True)
        parts = spatial_depth_residuals(xy, h, [np.arange(2)], 10., 1.)
        sum((r*r).sum() for r in parts).backward()
        self.assertTrue(torch.isfinite(xy.grad).all())
        self.assertTrue(torch.isfinite(h.grad).all())
        self.assertEqual(spatial_depth_residuals(xy, h, [np.array([0]), np.array([1])], 10., 1.), [])
        self.assertEqual(spatial_depth_residuals(xy, h, [np.arange(2)], 0., 1.), [])
        with self.assertRaises(ValueError):
            spatial_depth_residuals(xy, h, [np.arange(2)], 1., 0.)

    def test_periodic_angle_wrap(self):
        angles = torch.tensor([np.pi-.01, -np.pi+.01])
        self.assertLess(cost(angle_point_residuals(angles, [np.arange(2)], 10.)), .005)

    def test_angle_jump_and_gradient(self):
        angles = torch.tensor([0., 1.], requires_grad=True)
        residuals = angle_point_residuals(angles, [np.arange(2)], 10.)
        loss = sum((r*r).sum() for r in residuals)
        self.assertGreater(float(loss), 1.)
        loss.backward()
        self.assertTrue(torch.isfinite(angles.grad).all())

    def test_angle_off_and_stroke_separation(self):
        angles = torch.tensor([0., 1.])
        self.assertEqual(angle_point_residuals(angles, [np.arange(2)], 0.), [])
        self.assertEqual(angle_point_residuals(angles, [np.array([0]), np.array([1])], 10.), [])
        for weight in (-1., float('nan')):
            with self.assertRaises(ValueError):
                angle_point_residuals(angles, [np.arange(2)], weight)

    def test_translation_unpenalized(self):
        xy = torch.tensor([[0., 0.], [1., 0.], [2., 1.]])
        self.assertAlmostEqual(cost(trajectory_shape_residuals(
            xy, xy+4, [np.arange(3)], 3., 10.)), 0.)

    def test_reversal_penalized(self):
        xy = torch.tensor([[0., 0.], [1., 0.]])
        self.assertGreater(cost(trajectory_shape_residuals(
            xy, -xy, [np.arange(2)], 3., 10.)), 0.)

    def test_depth_jump_penalized(self):
        strokes = [np.arange(3)]
        constant = cost(trajectory_difference_residuals(torch.ones(3), strokes, 10., 10.))
        jump = cost(trajectory_difference_residuals(torch.tensor([0., 1., 0.]), strokes, 10., 10.))
        self.assertEqual(constant, 0.)
        self.assertGreater(jump, 0.)

    def test_lift_between_strokes_unpenalized(self):
        values = torch.tensor([0., 0., 1., 1.])
        self.assertEqual(cost(trajectory_difference_residuals(
            values, [np.array([0, 1]), np.array([2, 3])], 10., 10.)), 0.)


if __name__ == '__main__':
    unittest.main()
