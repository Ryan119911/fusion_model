"""Small CPU probe of the actual Beta posture decoder at a physical bound."""
import sys
from pathlib import Path
import unittest
import numpy as np
import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
               '/home/robot/coppeliasim/machine_learning/model']
from fusion_model_ros2_beta.joint_optimizer import PaperPSOCLM, difference_column, project_posture_decisions
from verify_joint_optimizer_guard import Renderer


class BoundDerivativeTest(unittest.TestCase):
    def test_projection_restores_inward_response_on_plateau(self):
        optimizer = PaperPSOCLM(Renderer(1.), order=1)
        matrices, indices, _, _ = optimizer._build_layout(np.array([0, 0]))
        vector = torch.tensor([20., 20., 0., 0., 0., 0., 8., -9.])
        original = vector.clone()
        def decode(v):
            return optimizer._decode(v[:6], matrices, indices, 2)[0][:, 0]
        old = difference_column(decode, vector, 0, .05, decode(vector), 'central')
        projected = project_posture_decisions(vector, 6, optimizer.POSTURE_BOUND_EPS)
        new = difference_column(decode, projected, 0, .05, decode(projected), 'central')
        self.assertEqual(float(torch.linalg.vector_norm(old)), 0.)
        self.assertGreater(float(torch.linalg.vector_norm(new)), .01)
        torch.testing.assert_close(decode(vector), decode(projected))
        torch.testing.assert_close(vector, original)
        torch.testing.assert_close(projected[6:], vector[6:])
        torch.testing.assert_close(project_posture_decisions(projected, 6, .02), projected)

    def test_upper_bound_forward_difference_misses_inward_response(self):
        optimizer = PaperPSOCLM(Renderer(1.), order=1)
        matrices, indices, _, _ = optimizer._build_layout(np.array([0, 0]))
        vector = torch.zeros(6)
        eps = optimizer.POSTURE_BOUND_EPS
        vector[:2] = float(np.log((1-eps)/eps))
        def decode(v):
            return optimizer._decode(v, matrices, indices, 2)[0][:, 0]
        step = .01 * (1 + abs(float(vector[0])))
        forward = difference_column(decode, vector, 0, step, decode(vector), 'forward')
        central = difference_column(decode, vector, 0, step, decode(vector), 'central')
        print(dict(forward=forward.tolist(), central=central.tolist()))
        self.assertLess(float(torch.linalg.vector_norm(forward)), 1e-3)
        self.assertGreater(float(torch.linalg.vector_norm(central)), .01)

    def test_linear_response_and_no_mutation(self):
        vector = torch.tensor([2., 3.])
        original = vector.clone()
        fn = lambda v: 3*v
        for scheme in ('forward', 'central'):
            result = difference_column(fn, vector, 0, .01, fn(vector), scheme)
            torch.testing.assert_close(result, torch.tensor([3., 0.]), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(vector, original)
        with self.assertRaises(ValueError):
            difference_column(fn, vector, 0, 0., fn(vector), 'central')


if __name__ == '__main__':
    unittest.main()
