"""Run under the model Python environment, no ROS node or motion."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
               '/home/robot/coppeliasim/machine_learning/model']
from fusion_model_ros2_beta.joint_optimizer import PaperPSOCLM


class Renderer:
    def __init__(self, gain):
        self.bbsmg = torch.nn.Linear(1,1)
        self.dynamic = SimpleNamespace(longitudinal_scale=.1655, transverse_scale=.1655)
        self.image_size = 128
        self.gain = gain
    def __call__(self, xy, pose, stroke, gamma):
        return torch.ones(1,1,8,8)*gamma.sum()*self.gain


class GammaGuardTest(unittest.TestCase):
    def test_equal_scale_but_gamma_sensitive_allowed(self):
        PaperPSOCLM(Renderer(1.), optimize_gamma=True)
    def test_unobservable_rotation_rejected(self):
        with self.assertRaisesRegex(ValueError, 'no usable forward response'):
            PaperPSOCLM(Renderer(0.), optimize_gamma=True)
    def test_nonfinite_response_rejected(self):
        with self.assertRaisesRegex(ValueError, 'no usable forward response'):
            PaperPSOCLM(Renderer(float('nan')), optimize_gamma=True)


if __name__ == '__main__':
    unittest.main()
