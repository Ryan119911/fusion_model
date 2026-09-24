"""Use the Beta optimizer copy with the original model CLI and renderer."""
import runpy
import sys
import argparse
import os
from pathlib import Path

model_root = Path(os.environ.get('BETA_INVERSION_MODEL_ROOT', '/home/robot/coppeliasim/machine_learning/model'))
sys.path.insert(0, str(model_root))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fusion_model_ros2_beta.joint_optimizer import PaperPSOCLM
import optim.paper_psoc_lm as optimizer
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--beta_angle_point_weight', type=float, default=0.0)
parser.add_argument('--beta_depth_spatial_weight', type=float, default=0.0)
parser.add_argument('--beta_depth_mm_per_pixel', type=float, default=1.0)
parser.add_argument('--beta_difference_scheme', choices=['forward', 'central'], default='forward')
parser.add_argument('--beta_project_posture_logits', action='store_true')
parser.add_argument('--beta_neural_domain_weight', type=float, default=0.0)
parser.add_argument('--beta_hard_neural_domain', action='store_true')
beta_args, remaining = parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining


class ConfiguredBetaOptimizer(PaperPSOCLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, angle_point_weight=beta_args.beta_angle_point_weight,
                         depth_spatial_weight=beta_args.beta_depth_spatial_weight,
                         depth_mm_per_pixel=beta_args.beta_depth_mm_per_pixel,
                         project_posture_logits=beta_args.beta_project_posture_logits,
                         neural_domain_weight=beta_args.beta_neural_domain_weight,
                         hard_neural_domain=beta_args.beta_hard_neural_domain,
                         difference_scheme=beta_args.beta_difference_scheme, **kwargs)


optimizer.PaperPSOCLM = ConfiguredBetaOptimizer
runpy.run_path(str(model_root/'tools/invert_paper_trajectory.py'), run_name='__main__')
