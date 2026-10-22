"""Generate and plan the original-target UR10 case without publishing motion."""
import argparse
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fusion_model_ros2_beta.offline_fontsize_inversion import OfflineInversionConfig
from fusion_model_ros2_beta.inversion_backend import make_inversion_backend
from fusion_model_ros2_beta.trajectory_catalog import TrajectoryEntry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model-root', default='/home/robot/coppeliasim/machine_learning/model')
    parser.add_argument('--python', default='/home/robot/miniconda3/envs/ddpm/bin/python')
    parser.add_argument('--size', type=float, default=.29)
    parser.add_argument('--steps', type=int, default=16)
    parser.add_argument('--candidate', help='Audit and plan an existing exact-size, in-domain candidate')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = dict(size_m=args.size, steps=args.steps, motion_published=False, passed=False)
    try:
        root = Path(args.model_root)
        config = OfflineInversionConfig(model_root=root, python_executable=Path(args.python),
            bbsmg_checkpoint=root/'outputs/paper_bbsmg_general_v16_pose_dense/bbsmg_best.pt',
            cache_root=output/'cache', timeout_s=7200, order=11)
        backend = make_inversion_backend(config, backend='joint_target',
            original_target_manifest=args.manifest, joint_steps=args.steps)
        record = next(iter(backend.references.values()))
        source = TrajectoryEntry(character=record['character'], sample_id=record['sample_id'],
            status='ready', trajectory_csv=Path(record['source_trajectory']))
        if args.candidate:
            from fusion_model_ros2_beta.joint_candidate import load_joint_candidate
            candidate = load_joint_candidate(args.candidate, require_training_domain=True)
            if abs(candidate.metadata['font_size_m'] - args.size) > 1e-9:
                raise ValueError('candidate physical size does not match requested size')
        else:
            candidate = backend.generate(source, args.size, progress=print)
        report['candidate'] = str(candidate.output_dir)
        report['neural_audit'] = json.loads((candidate.output_dir/'neural_ink_audit.json').read_text())
        (output/'generation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
        import rclpy
        from fusion_model_ros2_beta.brush_trajectory_driver import BrushTrajectoryDriver
        rclpy.init(args=['--ros-args', '-p', 'enable_input_page:=false',
            '-p', 'strict_ik:=true', '-p', 'publish_on_start:=false'])
        node = BrushTrajectoryDriver()
        try:
            node._build_targets(entries=[candidate], layout_mode='horizontal', font_size_m=args.size)
            report['targets'] = len(node._targets)
            report['duration_s'] = sum(t.duration_s for t in node._targets)
            report['trajectory'] = [dict(positions=list(t.joints), duration_s=t.duration_s) for t in node._targets]
            report['passed'] = True
        finally:
            node.destroy_node()
            rclpy.shutdown()
    except Exception as exc:
        report['error'] = str(exc)
        traceback.print_exc()
    finally:
        (output/'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
