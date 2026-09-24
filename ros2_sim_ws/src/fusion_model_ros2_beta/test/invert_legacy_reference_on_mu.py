"""Isolated same-size inversion experiment; never writes production caches."""
import json
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import hashlib

parser = argparse.ArgumentParser()
parser.add_argument('--root', default='/home/robot/ros2_ws/evaluation/legacy_recovery_20260919/area_400')
parser.add_argument('--joint', action='store_true')
parser.add_argument('--character-directory', default=None)
parser.add_argument('--metric-units', action='store_true')
parser.add_argument('--steps', type=int, default=15)
parser.add_argument('--spatial-depth', action='store_true')
parser.add_argument('--central-difference', action='store_true')
parser.add_argument('--warm-start-v3', action='store_true')
parser.add_argument('--warm-start-v5', action='store_true')
parser.add_argument('--warm-start-v6', action='store_true')
parser.add_argument('--warm-start-suffix', default=None,
                    help='resume controls from a validated same-size experiment directory')
parser.add_argument('--foreground-weight', type=float, default=3.0,
                    help='target pixel emphasis in model inversion, not rendered width')
parser.add_argument('--project-posture-logits', action='store_true')
parser.add_argument('--balanced-pose', action='store_true',
                    help='less foreground-biased loss plus decoded periodic angle continuity')
parser.add_argument('--regularized', action='store_true',
                    help='experimental shape and continuity penalties, not robot limits')
args = parser.parse_args()
if sum((args.warm_start_v3, args.warm_start_v5, args.warm_start_v6, bool(args.warm_start_suffix))) > 1:
    parser.error('choose one warm-start source')
if args.warm_start_suffix:
    import re
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', args.warm_start_suffix):
        parser.error('warm-start suffix must be a plain experiment directory name')
    if not (args.project_posture_logits and args.central_difference and args.balanced_pose):
        parser.error('resumed experiments require projected balanced central mode')
if not 0 <= args.foreground_weight <= 12:
    parser.error('--foreground-weight must be finite and between 0 and 12')
if args.foreground_weight != 3.0 and not args.balanced_pose:
    parser.error('--foreground-weight requires --balanced-pose')
if args.warm_start_v6 and not (args.project_posture_logits and args.central_difference and args.balanced_pose):
    parser.error('--warm-start-v6 requires projected balanced central mode')
if (args.project_posture_logits or args.warm_start_v5) and not (args.balanced_pose and args.central_difference):
    parser.error('projected/warm-v5 experiments require balanced central mode')
if (args.central_difference or args.warm_start_v3) and not args.balanced_pose:
    parser.error('central/warm-start experiments require --balanced-pose')
if args.spatial_depth and not args.balanced_pose:
    parser.error('--spatial-depth requires --balanced-pose')
if args.balanced_pose and not args.regularized:
    parser.error('--balanced-pose requires --regularized')
if args.regularized and not (args.joint and args.metric_units):
    parser.error('--regularized requires --joint --metric-units')
root = Path(args.root)
script = Path('/home/robot/ros2_ws/src/fusion_model_ros2_beta/fusion_model_ros2_beta/export_neural_ink.py')
for char in ([args.character_directory] if args.character_directory else ('char_u6b66', 'char_u6c49')):
    baseline = root/char
    suffix = f'same_size_joint_metric_guard_v2_{args.steps}' if args.metric_units else 'same_size_joint_15'
    if args.regularized:
        suffix += '_continuity_v2_objective_selection'
    if args.balanced_pose:
        suffix += '_balanced_pose_v3'
    if args.spatial_depth:
        suffix += '_spatial_depth_v4'
    if args.central_difference:
        suffix += '_central_v5'
    if args.warm_start_v3:
        suffix += '_warm_v3'
    if args.project_posture_logits:
        suffix += '_projected_v6'
    if args.warm_start_v5:
        suffix += '_warm_v5'
    if args.warm_start_v6:
        suffix += '_warm_v6'
    if args.foreground_weight != 3.0:
        suffix += '_fg_' + format(args.foreground_weight, 'g').replace('.', 'p')
    if args.warm_start_suffix:
        suffix += '_resume_' + hashlib.sha256(args.warm_start_suffix.encode()).hexdigest()[:10]
    output = baseline/(suffix if args.joint else 'same_size_inversion_60')
    output.mkdir(exist_ok=False)
    metadata = json.loads((baseline/'offline_inversion.json').read_text())
    cmd = metadata['command'][:]
    for flag, value in {
        '--trajectory_csv': metadata['source_trajectory'],
        '--initial_pose_csv': str(baseline/'inversion_trajectory.csv'),
        '--target_image': str(baseline/'scaled_target.png'),
        '--character': metadata['character'], '--sample_id': metadata['sample_id'],
        '--output_dir': str(output), '--max_steps': '60', '--optimization_size': '128',
    }.items():
        cmd[cmd.index(flag)+1] = value
    if args.warm_start_v3:
        initial = baseline/'same_size_joint_metric_guard_v2_15_continuity_v2_objective_selection_balanced_pose_v3'/'inversion_trajectory.csv'
        cmd[cmd.index('--initial_pose_csv')+1] = str(initial)
        metadata['warm_start_sha256'] = hashlib.sha256(initial.read_bytes()).hexdigest()
    if args.warm_start_v5:
        initial = baseline/'same_size_joint_metric_guard_v2_8_continuity_v2_objective_selection_balanced_pose_v3_spatial_depth_v4_central_v5_warm_v3'/'inversion_trajectory.csv'
        cmd[cmd.index('--initial_pose_csv')+1] = str(initial)
        metadata['warm_start_sha256'] = hashlib.sha256(initial.read_bytes()).hexdigest()
    if args.warm_start_v6:
        initial = baseline/'same_size_joint_metric_guard_v2_8_continuity_v2_objective_selection_balanced_pose_v3_spatial_depth_v4_central_v5_projected_v6_warm_v5'/'inversion_trajectory.csv'
        cmd[cmd.index('--initial_pose_csv')+1] = str(initial)
        metadata['warm_start_sha256'] = hashlib.sha256(initial.read_bytes()).hexdigest()
    if args.warm_start_suffix:
        seed = baseline/args.warm_start_suffix
        seed_meta = json.loads((seed/'offline_inversion.json').read_text())
        for key in ('character', 'sample_id', 'font_size_m', 'reference_font_size_m', 'source_trajectory'):
            if seed_meta[key] != metadata[key]:
                raise ValueError('warm-start source mismatch: '+key)
        if (seed/'scaled_target.png').read_bytes() != (baseline/'scaled_target.png').read_bytes():
            raise ValueError('warm-start target mismatch')
        seed_audit = json.loads((seed/'neural_ink_audit.json').read_text())
        if seed_audit['checkpoint_sha256'] != '30d5e0c37dc7b7913c02fea26930babafa46a4a07b5f698520df277a04d5b717':
            raise ValueError('warm-start checkpoint is not pinned V16')
        seed_controls = json.loads((seed/'control_change_audit.json').read_text())
        if not seed_controls['candidate_checks']['eligible_for_further_validation']:
            raise ValueError('warm-start source failed geometry validation')
        initial = seed/'inversion_trajectory.csv'
        cmd[cmd.index('--initial_pose_csv')+1] = str(initial)
        metadata['warm_start_source'] = str(seed)
        metadata['warm_start_sha256'] = hashlib.sha256(initial.read_bytes()).hexdigest()
    if args.joint:
        cmd.remove('--fused_pose_from_height')
        cmd[cmd.index('--field_mode')+1] = 'all'
        cmd[cmd.index('--max_steps')+1] = str(args.steps)
        cmd.extend(['--optimize_xy', '--xy_max_offset_px', '2.0', '--optimize_gamma'])
    if args.metric_units:
        cmd[2] = '/home/robot/ros2_ws/src/fusion_model_ros2_beta/fusion_model_ros2_beta/run_joint_inversion.py'
        scale = .01/(20*(.29/(128-32)))
        for flag in ('--footprint_longitudinal_scale','--footprint_transverse_scale'):
            cmd[cmd.index(flag)+1] = str(scale)
        metadata['unit_bridge'] = {'regression_unit_m': .01, 'pixel_size_m': .29/96,
                                  'training_pixels_per_unit':20, 'scale':scale,
                                  'scope':'legacy simulation unit convention, not hardware calibration'}
    metadata['command'] = cmd
    if args.central_difference:
        cmd.extend(['--beta_difference_scheme', 'central'])
    if args.project_posture_logits:
        cmd.append('--beta_project_posture_logits')
    if args.regularized:
        # Fixed experimental weights, not calibrated physical speed limits.
        penalties = {
            '--xy_segment_length_weight': 3.0,
            '--xy_segment_direction_weight': 10.0,
            '--h_point_velocity_weight': 10.0,
            '--h_point_acceleration_weight': 10.0,
            '--alpha_smoothness_weight': 1.0,
            '--beta_smoothness_weight': 1.0,
            '--gamma_smoothness_weight': 1.0,
        }
        for flag, value in penalties.items():
            if flag in cmd:
                cmd[cmd.index(flag)+1] = str(value)
            else:
                cmd.extend([flag, str(value)])
        metadata['experimental_continuity_weights'] = penalties
    if args.balanced_pose:
        cmd[cmd.index('--pixel_weight')+1] = str(args.foreground_weight)
        cmd.extend(['--beta_angle_point_weight', '10.0'])
        metadata['balanced_pose_experiment'] = dict(pixel_weight=args.foreground_weight, angle_point_weight=10.0)
    if args.spatial_depth:
        mm_per_pixel = 1000*.29/96
        cmd.extend(['--beta_depth_spatial_weight', '10.0',
                    '--beta_depth_mm_per_pixel', str(mm_per_pixel)])
        metadata['spatial_depth_experiment'] = dict(weight=10.0, mm_per_pixel=mm_per_pixel,
                                                  length_floor_mm=.25, physical_velocity_limit=False)
    implementation = Path(__file__).resolve().parents[1]/'fusion_model_ros2_beta'
    metadata['launch_implementation_sha256'] = {
        name: hashlib.sha256((implementation/name).read_bytes()).hexdigest()
        for name in ('joint_optimizer.py', 'run_joint_inversion.py')
    }
    (output/'experiment_request.json').write_text(json.dumps(metadata, indent=2))
    with (output/'process.log').open('w') as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
    sys.path.insert(0, '/home/robot/ros2_ws/src/fusion_model_ros2_beta')
    from fusion_model_ros2_beta.offline_fontsize_inversion import export_physical_trajectory, _read_pose_rows, _xy_reference
    _, rows = _read_pose_rows(Path(metadata['source_trajectory']))
    x, y, span = _xy_reference(rows)
    export_physical_trajectory(output/'inversion_trajectory.csv', output/'physical_trajectory.csv',
        source_center_x=x, source_center_y=y, source_span=span,
        reference_font_size_m=.29, font_size_m=metadata['font_size_m'])
    shutil.copyfile(baseline/'scaled_target.png', output/'scaled_target.png')
    (output/'offline_inversion.json').write_text(json.dumps(metadata, indent=2))
    if not args.joint:
        subprocess.run([sys.executable, str(script), '--model-root', '/home/robot/coppeliasim/machine_learning/model',
                        '--output-dir', str(output)], check=True)
    else:
        report = json.loads((output/'inversion_report.json').read_text())
        print(json.dumps({'character':metadata['character'], 'metrics':report['metrics'],
                          'optimized_fields':report['optimized_fields'], 'lm':report['lm']['message']}))
