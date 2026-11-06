#!/usr/bin/env python3
"""Fixed-direction brush calibration. Planning/analysis are offline; run is preview by default."""
import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
from scipy.spatial.transform import Rotation

from brush_buffered_motion import compile_program

ROOT = Path(__file__).resolve().parents[1]
CONVENTION = "Rot(paper_v,beta_exec) @ Rot(paper_u,alpha_exec) @ R0"


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write_csv(path, rows):
    if not rows:
        raise ValueError('no rows to export')
    with Path(path).open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_config(config):
    if config.get('schema') != 'brush_longline_experiment_v1':
        raise ValueError('unsupported experiment schema')
    g, m, inputs = config['geometry'], config['motion'], config['inputs']
    axes = np.column_stack([g[k] for k in ('paper_u_base', 'paper_v_base', 'paper_normal_base')])
    if axes.shape != (3, 3) or not np.isfinite(axes).all() or not np.allclose(axes.T @ axes, np.eye(3), atol=1e-6) or np.linalg.det(axes) < .999:
        raise ValueError('paper axes must be finite right-handed orthonormal unit vectors')
    for name in ('paper_origin_base_m', 'tip_tool_m', 'baseline_rotation_rad'):
        a = np.asarray(g[name], float)
        if a.shape != (3,) or not np.isfinite(a).all():
            raise ValueError('invalid geometry: ' + name)
    bounds = np.asarray(g['paper_bounds_uv_mm'], float)
    start = np.asarray(m['start_uv_mm'], float)
    if bounds.shape != (4,) or start.shape != (2,) or not np.isfinite(np.r_[bounds, start]).all() or not (bounds[0] < bounds[1] and bounds[2] < bounds[3]):
        raise ValueError('invalid paper bounds/start')
    for name, upper in [('length_mm', 1000), ('speed_mm_s', 30), ('sample_spacing_mm', 5),
                        ('standoff_mm', 100), ('touch_speed_mm_s', 3), ('lift_speed_mm_s', 30),
                        ('acceleration_mm_s2', 100), ('max_angular_speed_deg_s', 10), ('force_stop_n', 100)]:
        value = m[name]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 < value <= upper:
            raise ValueError('invalid motion setting: ' + name)
    if m['force_stop_n'] < 1:
        raise ValueError('existing force guard requires 1..100 N')
    if not (bounds[0] <= start[0] < start[0] + m['length_mm'] <= bounds[1] and bounds[2] <= start[1] <= bounds[3]):
        raise ValueError('straight line leaves measured paper bounds')
    if m['sample_spacing_mm'] > m['length_mm'] / 4:
        raise ValueError('sample spacing too coarse for line')
    for key in ('H_mm', 'alpha_deg', 'beta_deg'):
        values = np.array([inputs[key][v] for v in ('min', 'baseline', 'max')], float)
        if not np.isfinite(values).all() or not values[0] < values[2] or not values[0] <= values[1] <= values[2]:
            raise ValueError('invalid input bounds: ' + key)
    if not 0 <= inputs['H_mm']['min'] < inputs['H_mm']['max'] <= 25 or max(abs(inputs[k][v]) for k in ('alpha_deg', 'beta_deg') for v in ('min', 'max')) > 45:
        raise ValueError('H must stay in 0..25mm; declared execution tilts must remain within 45 degrees')
    z = config['zero_scan']
    if not np.isfinite([z['start_H_mm'], z['end_H_mm']]).all() or not -m['standoff_mm'] < z['start_H_mm'] < 0 < z['end_H_mm'] <= inputs['H_mm']['max']:
        raise ValueError('zero scan must start clear and end inside declared H envelope')
    return axes


def knots_for(config, kind):
    values = np.array([config['inputs'][k]['baseline'] for k in ('H_mm', 'alpha_deg', 'beta_deg')], float)
    if kind == 'zero':
        a, b = values.copy(), values.copy()
        a[0], b[0] = config['zero_scan']['start_H_mm'], config['zero_scan']['end_H_mm']
        return np.array([0., 1.]), np.array([a, b]), True
    fractions = np.array([0., .1, .25, .4, .55, .7, .85, 1.])
    result = np.tile(values, (len(fractions), 1))
    keys = ('H_mm', 'alpha_deg', 'beta_deg')
    if kind in ('h', 'alpha', 'beta'):
        index = ('h', 'alpha', 'beta').index(kind)
        low, high = config['inputs'][keys[index]]['min'], config['inputs'][keys[index]]['max']
        result[:, index] = [values[index], values[index], low, high, high, low, values[index], values[index]]
    elif kind == 'joint':
        normalized = np.array([[.5,.5,.5],[.5,.5,.5],[.2,.8,.3],[.8,.3,.7],[.4,.2,.8],[.7,.7,.2],[.5,.5,.5],[.5,.5,.5]])
        for i, key in enumerate(keys):
            low, high = config['inputs'][key]['min'], config['inputs'][key]['max']
            result[:, i] = low + normalized[:, i] * (high - low)
        result[[0, 1, -2, -1]] = values
    else:
        raise ValueError('unknown scan kind')
    return fractions, result, False


def input_curve(config, kind):
    length = config['motion']['length_mm']
    count = int(math.ceil(length / config['motion']['sample_spacing_mm'])) + 1
    if count > 9990:
        raise ValueError('route exceeds compiler waypoint budget')
    s = np.linspace(0, length, count)
    if config.get('knots') is not None:
        knots = config['knots']
        x = np.array([k['fraction'] for k in knots], float)
        y = np.array([[k[v] for v in ('H_mm', 'alpha_deg', 'beta_deg')] for k in knots], float)
        linear = kind == 'zero'
    else:
        x, y, linear = knots_for(config, kind)
    if x.ndim != 1 or len(x) < 2 or y.shape != (len(x), 3) or not np.isfinite(np.r_[x, y.ravel()]).all() or x[0] != 0 or x[-1] != 1 or np.any(np.diff(x) <= 0):
        raise ValueError('knots require increasing fractions from exactly 0 to 1 and finite H/alpha/beta')
    for i, key in enumerate(('H_mm', 'alpha_deg', 'beta_deg')):
        low = config['zero_scan']['start_H_mm'] if kind == 'zero' and i == 0 else config['inputs'][key]['min']
        high = config['inputs'][key]['max']
        if np.any(y[:, i] < low) or np.any(y[:, i] > high):
            raise ValueError('knot outside declared envelope: ' + key)
    if kind == 'zero' and (np.any(np.diff(y[:, 0]) < 0) or np.any(np.ptp(y[:, 1:], axis=0) > 1e-12)):
        raise ValueError('zero scan must descend monotonically at fixed posture')
    if kind in ('h', 'alpha', 'beta'):
        changing = ('h', 'alpha', 'beta').index(kind)
        if any(np.ptp(y[:, i]) > 1e-12 for i in range(3) if i != changing):
            raise ValueError('single-variable scan changes an additional input')
    j = np.clip(np.searchsorted(x, s / length, side='right') - 1, 0, len(x) - 2)
    t = (s / length - x[j]) / (x[j + 1] - x[j])
    weight = t if linear else 10 * t**3 - 15 * t**4 + 6 * t**5
    return s, y[j] + weight[:, None] * (y[j + 1] - y[j])


def build_plan(config, kind, trial_id, split='train', repeat_group=None):
    axes = validate_config(config)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', trial_id) or split not in ('train', 'holdout'):
        raise ValueError('safe trial id and train/holdout split required')
    s, inputs = input_curve(config, kind)
    g, m = config['geometry'], config['motion']
    origin, tip = np.array(g['paper_origin_base_m']), np.array(g['tip_tool_m'])
    u, v, n = axes.T
    R0 = Rotation.from_rotvec(g['baseline_rotation_rad'])
    rotations = Rotation.from_rotvec(inputs[:, 2, None] * math.pi / 180 * v) * Rotation.from_rotvec(inputs[:, 1, None] * math.pi / 180 * u) * R0
    xy = np.array(m['start_uv_mm'])[None, :] + np.column_stack((s, np.zeros_like(s)))
    natural_tip = origin + xy[:, 0, None] / 1000 * u + xy[:, 1, None] / 1000 * v - inputs[:, 0, None] / 1000 * n
    poses = np.column_stack((natural_tip - rotations.apply(tip), rotations.as_rotvec()))
    angular_speed = (rotations[:-1].inv() * rotations[1:]).magnitude() / (np.diff(s) / m['speed_mm_s'])
    if np.any(angular_speed > math.radians(m['max_angular_speed_deg_s'])):
        raise ValueError('posture change exceeds declared angular speed; slow down or lengthen transitions')
    points = []
    def append(pose, phase, contact, speed):
        points.append({'phase': phase, 'pose_m_rad': list(map(float, pose)), 'contact': contact,
                       'stroke_id': 0 if contact else None, 'speed_m_s': float(speed),
                       'acceleration_m_s2': m['acceleration_mm_s2'] / 1000})
    first_air = poses[0].copy(); first_air[:3] += n * (m['standoff_mm'] + inputs[0, 0]) / 1000
    append(first_air, 'longline_start_air', False, m['lift_speed_mm_s'] / 1000)
    append(poses[0], 'longline_touch_or_search', True, m['touch_speed_mm_s'] / 1000)
    rows = []
    for i in range(len(s)):
        if i:
            # MoveL speed is flange translation speed. Compensate its segment
            # length for a nominal constant natural-tip speed; feedback remains truth.
            speed = np.linalg.norm(poses[i, :3] - poses[i-1, :3]) / (s[i] - s[i-1]) * m['speed_mm_s']
            if not 1e-6 < speed <= .030:
                raise ValueError('required flange speed outside existing 30mm/s compiler limit')
            append(poses[i], 'longline_scan', True, speed)
        rows.append({'trial_id': trial_id, 'split': split, 'repeat_group': repeat_group or trial_id,
                     's_mm': float(s[i]), 'H_cmd_mm': float(inputs[i, 0]),
                     'alpha_cmd_rad': float(math.radians(inputs[i, 1])), 'beta_cmd_rad': float(math.radians(inputs[i, 2])),
                     'paper_u_mm': float(xy[i, 0]), 'paper_v_mm': float(xy[i, 1]),
                     'nominal_t_s': float(s[i] / m['speed_mm_s']), 'waypoint_index': i + 1})
    lifted = poses[-1].copy(); lifted[:3] += n * (m['standoff_mm'] + inputs[-1, 0]) / 1000
    append(lifted, 'longline_lift', False, m['lift_speed_mm_s'] / 1000)
    # Compile once offline to reject invalid blend geometry as well as malformed poses.
    compile_program(points, '192.0.2.10', 50071)
    return {'schema': 'brush_longline_plan_v1', 'trial_id': trial_id, 'kind': kind, 'split': split,
            'repeat_group': repeat_group or trial_id, 'rotation_convention': CONVENTION,
            'config': copy.deepcopy(config), 'config_sha256': digest(config), 'points': points, 'samples': rows,
            'motion_sent': False, 'contact_flags_mean': 'contact possible/allowed, not measured contact',
            'robot_execution_ready': False}


def export_plan(config_path, kind, trial_id, output, split='train', repeat_group=None):
    plan = build_plan(read(config_path), kind, trial_id, split, repeat_group)
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    save(out / 'plan.json', plan)
    write_csv(out / 'commanded_inputs.csv', plan['samples'])
    # An inspectable 2-D route plot without a plotting dependency.
    series = []
    colors = ['#246A9C', '#358366', '#926A2D']
    for key, color in zip(('H_cmd_mm', 'alpha_cmd_rad', 'beta_cmd_rad'), colors):
        values = np.array([r[key] for r in plan['samples']]); span = float(np.ptp(values))
        yy = 155 - (values - values.min()) / (span if span else 1.) * 100
        x = np.array([r['s_mm'] for r in plan['samples']]) / plan['config']['motion']['length_mm'] * 680 + 40
        series.append('<polyline points="' + ' '.join(f'{xx:.2f},{y:.2f}' for xx, y in zip(x, yy)) + f'" fill="none" stroke="{color}" stroke-width="2"/>')
    (out / 'input_preview.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg" width="760" height="210"><rect width="760" height="210" fill="white"/><text x="35" y="25">Normalized inputs: H (blue), alpha (green), beta (brown)</text>' + ''.join(series) + '</svg>')
    return {'plan': str((out / 'plan.json').resolve()), 'waypoints': len(plan['points']),
            'start_air_pose_m_rad': plan['points'][0]['pose_m_rad'], 'motion_sent': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    init = commands.add_parser('init', help='create an editable example configuration; no hardware access')
    init.add_argument('--output', type=Path, required=True)
    init.add_argument('--candidate', type=Path, help='import explicitly chosen existing candidate paper/TCP geometry')
    plan = commands.add_parser('plan', help='offline continuous H/alpha/beta route generation')
    plan.add_argument('--config', type=Path, required=True)
    plan.add_argument('--kind', choices=['zero', 'h', 'alpha', 'beta', 'joint'], required=True)
    plan.add_argument('--trial-id', required=True)
    plan.add_argument('--split', choices=['train', 'holdout'], default='train')
    plan.add_argument('--repeat-group')
    plan.add_argument('--output', type=Path, required=True)
    prepare = commands.add_parser('prepare', help='jx live read-only path audit; never sends motion')
    prepare.add_argument('--plan', type=Path, required=True)
    prepare.add_argument('--force-reference', type=Path, required=True)
    prepare.add_argument('--watchdog', type=Path, required=True)
    prepare.add_argument('--guard-host', required=True)
    prepare.add_argument('--robot-scripts', type=Path, required=True)
    prepare.add_argument('--output', type=Path, required=True)
    run = commands.add_parser('run', help='preview by default; --execute performs ONE guarded line with cameras')
    run.add_argument('--prepared', type=Path, required=True)
    run.add_argument('--cameras', type=Path)
    run.add_argument('--robot-scripts', type=Path)
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--execute', action='store_true')
    air = commands.add_parser('record-air', help='jx stationary unloaded reference acquisition; no motion or tare')
    air.add_argument('--robot-scripts', type=Path, required=True)
    air.add_argument('--unloaded-confirmed', action='store_true', required=True)
    air.add_argument('--output', type=Path, required=True)
    force = commands.add_parser('fit-force', help='offline fit from measured air samples plus independent holdouts')
    force.add_argument('--samples', type=Path, nargs='+', required=True)
    force.add_argument('--holdout', type=Path, nargs='+', required=True)
    force.add_argument('--maximum-error-n', type=float, required=True)
    force.add_argument('--maximum-translation-m', type=float, required=True)
    force.add_argument('--output', type=Path, required=True)
    cameras = commands.add_parser('camera-check', help='record camera-only preview, without robot access')
    cameras.add_argument('--cameras', type=Path, required=True)
    cameras.add_argument('--duration-s', type=float, default=2.)
    cameras.add_argument('--output', type=Path, required=True)
    for name in ('measure', 'side', 'zero', 'fit', 'predict', 'demo'):
        cmd = commands.add_parser(name, help='offline ' + name)
        cmd.add_argument('--output', type=Path, required=True)
        if name == 'measure':
            cmd.add_argument('--before', type=Path, required=True); cmd.add_argument('--after', type=Path, required=True)
            cmd.add_argument('--calibration', type=Path, required=True); cmd.add_argument('--plan', type=Path, required=True)
            cmd.add_argument('--feedback', type=Path, help='actual feedback JSONL; omit only for image-only measurement')
            cmd.add_argument('--before-pose', type=Path, help='measured pose JSON for hand-mounted camera calibration')
            cmd.add_argument('--after-pose', type=Path, help='measured pose JSON for hand-mounted camera calibration')
        elif name == 'side':
            cmd.add_argument('--video', type=Path, required=True); cmd.add_argument('--frames', type=Path, required=True)
            cmd.add_argument('--calibration', type=Path, required=True)
        elif name == 'zero':
            cmd.add_argument('--side-csv', type=Path, required=True); cmd.add_argument('--feedback', type=Path, required=True)
            cmd.add_argument('--plan', type=Path, required=True); cmd.add_argument('--gap-threshold-mm', type=float, default=.1)
            cmd.add_argument('--consecutive-frames', type=int, default=3)
        elif name == 'fit':
            cmd.add_argument('--csv', type=Path, nargs='+', required=True); cmd.add_argument('--ridge', type=float, default=1e-6)
        elif name == 'predict':
            cmd.add_argument('--model', type=Path, required=True); cmd.add_argument('--csv', type=Path, required=True)
    args = p.parse_args()
    if args.command == 'init':
        config = read(ROOT / 'examples/brush_longline_experiment.json')
        if args.candidate:
            c = read(args.candidate); paper, tcp = c['paper'], c['tcp']
            config['geometry'].update({
                'paper_origin_base_m': paper['reconstructed_center_tip_base_m'], 'paper_u_base': paper['paper_u_base'],
                'paper_v_base': paper['paper_v_base'], 'paper_normal_base': paper['paper_normal_base'],
                'tip_tool_m': tcp['translation_m'], 'baseline_rotation_rad': c['conversion']['baseline_tool0_rotation_rad']})
            config['geometry_status'] = 'imported_requires_current_setup_check'
            config['geometry_source'] = {'path': str(args.candidate.resolve()), 'sha256': hashlib.sha256(args.candidate.read_bytes()).hexdigest()}
            config['description'] = 'Imported geometry; confirm current setup and replace illustrative input limits before live use.'
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save(args.output, config); result = {'configuration': str(args.output), 'motion_sent': False}
    elif args.command == 'plan':
        result = export_plan(args.config, args.kind, args.trial_id, args.output, args.split, args.repeat_group)
    elif args.command in ('prepare', 'run', 'record-air', 'fit-force', 'camera-check'):
        from brush_longline_runtime import prepare_live, run_trial, record_air, fit_force, camera_check
        if args.command == 'prepare':
            result = prepare_live(args.plan, args.force_reference, args.watchdog, args.guard_host, args.robot_scripts, args.output)
        elif args.command == 'run':
            result = run_trial(args.prepared, args.output, args.execute, args.cameras, args.robot_scripts)
        elif args.command == 'record-air':
            result = record_air(args.robot_scripts, args.output)
        elif args.command == 'fit-force':
            result = fit_force(args.samples, args.holdout, args.maximum_error_n, args.maximum_translation_m, args.output)
        else:
            result = camera_check(args.cameras, args.duration_s, args.output)
    else:
        import brush_longline_measure as processing
        result = processing.dispatch(args)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    if args.command == 'fit-force' and result.get('passed') is not True:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
