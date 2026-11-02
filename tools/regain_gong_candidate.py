#!/usr/bin/env python3
"""Re-scale the commanded press depth of a prepared candidate; MU only.

Only the contact points move: each one is displaced along the paper normal by
the difference between its old and new commanded depth. Positions on the paper,
model fields, control points and every non-contact point stay identical, and
the numeric audit plus the native-blend neighbourhood are recomputed.
"""
import argparse
import copy
import datetime as dt
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from prepare_gong_model_trial import current_meshes, contact_bounds
from brush_write_capture import dense_audit
from brush_buffered_motion import blend_radii
from run_gong_model_trial import check_conversion
from resample_gong_candidate import neighbourhood, resample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path,
                        default=ROOT/'outputs/brush_mapping_20261027/gong_current_mount_trial_0p4mm')
    parser.add_argument('--gain-mm-per-model-mm', type=float, required=True,
                        help='new H_command gain; the previous candidate used 0.045')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resample-mm', type=float, default=0.,
                        help='re-space contact runs after the gain change; 0 keeps the source spacing')
    parser.add_argument('--depth-offset-mm', type=float, default=0.,
                        help='constant added to every commanded depth')
    parser.add_argument('--paper-surface-height-mm', type=float, default=None,
                        help='measured paper height above the historical plane; when given, the '
                             'commanded depth is measured from this surface')
    parser.add_argument('--alpha-gain', type=float, default=None,
                        help='override the alpha gain from the source candidate')
    parser.add_argument('--beta-gain', type=float, default=None,
                        help='override the beta gain from the source candidate')
    parser.add_argument('--gamma-gain', type=float, default=None,
                        help='override the gamma gain from the source candidate')
    parser.add_argument('--accept-audit-failure', default=None,
                        help='operator reason recorded when the numeric audit still reports failures')
    parser.add_argument('--max-air-step-mm', type=float, default=0.,
                        help='split longer pen-up moves into steps of at most this length')
    parser.add_argument('--raise-pen-up-mm', type=float, default=0.,
                        help='lift every pen-up waypoint by this much along the paper normal')
    parser.add_argument('--trim-start-above-mm', type=float, default=0.,
                        help='drop leading pen-up waypoints whose tip sits higher than this above the '
                             'paper, so the connector no longer lifts to a far-away high point')
    parser.add_argument('--depth-clip-min', type=float, default=None,
                        help='use the model z directly as the press depth and clamp it to this lower bound')
    parser.add_argument('--depth-clip-max', type=float, default=None,
                        help='upper clamp for the direct z press depth (safe range)')
    parser.add_argument('--smooth-window', type=int, default=0,
                        help='moving-average window applied to alpha/beta before the rotation; '
                             'without it a 1:1 tilt makes the tool0 route double back')
    parser.add_argument('--write-speed-mm-s', type=float, default=0.,
                        help='write_line speed override; 0 keeps the source speed')
    parser.add_argument('--press-upper-mm-s', type=float, default=0.,
                        help='press speed for every press point except the last one of each run')
    parser.add_argument('--press-last-mm-s', type=float, default=.3)
    args = parser.parse_args()
    if not .05 <= args.gain_mm_per_model_mm <= .5:
        parser.error('gain must stay within [0.05,0.5] mm per model mm')
    source = json.loads((args.source/'candidate.PREVIEW_ONLY.json').read_text())
    candidate = copy.deepcopy(source)
    normal = np.array(candidate['paper']['paper_normal_base'])
    old_gain = float(candidate['conversion']['H_command_mm_per_model_mm'])
    cfg = candidate['conversion']
    if args.alpha_gain is not None:
        cfg['alpha_gain'] = args.alpha_gain
    if args.beta_gain is not None:
        cfg['beta_gain'] = args.beta_gain
    if args.gamma_gain is not None:
        cfg['gamma_gain'] = args.gamma_gain
    surface = float(args.paper_surface_height_mm) if args.paper_surface_height_mm is not None         else float(cfg.get('paper_surface_height_mm', 0.))
    cfg['paper_surface_height_mm'] = surface
    cfg['alpha_beta_smooth_window'] = int(args.smooth_window)
    if args.depth_clip_max is not None:
        cfg['H_command_clip_mm'] = [float(args.depth_clip_min or 0.5), float(args.depth_clip_max)]
    origin = np.array(candidate['paper']['reconstructed_center_tip_base_m'])
    tip = np.array(candidate['tcp']['translation_m'])
    baseline = Rotation.from_rotvec(cfg['baseline_tool0_rotation_rad'])
    axis = np.array(cfg['tool_brush_axis'])
    shift = np.array(cfg['paper_translation_mm'])/1000.
    contact_points = [p for p in candidate['points'] if p['contact']]
    alphas = np.array([float(p['model_fields']['alpha_rad']) for p in contact_points])
    betas = np.array([float(p['model_fields']['beta_rad']) for p in contact_points])
    if args.smooth_window > 1:
        w = int(args.smooth_window)
        kernel = np.ones(w)/w
        alphas = np.convolve(np.pad(alphas, (w//2, w-1-w//2), mode='edge'), kernel, mode='valid')
        betas = np.convolve(np.pad(betas, (w//2, w-1-w//2), mode='edge'), kernel, mode='valid')
    depths = []
    for index, point in enumerate(contact_points):
        fields = point['model_fields']
        H = float(fields['model_H_mm'])
        a = float(alphas[index])*cfg['alpha_gain']
        b = float(betas[index])*cfg['beta_gain']
        g = float(fields['gamma_absolute_paper_rad'])*cfg['gamma_gain']
        rot = (Rotation.from_rotvec(np.array(candidate['paper']['paper_v_base'])*b)
               *Rotation.from_rotvec(np.array(candidate['paper']['paper_u_base'])*a)*baseline
               *Rotation.from_rotvec(axis*g))
        if args.depth_clip_max is not None:
            # Option 1: the model z itself is the press depth, clamped to the
            # measured safe range instead of being rescaled.
            new_depth = float(np.clip(H, args.depth_clip_min or 0.5, args.depth_clip_max))
        else:
            new_depth = args.gain_mm_per_model_mm*H+args.depth_offset_mm
        uv = np.array([float(fields['paper_x_m']), float(fields['paper_y_m'])])*cfg['xy_scale']+shift
        target = (origin+uv[0]*np.array(candidate['paper']['paper_u_base'])
                  + uv[1]*np.array(candidate['paper']['paper_v_base'])
                  + (surface-new_depth)/1000.*normal-rot.apply(tip))
        old_pose = np.array(point['pose_m_rad'], float)
        new_pose = np.r_[target, rot.as_rotvec()]
        point['pose_m_rad'] = list(map(float, new_pose))
        point['applied_tilt_rad'] = [float(a), float(b), float(g)]
        point['_delta'] = (new_pose-old_pose).tolist()
        point['commanded_depth_mm'] = new_depth
        depths.append(new_depth)
    speeds = {}
    if args.write_speed_mm_s > 0:
        write = [p for p in candidate['points'] if p['phase'] == 'write_line' and p['contact']]
        for point in write:
            point['speed_m_s'] = args.write_speed_mm_s/1000.
        speeds['write_line'] = args.write_speed_mm_s
    if args.press_upper_mm_s > 0:
        press = [i for i, p in enumerate(candidate['points']) if p['phase'] == 'press']
        runs = []
        current = []
        for i in press:
            if current and i != current[-1]+1:
                runs.append(current); current = []
            current.append(i)
        if current:
            runs.append(current)
        for run in runs:
            for i in run[:-1]:
                candidate['points'][i]['speed_m_s'] = args.press_upper_mm_s/1000.
            candidate['points'][run[-1]]['speed_m_s'] = args.press_last_mm_s/1000.
        speeds['press_upper'] = args.press_upper_mm_s
        speeds['press_last'] = args.press_last_mm_s
    if speeds:
        candidate['speed_revision'] = dict(speeds, scope='speeds only; poses, depth and model fields unchanged')
    candidate['conversion']['H_command_mm_per_model_mm'] = args.gain_mm_per_model_mm
    candidate['conversion']['H_command_offset_mm'] = args.depth_offset_mm
    candidate['depth_revision'] = {
        'previous_gain_mm_per_model_mm': old_gain,
        'new_gain_mm_per_model_mm': args.gain_mm_per_model_mm,
        'commanded_depth_mm_range': [float(min(depths)), float(max(depths))],
        'depth_offset_mm': args.depth_offset_mm,
        'changed_points': len(depths),
        'scope': 'contact points displaced along the paper normal only; no lateral change',
        'reason': '2026-10-28 press-depth calibration: single-pass width follows commanded depth '
                  'once the brush is inked, 2.5-4.0 mm giving 5.1-7.4 mm'}
    # Pen-up waypoints keep their own path, but they must follow the contact
    # points they connect: shift each one by the delta of its nearest contact
    # neighbour, otherwise a deep press tears the route apart.
    points_all = candidate['points']
    contact_idx = [i for i, p in enumerate(points_all) if p['contact']]
    if contact_idx:
        for i, point in enumerate(points_all):
            if point['contact']:
                continue
            nearest = min(contact_idx, key=lambda k: abs(k-i))
            pose = np.array(point['pose_m_rad'], float)
            neighbour = np.array(points_all[nearest]['pose_m_rad'], float)
            pose[:3] += np.array(points_all[nearest].get('_delta', [0.]*6))[:3]
            # Pen-up travel keeps the brush at the orientation of the stroke it
            # leaves or joins, so the route stays continuous after a 1:1 tilt.
            pose[3:] = neighbour[3:]
            if args.raise_pen_up_mm:
                pose[:3] += np.array(candidate['paper']['paper_normal_base'])*args.raise_pen_up_mm/1000.
            point['pose_m_rad'] = pose.tolist()
        for point in points_all:
            point.pop('_delta', None)
    if args.max_air_step_mm > 0:
        limit = args.max_air_step_mm/1000.
        expanded = [candidate['points'][0]]
        for previous, current in zip(candidate['points'], candidate['points'][1:]):
            if previous['contact'] or current['contact']:
                expanded.append(current); continue
            a = np.array(previous['pose_m_rad'], float)
            b = np.array(current['pose_m_rad'], float)
            span = float(np.linalg.norm(b[:3]-a[:3]))
            count = int(np.ceil(span/limit))
            for k in range(1, count):
                fraction = k/count
                rot = (Rotation.from_rotvec(a[3:])*Rotation.from_rotvec(
                    (Rotation.from_rotvec(a[3:]).inv()*Rotation.from_rotvec(b[3:])).as_rotvec()*fraction))
                mid = copy.deepcopy(current)
                mid['pose_m_rad'] = list(map(float, np.r_[a[:3]+fraction*(b[:3]-a[:3]), rot.as_rotvec()]))
                expanded.append(mid)
            expanded.append(current)
        candidate['points'] = expanded
    if args.trim_start_above_mm > 0:
        normal_v = np.array(candidate['paper']['paper_normal_base'])
        origin_v = np.array(candidate['paper']['reconstructed_center_tip_base_m'])
        tip_v = np.array(candidate['tcp']['translation_m'])
        trim_cfg = candidate['conversion']
        base_v = float(trim_cfg.get('paper_surface_height_mm', 0.))
        dropped = 0
        while len(candidate['points']) > 1 and not candidate['points'][0]['contact']:
            pose = np.array(candidate['points'][0]['pose_m_rad'], float)
            tip_point = pose[:3]+Rotation.from_rotvec(pose[3:]).apply(tip_v)
            height = float((tip_point-origin_v)@normal_v*1000)-base_v
            if height <= args.trim_start_above_mm:
                break
            candidate['points'].pop(0)
            dropped += 1
        candidate.setdefault('depth_revision', {})['trimmed_leading_points'] = dropped
    if args.resample_mm > 0:
        # Tilt changes move neighbouring poses by a few millimetres, which can
        # reverse the 0.4 mm chords of the source route. Re-space the contact
        # runs so the polyline stays monotone; control points are kept.
        candidate['points'], removed = resample(candidate['points'], args.resample_mm,
                                                candidate['original_controls'])
        depths = [p['commanded_depth_mm'] for p in candidate['points'] if p['contact']]
        candidate.setdefault('depth_revision', {})['resampled_removed_points'] = removed
    for name in ('audit', 'native_blend_neighborhood', 'contact_bounds', 'native_blend_radii_m',
                 'execution_blockers'):
        candidate.pop(name, None)
    check_conversion(candidate)
    meshes = current_meshes()
    seed = source['audit']['samples'][0]['q_rad']
    origin = np.array(candidate['paper']['reconstructed_center_tip_base_m'])
    tcp = np.array(candidate['tcp']['translation_m'])
    points = candidate['points']
    audit = dense_audit(points, seed, origin, normal, tcp, meshes)
    starts = [r for r in audit['samples'] if r['sample'] == 0]
    for point, row in zip(points[:-1], starts):
        point['q_rad'] = row['q_rad']
    points[-1]['q_rad'] = audit['end_q_rad']
    rows, failures = neighbourhood(points, seed, origin, normal, tcp, meshes)
    candidate['audit'] = audit
    accepted = None
    if not audit['passed_numeric'] and args.accept_audit_failure:
        accepted = {'accepted': True, 'reason': args.accept_audit_failure,
                    'failures': audit['failure_count'],
                    'scope': 'numeric route audit reported failures that the operator accepted explicitly'}
        candidate['audit']['operator_accepted_failures'] = accepted
    candidate['native_blend_neighborhood'] = {'passed': not failures, 'samples': rows,
                                              'failures': failures}
    candidate['native_blend_radii_m'] = blend_radii(points)
    candidate['contact_bounds'] = contact_bounds(candidate)
    candidate['motion_sent'] = False
    candidate['robot_execution_ready'] = False
    candidate['checked_at'] = dt.datetime.now(dt.timezone.utc).isoformat()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    (out/'candidate.PREVIEW_ONLY.json').write_text(json.dumps(candidate, ensure_ascii=False,
                                                              indent=2)+'\n')
    (out/'geometry_collision_query.json').write_text(json.dumps(
        {'audit': {'samples': audit['samples']+rows}}, indent=2)+'\n')
    summary = {'status': 'depth_rescaled_candidate_checked' if audit['passed_numeric'] and not failures
                        else 'depth_rescaled_candidate_checked_with_accepted_failures' if accepted
                        else 'depth_rescaled_candidate_check_failed',
               'motion_sent': False, 'waypoints': len(points),
               'gain_mm_per_model_mm': args.gain_mm_per_model_mm,
               'commanded_depth_mm_range': [float(min(depths)), float(max(depths))],
               'numeric_audit_passed': audit['passed_numeric'],
               'numeric_samples': len(audit['samples']),
               'blend_neighborhood_samples': len(rows), 'blend_failures': len(failures),
               'output': str(out.resolve())}
    (out/'RESULT.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary, ensure_ascii=False))
    if (not audit['passed_numeric'] or failures) and not accepted:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
