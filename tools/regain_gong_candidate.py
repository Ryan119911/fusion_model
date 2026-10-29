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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from prepare_gong_model_trial import current_meshes, contact_bounds
from brush_write_capture import dense_audit
from brush_buffered_motion import blend_radii
from run_gong_model_trial import check_conversion
from resample_gong_candidate import neighbourhood


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path,
                        default=ROOT/'outputs/brush_mapping_20261027/gong_current_mount_trial_0p4mm')
    parser.add_argument('--gain-mm-per-model-mm', type=float, required=True,
                        help='new H_command gain; the previous candidate used 0.045')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--depth-offset-mm', type=float, default=0.,
                        help='constant added to every commanded depth, e.g. the measured paper height offset')
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
    depths = []
    for point in candidate['points']:
        if not point['contact']:
            continue
        H = float(point['model_fields']['model_H_mm'])
        new_depth = args.gain_mm_per_model_mm*H+args.depth_offset_mm
        delta = new_depth-float(point['commanded_depth_mm'])
        pose = np.array(point['pose_m_rad'], float)
        pose[:3] -= normal*delta/1000.
        point['pose_m_rad'] = pose.tolist()
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
    if not audit['passed_numeric'] or failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
