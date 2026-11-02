#!/usr/bin/env python3
"""Build a sub-candidate from one index range of a prepared route; MU only.

Used to run a single stroke when the whole six-DOF route cannot fit inside the
production 1800 s program ceiling. The slice keeps the original poses and
rebuilds its own numeric audit, native-blend neighbourhood and contact bounds.
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
from resample_gong_candidate import neighbourhood


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--start', type=int, required=True, help='first waypoint index to keep')
    parser.add_argument('--end', type=int, default=None, help='last waypoint index, inclusive')
    parser.add_argument('--split-rotation-deg', type=float, default=0.,
                        help='subdivide pen-up segments so no step rotates more than this')
    parser.add_argument('--air-speed-mm-s', type=float, default=0.,
                        help='override the pen-up waypoint speed so rotation-only steps do not crawl')
    parser.add_argument('--raise-air-mm', type=float, default=0.,
                        help='lift every pen-up waypoint by this much so the connector sees a safe start height')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = json.loads((args.source/'candidate.PREVIEW_ONLY.json').read_text())
    candidate = copy.deepcopy(source)
    points = source['points']
    end = len(points)-1 if args.end is None else args.end
    if not 0 <= args.start < end <= len(points)-1:
        parser.error('invalid slice range')
    sliced = copy.deepcopy(points[args.start:end+1])
    if sliced[0]['contact']:
        parser.error('slice must start in free space')
    if args.split_rotation_deg:
        limit_rot = np.deg2rad(args.split_rotation_deg)
        expanded = [sliced[0]]
        for previous, current in zip(sliced, sliced[1:]):
            if previous['contact'] or current['contact']:
                expanded.append(current); continue
            a = np.array(previous['pose_m_rad'], float)
            b = np.array(current['pose_m_rad'], float)
            rel = (Rotation.from_rotvec(a[3:]).inv()*Rotation.from_rotvec(b[3:])).magnitude()
            count = int(np.ceil(rel/limit_rot))
            for k in range(1, max(count, 1)):
                fraction = k/max(count, 1)
                rot = Rotation.from_rotvec(a[3:])*Rotation.from_rotvec(
                    (Rotation.from_rotvec(a[3:]).inv()*Rotation.from_rotvec(b[3:])).as_rotvec()*fraction)
                mid = copy.deepcopy(current)
                mid['pose_m_rad'] = list(map(float, np.r_[a[:3]+fraction*(b[:3]-a[:3]), rot.as_rotvec()]))
                expanded.append(mid)
            expanded.append(current)
        sliced = expanded
    if args.air_speed_mm_s:
        # A pen-up step that barely translates but rotates a lot would crawl at a
        # translation-based speed; give those steps a workable command speed.
        for point in sliced:
            if not point['contact']:
                point['speed_m_s'] = args.air_speed_mm_s/1000.
    if args.raise_air_mm:
        normal_s = np.array(candidate['paper']['paper_normal_base'])
        for point in sliced:
            if not point['contact']:
                pose = np.array(point['pose_m_rad'], float)
                pose[:3] += normal_s*args.raise_air_mm/1000.
                point['pose_m_rad'] = pose.tolist()
    candidate['points'] = sliced
    candidate['slice'] = {'source': str(args.source.resolve()), 'start': args.start, 'end': end,
                          'kept': len(sliced),
                          'scope': 'single-stroke sub-route; poses unchanged'}
    for name in ('audit', 'native_blend_neighborhood', 'contact_bounds', 'native_blend_radii_m',
                 'execution_blockers'):
        candidate.pop(name, None)
    check_conversion(candidate)
    meshes = current_meshes()
    seed = source['audit']['samples'][0]['q_rad']
    origin = np.array(candidate['paper']['reconstructed_center_tip_base_m'])
    normal = np.array(candidate['paper']['paper_normal_base'])
    tcp = np.array(candidate['tcp']['translation_m'])
    audit = dense_audit(sliced, seed, origin, normal, tcp, meshes)
    starts = [r for r in audit['samples'] if r['sample'] == 0]
    for point, row in zip(sliced[:-1], starts):
        point['q_rad'] = row['q_rad']
    sliced[-1]['q_rad'] = audit['end_q_rad']
    rows, failures = neighbourhood(sliced, seed, origin, normal, tcp, meshes)
    candidate['audit'] = audit
    accepted = source['audit'].get('operator_accepted_failures')
    if accepted:
        candidate['audit']['operator_accepted_failures'] = accepted
    candidate['native_blend_neighborhood'] = {'passed': not failures, 'samples': rows,
                                              'failures': failures}
    candidate['native_blend_radii_m'] = blend_radii(sliced)
    candidate['contact_bounds'] = contact_bounds(candidate)
    candidate['checked_at'] = dt.datetime.now(dt.timezone.utc).isoformat()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    (out/'candidate.PREVIEW_ONLY.json').write_text(json.dumps(candidate, ensure_ascii=False,
                                                              indent=2)+'\n')
    (out/'geometry_collision_query.json').write_text(json.dumps(
        {'audit': {'samples': audit['samples']+rows}}, indent=2)+'\n')
    accepted_ok = bool(accepted and accepted.get('accepted'))
    summary = {'status': 'slice_checked' if audit['passed_numeric'] and not failures
                        else 'slice_checked_with_accepted_failures' if accepted_ok
                        else 'slice_check_failed',
               'waypoints': len(sliced), 'source_start': args.start, 'source_end': end,
               'numeric_samples': len(audit['samples']),
               'blend_samples': len(rows), 'blend_failures': len(failures),
               'numeric_audit_passed': audit['passed_numeric'],
               'output': str(out.resolve())}
    (out/'RESULT.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary, ensure_ascii=False))
    if (not audit['passed_numeric'] or failures) and not accepted_ok:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
