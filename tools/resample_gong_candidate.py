#!/usr/bin/env python3
"""Re-space the writing polyline of a prepared candidate; MU only, no motion.

Only contact runs are thinned to the requested chord length. Every original
6+4+6 control point is kept, non-contact points are untouched, and the numeric
audit plus the native-blend neighbourhood are recomputed for the new route.
The output is a new folder; nothing is overwritten.
"""
import argparse
import copy
import datetime as dt
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from prepare_gong_model_trial import current_meshes, contact_bounds
from brush_write_capture import dense_audit
from brush_buffered_motion import blend_radii
from run_gong_model_trial import check_conversion


def runs(points):
    """Contiguous contact runs of one stroke, as index lists."""
    out = []
    current = []
    for index, point in enumerate(points):
        if point['contact']:
            if current and (index != current[-1]+1 or
                            point['stroke_id'] != points[current[-1]]['stroke_id']):
                out.append(current); current = []
            current.append(index)
        elif current:
            out.append(current); current = []
    if current:
        out.append(current)
    return out


def resample(points, step_mm, controls):
    """Thin each contact run to step_mm chords, always keeping control points."""
    result = []
    control_keys = {(int(c['stroke_id']), round(float(c['paper_x_m']), 9), round(float(c['paper_y_m']), 9),
                     round(float(c['model_H_mm']), 9), round(float(c['alpha_rad']), 9),
                     round(float(c['beta_rad']), 9), round(float(c['gamma_absolute_paper_rad']), 9))
                    for c in controls}
    removed = 0
    index = 0
    runs_list = runs(points)
    boundaries = {r[0] for r in runs_list} | {r[-1] for r in runs_list}
    for run in runs_list:
        kept = [run[0]]
        travelled = 0.
        for a, b in zip(run, run[1:]):
            travelled += float(np.linalg.norm(np.array(points[b]['pose_m_rad'][:3]) -
                                              np.array(points[a]['pose_m_rad'][:3])))
            fields = points[b]['model_fields']
            key = (int(points[b]['stroke_id']), round(float(fields['paper_x_m']), 9),
                   round(float(fields['paper_y_m']), 9), round(float(fields['model_H_mm']), 9),
                   round(float(fields['alpha_rad']), 9), round(float(fields['beta_rad']), 9),
                   round(float(fields['gamma_absolute_paper_rad']), 9))
            if travelled >= step_mm/1000 or b == run[-1] or key in control_keys:
                kept.append(b); travelled = 0.
        removed += len(run)-len(kept)
        result.extend(kept)
    keep = set(result)
    order = [i for i in range(len(points)) if i in keep or not points[i]['contact']]
    return [copy.deepcopy(points[i]) for i in order], removed


def neighbourhood(points, seed, origin, normal, tcp, meshes):
    """Same ±axis blend sampling used for the current-mount geometry evidence."""
    from scipy.spatial import ConvexHull
    from brush_model_start import connect_start  # noqa: F401  (import check only)
    from fusion_model_ros2_beta.ur10_wrist_cost import ik, fk_jacobian, LOWER, UPPER
    cloud = np.concatenate([m.reshape(-1, 3) for m in meshes.values()])
    cloud = cloud[ConvexHull(cloud).vertices]
    rows = []
    failures = []
    for index, radius in enumerate(blend_radii(points)):
        if not radius:
            continue
        source = points[index]
        pose = np.array(source['pose_m_rad'])
        rotation = R.from_rotvec(pose[3:])
        for direction in np.r_[np.eye(3), -np.eye(3)]:
            target = np.eye(4)
            target[:3, :3] = rotation.as_matrix()
            target[:3, 3] = pose[:3]+radius*direction
            q, ok = ik(target, source['q_rad'])
            _, jac = fk_jacobian(q)
            margin = float(np.min(np.minimum(q-LOWER, UPPER-q)))
            singular = float(np.linalg.svd(jac, compute_uv=False)[-1])
            gap = float(np.min((cloud@target[:3, :3].T+target[:3, 3]-origin)@normal))
            row = {'phase': 'native_blend_neighborhood', 'source_waypoint_index': index,
                   'direction': direction.tolist(), 'radius_m': float(radius),
                   'pose_m_rad': np.r_[target[:3, 3], pose[3:]].tolist(), 'q_rad': q.tolist(),
                   'min_singular': singular, 'joint_margin_deg': float(np.rad2deg(margin)),
                   'cad_paper_clearance_mm': gap*1000}
            rows.append(row)
            if not ok or margin < np.deg2rad(5) or singular < .02 or gap < .020:
                failures.append(row)
    return rows, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path,
                        default=ROOT/'outputs/brush_mapping_20261027/gong_current_mount_trial')
    parser.add_argument('--step-mm', type=float, default=.4)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not .2 <= args.step_mm <= 2.:
        parser.error('--step-mm must stay within [0.2,2] mm')
    source = json.loads((args.source/'candidate.PREVIEW_ONLY.json').read_text())
    points, removed = resample(source['points'], args.step_mm, source['original_controls'])
    candidate = copy.deepcopy(source)
    candidate['points'] = points
    for name in ('audit', 'native_blend_neighborhood', 'contact_bounds', 'native_blend_radii_m',
                 'execution_blockers'):
        candidate.pop(name, None)
    candidate['resampling'] = {'step_mm': args.step_mm, 'removed_points': removed,
                               'scope': 'contact runs only; control points and non-contact points kept',
                               'reason': '0.05 mm chords let the controller pass this route faster than '
                                         'the commanded 1 mm/s, which overran the 0.8 mm tracking envelope'}
    check_conversion(candidate)
    meshes = current_meshes()
    seed = source['audit']['samples'][0]['q_rad']
    origin = np.array(candidate['paper']['reconstructed_center_tip_base_m'])
    normal = np.array(candidate['paper']['paper_normal_base'])
    tcp = np.array(candidate['tcp']['translation_m'])
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
    (out/'candidate.PREVIEW_ONLY.json').write_text(json.dumps(candidate, ensure_ascii=False, indent=2)+'\n')
    (out/'geometry_collision_query.json').write_text(json.dumps(
        {'audit': {'samples': audit['samples']+rows}}, indent=2)+'\n')
    summary = {'status': 'resampled_candidate_checked' if audit['passed_numeric'] and not failures
                        else 'resampled_candidate_check_failed',
               'motion_sent': False, 'waypoints': len(points), 'removed_points': removed,
               'step_mm': args.step_mm, 'numeric_audit_passed': audit['passed_numeric'],
               'numeric_samples': len(audit['samples']), 'blend_neighborhood_samples': len(rows),
               'blend_failures': len(failures),
               'control_counts': candidate['control_counts'], 'output': str(out.resolve())}
    (out/'RESULT.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary, ensure_ascii=False))
    if not audit['passed_numeric'] or failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
