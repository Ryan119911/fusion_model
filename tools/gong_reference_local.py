#!/usr/bin/env python3
"""Local suspended-force reference built only from measured air postures.

Three measured postures cannot identify a ten-term quadratic wrench model, so this
module refuses to pretend otherwise. It fits a first-order (affine) map in the
normalized relative rotation vector and reports two things it can actually
measure: the moving-force error along the traversed air segments, and how far the
requested writing postures lie outside the measured span. The latter is reported
as an explicit extrapolation with a numeric bound, never relabelled as coverage.

The 10 N value here is a reference-error acceptance limit. The existing 10 N
motion force stop is a different quantity and is untouched.
"""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

LIMIT_N = 10.


def basis(rotations, baseline, kind):
    """Design matrix in the same /4 degree relative rotation vector convention."""
    rel = (Rotation.from_rotvec(baseline).inv() * Rotation.from_rotvec(
        np.atleast_2d(rotations))).as_rotvec() / np.deg2rad(4)
    a, b, c = rel.T
    if kind == 'affine':
        return np.column_stack((np.ones(len(rel)), a, b, c)), rel
    if kind == 'linear':
        return np.column_stack((a, b, c)), rel
    if kind == 'quadratic':
        return np.column_stack((np.ones(len(rel)), a, b, c, a*a, b*b, c*c, a*b, a*c, b*c)), rel
    raise ValueError('unknown local reference basis: '+repr(kind))


def fit(rotations, wrench, kind='linear'):
    """First-order local reference about the first measured posture.

    'linear' differences against the first posture's measured wrench, so a
    three-posture record fits a two-direction gradient with minimum norm and the
    rank deficiency is reported instead of being hidden. 'affine' and 'quadratic'
    keep an intercept and need four and ten postures respectively.
    """
    rotations = np.atleast_2d(np.asarray(rotations, float))
    wrench = np.atleast_2d(np.asarray(wrench, float))
    if rotations.shape[1] != 3 or wrench.shape != (len(rotations), 6):
        raise ValueError('pose/wrench shapes do not match')
    baseline = rotations[0].copy()
    design, rel = basis(rotations, baseline, kind)
    rank = int(np.linalg.matrix_rank(design))
    required = {'linear': 2, 'affine': 4, 'quadratic': 10}[kind]
    if rank < required:
        raise ValueError('%s basis needs rank %d, measured rank is %d from %d postures'
                         % (kind, required, rank, len(rotations)))
    if rank == design.shape[1]:
        coefficients = np.linalg.lstsq(design, wrench, rcond=None)[0]
        offset = np.zeros(6)
    else:
        offset = wrench[0].copy()
        coefficients = np.linalg.lstsq(design, wrench-offset, rcond=None)[0]
    residual = design@coefficients + offset - wrench
    return {'basis_kind': kind, 'baseline_rotation_rad': baseline.tolist(),
            'coefficients': coefficients.tolist(), 'offset': offset.tolist(),
            'feature_columns': int(design.shape[1]), 'measured_rank': rank,
            'posture_count': int(len(rotations)),
            'unmeasured_directions': int(max(0, 3-rank)) if kind == 'linear' else 0,
            'condition': float(np.linalg.cond(design)),
            'static_residual_force_n': float(np.max(np.linalg.norm(residual[:, :3], axis=1))),
            'degrees_of_freedom': max(0, int(len(rotations))-design.shape[1]),
            'reference_features': rel.tolist(),
            'max_pairwise_force_spread_n': float(max(
                np.linalg.norm(wrench[i, :3]-wrench[j, :3])
                for i in range(len(wrench)) for j in range(i+1, len(wrench))) if len(wrench) > 1 else 0.),
            'scope': 'first-order local air reference about the first measured posture; a measured '
                     'rank below three leaves one rotation direction out of the fit, which is '
                     'reported rather than interpolated'}


def predict(local_fit, rotations):
    design, _ = basis(np.atleast_2d(rotations), np.array(local_fit['baseline_rotation_rad']),
                      local_fit['basis_kind'])
    return design@np.array(local_fit['coefficients'])+np.array(local_fit.get('offset', [0.]*6))


def moving_errors(run, references, local_fit):
    """Independent force error on every paired moving raw row of the same records."""
    errors, unpaired = [], 0
    for reference in references:
        folder = Path(reference.get('reference_source_directory', Path(run)/reference['reference_id']))
        feedback = [json.loads(s) for s in (folder/'controller_feedback.jsonl').read_text().splitlines()]
        times = np.array([r['host_unix_ns']/1e9 for r in feedback])
        poses = np.array([r['actual_tool0_pose_m_rad'] for r in feedback])
        velocity = np.array([r['tcp_velocity'] for r in feedback])
        raw = [json.loads(s) for s in (folder/'raw_force.jsonl').read_text().splitlines()]
        stamps = np.array([r['received_unix_s'] for r in raw])
        wrench = np.array([r['wrench_n_nm'] for r in raw])
        index = np.searchsorted(times, stamps).clip(0, len(times)-1)
        left = (index-1).clip(0, len(times)-1)
        index = np.where(abs(times[left]-stamps) < abs(times[index]-stamps), left, index)
        paired = abs(times[index]-stamps) <= .025
        moving = ((np.linalg.norm(velocity[index, :3], axis=1) > .0001) |
                  (np.linalg.norm(velocity[index, 3:], axis=1) > .001))
        use = paired & moving
        unpaired += int(np.sum(~paired))
        if np.any(use):
            residual = predict(local_fit, poses[index[use], 3:]) - wrench[use]
            errors.extend(np.linalg.norm(residual[:, :3], axis=1).tolist())
    if not errors:
        raise ValueError('no independent moving raw force rows in these records')
    values = np.array(errors)
    return {'moving_paired_rows': int(len(values)), 'unpaired_raw_rows': unpaired,
            'moving_max_force_error_n': float(values.max()),
            'moving_p99_force_error_n': float(np.quantile(values, .99))}


def _segment_distance(point, first, second):
    step = second-first
    t = np.clip((point-first)@step/(step@step), 0., 1.)
    return float(np.linalg.norm(point-(first+t*step)))


def _distance_to_span(point, vertices):
    """Distance to the convex hull of the measured postures (1, 2 or >=3 points)."""
    if len(vertices) == 1:
        return float(np.linalg.norm(point-vertices[0]))
    if len(vertices) == 2:
        return _segment_distance(point, vertices[0], vertices[1])
    if len(vertices) > 3:
        from scipy.spatial import Delaunay
        hull = Delaunay(vertices)
        if np.all(hull.find_simplex(np.atleast_2d(point)) >= 0):
            return 0.
        pairs = [(i, j) for i in range(len(vertices)) for j in range(i+1, len(vertices))]
        return min(_segment_distance(point, vertices[i], vertices[j]) for i, j in pairs)
    a, b, c = vertices
    normal = np.cross(b-a, c-a)
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        return min(_segment_distance(point, a, b), _segment_distance(point, b, c),
                   _segment_distance(point, c, a))
    normal = normal/norm
    offset = float((point-a)@normal)
    planar = point-offset*normal
    uv, _, _, _ = np.linalg.lstsq(np.column_stack((b-a, c-a)), planar-a, rcond=None)
    if uv[0] >= -1e-9 and uv[1] >= -1e-9 and uv.sum() <= 1+1e-9:
        return abs(offset)
    return min(_segment_distance(point, a, b), _segment_distance(point, b, c),
               _segment_distance(point, c, a))


def coverage(local_fit, trial_rotations):
    """Reported as extrapolation, not as validated coverage."""
    reference = np.array(local_fit['reference_features'])
    query = basis(np.atleast_2d(trial_rotations), np.array(local_fit['baseline_rotation_rad']),
                  local_fit['basis_kind'])[1]
    distances = np.array([_distance_to_span(row, reference) for row in query])
    rows = 0 if local_fit['basis_kind'] == 'linear' else 1
    gradient = np.linalg.svd(np.array(local_fit['coefficients'])[rows:rows+3, :3], compute_uv=False)[0]
    worst = float(distances.max())
    return {'complete': bool(worst <= 1e-9), 'outside_fraction': float(np.mean(distances > 1e-9)),
            'max_distance_feature_unit': worst, 'max_distance_deg': float(worst*4),
            'p99_distance_deg': float(np.quantile(distances, .99)*4),
            'measured_gradient_n_per_feature_unit': float(gradient),
            'extrapolation_bound_n': float(worst*gradient),
            'scope': 'first-order force-variation bound over the measured gradient, applied to the '
                     'distance outside the measured posture span; not a measurement at those postures'}
