#!/usr/bin/env python3
"""Fit saved contact heights for analysis; never command or validate a robot."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull


def fit_contacts(records):
    if len(records) < 3:
        raise ValueError('At least three saved contact points are required')
    indices = [r['index'] for r in records]
    if len(set(indices)) != len(indices):
        raise ValueError('Duplicate contact indices')
    xyz = np.asarray([r['pen_tip_base_m'] for r in records], dtype=float)
    if xyz.shape != (len(records), 3) or not np.all(np.isfinite(xyz)):
        raise ValueError('Contact coordinates must be finite XYZ values in metres')
    center = xyz.mean(axis=0)
    design = np.column_stack((xyz[:, :2] - center[:2], np.ones(len(xyz))))
    coef, _, rank, _ = np.linalg.lstsq(design, xyz[:, 2] - center[2], rcond=None)
    if rank != 3:
        raise ValueError('The XY locations do not uniquely determine a height plane')
    a, b, intercept = coef
    c = center[2] + intercept - a * center[0] - b * center[1]
    normal = np.array([-a, -b, 1.0])
    normal /= np.linalg.norm(normal)
    predicted = a * xyz[:, 0] + b * xyz[:, 1] + c
    residual = (xyz[:, 2] - predicted) * 1000
    perpendicular = residual * normal[2]
    hull = ConvexHull(xyz[:, :2])
    loo = []
    if len(xyz) >= 4:
        for i, index in enumerate(indices):
            keep = np.arange(len(xyz)) != i
            fitted, _, remaining_rank, _ = np.linalg.lstsq(
                design[keep], xyz[keep, 2] - center[2], rcond=None)
            if remaining_rank == 3:
                error = (xyz[i, 2] - center[2] - design[i] @ fitted) * 1000
                loo.append({'index': index, 'height_error_mm': float(error)})
    # Check consistency of the reported equation and least-squares residuals.
    if not np.allclose(design.T @ (residual / 1000), 0, atol=1e-12, rtol=0):
        raise RuntimeError('Least-squares consistency check failed')
    return {
        'equation_mm': 'z_base_mm = a*x_base_mm + b*y_base_mm + c_mm',
        'a': float(a), 'b': float(b), 'c_mm': float(c * 1000), 'c_m': float(c),
        'normal_base': normal.tolist(),
        'tilt_deg': float(np.degrees(np.arctan(np.hypot(a, b)))),
        'point_count': len(xyz), 'rank': int(rank),
        'reference_xy_mm': (center[:2] * 1000).tolist(),
        'height_at_reference_mm': float((a * center[0] + b * center[1] + c) * 1000),
        'xy_bounds_mm': (np.array([xyz[:, :2].min(axis=0), xyz[:, :2].max(axis=0)]) * 1000).tolist(),
        'xy_convex_hull_mm': (xyz[hull.vertices, :2] * 1000).tolist(),
        'xy_coverage_area_mm2': float(hull.volume * 1e6),
        'height_range_at_sampled_xy_mm': [float(predicted.min() * 1000), float(predicted.max() * 1000)],
        'rms_height_residual_mm': float(np.sqrt(np.mean(residual ** 2))),
        'max_abs_height_residual_mm': float(np.abs(residual).max()),
        'rms_perpendicular_residual_mm': float(np.sqrt(np.mean(perpendicular ** 2))),
        'max_abs_perpendicular_residual_mm': float(np.abs(perpendicular).max()),
        'points': [{'index': index, 'measured_xyz_mm': (point * 1000).tolist(),
                    'fitted_z_mm': float(height * 1000), 'height_residual_mm': float(error),
                    'perpendicular_residual_mm': float(distance)}
                   for index, point, height, error, distance in
                   zip(indices, xyz, predicted, residual, perpendicular)],
        'leave_one_out': loo,
        'independent_validation_count': 0,
        'meaning': 'Least-squares loaded pen-tip contact plane, including TCP and pen compliance errors',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--paper-has-moved', action='store_true')
    args = parser.parse_args()
    source = args.session.read_bytes()
    session = json.loads(source)
    if session.get('kind') != 'two_corner_xy_rectangle_force_paper_plane':
        parser.error('Expected a saved two-corner contact calibration session')
    records = sorted(session['accepted_points'], key=lambda r: r['index'])
    report = {
        'kind': 'partial_paper_plane_height_analysis',
        'status': 'fit_only_without_independent_validation',
        'created_at': dt.datetime.now(dt.timezone.utc).isoformat(),
        'source_session': str(args.session.resolve()),
        'source_session_sha256': hashlib.sha256(source).hexdigest(),
        'analysis_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_points': records,
        'tcp_calibration_sha256': session['tcp_calibration_sha256'],
        'robot_identity': session.get('robot_identity'),
        'paper_moved_after_measurements': args.paper_has_moved,
        'current_paper_placement_validated': False,
        'formal_12_point_calibration_complete': False,
        'controller_commands_sent': False,
        'fit': fit_contacts(records),
    }
    # Exclusive creation preserves earlier fits and the source acquisition file.
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
