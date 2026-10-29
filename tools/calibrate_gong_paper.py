#!/usr/bin/env python3
"""Fit the current paper plane from one Mech-Eye cloud; MU only.

Reads a capture folder (cloud_xyz_m.npy) plus the tool0 pose that produced it,
transforms the cloud into the robot base frame with the 2026-10-27 hand-eye
record, fits the dominant plane, and reports how that plane differs from the
historical 2026-10-23 paper frame: height offset and tilt.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
HAND_EYE_TRANSLATION = np.array([0.04707493838088885, 0.15015738159018904, 0.16350798747456627])
HAND_EYE_ROTATION = np.array([[-0.9999468780239723, 0.00894532352590486, -0.00512077309857878],
                              [-0.008882899077318561, -0.9998875041985456, -0.012086068491636742],
                              [-0.0052283108259184525, -0.012039939145164207, 0.9999138486046124]])


def fit_plane(points):
    centre = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points-centre, full_matrices=False)
    normal = vh[-1]
    return centre, normal/np.linalg.norm(normal)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--tool-pose', nargs=6, type=float, required=True)
    parser.add_argument('--candidate', type=Path,
                        default=ROOT/'outputs/brush_mapping_20261027/gong_current_mount_trial_h022_fast/candidate.PREVIEW_ONLY.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    candidate = json.loads(args.candidate.read_text())
    u = np.array(candidate['paper']['paper_u_base'])
    v = np.array(candidate['paper']['paper_v_base'])
    n_old = np.array(candidate['paper']['paper_normal_base'])
    origin = np.array(candidate['paper']['reconstructed_center_tip_base_m'])
    cloud = np.load(args.capture/'cloud_xyz_m.npy').reshape(-1, 3)
    valid = np.isfinite(cloud).all(axis=1) & (np.abs(cloud).sum(axis=1) > 1e-6)
    camera = cloud[valid]
    tool = np.array(args.tool_pose, float)
    rotation = Rotation.from_rotvec(tool[3:]).as_matrix()
    in_base = (camera@HAND_EYE_ROTATION.T+HAND_EYE_TRANSLATION)@rotation.T+tool[:3]
    # Keep the dominant surface: the paper is what sits perpendicular to the
    # optical axis at the commanded standoff.
    axis = rotation@HAND_EYE_ROTATION@np.array([0., 0., 1.])
    along = in_base@axis
    offset = float(np.median(along))
    near = np.abs(along-offset) < 0.010
    points = in_base[near]
    for _ in range(3):
        centre, normal = fit_plane(points)
        if normal@n_old < 0:
            normal = -normal
        residual = np.abs((points-centre)@normal)
        keep = residual < max(0.0015, float(np.percentile(residual, 90)))
        points = points[keep]
    centre, normal = fit_plane(points)
    if normal@n_old < 0:
        normal = -normal
    residual = np.abs((points-centre)@normal)
    cos_tilt = float(np.clip(normal@n_old, -1, 1))
    tilt_deg = float(np.degrees(np.arccos(cos_tilt)))
    # Height of the new paper along the historical normal, sampled on the
    # historical plane at the character's own (u, v) footprint.
    contact = [p for p in candidate['points'] if p['contact']]
    uv = np.array([[p['model_fields']['paper_x_m']*candidate['conversion']['xy_scale'],
                    p['model_fields']['paper_y_m']*candidate['conversion']['xy_scale']]
                   for p in contact])
    uv += np.array(candidate['conversion']['paper_translation_mm'])/1000.
    samples = origin+uv[:, :1]*u+uv[:, 1:2]*v
    heights = []
    for point in samples:
        denom = normal@n_old
        if abs(denom) < 1e-6:
            heights.append(float('nan'))
            continue
        t = (normal@centre-normal@point)/denom
        heights.append(float(t*1000))
    heights = np.array(heights)
    plane_in_old = {'normal_dot_old': cos_tilt, 'tilt_deg': tilt_deg,
                    'height_at_character_mm': float(np.nanmedian(heights)),
                    'height_p05_mm': float(np.nanpercentile(heights, 5)),
                    'height_p95_mm': float(np.nanpercentile(heights, 95)),
                    'points_used': int(len(points)),
                    'residual_p50_mm': float(np.median(residual)*1000),
                    'residual_p95_mm': float(np.percentile(residual, 95)*1000),
                    'plane_point_m': centre.tolist(), 'plane_normal': normal.tolist(),
                    'scope': 'dominant plane within 10 mm of the commanded standoff; table and '
                             'fixtures outside that band are excluded'}
    record = {'status': 'paper_plane_fitted', 'capture': str(args.capture),
              'tool_pose_m_rad': tool.tolist(), 'plane': plane_in_old,
              'historical_plane': {'origin_m': origin.tolist(), 'normal': n_old.tolist()},
              'interpretation': ('height_at_character_mm is how far the current sheet sits above '
                                 '(positive) or below (negative) the historical plane measured along '
                                 'the historical normal at the character footprint')}
    args.output.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({k: plane_in_old[k] for k in ('tilt_deg', 'height_at_character_mm',
                                                   'height_p05_mm', 'height_p95_mm',
                                                   'residual_p50_mm', 'points_used')}))


if __name__ == '__main__':
    main()
