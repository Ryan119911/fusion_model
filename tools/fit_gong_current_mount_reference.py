#!/usr/bin/env python3
"""Build one local air-force acceptance record for a prepared trial candidate.

No SSH, ROS graph, controller or camera access. The output is a new evidence file
and never overwrites an existing one. Coverage is reported as extrapolation; it is
only labelled complete when the measured postures actually contain the trial
postures. Accepting a declared extrapolation requires an explicit reason string
that is stored in the record.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gong_reference_local import LIMIT_N, coverage, fit, moving_errors


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(run, candidate, model_kind='linear', limit_n=LIMIT_N, accept_extrapolation=None):
    run = Path(run)
    result = json.loads((run/'result.json').read_text())
    references = result.get('completed_references') or []
    if len(references) < 3:
        raise ValueError('at least three completed static air references required')
    if result.get('force_tare_performed') not in (False, None):
        raise ValueError('a tared session cannot provide a gravity-inclusive reference')
    if result.get('contact_force_map_verified') not in (False, None):
        raise ValueError('this record must not claim a verified contact force map')
    poses = np.array([r['reference']['actual_pose_m_rad'] for r in references])
    wrench = np.array([r['reference']['median_n_nm'] for r in references])
    local_fit = fit(poses[:, 3:], wrench, model_kind)
    moving = moving_errors(run, references, local_fit)
    trial = json.loads(Path(candidate).read_text())
    trial_rotations = [p['pose_m_rad'][3:] for p in trial['points']]
    span = coverage(local_fit, trial_rotations)
    bound_ok = span['extrapolation_bound_n'] <= limit_n
    moving_ok = moving['moving_max_force_error_n'] <= limit_n
    declared = None
    if not span['complete']:
        if not bound_ok:
            raise ValueError('extrapolation bound %.2f N exceeds the %.1f N limit; collect the missing postures'
                             % (span['extrapolation_bound_n'], limit_n))
        if not accept_extrapolation:
            raise ValueError('writing postures are %.2f deg outside the measured span; pass '
                             '--accept-extrapolation with the operator reason to record that decision'
                             % span['max_distance_deg'])
        declared = {'accepted': True, 'reason': accept_extrapolation,
                    'bound_n': span['extrapolation_bound_n'],
                    'max_distance_deg': span['max_distance_deg'],
                    'scope': 'first-order bound from the measured gradient; the writing postures were '
                             'not measured at this installation'}
    passed = bool(moving_ok and bound_ok and (span['complete'] or declared))
    return {
        'status': ('passed_local_air_reference_validation' if span['complete'] and passed else
                   'passed_local_air_reference_validation_with_declared_extrapolation' if passed else
                   'failed_local_air_reference_validation'),
        'passed': passed,
        'model_kind': {'linear': 'first_order_linear_relative_rotvec',
                       'affine': 'affine_relative_rotvec'}.get(model_kind, model_kind),
        'reference_error_limit_n': float(limit_n),
        'force_stop_threshold_n': 100.,
        'force_stop_scope': 'operator decision 2026-10-28: every brush program uses a 100 N force '
                            'increment stop; the 10 N value below is only this record''s reference-error limit',
        'static_reference_count': int(len(references)),
        'static_reference_ids': [r['reference_id'] for r in references],
        'static_roles': [r.get('role') for r in references],
        'moving_paired_rows': moving['moving_paired_rows'],
        'unpaired_raw_rows': moving['unpaired_raw_rows'],
        'moving_max_force_error_n': moving['moving_max_force_error_n'],
        'moving_p99_force_error_n': moving['moving_p99_force_error_n'],
        'local_fit': local_fit,
        'coverage': span,
        'declared_extrapolation': declared,
        'trial_candidate': str(Path(candidate).resolve()),
        'trial_candidate_sha256': sha256(candidate),
        'scope': ('measured local air postures at the current camera installation only; the '
                  'first-order map is an initial reference, not a brush H or contact-force map'),
        'write_force_reference_verified': False,
        'brush_H_mapping_accepted': False,
        'run': str(run.resolve()),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True, help='remote_evidence folder with result.json')
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model-kind', choices=('linear', 'affine', 'quadratic'), default='linear')
    p.add_argument('--limit-n', type=float, default=LIMIT_N)
    p.add_argument('--accept-extrapolation', help='operator reason recorded when the trial postures '
                                                  'lie outside the measured posture span')
    a = p.parse_args()
    if not 0 < a.limit_n <= 10:
        p.error('reference error limit must remain within (0,10] N')
    record = build(a.run, a.candidate, a.model_kind, a.limit_n, a.accept_extrapolation)
    with a.output.open('x') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    summary = {k: v for k, v in record.items() if k not in ('local_fit',)}
    summary['coverage'] = {k: v for k, v in record['coverage'].items() if k != 'scope'}
    print(json.dumps(summary, ensure_ascii=False))
    if not record['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
