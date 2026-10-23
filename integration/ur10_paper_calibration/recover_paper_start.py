#!/usr/bin/env python3
"""Guarded upward release and optional high-clearance orientation recovery.

Uses the last probe's original free-space force baseline, not a new contact
tare. The initial lift has no XY/orientation change. A <=2 degree orientation
change is restored only after reaching high clearance. Dry-run is the default.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import calibrate_paper_plane_auto as app
import probe_paper_height as probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    session = json.loads(args.session.read_text())
    app.validate_session(session)
    calibration_path = Path(session['tcp_calibration'])
    if hashlib.sha256(calibration_path.read_bytes()).hexdigest() != session['tcp_calibration_sha256']:
        raise RuntimeError('TCP calibration changed')
    previous = Path(session['attempts'][-1]['measurement'])
    old = json.loads(previous.read_text())
    baseline = np.asarray(old['baseline']['median_n_nm'])
    if not np.allclose(old['active_tcp_offset_m_rad'], session['active_tcp_offset_m_rad'], atol=1e-9, rtol=0):
        raise RuntimeError('last free baseline belongs to a different TCP')
    host = session['robot_identity']['host']
    tip = np.asarray(session['flange_to_pen_tip_m'])
    reader, force = None, None
    report = {'kind': 'bounded_upward_paper_start_recovery', 'motion_sent': False,
              'source_session': str(args.session), 'baseline_source': str(previous),
              'original_baseline_n_nm': baseline.tolist(), 'status': 'preflight',
              'overload_delta_n': 5, 'step_mm': 2, 'initial_raise_mm': 4, 'settle_frames': []}
    output = Path.home() / '.handoff/reports' / ('paper-start-recovery-' + dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json')
    try:
        report['controller'] = probe.controller_ready(host)
        info = app.capture(host, 30011, 1)
        offset = np.asarray(info['active_tcp_offset_m_rad'])
        if not np.allclose(offset, session['active_tcp_offset_m_rad'], atol=1e-9, rtol=0):
            raise RuntimeError('active TCP changed')
        reader = probe.RealtimeReader(host, 30013)
        q, tcp, point = app.state_tip(reader, offset, tip)
        expected = Rotation.from_rotvec(session['tcp_orientation_rotvec'])
        orientation_error = (Rotation.from_rotvec(tcp[3:]).inv() * expected).magnitude()
        report['orientation_difference_deg'] = float(np.rad2deg(orientation_error))
        # A small manual orientation change can be restored only after raising
        # at the CURRENT orientation to a separately checked high clearance.
        # The contact collector's 0.2/0.1 degree gates are unchanged.
        if orientation_error > np.deg2rad(2):
            raise RuntimeError('orientation changed by more than 2 degrees; automatic recovery is not applicable')
        low, high = (np.asarray(session['rectangle'][key]) for key in ('outer_min_xy_m', 'outer_max_xy_m'))
        gap = point[2] - app.estimate_height(session, point[:2])
        report['start_pen_tip_base_m'] = point.tolist()
        report['estimated_gap_mm'] = float(gap * 1000)
        if not 0 <= gap < .0015 or np.any(point[:2] < low) or np.any(point[:2] > high):
            raise RuntimeError('upward recovery only applies to an interior start 0..1.5 mm above the estimated plane')
        final = tcp.copy(); final[2] += .004
        report['path_gate'] = probe.model_gate(q, tcp, final, offset)
        force = probe.RawForce()
        until = probe.time.monotonic() + 10
        while True:
            try:
                report['initial_force_delta_n'] = probe.force_level(force, baseline)
                break
            except probe.ForceGuardError:
                raise
            except RuntimeError:
                if probe.time.monotonic() >= until:
                    raise
                probe.time.sleep(.1)
        _, rechecked, _ = app.state_tip(reader, offset, tip)
        if np.linalg.norm(rechecked[:3] - tcp[:3]) > .0001:
            raise RuntimeError('robot moved during recovery preflight')
        if not args.execute:
            report['status'] = 'dry_run_pass'
            return
        for step in (1, 2):
            endpoint = tcp.copy(); endpoint[2] += step * .002
            report['motion_sent'] = True
            report['pending_step'] = step
            probe.save(output, report)
            probe.move_step(reader, force, baseline, host, endpoint, .0005, False)
            probe.wait_stopped(reader, force, baseline, report['settle_frames'])
            report['completed_steps'] = step
            probe.save(output, report)
        _, actual, end_point = app.state_tip(reader, offset, tip)
        report['end_pen_tip_base_m'] = end_point.tolist()
        report['end_force_delta_n'] = probe.force_level(force, baseline)
        if np.linalg.norm(actual[:3] - final[:3]) > .0001:
            raise RuntimeError('upward recovery did not reach its endpoint')
        report['status'] = 'raised_4mm'
        if orientation_error > np.deg2rad(.2):
            known_z = [p['pen_tip_base_m'][2] for p in session['corners'] + session['accepted_points']]
            clearance_z = max(max(known_z) + .020, float(end_point[2]))
            raised = actual.copy(); raised[2] += clearance_z - end_point[2]
            q, actual, _ = app.state_tip(reader, offset, tip)
            report['clearance_path_gate'] = probe.model_gate(q, actual, raised, offset)
            start = actual.copy()
            count = max(1, int(math.ceil((raised[2] - start[2]) / .002)))
            for index in range(1, count + 1):
                endpoint = start + (raised - start) * index / count
                probe.move_step(reader, force, baseline, host, endpoint, .0005, False)
                probe.wait_stopped(reader, force, baseline, report['settle_frames'])
                report['completed_clearance_steps'] = index
                probe.save(output, report)
            q, actual, high_point = app.state_tip(reader, offset, tip)
            if high_point[2] < clearance_z - .0001:
                raise RuntimeError('high clearance not reached; orientation restoration forbidden')
            # Rebaseline only after the verified free-space lift, before the
            # small orientation correction, never at the loaded starting tip.
            free_baseline, report['high_clearance_free_baseline'] = force.baseline()
            current_rotation = Rotation.from_rotvec(actual[3:])
            delta = (current_rotation.inv() * expected).as_rotvec()
            final_pose = app.tcp_target(high_point, expected.as_rotvec(), offset, tip)
            count = max(1, int(math.ceil(np.linalg.norm(delta) / np.deg2rad(.15))),
                        int(math.ceil(np.linalg.norm(final_pose[:3] - actual[:3]) / .0015)))
            report['orientation_segments'] = []
            for index in range(1, count + 1):
                rotation = current_rotation * Rotation.from_rotvec(delta * index / count)
                endpoint = app.tcp_target(high_point, rotation.as_rotvec(), offset, tip)
                q, prior, _ = app.state_tip(reader, offset, tip)
                if np.linalg.norm(endpoint[:3] - prior[:3]) > .0025:
                    raise RuntimeError('orientation segment exceeds finite translation envelope')
                gate = probe.model_gate(q, prior, endpoint, offset)
                entry = {'index': index, 'path_gate': gate, 'target_tcp_m_rad': endpoint.tolist()}
                report['orientation_segments'].append(entry)
                probe.save(output, report)
                app.travel_segment(reader, force, free_baseline, host, endpoint, endpoint[3:], .001, .0005, entry)
                probe.wait_stopped(reader, force, free_baseline, report['settle_frames'])
                _, checked, _ = app.state_tip(reader, offset, tip)
                if (Rotation.from_rotvec(checked[3:]).inv() * rotation).magnitude() > np.deg2rad(.05):
                    raise RuntimeError('orientation recovery segment did not settle at its target')
            _, checked, restored_point = app.state_tip(reader, offset, tip)
            if (Rotation.from_rotvec(checked[3:]).inv() * expected).magnitude() > np.deg2rad(.05):
                raise RuntimeError('original pen orientation was not restored')
            report['restored_pen_tip_base_m'] = restored_point.tolist()
            report['total_raise_mm'] = float((restored_point[2] - point[2]) * 1000)
            report['status'] = 'original_orientation_restored_at_clearance'
    except BaseException as exc:
        if report['motion_sent']:
            probe.stop(host)
        report['status'], report['error'] = 'stopped_on_error', str(exc)
        if hasattr(exc, 'diagnostics'):
            report['diagnostics'] = exc.diagnostics
        raise
    finally:
        if force is not None:
            report['force_trace'] = probe.force_snapshot(force, baseline)
            force.close()
        if reader is not None:
            reader.close()
        probe.close_commands()
        probe.save(output, report)
        print('RECOVERY:', output, report['status'], flush=True)


if __name__ == '__main__':
    main()
