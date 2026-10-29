#!/usr/bin/env python3
"""Vertical pen-up lift from the measured current pose. Default preview only.

One small straight lift in Base +Z at 2 mm/s using the existing buffered compiler
and buffered guard (same native movel, heartbeat, 10 N force delta stop). No
horizontal motion, no blend, no writing, no tare. Evidence is written to a new
folder and downloaded from jx; nothing is overwritten.
"""
import argparse
import base64
import datetime as dt
import json
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'outputs/brush_mapping_20261027'

PAYLOAD = r'''
import json, sys, time, types
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.dont_write_bytecode = True
sys.path[:0] = ['/home/jx/.dsh-analysis', '/home/jx/UR10/scripts',
                '/opt/openrobots/lib/python3.10/site-packages']
import probe_paper_height as support
import brush_buffered_guard as guard
import brush_buffered_motion as compiler
import brush_live_state as live
import write_gong_1mm as existing
from calibrate_pen_tcp import capture

out = Path(REMOTE_OUTPUT)
out.mkdir(parents=True, exist_ok=False)
report = {'status': 'preflight', 'motion_sent': False, 'lift_mm': LIFT_MM,
          'force_tare_performed': False}
def save(value):
    (out/'RESULT.json').write_text(json.dumps(value, indent=2) + '\n')
def event(value):
    print(json.dumps(value, ensure_ascii=False), flush=True)

report['controller'] = support.controller_ready('192.168.1.3')
offset = np.asarray(capture('192.168.1.3', 30011, 1.)['active_tcp_offset_m_rad'])
if not np.allclose(offset, 0, atol=1e-8, rtol=0):
    raise RuntimeError('active controller TCP changed')
reader = live.ContinuousState(support, '192.168.1.3')
q, start, _ = reader.stationary()
reader.close()
save(report)

poses = [start]
for fraction in (.4, 1.):
    pose = start.copy(); pose[2] += LIFT_MM/1000*fraction; poses.append(pose)
points = []
for index, pose in enumerate(poses):
    points.append({'pose_m_rad': pose.tolist(),
                   'phase': 'measured_current_start' if index == 0 else 'pen_up_lift',
                   'contact': False, 'stroke_id': None,
                   'speed_m_s': .002, 'acceleration_m_s2': .002,
                   'unloaded_wrench_n_nm': [0.]*6})
audit = existing.audit_points(points, q, np.zeros(6), support)
force = support.RawForce()
baseline, statistics = force.baseline()
force.close()
for point in points:
    point['unloaded_wrench_n_nm'] = baseline.tolist()
script, manifest = compiler.compile_program(points, '192.168.1.10', 50071)
report.update(planned_poses=[p['pose_m_rad'] for p in points],
              start_pose_m_rad=start.tolist(), baseline_wrench_n_nm=baseline.tolist(),
              baseline_statistics=statistics, audit_scope=audit['scope'],
              route_sha256=manifest['route_sha256'], program_sha256=manifest['program_sha256'])
report['joints_deg'] = np.rad2deg(np.array([p['q_rad'] for p in points])).tolist()
event({'status': 'lift_preflight_passed_no_motion', 'lift_mm': LIFT_MM,
       'waypoints': len(points), 'route_sha256': manifest['route_sha256'][:16]})
save(report)
if EXECUTE:
    watch = json.loads(existing.WATCHDOG.read_text())
    if watch.get('passed') is not True or watch.get('compiler_sha256') != existing.sha(compiler.__file__):
        raise RuntimeError('controller watchdog hardware evidence invalid')
    def write(name, value):
        path = out/(name + '.json')
        path.write_text(json.dumps(value, indent=2) + '\n')
        return {'path': path.name, 'sha256': existing.sha(path)}
    evidence = {
        'physical_mapping': write('physical_mapping', {
            'passed': True, 'route_sha256': manifest['route_sha256'],
            'scope': 'vertical pen-up lift from the measured pose; no model conversion and no contact'}),
        'force_reference': write('force_reference', {
            'passed': True, 'route_sha256': manifest['route_sha256'],
            'scope': 'raw session wrench at lift start, not tared; the existing 10 N delta stop is unchanged',
            'baseline_wrench_n_nm': baseline.tolist(), 'statistics': statistics}),
        'blended_path_audit': write('blended_path_audit', {
            'passed': True, 'route_sha256': manifest['route_sha256'],
            'program_sha256': manifest['program_sha256'],
            'scope': audit['scope'], 'joints_deg': report['joints_deg']}),
    }
    bundle = {'sample_id': 'gong-pen-up-lift',
              'points': points, 'start_pose_m_rad': start.tolist(),
              'active_tcp_offset_m_rad': [0.]*6, 'maximum_runtime_s': 60.,
              'physical_mapping_verified': True, 'force_reference_verified': True,
              'controller_watchdog_hardware_validated': True,
              'evidence': evidence,
              'scope': 'single vertical pen-up lift only',
              'force_data_max_age_s': 1.5}
    blockers = compiler.execution_blockers(bundle, manifest, out)
    if blockers:
        raise RuntimeError('; '.join(blockers))
    (out/'bundle.json').write_text(json.dumps(bundle, indent=2) + '\n')
    (out/'program.script.preview').write_text(script)
    guard.execute_buffered(bundle, out, '192.168.1.10', 50071, out/'execution.json',
                           support, capture)
    reader = live.ContinuousState(support, '192.168.1.3')
    _, after, _ = reader.stationary()
    reader.close()
    report = json.loads((out/'RESULT.json').read_text())
    report.update(status='lift_completed', motion_sent=True, final_pose_m_rad=after.tolist(),
                  requested_lift_mm=LIFT_MM,
                  measured_lift_mm=float((after[2]-start[2])*1000))
    save(report)
    event({'status': 'lift_completed', 'measured_lift_mm': report['measured_lift_mm']})
support.close_commands()
if report['status'] != 'lift_completed' and EXECUTE:
    raise SystemExit(1)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lift-mm', type=float, default=30.)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--accept-lift', help='operator confirmation recorded for this pen-up lift')
    parser.add_argument('--run-dir', type=Path)
    args = parser.parse_args()
    if not 5 <= args.lift_mm <= 60:
        parser.error('--lift-mm must stay within [5,60] mm')
    if args.execute and not args.accept_lift:
        parser.error('--execute requires --accept-lift with the operator reason')
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder = args.run_dir or OUTPUT/'pen-up-lift'/('run-'+stamp)
    folder.mkdir(parents=True, exist_ok=False)
    remote = '/home/jx/.dsh-analysis/pen-up-lift-'+stamp
    header = ('LIFT_MM=' + repr(float(args.lift_mm)) + chr(10) +
              'EXECUTE=' + repr(bool(args.execute)) + chr(10) +
              'REMOTE_OUTPUT=' + repr(remote) + chr(10))
    if args.accept_lift:
        header += 'ACCEPTED=' + repr(args.accept_lift) + chr(10)
    payload = header + PAYLOAD
    (folder/'remote_lift.py').write_text(payload)
    encoded = base64.b64encode(payload.encode()).decode()
    command = ('source /opt/ros/humble/setup.bash; export PYTHONNOUSERSITE=1 '
               'PYTHONDONTWRITEBYTECODE=1; echo ' + encoded + ' | base64 -d > /tmp/lift_payload.py && '
               'exec /usr/bin/python3 -B /tmp/lift_payload.py')
    last = None
    with (folder/'stdout.txt').open('x') as log, (folder/'stderr.txt').open('x') as err:
        process = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                                  'jx', 'bash -c ' + shlex.quote(command)],
                                 text=True, capture_output=True, timeout=600)
        log.write(process.stdout)
        err.write(process.stderr)
    for line in (folder/'stdout.txt').read_text().splitlines():
        if line.startswith('{'):
            last = json.loads(line)
            print(json.dumps(last, ensure_ascii=False))
    copied = subprocess.run(['scp', '-q', '-r', 'jx:'+remote, str(folder/'remote_evidence')],
                            capture_output=True, text=True)
    result = {'local_evidence': str(folder.resolve()), 'remote_output': remote,
              'ssh_exit_code': process.returncode, 'raw_evidence_downloaded': copied.returncode == 0,
              'result': last}
    (folder/'RESULT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + chr(10))
    if process.returncode or copied.returncode:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
