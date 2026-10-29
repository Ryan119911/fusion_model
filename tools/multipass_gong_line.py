#!/usr/bin/env python3
"""Multi-pass line test: can repeated passes widen a stroke to 5-8 mm?

Writes short parallel lines at one commanded depth, either repeating the exact
same path or repeating it with a lateral offset, so the ink width can be
measured afterwards. Default preview; motion needs --execute and --accept-test.
"""
import argparse
import base64
import datetime as dt
import json
import shlex
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'outputs/brush_mapping_20261027'

PAYLOAD = r'''
import json, sys, time
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

out = Path(REMOTE_OUTPUT); out.mkdir(parents=True, exist_ok=False)
report = {'status': 'preview', 'motion_sent': False, 'variants': VARIANTS}
def save(v):
    (out/'RESULT.json').write_text(json.dumps(v, indent=2) + '\n')
def event(v):
    print(json.dumps(v, ensure_ascii=False), flush=True)

support.controller_ready('192.168.1.3')
offset0 = np.asarray(capture('192.168.1.3', 30011, 1.)['active_tcp_offset_m_rad'])
if not np.allclose(offset0, 0, atol=1e-8, rtol=0):
    raise RuntimeError('active controller TCP changed')
reader = live.ContinuousState(support, '192.168.1.3')
q, start, _ = reader.stationary(); reader.close()

origin = np.array(ORIGIN); normal = np.array(NORMAL); u = np.array(U_AXIS); v = np.array(V_AXIS)
tip = np.array(TIP); rot = np.array(BASELINE, float)
R = Rotation.from_rotvec(rot)
points = [{'pose_m_rad': list(map(float, start)), 'phase': 'measured_current_start',
           'contact': False, 'stroke_id': None, 'speed_m_s': .010,
           'acceleration_m_s2': .100, 'unloaded_wrench_n_nm': [0.]*6}]

def add(tip_point, phase, speed, contact, stroke):
    tool0 = np.asarray(tip_point, float)-R.apply(tip)
    if points and np.allclose(points[-1]['pose_m_rad'], np.r_[tool0, rot], atol=1e-9, rtol=0):
        return
    points.append({'pose_m_rad': list(map(float, np.r_[tool0, rot])), 'phase': phase,
                   'contact': contact, 'stroke_id': stroke, 'speed_m_s': speed,
                   'acceleration_m_s2': .100, 'unloaded_wrench_n_nm': [0.]*6})

stroke = 0
for variant in VARIANTS:
    for pass_offset in variant['offsets_mm']:
        base = origin+u*(variant['u_start_mm']/1000)+v*((variant['v_mm']+pass_offset)/1000)
        left = base+normal*(STANDOFF/1000)
        right = base+u*(variant['length_mm']/1000)+normal*(STANDOFF/1000)
        add(left, 'mp_approach', .005, False, None)
        add(left-normal*((STANDOFF+DEPTH)/1000), 'mp_touch', .0005, False, None)
        add(right-normal*((STANDOFF+DEPTH)/1000), 'mp_line', .010, True, stroke)
        add(right, 'mp_lift', .005, False, None)
    stroke += 1

force = support.RawForce()
start_t = time.monotonic(); deadline = start_t+10
while time.monotonic() < deadline:
    time.sleep(.25)
    with force.lock:
        n = sum(1 for r in force.rows if r[0] >= start_t)
    if n >= 200 and time.monotonic()-start_t >= 2:
        break
with force.lock:
    rows = [r for r in force.rows if r[0] >= start_t]
if len(rows) < 200:
    raise RuntimeError('force rate too low for a baseline window')
values = np.asarray([r[3] for r in rows], float)
median = np.median(values, axis=0)
filtered = np.asarray([np.median(values[i-4:i+1, :3], axis=0) for i in range(4, len(values))])
noise = float(np.quantile(np.linalg.norm(filtered-median[:3], axis=1), .99))
statistics = {'count': len(rows), 'median_n_nm': median.tolist(), 'std_n_nm': values.std(axis=0).tolist(),
              'filtered_noise_p99_n': noise, 'accepted_limit_n': BASELINE_LIMIT_N}
force.close()
if noise > BASELINE_LIMIT_N:
    raise RuntimeError('baseline noise %.3f N above the accepted %.3f N' % (noise, BASELINE_LIMIT_N))
for p in points:
    p['unloaded_wrench_n_nm'] = median.tolist()

audit = existing.audit_points(points, q, np.zeros(6), support)
script, manifest = compiler.compile_program(points, '192.168.1.10', 50071)
report.update(waypoints=len(points), route_sha256=manifest['route_sha256'],
              joints_deg=np.rad2deg(np.array([p['q_rad'] for p in points])).tolist(),
              baseline_wrench_n_nm=median.tolist(), baseline_statistics=statistics,
              audit_scope=audit['scope'], depth_mm=DEPTH, standoff_mm=STANDOFF)
event({'status': 'multipass_preflight_passed_no_motion', 'waypoints': len(points)})
save(report)
if EXECUTE:
    watch = json.loads(existing.WATCHDOG.read_text())
    if watch.get('passed') is not True or watch.get('compiler_sha256') != existing.sha(compiler.__file__):
        raise RuntimeError('controller watchdog hardware evidence invalid')
    def write(name, value):
        path = out/(name+'.json'); path.write_text(json.dumps(value, indent=2) + '\n')
        return {'path': path.name, 'sha256': existing.sha(path)}
    evidence = {
        'physical_mapping': write('physical_mapping', {'passed': True,
            'route_sha256': manifest['route_sha256'],
            'scope': 'multi-pass widening test lines; commanded depth, not measured penetration'}),
        'force_reference': write('force_reference', {'passed': True,
            'route_sha256': manifest['route_sha256'],
            'scope': 'raw session wrench; the 10 N delta stop is unchanged',
            'baseline_wrench_n_nm': median.tolist(), 'statistics': statistics}),
        'blended_path_audit': write('blended_path_audit', {'passed': True,
            'route_sha256': manifest['route_sha256'], 'program_sha256': manifest['program_sha256'],
            'scope': audit['scope'], 'joints_deg': report['joints_deg']})}
    bundle = {'sample_id': 'multipass-line', 'points': points, 'start_pose_m_rad': start.tolist(),
              'active_tcp_offset_m_rad': [0.]*6, 'maximum_runtime_s': 900.,
              'physical_mapping_verified': True, 'force_reference_verified': True,
              'controller_watchdog_hardware_validated': True, 'evidence': evidence,
              'scope': 'multi-pass widening test', 'force_data_max_age_s': 1.5}
    blockers = compiler.execution_blockers(bundle, manifest, out)
    if blockers:
        raise RuntimeError('; '.join(blockers))
    (out/'bundle.json').write_text(json.dumps(bundle, indent=2) + '\n')
    (out/'program.script.preview').write_text(script)
    status = 'multipass_completed'
    try:
        guard.execute_buffered(bundle, out, '192.168.1.10', 50071, out/'execution.json', support, capture)
    except BaseException as exc:
        status = 'stopped_on_error'
        report['error'] = type(exc).__name__ + ': ' + str(exc)
    reader = live.ContinuousState(support, '192.168.1.3')
    _, after, _ = reader.stationary(); reader.close()
    report.update(status=status, motion_sent=True, final_pose_m_rad=after.tolist())
    save(report)
    event({'status': status, 'error': report.get('error')})
support.close_commands()
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path,
                        default=OUTPUT/'gong_current_mount_trial_fast/candidate.PREVIEW_ONLY.json')
    parser.add_argument('--depth-mm', type=float, default=.8)
    parser.add_argument('--standoff-mm', type=float, default=3.)
    parser.add_argument('--length-mm', type=float, default=25.)
    parser.add_argument('--u-start-mm', type=float, default=80.)
    parser.add_argument('--v-start-mm', type=float, default=15.)
    parser.add_argument('--gap-mm', type=float, default=15.)
    parser.add_argument('--variant', action='append', default=None,
                        help='one variant as comma-separated lateral offsets in mm, e.g. 0 or 0,-1.5,1.5; '
                             'a leading minus must not start the argument')
    parser.add_argument('--baseline-limit-n', type=float, default=.3)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--accept-test', help='operator confirmation recorded for these test lines')
    args = parser.parse_args()
    if args.execute and not args.accept_test:
        parser.error('--execute requires --accept-test with the operator reason')
    if not 0 < args.depth_mm <= 1.5:
        parser.error('--depth-mm must stay within (0,1.5] mm')
    specs = args.variant or ['0', '0,-1.5,1.5', '0,-3,3']
    variants = []
    for index, spec in enumerate(specs):
        offsets = [float(x) for x in spec.split(',')]
        if len(offsets) > 6:
            parser.error('at most six passes per variant')
        variants.append({'u_start_mm': args.u_start_mm, 'v_mm': args.v_start_mm+args.gap_mm*index,
                         'length_mm': args.length_mm, 'offsets_mm': offsets})
    if variants[-1]['v_mm']+max(max(abs(x) for x in var['offsets_mm']) for var in variants) > 100:
        parser.error('test pattern leaves the nominal A4 half height of 105 mm')
    cand = json.loads(args.candidate.read_text())
    u = np.array(cand['paper']['paper_u_base']); v = np.array(cand['paper']['paper_v_base'])
    n = np.array(cand['paper']['paper_normal_base'])
    origin = np.array(cand['paper']['reconstructed_center_tip_base_m'])
    tip = np.array(cand['tcp']['translation_m'])
    baseline = np.array(cand['conversion']['baseline_tool0_rotation_rad'], float)
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder = OUTPUT/'multipass'/('run-'+stamp)
    folder.mkdir(parents=True, exist_ok=False)
    remote = '/home/jx/.dsh-analysis/multipass-'+stamp
    header = ('BASELINE_LIMIT_N=' + repr(float(args.baseline_limit_n)) + chr(10) +
              'VARIANTS=' + repr(variants) + chr(10) +
              'DEPTH=' + repr(float(args.depth_mm)/1000) + chr(10) +
              'STANDOFF=' + repr(float(args.standoff_mm)) + chr(10) +
              'ORIGIN=' + repr(origin.tolist()) + chr(10) +
              'NORMAL=' + repr(n.tolist()) + chr(10) +
              'U_AXIS=' + repr(u.tolist()) + chr(10) +
              'V_AXIS=' + repr(v.tolist()) + chr(10) +
              'TIP=' + repr(tip.tolist()) + chr(10) +
              'BASELINE=' + repr(baseline.tolist()) + chr(10) +
              'EXECUTE=' + repr(bool(args.execute)) + chr(10) +
              'REMOTE_OUTPUT=' + repr(remote) + chr(10))
    payload = header+PAYLOAD
    (folder/'remote_multipass.py').write_text(payload)
    (folder/'request.json').write_text(json.dumps({'variants': variants, 'depth_mm': args.depth_mm,
        'standoff_mm': args.standoff_mm, 'baseline_limit_n': args.baseline_limit_n,
        'motion_sent': False}, indent=2) + chr(10))
    encoded = base64.b64encode(payload.encode()).decode()
    command = ('source /opt/ros/humble/setup.bash; source /home/jx/colcon_ws/install/setup.bash; '
               'export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1; '
               'echo ' + encoded + ' | base64 -d > /tmp/multipass.py && exec /usr/bin/python3 -B /tmp/multipass.py')
    with (folder/'stdout.txt').open('x') as log, (folder/'stderr.txt').open('x') as err:
        process = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', 'jx',
                                  'bash -c ' + shlex.quote(command)], text=True, capture_output=True, timeout=1200)
        log.write(process.stdout); err.write(process.stderr)
    last = None
    for line in (folder/'stdout.txt').read_text().splitlines():
        if line.startswith('{'):
            last = json.loads(line); print(line)
    copied = subprocess.run(['scp', '-q', '-r', 'jx:'+remote, str(folder/'remote_evidence')],
                            capture_output=True, text=True)
    result = {'local_evidence': str(folder.resolve()), 'ssh_exit_code': process.returncode,
              'raw_evidence_downloaded': copied.returncode == 0, 'result': last}
    (folder/'RESULT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + chr(10))
    if process.returncode or copied.returncode:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
