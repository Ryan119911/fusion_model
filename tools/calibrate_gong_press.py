#!/usr/bin/env python3
"""Write a short press-depth ladder for H calibration. Default preview only.

Each line is drawn at a fixed orientation along paper +u at one commanded
penetration below the historical paper plane. This is a calibration probe, not
model writing; it deliberately exceeds the 1 mm trial envelope (operator-approved
2026-10-28) and is capped at --max-depth-mm.
"""
import argparse
import base64
import datetime as dt
import json
import shlex
import subprocess
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

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
report = {'status': 'preview', 'motion_sent': False, 'depths_mm': DEPTHS}
def save(v):
    (out/'RESULT.json').write_text(json.dumps(v, indent=2) + '\n')
def event(v):
    print(json.dumps(v, ensure_ascii=False), flush=True)

support.controller_ready('192.168.1.3')
offset = np.asarray(capture('192.168.1.3', 30011, 1.)['active_tcp_offset_m_rad'])
if not np.allclose(offset, 0, atol=1e-8, rtol=0):
    raise RuntimeError('active controller TCP changed')
reader = live.ContinuousState(support, '192.168.1.3'); q, start, _ = reader.stationary(); reader.close()

origin = np.array(ORIGIN); normal = np.array(NORMAL); u = np.array(U_AXIS); v = np.array(V_AXIS)
tip = np.array(TIP); rot = np.array(BASELINE, float)
points = [{'pose_m_rad': start.tolist(), 'phase': 'measured_current_start', 'contact': False,
           'stroke_id': None, 'speed_m_s': .010, 'acceleration_m_s2': .100,
           'unloaded_wrench_n_nm': [0.]*6}]

_rotation = Rotation.from_rotvec(rot)
def add(tip_point, phase, speed, contact, stroke):
    tool0 = tip_point-_rotation.apply(tip)
    points.append({'pose_m_rad': list(map(float, np.r_[tool0, rot])), 'phase': phase,
                   'contact': contact, 'stroke_id': stroke, 'speed_m_s': speed,
                   'acceleration_m_s2': .100, 'unloaded_wrench_n_nm': [0.]*6})

for index, depth in enumerate(DEPTHS):
    vv = V_START+GAP*index
    left = origin+u*(U_START/1000)+v*(vv/1000)
    right = origin+u*((U_START+LENGTH)/1000)+v*(vv/1000)
    add(left+normal*.003, 'cal_approach', .005, False, None)
    add(left-normal*(depth/1000), 'cal_touch', .0005, False, None)
    add(right-normal*(depth/1000), 'cal_line', .010, True, index)
    add(right+normal*.003, 'cal_lift', .005, False, None)

def measured_baseline(force, limit_n):
    start = time.monotonic(); deadline = start+10
    while time.monotonic() < deadline:
        time.sleep(.25)
        with force.lock:
            count = sum(1 for r in force.rows if r[0] >= start)
        if count >= 200 and time.monotonic()-start >= 2:
            break
    with force.lock:
        rows = [r for r in force.rows if r[0] >= start]
    if len(rows) < 200:
        raise RuntimeError('force rate too low for a baseline window')
    values = np.asarray([r[3] for r in rows], float)
    median = np.median(values, axis=0)
    filtered = np.asarray([np.median(values[i-4:i+1, :3], axis=0) for i in range(4, len(values))])
    noise = float(np.quantile(np.linalg.norm(filtered-median[:3], axis=1), .99))
    stats = {'count': len(rows), 'median_n_nm': median.tolist(),
             'std_n_nm': values.std(axis=0).tolist(), 'filtered_noise_p99_n': noise,
             'accepted_limit_n': float(limit_n),
             'scope': 'calibration ladder; probe default is 0.3 N, the 10 N stop is unchanged'}
    if noise > limit_n:
        raise RuntimeError('baseline noise %.3f N above the accepted %.3f N' % (noise, limit_n))
    return median, stats


force = support.RawForce(); baseline, statistics = measured_baseline(force, BASELINE_LIMIT_N); force.close()
audit = existing.audit_points(points, q, np.zeros(6), support)
for p in points:
    p['unloaded_wrench_n_nm'] = baseline.tolist()
script, manifest = compiler.compile_program(points, '192.168.1.10', 50071)
report.update(waypoints=len(points), route_sha256=manifest['route_sha256'],
              joints_deg=np.rad2deg(np.array([p['q_rad'] for p in points])).tolist(),
              baseline_wrench_n_nm=baseline.tolist(), audit_scope=audit['scope'],
              pattern={'u_start_mm': U_START, 'length_mm': LENGTH, 'gap_mm': GAP, 'v_start_mm': V_START})
event({'status': 'press_ladder_preflight_passed_no_motion', 'waypoints': len(points)})
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
            'scope': 'press-depth calibration ladder; penetration is commanded, not measured'}),
        'force_reference': write('force_reference', {'passed': True,
            'route_sha256': manifest['route_sha256'],
            'scope': 'raw session wrench; the 10 N delta stop is unchanged',
            'baseline_wrench_n_nm': baseline.tolist(), 'statistics': statistics}),
        'blended_path_audit': write('blended_path_audit', {'passed': True,
            'route_sha256': manifest['route_sha256'], 'program_sha256': manifest['program_sha256'],
            'scope': audit['scope'], 'joints_deg': report['joints_deg']})}
    bundle = {'sample_id': 'press-ladder', 'points': points, 'start_pose_m_rad': start.tolist(),
              'active_tcp_offset_m_rad': [0.]*6, 'maximum_runtime_s': 300.,
              'physical_mapping_verified': True, 'force_reference_verified': True,
              'controller_watchdog_hardware_validated': True, 'evidence': evidence,
              'force_stop_n': FORCE_STOP_N,
              'scope': 'press-depth calibration ladder', 'force_data_max_age_s': 1.5}
    blockers = compiler.execution_blockers(bundle, manifest, out)
    if blockers:
        raise RuntimeError('; '.join(blockers))
    (out/'bundle.json').write_text(json.dumps(bundle, indent=2) + '\n')
    (out/'program.script.preview').write_text(script)
    guard.execute_buffered(bundle, out, '192.168.1.10', 50071, out/'execution.json', support, capture)
    reader = live.ContinuousState(support, '192.168.1.3'); q2, after, _ = reader.stationary(); reader.close()
    report.update(status='press_ladder_completed', motion_sent=True, final_pose_m_rad=after.tolist())
    save(report)
    event({'status': 'press_ladder_completed'})
support.close_commands()
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path,
                        default=OUTPUT/'gong_current_mount_trial_fast/candidate.PREVIEW_ONLY.json')
    parser.add_argument('--depths', nargs='+', type=float, default=[0.5, 1.0, 1.5, 2.0])
    parser.add_argument('--max-depth-mm', type=float, default=2.0)
    parser.add_argument('--length-mm', type=float, default=25.)
    parser.add_argument('--gap-mm', type=float, default=15.)
    parser.add_argument('--u-start-mm', type=float, default=80.)
    parser.add_argument('--v-start-mm', type=float, default=-40.)
    parser.add_argument('--baseline-limit-n', type=float, default=.3,
                        help='baseline noise acceptance; the probe default is 0.3 N')
    parser.add_argument('--force-stop-n', type=float, default=10.,
                        help='live force increment stop for this test only; default 10 N')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--accept-ladder', help='operator confirmation recorded for these calibration strokes')
    args = parser.parse_args()
    if args.execute and not args.accept_ladder:
        parser.error('--execute requires --accept-ladder with the operator reason')
    if max(args.depths) > args.max_depth_mm:
        parser.error('depth exceeds the approved --max-depth-mm')
    if not 0 < args.force_stop_n <= 100:
        parser.error('--force-stop-n must stay within (0,100] N')
    if min(args.depths) <= 0:
        parser.error('depths must be positive')
    cand = json.loads(args.candidate.read_text())
    u = np.array(cand['paper']['paper_u_base']); v = np.array(cand['paper']['paper_v_base'])
    n = np.array(cand['paper']['paper_normal_base'])
    origin = np.array(cand['paper']['reconstructed_center_tip_base_m'])
    tip = np.array(cand['tcp']['translation_m'])
    baseline = np.array(cand['conversion']['baseline_tool0_rotation_rad'], float)
    # Bounds come from the measured sheet, not from the nominal A4 half sizes:
    # the 2026-10-28 Mech-Eye cloud covers u 2..198 mm and v -62..104 mm.
    u_end = args.u_start_mm+args.length_mm
    v_end = args.v_start_mm+args.gap_mm*(len(args.depths)-1)
    if u_end > 195 or args.u_start_mm < 0 or args.v_start_mm < -62 or v_end > 102:
        parser.error('calibration pattern leaves the measured paper (u 2..198, v -62..104 mm)')
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder = OUTPUT/'press-ladder'/('run-'+stamp)
    folder.mkdir(parents=True, exist_ok=False)
    remote = '/home/jx/.dsh-analysis/press-ladder-'+stamp
    header = ('BASELINE_LIMIT_N=' + repr(float(args.baseline_limit_n)) + chr(10) +
              'FORCE_STOP_N=' + repr(float(args.force_stop_n)) + chr(10) +
              'DEPTHS=' + repr([float(x) for x in args.depths]) + chr(10) +
              'LENGTH=' + repr(float(args.length_mm)) + chr(10) +
              'GAP=' + repr(float(args.gap_mm)) + chr(10) +
              'U_START=' + repr(float(args.u_start_mm)) + chr(10) +
              'V_START=' + repr(float(args.v_start_mm)) + chr(10) +
              'ORIGIN=' + repr(origin.tolist()) + chr(10) +
              'NORMAL=' + repr(n.tolist()) + chr(10) +
              'U_AXIS=' + repr(u.tolist()) + chr(10) +
              'V_AXIS=' + repr(v.tolist()) + chr(10) +
              'TIP=' + repr(tip.tolist()) + chr(10) +
              'BASELINE=' + repr(baseline.tolist()) + chr(10) +
              'EXECUTE=' + repr(bool(args.execute)) + chr(10) +
              'REMOTE_OUTPUT=' + repr(remote) + chr(10))
    payload = header+PAYLOAD
    (folder/'remote_ladder.py').write_text(payload)
    (folder/'request.json').write_text(json.dumps({'depths_mm': args.depths,
        'length_mm': args.length_mm, 'gap_mm': args.gap_mm,
        'u_start_mm': args.u_start_mm, 'v_start_mm': args.v_start_mm,
        'max_depth_mm': args.max_depth_mm, 'motion_sent': False}, indent=2) + chr(10))
    encoded = base64.b64encode(payload.encode()).decode()
    command = ('source /opt/ros/humble/setup.bash; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1; '
               'echo ' + encoded + ' | base64 -d > /tmp/ladder.py && exec /usr/bin/python3 -B /tmp/ladder.py')
    with (folder/'stdout.txt').open('x') as log, (folder/'stderr.txt').open('x') as err:
        process = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', 'jx',
                                  'bash -c ' + shlex.quote(command)], text=True, capture_output=True, timeout=900)
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
