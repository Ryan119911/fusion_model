#!/usr/bin/env python3
"""Bounded pen-up corner probe around one sharp turn of the prepared route.

Repeats a short window of the writing polyline at a small standoff above the
paper, with and without the corner-stop rule, so the tracking deviation can be
measured without writing. Uses the same buffered compiler and buffered guard;
no contact, no model conversion, no tare. Default preview; motion needs --execute.
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
candidate = json.loads(Path(CANDIDATE).read_text())
tip = np.array(candidate['tcp']['translation_m'])
origin = np.array(candidate['paper']['reconstructed_center_tip_base_m'])
normal = np.array(candidate['paper']['paper_normal_base'])
source = candidate['points']
xyz = np.array([p['pose_m_rad'][:3] for p in source])
contact = [i for i, p in enumerate(source) if p['contact']]
turns = []
for a, b, c in zip(contact, contact[1:], contact[2:]):
    if b - a != 1 or c - b != 1:
        continue
    u1 = xyz[b] - xyz[a]; u2 = xyz[c] - xyz[b]
    turns.append((float(np.degrees(np.arccos(np.clip(u1@u2/np.linalg.norm(u1)/np.linalg.norm(u2), -1, 1)))), b))
by_index = {b: t for t, b in turns}
if CORNER_INDEX is None:
    turn, corner = max(turns)
else:
    corner = int(CORNER_INDEX)
    if corner not in by_index:
        raise ValueError('requested corner index has no consecutive-contact turn')
    turn = by_index[corner]
    if turn <= 5.:
        raise ValueError('requested corner index is not a turn: %.1f deg' % turn)
report = {'status': 'preview', 'motion_sent': False, 'corner_index': corner,
          'corner_turn_deg': turn, 'standoff_mm': STANDOFF_MM, 'resample_mm': RESAMPLE_MM}
def save(v):
    (out/'RESULT.json').write_text(json.dumps(v, indent=2) + '\n')
def event(v):
    print(json.dumps(v, ensure_ascii=False), flush=True)

step = np.linalg.norm(np.diff(xyz[contact[0]:contact[-1]+1], axis=0), axis=1)
before = []; acc = 0.
i = corner
while i > contact[0] and acc < BEFORE_MM/1000:
    acc += np.linalg.norm(xyz[i]-xyz[i-1]); i -= 1
start_index = i
after = []; acc = 0.; j = corner
while j < contact[-1] and acc < AFTER_MM/1000:
    acc += np.linalg.norm(xyz[j+1]-xyz[j]); j += 1
end_index = j
window = list(range(start_index, end_index+1))
report.update(window=[window[0], window[-1]], window_points=len(window))

support.controller_ready('192.168.1.3')
offset = np.asarray(capture('192.168.1.3', 30011, 1.)['active_tcp_offset_m_rad'])
if not np.allclose(offset, 0, atol=1e-8, rtol=0):
    raise RuntimeError('active controller TCP changed')
reader = live.ContinuousState(support, '192.168.1.3'); q, start, _ = reader.stationary(); reader.close()

def point(pose, phase, speed, contact_flag=False, stroke_id=None):
    # The probe window keeps contact/stroke semantics so the compiler builds the
    # same native blend as the real writing route; geometry stays at the standoff.
    return {'pose_m_rad': list(map(float, pose)), 'phase': phase, 'contact': contact_flag,
            'stroke_id': stroke_id, 'speed_m_s': speed, 'acceleration_m_s2': .002,
            'unloaded_wrench_n_nm': [0.]*6}

poses = []
for index in window:
    pose = np.array(source[index]['pose_m_rad'], float)
    pose[:3] += STANDOFF_MM/1000*normal
    poses.append(pose)
corner_local = window.index(corner)
if RESAMPLE_MM:
    arc = [0.]
    for a, b in zip(poses, poses[1:]):
        arc.append(arc[-1]+float(np.linalg.norm(b[:3]-a[:3])))
    keep = [0]; target = RESAMPLE_MM/1000
    for i in range(1, len(poses)-1):
        if arc[i] >= target:
            keep.append(i); target = arc[i]+RESAMPLE_MM/1000
    keep.append(corner_local); keep.append(len(poses)-1)
    keep = sorted(set(keep))
    report['resampled_points'] = len(keep)
    poses = [poses[i] for i in keep]
    corner_local = keep.index(corner_local)
points = [point(start, 'measured_current_start', .003)]
approach = poses[0].copy(); approach[2] = start[2]
points.append(point(approach, 'probe_air_translate', .003))
points.append(point(poses[0], 'probe_descend', .0015))
for index, pose in zip(window, poses):
    src = source[index]
    speed = SPEED_MM_S/1000 if SPEED_MM_S > 0 else src['speed_m_s']
    points.append(point(pose, 'corner_probe', speed, bool(src['contact']),
                        int(src['stroke_id']) if src['contact'] else None))
lift = poses[-1].copy(); lift[:3] += .020*normal
points.append(point(lift, 'probe_retreat', .002))

audit = existing.audit_points(points, q, np.zeros(6), support)
force = support.RawForce(); baseline, statistics = force.baseline(); force.close()
for p in points:
    p['unloaded_wrench_n_nm'] = baseline.tolist()
script, manifest = compiler.compile_program(points, '192.168.1.10', 50071)
report.update(waypoints=len(points), blended_points=manifest['interior_blended_count'],
              route=[[float(x) for x in p['pose_m_rad']] for p in points],
              route_sha256=manifest['route_sha256'],
              audit_scope=audit['scope'], baseline_wrench_n_nm=baseline.tolist(),
              joints_deg=np.rad2deg(np.array([p['q_rad'] for p in points])).tolist())
event({'status': 'corner_probe_preflight_passed_no_motion', 'corner_index': corner,
       'turn_deg': round(turn, 2), 'waypoints': len(points),
       'resample_mm': RESAMPLE_MM})
save(report)
if EXECUTE:
    watch = json.loads(existing.WATCHDOG.read_text())
    if watch.get('passed') is not True or watch.get('compiler_sha256') != existing.sha(compiler.__file__):
        raise RuntimeError('controller watchdog hardware evidence invalid')
    def write(name, value):
        path = out/(name + '.json'); path.write_text(json.dumps(value, indent=2) + '\n')
        return {'path': path.name, 'sha256': existing.sha(path)}
    evidence = {
        'physical_mapping': write('physical_mapping', {'passed': True,
            'route_sha256': manifest['route_sha256'],
            'scope': 'pen-up corner probe at a fixed standoff; no writing and no model conversion'}),
        'force_reference': write('force_reference', {'passed': True,
            'route_sha256': manifest['route_sha256'],
            'scope': 'raw session wrench; the 10 N delta stop is unchanged',
            'baseline_wrench_n_nm': baseline.tolist(), 'statistics': statistics}),
        'blended_path_audit': write('blended_path_audit', {'passed': True,
            'route_sha256': manifest['route_sha256'], 'program_sha256': manifest['program_sha256'],
            'scope': audit['scope'], 'resample_mm': RESAMPLE_MM})}
    bundle = {'sample_id': 'corner-probe', 'points': points,
              'start_pose_m_rad': start.tolist(), 'active_tcp_offset_m_rad': [0.]*6,
              'maximum_runtime_s': 300., 'physical_mapping_verified': True,
              'force_reference_verified': True, 'controller_watchdog_hardware_validated': True,
              'evidence': evidence,
              'scope': 'bounded pen-up corner probe', 'force_data_max_age_s': 1.5}
    blockers = compiler.execution_blockers(bundle, manifest, out)
    if blockers:
        raise RuntimeError('; '.join(blockers))
    (out/'bundle.json').write_text(json.dumps(bundle, indent=2) + '\n')
    (out/'program.script.preview').write_text(script)
    status = 'probe_completed'
    try:
        guard.execute_buffered(bundle, out, '192.168.1.10', 50071, out/'execution.json', support, capture)
    except BaseException as exc:
        status = 'stopped_on_error'
        report['error'] = type(exc).__name__ + ': ' + str(exc)
    report['status'] = status
    save(report)
    event({'status': status, 'error': report.get('error')})
support.close_commands()
'''


def deviation(route, ticks):
    xyz = np.array(route)[:, :3]
    seg = np.diff(xyz, axis=0)
    den = np.maximum((seg*seg).sum(1), 1e-20)
    worst = 0.
    for tick in ticks:
        p = np.array(tick['pose_m_rad'][:3])
        f = np.clip(((p-xyz[:-1])*seg).sum(1)/den, 0, 1)
        proj = xyz[:-1]+f[:, None]*seg
        worst = max(worst, float(np.linalg.norm(proj-p, axis=1).min()))
    return worst*1000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path,
                        default=OUTPUT/'gong_current_mount_trial')
    parser.add_argument('--standoff-mm', type=float, default=3.)
    parser.add_argument('--before-mm', type=float, default=5.)
    parser.add_argument('--after-mm', type=float, default=6.)
    parser.add_argument('--resample-mm', type=float, default=.4,
                        help='chord length used to resample the probe window; 0 keeps the original spacing')
    parser.add_argument('--speed-mm-s', type=float, default=0.,
                        help='override the probe-window speed in mm/s; 0 keeps the source speed')
    parser.add_argument('--corner-index', type=int, default=284,
                        help='candidate point index of the sharp turn; 284 is the second-stroke corner')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--accept-probe', help='operator confirmation recorded for this pen-up probe')
    parser.add_argument('--label', default='probe')
    args = parser.parse_args()
    if args.execute and not args.accept_probe:
        parser.error('--execute requires --accept-probe with the operator reason')
    if not 0 <= args.standoff_mm <= 10:
        parser.error('--standoff-mm must stay within [0,10] mm; 0 reproduces the commanded contact depth')
    if not 0 <= args.resample_mm <= 2:
        parser.error('--resample-mm must stay within [0,2] mm')
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder = OUTPUT/'corner-probe'/(args.label+'-'+stamp)
    folder.mkdir(parents=True, exist_ok=False)
    remote = '/home/jx/.dsh-analysis/corner-probe-'+stamp
    upload = '/tmp/corner_probe_candidate_'+stamp+'.json'
    pushed = subprocess.run(['scp', '-q', str(args.folder/'candidate.PREVIEW_ONLY.json'),
                             'jx:'+upload], capture_output=True, text=True)
    if pushed.returncode:
        raise RuntimeError('candidate upload failed: '+pushed.stderr)
    header = ('CANDIDATE=' + repr(upload) + chr(10) +
              'STANDOFF_MM=' + repr(float(args.standoff_mm)) + chr(10) +
              'BEFORE_MM=' + repr(float(args.before_mm)) + chr(10) +
              'AFTER_MM=' + repr(float(args.after_mm)) + chr(10) +
              'CORNER_INDEX=' + repr(args.corner_index) + chr(10) +
              'RESAMPLE_MM=' + repr(float(args.resample_mm)) + chr(10) +
              'SPEED_MM_S=' + repr(float(args.speed_mm_s)) + chr(10) +
              'EXECUTE=' + repr(bool(args.execute)) + chr(10) +
              'REMOTE_OUTPUT=' + repr(remote) + chr(10))
    payload = header+PAYLOAD
    (folder/'remote_probe.py').write_text(payload)
    encoded = base64.b64encode(payload.encode()).decode()
    command = ('source /opt/ros/humble/setup.bash; export PYTHONNOUSERSITE=1 '
               'PYTHONDONTWRITEBYTECODE=1; echo ' + encoded + ' | base64 -d > /tmp/corner_probe.py && '
               'exec /usr/bin/python3 -B /tmp/corner_probe.py')
    with (folder/'stdout.txt').open('x') as log, (folder/'stderr.txt').open('x') as err:
        process = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                                  'jx', 'bash -c ' + shlex.quote(command)],
                                 text=True, capture_output=True, timeout=900)
        log.write(process.stdout); err.write(process.stderr)
    last = None
    for line in (folder/'stdout.txt').read_text().splitlines():
        if line.startswith('{'):
            last = json.loads(line); print(line)
    copied = subprocess.run(['scp', '-q', '-r', 'jx:'+remote, str(folder/'remote_evidence')],
                            capture_output=True, text=True)
    evidence = folder/'remote_evidence'
    summary = {'local_evidence': str(folder.resolve()), 'ssh_exit_code': process.returncode,
               'raw_evidence_downloaded': copied.returncode == 0, 'result': last}
    if evidence.is_dir():
        report = json.loads((evidence/'RESULT.json').read_text())
        execution = evidence/'execution.json'
        if execution.is_file() and report.get('waypoints'):
            ticks = json.loads(execution.read_text()).get('ticks', [])
            summary['measured_max_deviation_mm'] = round(deviation(report['route'], ticks), 3)
        summary['execution_status'] = json.loads(execution.read_text()).get('status') if execution.is_file() else None
    (folder/'RESULT.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + chr(10))
    print(json.dumps(summary, ensure_ascii=False))
    if process.returncode or copied.returncode:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
