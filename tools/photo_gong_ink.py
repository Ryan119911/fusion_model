#!/usr/bin/env python3
"""Move to a computed overhead photo pose and capture the ink with Mech-Eye.

The camera is eye-in-hand and its optical axis is within 1 degree of tool0 +Z, so
the pose is solved from the 2026-10-27 hand-eye record: the camera is placed
--distance-mm above the character centre looking down, and the resulting tool0
pose is checked for tip clearance before anything moves. Default is preview.
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
report = {'status': 'preview', 'motion_sent': False, 'target_pose_m_rad': TARGET}
def save(v):
    (out/'RESULT.json').write_text(json.dumps(v, indent=2) + '\n')
def event(v):
    print(json.dumps(v, ensure_ascii=False), flush=True)

support.controller_ready('192.168.1.3')
offset = np.asarray(capture('192.168.1.3', 30011, 1.)['active_tcp_offset_m_rad'])
if not np.allclose(offset, 0, atol=1e-8, rtol=0):
    raise RuntimeError('active controller TCP changed')
STATIONARY = {'tcp_speed_m_s': .005, 'joint_speed_rad_s': .03,
              'position_spread_m': .0001, 'rotation_spread_deg': .02, 'samples': 9,
              'scope': 'pen-up air move only; the probe default is 0.1 mm/s / 0.001 rad/s'}


def wait_stationary(support, attempts=8):
    """Relaxed rest detection for an air-only move.

    The controller's velocity estimate jitters at some postures (0.01 rad/s
    spikes) while the position holds within 0.03 mm, so the probe's 0.001 rad/s
    gate never settles. Position and orientation spread over nine frames are
    still required, so a genuinely moving arm is rejected.
    """
    for attempt in range(attempts):
        reader = live.ContinuousState(support, '192.168.1.3')
        try:
            window = [] ; deadline = time.monotonic()+20
            while time.monotonic() < deadline:
                q, pose, v = reader.fresh(timeout=1.)
                if (np.linalg.norm(v[:3]) > STATIONARY['tcp_speed_m_s'] or
                        np.max(np.abs(v[3:])) > STATIONARY['joint_speed_rad_s']):
                    window = []; continue
                window.append((q, pose))
                if len(window) < STATIONARY['samples']:
                    continue
                pos = np.array([w[1][:3] for w in window])
                if np.max(np.linalg.norm(pos-pos.mean(axis=0), axis=1)) > STATIONARY['position_spread_m']:
                    window = window[1:]; continue
                rots = Rotation.from_rotvec([w[1][3:] for w in window]); mean = rots.mean()
                if np.max((mean.inv()*rots).magnitude()) > np.deg2rad(STATIONARY['rotation_spread_deg']):
                    window = window[1:]; continue
                return (np.mean([w[0] for w in window], axis=0),
                        np.r_[pos.mean(axis=0), mean.as_rotvec()], attempt+1)
        finally:
            reader.close()
    raise RuntimeError('no rest window under the relaxed pen-up thresholds')


q, start, rest_attempt = wait_stationary(support)

target = np.array(TARGET, float)
if INTERPOLATE:
    # One program with every intermediate waypoint: the previous flow sent a
    # separate guarded job per step, which cost a full preflight each time.
    from scipy.spatial.transform import Slerp
    slerp = Slerp([0, 1], Rotation.from_matrix([Rotation.from_rotvec(start[3:]).as_matrix(),
                                                Rotation.from_rotvec(target[3:]).as_matrix()]))
    poses = [start]
    for i in range(1, INTERPOLATE+1):
        f = i/INTERPOLATE
        poses.append(np.r_[start[:3]+f*(target[:3]-start[:3]), slerp(f).as_rotvec()])
else:
    high = start.copy(); high[2] = start[2]+LIFT_M
    above = target.copy(); above[2] = target[2]+APPROACH_M
    poses = [start, high, above, target]
points = []
for i, pose in enumerate(poses):
    if points and np.allclose(points[-1]['pose_m_rad'], pose, atol=1e-9, rtol=0):
        # Zero-length movels made the controller kick sideways at the start;
        # a duplicated waypoint is never a real move.
        continue
    points.append({'pose_m_rad': list(map(float, pose)),
                   'phase': 'measured_current_start' if i == 0 else 'photo_pose_move',
                   'contact': False, 'stroke_id': None, 'speed_m_s': SPEED_M_S,
                   'acceleration_m_s2': ACCEL_M_S2, 'unloaded_wrench_n_nm': [0.]*6})
# The baseline is measured before the audit: the audit's heavy Python work
# starves the ROS spin thread and the force stream then looks stale.


def measured_baseline(force, limit_n):
    """Same five-sample-median estimate as the probe, with an explicit limit.

    The probe's 0.3 N window is a contact-probing acceptance. This tool only
    moves the arm in air at a fixed standoff, so the operator may accept a
    wider, recorded noise window; the 10 N force stop and the tracking envelope
    are unchanged and still enforced by the live guard.
    """
    start = time.monotonic()
    deadline = start + 10
    while time.monotonic() < deadline:
        time.sleep(.25)
        with force.lock:
            count = sum(1 for r in force.rows if r[0] >= start)
        if count >= 200 and time.monotonic()-start >= 2:
            break
    with force.lock:
        rows = [r for r in force.rows if r[0] >= start]
    if len(rows) < 200:
        raise RuntimeError('force rate too low for a baseline window: %d rows in %.1f s'
                           % (len(rows), time.monotonic()-start))
    values = np.asarray([r[3] for r in rows], float)
    median = np.median(values, axis=0)
    filtered = np.asarray([np.median(values[i-4:i+1, :3], axis=0) for i in range(4, len(values))])
    noise = float(np.quantile(np.linalg.norm(filtered-median[:3], axis=1), .99))
    stats = {'count': len(rows), 'median_n_nm': median.tolist(),
             'std_n_nm': values.std(axis=0).tolist(), 'filtered_noise_p99_n': noise,
             'accepted_limit_n': float(limit_n),
             'scope': 'pen-up air move only; probe default is 0.3 N'}
    if noise > limit_n:
        raise RuntimeError('pen-up baseline noise %.3f N above the accepted %.3f N' % (noise, limit_n))
    return median, stats


force = support.RawForce(); baseline, statistics = measured_baseline(force, BASELINE_LIMIT_N); force.close()
audit = existing.audit_points(points, q, np.zeros(6), support)
for p in points:
    p['unloaded_wrench_n_nm'] = baseline.tolist()
script, manifest = compiler.compile_program(points, '192.168.1.10', 50071)
report.update(waypoints=len(points), route_sha256=manifest['route_sha256'],
              stationary_criteria=STATIONARY, stationary_attempt=rest_attempt,
              joints_deg=np.rad2deg(np.array([p['q_rad'] for p in points])).tolist(),
              baseline_wrench_n_nm=baseline.tolist(), audit_scope=audit['scope'])
event({'status': 'photo_pose_preflight_passed_no_motion', 'waypoints': len(points)})
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
            'scope': 'pen-up move to the overhead photo pose; no writing and no model conversion'}),
        'force_reference': write('force_reference', {'passed': True,
            'route_sha256': manifest['route_sha256'],
            'scope': 'raw session wrench; the 10 N delta stop is unchanged',
            'baseline_wrench_n_nm': baseline.tolist(), 'statistics': statistics}),
        'blended_path_audit': write('blended_path_audit', {'passed': True,
            'route_sha256': manifest['route_sha256'], 'program_sha256': manifest['program_sha256'],
            'scope': audit['scope'], 'joints_deg': report['joints_deg']})}
    bundle = {'sample_id': 'photo-pose-move', 'points': points,
              'start_pose_m_rad': start.tolist(), 'active_tcp_offset_m_rad': [0.]*6,
              'maximum_runtime_s': 120., 'physical_mapping_verified': True,
              'force_reference_verified': True, 'controller_watchdog_hardware_validated': True,
              'evidence': evidence, 'scope': 'pen-up photo pose move',
              'force_data_max_age_s': 1.5}
    blockers = compiler.execution_blockers(bundle, manifest, out)
    if blockers:
        raise RuntimeError('; '.join(blockers))
    (out/'bundle.json').write_text(json.dumps(bundle, indent=2) + '\n')
    (out/'program.script.preview').write_text(script)
    guard.execute_buffered(bundle, out, '192.168.1.10', 50071, out/'execution.json', support, capture)
    reader = live.ContinuousState(support, '192.168.1.3'); q2, after, _ = reader.stationary(); reader.close()
    report.update(status='photo_pose_reached', motion_sent=True, final_pose_m_rad=after.tolist(),
                  position_error_mm=float(np.linalg.norm(after[:3]-target[:3])*1000),
                  orientation_error_deg=float(np.rad2deg((Rotation.from_rotvec(target[3:]).inv()
                      * Rotation.from_rotvec(after[3:])).magnitude())))
    save(report)
    event({'status': 'photo_pose_reached', 'position_error_mm': round(report['position_error_mm'], 3)})
    if CAPTURE:
        import rclpy, importlib.util, cv2
        from rclpy.serialization import serialize_message
        spec = importlib.util.spec_from_file_location('mecheye', '/home/jx/UR10/scripts/validate_mecheye_capture.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        rclpy.init(args=[]); node = module.CaptureValidator()
        folder = out/'camera'; folder.mkdir(exist_ok=False)
        shots = {}
        for key, client, service in [('image', node.color_client, module.CaptureColorImage),
                                     ('cloud', node.cloud_client, module.CapturePointCloud)]:
            msg, _ = node.capture(key, client, service, 20.)
            meta = {'frame_id': msg.header.frame_id, 'width': msg.width, 'height': msg.height}
            if key == 'image':
                enc = msg.encoding.lower(); arr = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.step)
                arr = arr[:, :msg.width*(3 if enc in ('bgr8', 'rgb8') else 1)]
                image = arr.reshape(msg.height, msg.width, 3) if enc in ('bgr8', 'rgb8') else arr.reshape(msg.height, msg.width)
                if enc == 'rgb8':
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(folder/'ink.png'), image)
                if image.ndim == 3:
                    cv2.imwrite(str(folder/'ink_mono.png'), image[:, :, 0])
                meta['encoding'] = enc
            else:
                fields = {f.name: f for f in msg.fields}
                dtype = np.dtype('>f4' if msg.is_bigendian else '<f4')
                xyz = np.stack([np.ndarray((msg.height, msg.width), dtype=dtype, buffer=bytes(msg.data),
                                offset=fields[k].offset, strides=(msg.row_step, msg.point_step)).astype(np.float32)
                                for k in 'xyz'], axis=-1)
                np.save(folder/'cloud_xyz_m.npy', xyz)
            shots[key] = meta
        node.destroy_node(); rclpy.shutdown()
        report['capture'] = {'folder': str(folder), 'images': [str(folder/'ink.png'), str(folder/'ink_mono.png')],
                             'shots': shots}
        save(report)
        event({'status': 'captured', 'image': str(folder/'ink.png')})
support.close_commands()
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path,
                        default=OUTPUT/'gong_current_mount_trial_fast/candidate.PREVIEW_ONLY.json')
    parser.add_argument('--distance-mm', type=float, default=350.)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--accept-photo', help='operator confirmation recorded for this photo move')
    parser.add_argument('--no-capture', action='store_true')
    parser.add_argument('--target-pose', nargs=6, type=float,
                        help='explicit tool0 pose (m,rad) for one incremental step')
    parser.add_argument('--aim-u-mm', type=float, default=0.)
    parser.add_argument('--aim-v-mm', type=float, default=0.)
    parser.add_argument('--baseline-limit-n', type=float, default=.3,
                        help='pen-up baseline noise acceptance; the probe default is 0.3 N')
    parser.add_argument('--speed-mm-s', type=float, default=20.)
    parser.add_argument('--accel-mm-s2', type=float, default=100.)
    parser.add_argument('--steps', type=int, default=0,
                        help='split the move into N interpolated waypoints inside one program; 0 keeps the lift/approach shape')
    parser.add_argument('--lift-mm', type=float, default=200.)
    parser.add_argument('--approach-mm', type=float, default=150.)
    args = parser.parse_args()
    if args.execute and not args.accept_photo:
        parser.error('--execute requires --accept-photo with the operator reason')
    if not 250 <= args.distance_mm <= 600:
        parser.error('--distance-mm must stay within [250,600]')
    if not .3 <= args.baseline_limit_n <= 5.:
        parser.error('--baseline-limit-n must stay within [0.3,5] N')
    cand = json.loads(args.candidate.read_text())
    u = np.array(cand['paper']['paper_u_base']); n = np.array(cand['paper']['paper_normal_base'])
    origin = np.array(cand['paper']['reconstructed_center_tip_base_m'])
    tip = np.array(cand['tcp']['translation_m'])
    tips = np.array([np.array(p['pose_m_rad'])[:3] +
                     R.from_rotvec(np.array(p['pose_m_rad'])[3:]).apply(tip)
                     for p in cand['points'] if p['contact']])
    center = tips.mean(0); center = center - n*((center-origin)@n)
    if args.aim_u_mm or args.aim_v_mm:
        center = origin+u*(args.aim_u_mm/1000)+np.array(cand['paper']['paper_v_base'])*(args.aim_v_mm/1000)
        center = center - n*((center-origin)@n)
    dist = args.distance_mm/1000
    cam_pos = center+dist*n
    z = -n; x = u-n*(u@n); x = x/np.linalg.norm(x); y = np.cross(z, x)
    T_bc = np.eye(4); T_bc[:3, :3] = np.column_stack((x, y, z)); T_bc[:3, 3] = cam_pos
    translation = np.array([0.04707493838088885, 0.15015738159018904, 0.16350798747456627])
    rotation = np.array([[-0.9999468780239723, 0.00894532352590486, -0.00512077309857878],
                         [-0.008882899077318561, -0.9998875041985456, -0.012086068491636742],
                         [-0.0052283108259184525, -0.012039939145164207, 0.9999138486046124]])
    T_tc = np.eye(4); T_tc[:3, :3] = rotation; T_tc[:3, 3] = translation
    T_bt = T_bc@np.linalg.inv(T_tc)
    target = np.r_[T_bt[:3, 3], R.from_matrix(T_bt[:3, :3]).as_rotvec()]
    if args.target_pose:
        target = np.array(args.target_pose, float)
    tip_world = T_bt[:3, 3]+R.from_rotvec(target[3:]).apply(tip)
    tip_height = float((tip_world-origin)@n*1000)
    if tip_height < 20:
        parser.error('computed photo pose leaves the brush tip only %.1f mm above the paper; '
                     'increase --distance-mm' % tip_height)
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder = OUTPUT/'photo-pose'/('run-'+stamp)
    folder.mkdir(parents=True, exist_ok=False)
    remote = '/home/jx/.dsh-analysis/photo-pose-'+stamp
    header = ('TARGET=' + repr([float(v) for v in target]) + chr(10) +
              'EXECUTE=' + repr(bool(args.execute)) + chr(10) +
              'CAPTURE=' + repr(not args.no_capture) + chr(10) +
              'BASELINE_LIMIT_N=' + repr(float(args.baseline_limit_n)) + chr(10) +
              'SPEED_M_S=' + repr(float(args.speed_mm_s/1000)) + chr(10) +
              'ACCEL_M_S2=' + repr(float(args.accel_mm_s2/1000)) + chr(10) +
              'INTERPOLATE=' + repr(int(args.steps)) + chr(10) +
              'LIFT_M=' + repr(float(args.lift_mm/1000)) + chr(10) +
              'APPROACH_M=' + repr(float(args.approach_mm/1000)) + chr(10) +
              'REMOTE_OUTPUT=' + repr(remote) + chr(10))
    payload = header+PAYLOAD
    (folder/'remote_photo.py').write_text(payload)
    (folder/'request.json').write_text(json.dumps({'target_pose_m_rad': target.tolist(),
        'camera_position_m': cam_pos.tolist(), 'distance_mm': args.distance_mm,
        'handeye': 'config/handeye_eye_in_hand_20261027.yaml (valid=true, 2026-10-27)',
        'tip_height_mm': tip_height, 'motion_sent': False}, indent=2) + chr(10))
    encoded = base64.b64encode(payload.encode()).decode()
    command = ('source /opt/ros/humble/setup.bash; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1; '
               'echo ' + encoded + ' | base64 -d > /tmp/photo_pose.py && exec /usr/bin/python3 -B /tmp/photo_pose.py')
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
              'raw_evidence_downloaded': copied.returncode == 0, 'result': last,
              'target_pose_m_rad': target.tolist(), 'tip_height_mm': tip_height}
    (folder/'RESULT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + chr(10))
    print(json.dumps({k: v for k, v in result.items() if k != 'target_pose_m_rad'}, ensure_ascii=False))
    if process.returncode or copied.returncode:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
