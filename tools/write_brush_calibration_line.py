#!/usr/bin/env python3
"""One 10mm ink calibration line: fixed pose, 1mm paper-normal depression.

Uses today's operator zero and unloaded wrench plus the confirmed unchanged
paper axes. Default is preflight/export only; --execute sends one finite job.
No neural H/posture mapping is claimed. No automatic recovery or force tare.
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import socket
import sys
import time
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation
from brush_buffered_motion import compile_program, route_identity, execution_blockers
from brush_buffered_guard import execute_buffered
from brush_live_state import ContinuousState
from brush_trial_force import trial_force_level
from write_gong_1mm import audit_points

ROOT = Path.home()/'UR10'
FOLDER = ROOT/'outputs/brush_mapping_20261020'
SOURCES = {
    'zero': (FOLDER/'zero-03.json', 'b09da471fc6c598bd3551ffa1c425ef9beeecf033bde067cf84ba0572c6bc854'),
    'unloaded': (FOLDER/'unloaded-01.json', '301294166b96947624aca52475b23dc782d8f364865bf115107101c1aa29542d'),
    'paper': (ROOT/'outputs/gong_xy_water_trial_20261016/gong_geometry_ready.json', '4200ef2038962e21134890cb1f665f18ea8d2d6f6c9bb398bb56d2e0c3900d25')}
WATCHDOG = ROOT/'outputs/gong_1mm_continuous_20261016/watchdog_validation_v2.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_sources():
    result = {}
    for name, (path, expected) in SOURCES.items():
        if sha(path) != expected:
            raise ValueError('source changed: '+name)
        result[name] = json.loads(path.read_bytes())
    for name in ('zero', 'unloaded'):
        data = result[name]
        age = (dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(data['recorded_at'])).total_seconds()
        if data['status'] != 'recorded' or not 0 <= age <= 1800:
            raise ValueError('successful '+name+' must be from last 30 minutes')
    if result['unloaded']['zero_source']['sha256'] != SOURCES['zero'][1]:
        raise ValueError('unloaded reference not bound to this zero')
    return result


def build_points(zero, unloaded, paper, start, *, press_mm=1.,
                 line_speed_m_s=.030, tip_offset_m=None):
    if not np.isfinite([press_mm, line_speed_m_s]).all() or not 0 < press_mm <= 1 or not 0 < line_speed_m_s <= .030:
        raise ValueError('fixed-pose trial requires 0<press<=1mm and 0<speed<=30mm/s')
    tip_offset = np.asarray([0., 0., .315] if tip_offset_m is None else tip_offset_m, float)
    if tip_offset.shape != (3,) or not np.isfinite(tip_offset).all():
        raise ValueError('explicit finite 3D tip offset required')
    start = np.asarray(start, float)
    origin = np.asarray(zero['pose_sample']['flange_pose_m_rad'], float)
    reference = np.asarray(unloaded['pose_sample']['flange_pose_m_rad'], float)
    u = np.asarray(paper['paper_u_base'], float)
    normal = np.asarray(paper['paper_normal_base'], float)
    baseline = np.asarray(unloaded['raw_wrench_summary']['median_n_nm'], float)
    if any(x.shape != (6,) for x in (start, origin, reference, baseline)) or u.shape != (3,) or normal.shape != (3,):
        raise ValueError('invalid sample shape')
    if not np.isfinite(np.r_[start, origin, reference, baseline, u, normal]).all():
        raise ValueError('nonfinite sample')
    if abs(np.linalg.norm(u)-1)>1e-6 or abs(np.linalg.norm(normal)-1)>1e-6 or abs(u@normal)>1e-6 or normal[2]<.99:
        raise ValueError('invalid confirmed paper axes')
    for pose in (origin, reference):
        if (Rotation.from_rotvec(pose[3:]).inv()*Rotation.from_rotvec(start[3:])).magnitude()>np.deg2rad(.05):
            raise ValueError('fixed orientation changed >0.05deg')
    if np.linalg.norm(start[:3]-reference[:3])>.0002:
        raise ValueError('start differs from measured unloaded reference >0.2mm')
    # Preserve the legacy offset by default; callers may provide the accepted
    # natural-tip vector. Depth deltas still use the fresh local touch zero.
    origin = origin.copy()
    origin[:3] += Rotation.from_rotvec(origin[3:]).apply(tip_offset)-Rotation.from_rotvec(start[3:]).apply(tip_offset)
    origin[3:] = start[3:]
    displacement = start[:3]-origin[:3]
    height = float(displacement@normal)
    if not .001 <= height <= .005 or np.linalg.norm(displacement-height*normal)>.0002:
        raise ValueError('start must be 1..5mm clear at the recorded zero XY')
    near = origin.copy(); near[:3] += normal*.001
    pressed = origin.copy(); pressed[:3] -= normal*(press_mm/1000)
    end = pressed.copy(); end[:3] += u*.010
    lift = origin.copy(); lift[:3] += u*.010+normal*.003
    points = []
    for phase, pose, contact, speed, accel in [
            ('measured_start', start, False, .003, .002),
            ('near_zero', near, False, .0003, .002),
            ('press_1mm', pressed, True, .0003, .002),
            ('line_10mm', end, True, line_speed_m_s, .100),
            ('lift', lift, False, .003, .002)]:
        points.append({'phase':phase, 'pose_m_rad':pose.tolist(), 'contact':contact,
                       'stroke_id':0 if contact else None, 'speed_m_s':speed,
                       'acceleration_m_s2':accel, 'unloaded_wrench_n_nm':baseline.tolist()})
    return points, {'zero_flange_pose_m_rad':origin.tolist(), 'paper_u_base':u.tolist(),
                    'paper_normal_base':normal.tolist(), 'physical_press_mm':press_mm,
                    'tip_offset_for_small_angle_correction_m':tip_offset.tolist(),
                    'line_length_mm':10., 'final_clearance_mm':3.}


def full_force_snapshot(force):
    with force.lock:
        rows = list(force.rows)
    return {'scope':'all raw wrench rows retained by this finite job (capacity50000)',
            'rows':[{'received_monotonic_s':r[0], 'received_unix_s':r[1],
                     'source_unix_s':r[2], 'wrench_n_nm':r[3]} for r in rows]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--paper-unchanged', action='store_true')
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    if not args.paper_unchanged:
        p.error('requires operator confirmation of unchanged paper/tool and20mm blank along paper u')
    data = load_sources()
    watchdog = json.loads(WATCHDOG.read_bytes())
    if watchdog.get('passed') is not True or watchdog['compiler_sha256'] != sha(Path(__file__).with_name('brush_buffered_motion.py')):
        raise ValueError('existing controller watchdog validation does not match compiler')
    sys.path[:0] = [str(ROOT/'scripts'), str(Path.home()/'.dsh-analysis')]
    import probe_paper_height as support
    from calibrate_pen_tcp import capture
    import fcntl
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    def write(name, content):
        path = out/name
        with path.open('x') as stream:
            json.dump(content, stream, ensure_ascii=False, indent=2, allow_nan=False)
        return path
    preflight = {'status':'started', 'motion_sent':False}
    reader = force = None
    lockpath = ROOT/'outputs/paper_writing/brush-execution.lock'
    try:
        with lockpath.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            controller = support.controller_ready('192.168.1.3')
            sample = capture('192.168.1.3',30011,1)
            offset = np.asarray(sample['active_tcp_offset_m_rad'])
            if not np.allclose(offset,0,atol=1e-8,rtol=0):
                raise ValueError('active TCP changed from measured zero offset')
            reader = ContinuousState(support,'192.168.1.3')
            q, start, _ = reader.stationary()
            points, geometry = build_points(data['zero'],data['unloaded'],data['paper'],start)
            audit = audit_points(points,q,offset,support)
            with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as sock:
                sock.connect(('192.168.1.3',30013)); guard_host = sock.getsockname()[0]
            script, manifest = compile_program(points,guard_host,50071)
            fingerprint = route_identity(points)
            provenance = [{'path':str(path),'sha256':digest} for path,digest in SOURCES.values()]
            records = {
                'physical_mapping':{'passed':True,'scope':'physical1mm fixed-pose calibration sample; NOT neural-H or pose calibration',
                    'route_sha256':fingerprint,'geometry':geometry,'sources':provenance,'operator_paper_tool_unchanged':True},
                'force_reference':{'passed':True,'scope':'today fixed-pose unloaded reference only;10N vector increment guard',
                    'route_sha256':fingerprint,'source':provenance[1]},
                'blended_path_audit':{**audit,'route_sha256':fingerprint,'program_sha256':manifest['program_sha256']}}
            evidence = {}
            for name, record in records.items():
                path = write(name+'.json',record)
                evidence[name] = {'path':path.name,'sha256':sha(path)}
            bundle = {'scope':'one operator-authorized ink calibration line', 'points':points,
                'start_pose_m_rad':start.tolist(),'active_tcp_offset_m_rad':offset.tolist(),
                'maximum_runtime_s':60,'physical_mapping_verified':True,'force_reference_verified':True,
                'controller_watchdog_hardware_validated':True,'evidence':evidence,
                'watchdog_evidence':{'path':str(WATCHDOG),'sha256':sha(WATCHDOG)},
                'model_H_angles_executed':False,'model_H_mm':None,'calibration_accepted':False,
                'commanded_press_mm':1.,'line_length_mm':10.,'controller':controller}
            blockers = execution_blockers(bundle,manifest,out)
            if blockers:
                raise RuntimeError('; '.join(blockers))
            write('bundle.json',bundle); write('manifest.json',manifest)
            with (out/'program.script.preview').open('x') as stream:
                stream.write(script)
            force = support.RawForce(); deadline = time.monotonic()+10
            while True:
                try:
                    force.recent(); break
                except RuntimeError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(.02)
            delta = trial_force_level(force,points[0]['unloaded_wrench_n_nm'],10.)
            preflight.update(status='passed',force_increment_n=delta,guard_host=guard_host,
                             point_count=len(points),execute_requested=args.execute)
    except BaseException as exc:
        preflight.update(status='failed',error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        write('preflight.json',preflight)
        if force is not None and force.rclpy.ok(): force.close()
        if reader is not None: reader.close()
        support.close_commands()
    print(json.dumps(preflight),flush=True)
    if args.execute:
        # Reuse the unchanged live guard; retain its entire force window here.
        recorded_support = SimpleNamespace(**vars(support))
        recorded_support.force_snapshot = full_force_snapshot
        execute_buffered(bundle,out,guard_host,50071,out/'execution.json',recorded_support,capture)
        print('COMPLETE: 10mm calibration line at1mm depression; final lift verified',flush=True)
    else:
        print('PREFLIGHT ONLY: no motion sent',flush=True)


if __name__ == '__main__':
    main()
