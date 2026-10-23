#!/usr/bin/env python3
"""Offline camera-pose design; explicit live mode reuses existing guarded bundles.

Default runs never import ROS, trigger cameras, open robot sockets or send motion.
"""
import argparse
import datetime as dt
import json
import math
import struct
import time
from collections import deque
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ros2_sim_ws/src/fusion_model_ros2_beta'))
try:
    from fusion_model_ros2_beta.ur10_actual_kinematics import forward_kinematics, CALIBRATION_HASH
except ModuleNotFoundError as exc:
    if not exc.name.startswith('fusion_model_ros2_beta'): raise
    # The UR10 repository carries the unchanged factory module beside scripts.
    from ur10_actual_kinematics import forward_kinematics, CALIBRATION_HASH

LOWER = np.deg2rad([-360, -360, -180, -360, -360, -360]); UPPER = -LOWER
DATA = ROOT / 'outputs/brush_mapping_20261022'
UPSTREAM = ROOT if (ROOT/'config/handeye_eye_in_hand_20260921.yaml').is_file() else DATA/'write_capture_offline/upstream'


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def source_record(path):
    p = Path(path).resolve()
    result={'path': str(p), 'original_text': p.read_text()}
    if p.is_relative_to(ROOT):result['repository_relative_path']=str(p.relative_to(ROOT))
    return result


def matrix(pose):
    t = np.eye(4); t[:3, :3] = Rotation.from_rotvec(pose[3:]).as_matrix(); t[:3, 3] = pose[:3]
    return t


def pose(t):
    return np.r_[t[:3, 3], Rotation.from_matrix(t[:3, :3]).as_rotvec()]


def interpolate(a, b, linear_step=.01, angular_step_deg=3.):
    """Linear translation and shortest-arc orientation, as movel."""
    a, b = np.asarray(a), np.asarray(b)
    angle = (Rotation.from_rotvec(a[3:]).inv()*Rotation.from_rotvec(b[3:])).magnitude()
    count = max(1, math.ceil(np.linalg.norm(b[:3]-a[:3])/linear_step), math.ceil(angle/math.radians(angular_step_deg)))
    weights = np.linspace(0, 1, count+1)
    rotations = Slerp([0, 1], Rotation.from_rotvec([a[3:], b[3:]]))(weights)
    return np.column_stack((a[:3]+weights[:, None]*(b[:3]-a[:3]), rotations.as_rotvec()))


def solve(target, seed):
    def residual(q):
        t = forward_kinematics(q)
        return np.r_[t[:3, 3]-target[:3, 3], .35*Rotation.from_matrix(t[:3, :3].T@target[:3, :3]).as_rotvec()]
    result = least_squares(residual, np.clip(seed, LOWER+1e-8, UPPER-1e-8), bounds=(LOWER, UPPER),
                           max_nfev=120, ftol=1e-9, xtol=1e-9, gtol=1e-9)
    t = forward_kinematics(result.x)
    return result.x, float(np.linalg.norm(t[:3, 3]-target[:3, 3])), float(Rotation.from_matrix(t[:3, :3].T@target[:3, :3]).magnitude())


def jacobian(q):
    t = forward_kinematics(q); columns = []
    for i in range(6):
        delta = np.zeros(6); delta[i] = 1e-6; end = forward_kinematics(q+delta)
        columns.append(np.r_[(end[:3, 3]-t[:3, 3])/1e-6, Rotation.from_matrix(end[:3, :3]@t[:3, :3].T).as_rotvec()/1e-6])
    return np.array(columns).T


def cad_meshes(upstream=UPSTREAM):
    meshes = {}
    for path in (Path(upstream)/'model/solidworks/end_effector').glob('*.STL'):
        raw = path.read_bytes(); count = int.from_bytes(raw[80:84], 'little')
        dtype = np.dtype([('normal', '<f4', 3), ('vertices', '<f4', (3, 3)), ('attr', '<u2')])
        meshes[path.name] = np.frombuffer(raw, dtype=dtype, offset=84, count=count)['vertices'].astype(float)
    sensor = next(v for k, v in meshes.items() if '六轴力' in k)
    xyz = sensor.reshape(-1, 3); low, high = xyz.min(0), xyz.max(0)
    origin = np.array([(low[0]+high[0])/2, high[1], (low[2]+high[2])/2])
    rotation = Rotation.from_euler('z', 270, degrees=True).as_matrix()@np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    return {name: (value-origin)@rotation.T/1000 for name, value in meshes.items()}


def segment_hits_mesh(origin, ends, triangles):
    """Finite camera-to-paper ray/triangle intersections, no mesh dependency."""
    a = triangles[:, 0]; e1 = triangles[:, 1]-a; e2 = triangles[:, 2]-a; s = origin-a; hits = []
    for end in ends:
        direction = end-origin; h = np.cross(direction, e2); det = np.einsum('ij,ij->i', e1, h)
        valid = abs(det)>1e-12; inv = np.divide(1., det, out=np.zeros_like(det), where=valid)
        u = inv*np.einsum('ij,ij->i', s, h); q = np.cross(s, e1)
        v = inv*(q@direction); t = inv*np.einsum('ij,ij->i', e2, q)
        hits.append(bool(np.any(valid & (u>=0) & (v>=0) & (u+v<=1) & (t>1e-5) & (t<1-1e-5))))
    return hits


def optical_check(flange, camera, target, axes, tcp, meshes, distance, roi_mm=50.):
    u, v, n = axes; base_cam = flange@camera
    grid = np.linspace(-roi_mm/2000, roi_mm/2000, 5)
    roi = np.array([target+x*u+y*v for x in grid for y in grid])
    local = (np.column_stack((roi, np.ones(len(roi))))@np.linalg.inv(flange).T)[:, :3]
    camera_origin = camera[:3, 3]; blocked = {}
    for name, mesh in meshes.items():
        if '3D相机' not in name: blocked[name] = sum(segment_hits_mesh(camera_origin, local, mesh))
    shaft0 = np.array([0., 0., .22]); shaft1 = np.asarray(tcp); brush_hits = 0
    for end in local:
        ray = camera_origin+np.linspace(0, 1, 251)[:, None]*(end-camera_origin)
        w = np.clip((ray-shaft0)@(shaft1-shaft0)/np.sum((shaft1-shaft0)**2), 0, 1)
        brush_hits += int(np.min(np.linalg.norm(ray-(shaft0+w[:, None]*(shaft1-shaft0)), axis=1))<.0078)
    camera_roi = (np.column_stack((roi, np.ones(len(roi))))@np.linalg.inv(base_cam).T)[:, :3]
    inside = np.all((abs(camera_roi[:, 0]/camera_roi[:, 2])<220/300/2) &
                    (abs(camera_roi[:, 1]/camera_roi[:, 2])<160/300/2) & (camera_roi[:, 2]>0))
    return {'roi_size_mm': roi_mm, 'roi_ray_count': len(roi), 'cad_blocked_rays': blocked,
            'brush_envelope_blocked_rays': brush_hits, 'nominal_fov_covers_roi': bool(inside),
            'working_distance_m': distance, 'working_distance_status': '450mm nominal boundary of published lens ranges; installed focus/CameraInfo required',
            'optical_passed_nominal': not any(blocked.values()) and not brush_hits and bool(inside),
            'scope': '25 CAD rays, assumed shaft capsule, conservative nominal FoV; not physical optical validation'}


def dense_audit(points, seed, paper_origin, normal, tcp, meshes):
    from scipy.spatial import ConvexHull
    q = np.asarray(seed, float); rows = []; failures = []
    cloud = np.concatenate([v.reshape(-1, 3) for v in meshes.values()]); cloud = cloud[ConvexHull(cloud).vertices]
    tip = np.asarray(tcp); shaft0 = np.array([0., 0., .22])
    for index in range(1, len(points)):
        a, b = points[index-1], points[index]; values = interpolate(a['pose_m_rad'], b['pose_m_rad'])
        for j, value in enumerate(values):
            target = matrix(value); previous = q.copy(); q, pe, re = solve(target, q)
            margin = float(min(np.min(q-LOWER), np.min(UPPER-q)))
            singular = float(np.linalg.svd(jacobian(q), compute_uv=False)[-1])
            body = float(np.min((cloud@target[:3, :3].T+target[:3, 3]-paper_origin)@normal))
            tip_height = float((target[:3, 3]+target[:3, :3]@tip-paper_origin)@normal)
            shaft = float(min((target[:3, 3]+target[:3, :3]@shaft0-paper_origin)@normal, tip_height)-.0078)
            linear = np.linalg.norm(value[:3]-values[max(0, j-1), :3])
            angular = (Rotation.from_rotvec(value[3:]).inv()*Rotation.from_rotvec(values[max(0, j-1), 3:])).magnitude()
            duration = max(linear/b['speed_m_s'], 1e-6)
            rate = 0. if j==0 else float(np.max(abs(q-previous))/duration)
            row = {'phase': b['phase'], 'sample': j, 'q_rad': q.tolist(), 'pose_m_rad': value.tolist(),
                   'ik_position_m': pe, 'ik_rotation_rad': re, 'joint_margin_deg': math.degrees(margin),
                   'min_singular': singular, 'joint_step_rad': float(np.max(abs(q-previous))),
                   'joint_speed_estimate_rad_s': rate, 'cad_paper_clearance_mm': body*1000,
                   'tip_clearance_mm': tip_height*1000, 'shaft_capsule_clearance_mm': shaft*1000}
            invalid = pe>.0001 or re>.001 or margin<math.radians(5) or singular<.02 or rate>.6 or body<.02
            if b['phase'] not in ('lift_clear', 'write_line', 'press') and shaft<.01: invalid = True
            if row['joint_step_rad']>.6: invalid = True
            if invalid: failures.append(row)
            rows.append(row)
    return {'passed_numeric': not failures, 'failures': failures[:12], 'failure_count': len(failures),
            'samples': rows, 'end_q_rad': q.tolist(), 'external_obstacles_checked': False,
            'dynamic_acceleration_validated': False,
            'scope': 'factory FK, sampled continuous IK/Jacobian, limits/rate, CAD and shaft versus paper; arm self collision separate'}


def build_trial(paper, tcp, specification=None):
    """Explicit physical command trial; no neural-H conversion is inferred."""
    spec={'sample_id':'brush-depth-003-r1','split':'train','repeat_group':'depth-003',
          'commanded_depth_mm':.3,'line_length_mm':10.,'write_speed_mm_s':30.,
          'paper_start_uv_mm':[0.,0.],'commanded_tilt_delta_uv_rad':[0.,0.],
          'commanded_axis_delta_rad':0.}
    spec.update(specification or {})
    if spec['split'] not in ('train','holdout') or not spec['sample_id'] or not spec['repeat_group']:
        raise ValueError('sample id, split and repeat group required')
    scalars=np.array([spec[k] for k in ('commanded_depth_mm','line_length_mm','write_speed_mm_s','commanded_axis_delta_rad')],float)
    if not np.isfinite(scalars).all() or not 0<scalars[0]<=10 or not 0<scalars[1]<=50 or not 0<scalars[2]<=30:
        raise ValueError('invalid trial depth, length or existing speed envelope')
    start=np.asarray(spec['paper_start_uv_mm'],float);tilt=np.asarray(spec['commanded_tilt_delta_uv_rad'],float)
    if start.shape!=(2,) or tilt.shape!=(2,) or not np.isfinite(np.r_[start,tilt]).all():raise ValueError('invalid paper offset/tilt')
    u,v,n=[np.array(paper[k]) for k in ('paper_u_base','paper_v_base','paper_normal_base')]
    origin=np.array(paper['reconstructed_center_tip_base_m']);t=np.array(tcp['translation_m'])
    zero=matrix(paper['historical_center_zero_controller_pose_m_rad'])
    if np.any(tilt) or spec['commanded_axis_delta_rad']:
        brush=np.array(tcp['rotation_matrix'])
        zero[:3,:3]=(Rotation.from_rotvec(v*tilt[1]).as_matrix()@Rotation.from_rotvec(u*tilt[0]).as_matrix()
                       @zero[:3,:3]@brush@Rotation.from_euler('z',spec['commanded_axis_delta_rad']).as_matrix()@brush.T)
        zero[:3,3]=origin+(start[0]*u+start[1]*v)/1000-zero[:3,:3]@t
    else:
        # Preserve the recorded default geometry verbatim, including roundoff.
        zero[:3,3]+=(start[0]*u+start[1]*v)/1000
    write=pose(zero);length=spec['line_length_mm']/1000;depth=spec['commanded_depth_mm']/1000
    line_end=write.copy();line_end[:3]+=length*u
    clear=line_end.copy();clear[:3]+=.040*n
    pressed=write.copy();pressed[:3]-=depth*n
    pressed_end=pressed.copy();pressed_end[:3]+=length*u
    start_air=write.copy();start_air[:3]+=.040*n
    points=[]
    for phase,value,contact,speed in [('write_start_air',start_air,False,.003),('press',pressed,True,.0003),
            ('write_line',pressed_end,True,spec['write_speed_mm_s']/1000),('lift_clear',clear,False,.003)]:
        points.append({'phase':phase,'pose_m_rad':value.tolist(),'contact':contact,'stroke_id':0 if contact else None,
                       'speed_m_s':speed,'acceleration_m_s2':.100 if phase=='write_line' else .002})
    target=origin+((start[0]+spec['line_length_mm']/2)*u+start[1]*v)/1000
    return spec,points,target


def design(output, data=DATA, upstream=UPSTREAM, trial_spec=None):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    paper_file = Path(data)/'paper-frame-reuse-20261022.json'; tcp_file = Path(data)/'brush-pose-nominal-20261022.json'
    handeye_file = Path(upstream)/'config/handeye_eye_in_hand_20260921.yaml'
    paper, tcp, handeye = read(paper_file), read(tcp_file), yaml.safe_load(handeye_file.read_text())
    if handeye['transform_direction']!='tool0_from_camera': raise ValueError('wrong handeye direction')
    h = np.eye(4); h[:3, :3] = handeye['rotation_matrix']; h[:3, 3] = handeye['translation_m']
    origin = np.array(paper['reconstructed_center_tip_base_m'])
    u, v, n = [np.array(paper[k]) for k in ('paper_u_base', 'paper_v_base', 'paper_normal_base')]
    t = np.array(tcp['translation_m']); meshes = cad_meshes(upstream)
    trial,sample_write,target=build_trial(paper,tcp,trial_spec)
    clear=np.array(sample_write[-1]['pose_m_rad']);start_air=np.array(sample_write[0]['pose_m_rad'])
    seed = np.array([.035303656, -1.794883553, -1.574264828, -4.559593383, 4.657249928, -4.851251666])
    start_q, _, _ = solve(matrix(start_air), seed); candidates = []
    for roll in (0, 90, 180, 270):
        rcam = np.column_stack((u, -v, -n))@Rotation.from_euler('z', roll, degrees=True).as_matrix()
        distance = .450
        base_cam = np.eye(4); base_cam[:3, :3] = rcam; base_cam[:3, 3] = target+distance*n
        f = base_cam@np.linalg.inv(h)
        high = clear.copy(); high[:3] += .200*n
        turned = high.copy(); turned[:3] += .080*n; turned[3:] = pose(f)[3:]
        overhead = pose(f); overhead[:3] += .150*n; points = []
        for phase, value in [('lift_clear', clear), ('raise_for_camera', high),
                             ('reorient', turned), ('translate_above_photo', overhead), ('photo_pose', pose(f))]:
            points.append({'phase': phase, 'pose_m_rad': value.tolist(), 'contact': False,
                           'stroke_id': None, 'speed_m_s': .003, 'acceleration_m_s2': .002})
        optics = optical_check(f, h, target, (u, v, n), t, meshes, distance,max(50.,trial['line_length_mm']+30.))
        audit = dense_audit(sample_write+points[1:], start_q, origin, n, t, meshes)
        c = {'camera_roll_deg': roll, 'photo_tool0_pose_m_rad': pose(f).tolist(),
             'base_from_camera': base_cam.tolist(), 'target_base_m': target.tolist(),
             'points': points, 'planned_write':sample_write,'trial':trial, 'optics': optics, 'audit': audit}
        candidates.append(c); save(output/f'candidate_{roll}.json', c)
    eligible = [c for c in candidates if c['audit']['passed_numeric'] and c['optics']['optical_passed_nominal']]
    selected = min(eligible or candidates, key=lambda c: (c['audit']['failure_count'], sum(c['optics']['cad_blocked_rays'].values())))
    config = {'schema': 'brush_write_capture_v1', **trial,
              'physical_tilt_uv_rad': None, 'physical_axis_rotation_rad': None,
              'model_H_mm': None, 'measured_depth_paper_normal_mm': None, 'write_bundle': None, 'transfer_bundle': None,
              'photo_tool0_pose_m_rad': selected['photo_tool0_pose_m_rad'], 'base_from_camera_nominal': selected['base_from_camera'],
              'paper_frame': paper, 'tcp': tcp, 'handeye': handeye,
              'source_files': [source_record(p) for p in (paper_file, tcp_file, handeye_file,
                              Path(upstream)/'config/robot_attachments.preliminary-cad-20260921.yaml')],
              'planned_write': sample_write, 'planned_transfer': selected['points'],
              'nominal_numeric_passed': bool(eligible),
              'selected_candidate_roll_deg': selected['camera_roll_deg'],
              'camera_capture_reuse_script': '/home/jx/UR10/scripts/validate_mecheye_capture.py',
              'upstream_robot_scripts': '/home/jx/UR10/scripts',
              'execution_blockers': ['arm self collision has not been checked for this candidate',
                  'no fresh controller start/zero/unloaded references while powered off',
                  'no route-bound unloaded force references for camera reorientation', 'no approved write/transfer guarded bundles',
                  'camera focus variant and calibrated intrinsics require capture check',
                  'paper height and nominal brush orientation retain their prior unverified status'],
              'robot_execution_ready': False, 'motion_sent': False}
    save(output/'sample.json', config)
    save(output/'design_summary.json', {'nominal_candidates': [{'roll_deg': c['camera_roll_deg'],
         'numeric_passed': c['audit']['passed_numeric'], 'failure_count': c['audit']['failure_count'], 'optics': c['optics']} for c in candidates],
         'selected_roll_deg': selected['camera_roll_deg'], 'factory_calibration': CALIBRATION_HASH,
         'start_q_status': 'historical seed re-solved offline; robot off', 'motion_sent': False, 'runtime_blockers': config['execution_blockers']})
    return config


def run_workflow(sample, output, execute=False, backend=None):
    output = Path(output); output.mkdir(parents=True, exist_ok=False); save(output/'sample.json', sample)
    if not execute:
        result = {'sample_id': sample['sample_id'], 'status': 'preview_only', 'motion_sent': False,
                  'camera_triggered': False, 'execution_blockers': sample.get('execution_blockers', [])}
        save(output/'result.json', result); return result
    if backend is None: raise ValueError('live backend required')
    result = {'sample_id': sample['sample_id'], 'status': 'started', 'stages': [], 'started_at': dt.datetime.now(dt.timezone.utc).isoformat()}
    try:
        backend.preflight(sample, output)
        for name in ('write', 'transfer'):
            result['stages'].append({'stage': name, 'evidence': backend.move(name, sample, output)})
        backend.stationary(sample, output); backend.capture(sample, output); backend.stationary(sample, output)
        result['status'] = 'captured'
    except BaseException as exc:
        result.update(status='failed', error=type(exc).__name__+': '+str(exc)); raise
    finally:
        result['finished_at'] = dt.datetime.now(dt.timezone.utc).isoformat(); save(output/'result.json', result)
    return result


def select_checked_candidate(design_dir, candidate_file, collision_file, output):
    """Finalize an offline candidate; collision evidence binds original q rows."""
    design_dir, output = Path(design_dir), Path(output)
    sample = read(design_dir/'sample.json'); c = read(candidate_file); collision = read(collision_file)
    if collision.get('supplied_samples')!=c['audit']['samples']: raise ValueError('collision result is for a different sampled path')
    if c.get('planned_write',sample['planned_write'])!=sample['planned_write']:raise ValueError('candidate belongs to a different writing trial')
    if not c['audit']['passed_numeric'] or not c['optics']['optical_passed_nominal']: raise ValueError('candidate numeric/optical check failed')
    if not all(m['passed_20mm'] for m in collision['models'].values()): raise ValueError('self collision check failed')
    sample.update(photo_tool0_pose_m_rad=c['photo_tool0_pose_m_rad'], base_from_camera_nominal=c['base_from_camera'],
                  selected_candidate_roll_deg=c['camera_roll_deg'], planned_transfer=c['points'],
                  offline_collision_check=source_record(collision_file), nominal_numeric_passed=True,
                  physical_tilt_uv_rad=None)
    sample['execution_blockers']=[b for b in sample['execution_blockers'] if b!='arm self collision has not been checked for this candidate']
    save(output,sample)
    return sample


class LiveBackend:
    def __init__(self):
        self.bundles = {}; self.stationary_samples = []

    def preflight(self, sample, out):
        import socket
        from brush_buffered_motion import compile_program, execution_blockers
        if sample.get('execution_blockers'): raise ValueError('unresolved: '+'; '.join(sample['execution_blockers']))
        references = sample.get('fresh_reference_files', {})
        if set(references)!= {'zero', 'unloaded'}: raise ValueError('fresh zero/unloaded references required')
        for name, filename in references.items():
            ref = read(filename)
            age = (dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(ref['recorded_at'])).total_seconds()
            if ref.get('status')!='recorded' or not 0<=age<=1800: raise ValueError('reference stale: '+name)
        zero=read(references['zero']);z=matrix(zero['pose_sample']['flange_pose_m_rad'])
        tcp=np.array(sample['tcp']['translation_m']);n=np.array(sample['paper_frame']['paper_normal_base'])
        pressed=next(p for p in sample['planned_write'] if p['contact']);m=matrix(pressed['pose_m_rad'])
        expected_tip=m[:3,3]+m[:3,:3]@tcp+n*sample['commanded_depth_mm']/1000
        if np.linalg.norm(z[:3,3]+z[:3,:3]@tcp-expected_tip)>.0002:raise ValueError('fresh zero disagrees with planned writing site')
        sys.path[:0] = [sample['upstream_robot_scripts'], '/home/jx/.dsh-analysis']
        import probe_paper_height as support
        from calibrate_pen_tcp import capture
        self.support, self.pose_capture = support, capture
        for item in sample['source_files']:
            source=ROOT/item['repository_relative_path'] if 'repository_relative_path' in item else Path(item['path'])
            if source.read_text()!=item['original_text']: raise ValueError('source configuration changed')
        for name in ('write', 'transfer'):
            path = Path(sample[name+'_bundle']); bundle = read(path)
            if bundle.get('sample_id')!=sample['sample_id']: raise ValueError('bundle sample id mismatch')
            if bundle.get('commanded_press_mm')!=sample['commanded_depth_mm']: raise ValueError('bundle depth metadata mismatch')
            planned = sample['planned_'+name]
            if len(bundle['points'])!=len(planned): raise ValueError('bundle route length differs from sample')
            for point, expected in zip(bundle['points'], planned):
                for field in ('pose_m_rad', 'contact', 'speed_m_s', 'acceleration_m_s2'):
                    if point[field]!=expected[field]: raise ValueError('bundle route differs from sample: '+field)
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route:
                route.connect(('192.168.1.3', 30013)); host = route.getsockname()[0]
            _, manifest = compile_program(bundle['points'], host, 50071)
            errors = execution_blockers(bundle, manifest, path.parent)
            if errors: raise ValueError('; '.join(errors))
            self.bundles[name] = path, bundle, host
        wp = np.array(self.bundles['write'][1]['points'][-1]['pose_m_rad'])
        tp = np.array(self.bundles['transfer'][1]['points'][0]['pose_m_rad'])
        if np.linalg.norm(wp[:3]-tp[:3])>.0001 or (Rotation.from_rotvec(wp[3:]).inv()*Rotation.from_rotvec(tp[3:])).magnitude()>.001: raise ValueError('write lift and transfer start differ')
        end = np.array(self.bundles['transfer'][1]['points'][-1]['pose_m_rad']); target = np.array(sample['photo_tool0_pose_m_rad'])
        if np.linalg.norm(end[:3]-target[:3])>.0001 or (Rotation.from_rotvec(end[3:]).inv()*Rotation.from_rotvec(target[3:])).magnitude()>.001: raise ValueError('photo pose differs from transfer end')
        save(out/'used_bundles.json', {k: {'source': str(v[0]), 'bundle': v[1]} for k, v in self.bundles.items()})

    def move(self, name, sample, out):
        from types import SimpleNamespace
        from brush_buffered_guard import execute_buffered
        from write_brush_calibration_line import full_force_snapshot
        path, bundle, host = self.bundles[name]; target = out/(name+'_execution.json')
        support = SimpleNamespace(**vars(self.support)); support.force_snapshot = full_force_snapshot
        # Preserve all samples on disk. The original force deque is bounded and
        # alone would discard early data on a slow camera-transfer trajectory.
        original_force = self.support.RawForce
        original_fresh = self.support.fresh
        with (out/(name+'_raw_force.jsonl')).open('x') as force_file, (out/(name+'_controller_feedback.jsonl')).open('x') as state_file:
            class RecordingForce(original_force):
                def __init__(self):
                    super().__init__()
                    with self.lock: self.rows = RecordingRows(self.rows, force_file, sample['sample_id'])
            def fresh(reader, *args, **kwargs):
                result = original_fresh(reader, *args, **kwargs)
                state_file.write(json.dumps({'sample_id': sample['sample_id'], 'host_unix_ns': time.time_ns(),
                    'host_monotonic_ns': time.monotonic_ns(), 'controller_time_s': struct.unpack_from('>d',reader.last_frame,4)[0],
                    'actual_q_rad': result[0].tolist(), 'actual_tool0_pose_m_rad': result[1].tolist(),
                    'tcp_velocity': result[2].tolist()})+'\n')
                return result
            support.RawForce = RecordingForce; support.fresh = fresh
            execute_buffered(bundle, path.parent, host, 50071, target, support, self.pose_capture)
        if read(target)['status']!='completed_buffered_program': raise RuntimeError('move did not complete')
        return str(target)

    def stationary(self, sample, out):
        state = self.pose_capture('192.168.1.3', 30011, 1.)
        if state['max_motion_mm']>.1 or state['max_motion_deg']>.05: raise RuntimeError('camera not stationary')
        if not np.allclose(state['active_tcp_offset_m_rad'], 0, atol=1e-8, rtol=0): raise RuntimeError('expected zero active TCP')
        actual = np.array(state['flange_pose_m_rad']); target = np.array(sample['photo_tool0_pose_m_rad'])
        if np.linalg.norm(actual[:3]-target[:3])>.0001 or (Rotation.from_rotvec(actual[3:]).inv()*Rotation.from_rotvec(target[3:])).magnitude()>math.radians(.1): raise RuntimeError('photo position mismatch')
        self.stationary_samples.append({'host_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'sample': state})
        save(out/f'photo_stationary_{len(self.stationary_samples)}.json', self.stationary_samples[-1])

    def capture(self, sample, out):
        import threading
        from brush_live_state import ContinuousState
        from brush_capture_measure import capture_raw
        # Bracketed snapshots alone would miss motion during sequential captures.
        reader=ContinuousState(self.support,'192.168.1.3');finished=threading.Event();errors=[]
        target=np.array(sample['photo_tool0_pose_m_rad'])
        # A newly opened receive-only socket needs its first valid window before
        # the 150 ms ongoing-stream deadline applies. Never trigger the camera
        # while this initial window is still pending.
        try: reader.stationary()
        except BaseException:
            reader.close()
            raise
        with (out/'capture_controller_feedback.jsonl').open('x') as stream:
            def monitor():
                try:
                    while not finished.is_set():
                        q,p,v=reader.fresh(timeout=.15)
                        stream.write(json.dumps({'sample_id':sample['sample_id'],'host_unix_ns':time.time_ns(),
                            'actual_q_rad':q.tolist(),'actual_tool0_pose_m_rad':p.tolist(),'tcp_velocity':v.tolist()})+'\n')
                        angle=(Rotation.from_rotvec(p[3:]).inv()*Rotation.from_rotvec(target[3:])).magnitude()
                        if np.linalg.norm(v[:3])>.0001 or np.linalg.norm(v[3:])>.001 or np.linalg.norm(p[:3]-target[:3])>.0001 or angle>math.radians(.1):
                            raise RuntimeError('robot moved during camera capture')
                except BaseException as exc:errors.append(str(exc))
            monitor_thread=threading.Thread(target=monitor,daemon=True);monitor_thread.start()
            try:capture_raw(sample, out, self.stationary_samples[-1])
            finally:
                finished.set();monitor_thread.join(timeout=1.);reader.close()
            if errors:raise RuntimeError('; '.join(errors))


class RecordingRows(deque):
    """Keep the existing bounded force window and a complete streaming record."""
    def __init__(self, rows, stream, sample_id=None):
        super().__init__(maxlen=rows.maxlen); self.stream=stream;self.sample_id=sample_id
        for row in rows: self.append(row)

    def append(self, row):
        super().append(row)
        self.stream.write(json.dumps({'sample_id':self.sample_id,'received_monotonic_s':row[0], 'received_unix_s':row[1],
                                     'source_unix_s':row[2], 'wrench_n_nm':row[3]})+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--design', action='store_true'); p.add_argument('--sample', type=Path)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--execute', action='store_true')
    p.add_argument('--data-dir',type=Path,default=DATA);p.add_argument('--upstream-dir',type=Path,default=UPSTREAM)
    p.add_argument('--select-candidate', type=Path); p.add_argument('--collision-report', type=Path)
    p.add_argument('--trial-spec',type=Path,help='explicit sample id, physical commands, paper offset and split JSON; no H mapping')
    args = p.parse_args()
    if args.select_candidate:
        if args.execute or not args.sample or not args.collision_report: p.error('selection needs --sample design/sample.json and --collision-report; offline only')
        select_checked_candidate(args.sample.parent,args.select_candidate,args.collision_report,args.output)
        print('OFFLINE candidate selected; runtime references remain pending')
    elif args.design:
        if args.execute or args.sample: p.error('--design is offline only')
        result = design(args.output,args.data_dir,args.upstream_dir,read(args.trial_spec) if args.trial_spec else None)
        print(json.dumps({'selected_roll_deg': result['selected_candidate_roll_deg'], 'nominal_numeric_passed': result['nominal_numeric_passed'], 'motion_sent': False}))
    else:
        if not args.sample: p.error('--sample required')
        print(json.dumps(run_workflow(read(args.sample), args.output, args.execute, LiveBackend() if args.execute else None)))


if __name__=='__main__': main()
