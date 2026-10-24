"""Air-only first-reference acquisition. Never certifies a writing force map.

The production writing guard remains unchanged. This entry watches *raw wrench
change from the measured session start*, including gravity, rather than claiming
an already calibrated gravity-corrected contact force. Geometry forbids contact.
No CLI: a caller must supply the audited route and an existing watchdog record.
"""
import json
import math
import re
import socket
import struct
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from brush_buffered_motion import compile_program
from brush_buffered_guard import health_tick
from brush_live_state import ContinuousState


def compile_air_program(points):
    """Keep the original watchdog scaffold; give air MoveL explicit durations.

    Near-pure rotations must not derive runtime from a tiny flange translation.
    The source compiler manifest describes the unmodified source, not this
    timed program. Actual program text is saved and sent directly.
    """
    source, original = compile_program(points, '192.168.1.10', 50071)
    durations = [2.]
    for a, b in zip(points, points[1:]):
        pa, pb = [np.array(p['pose_m_rad']) for p in (a,b)]
        distance = np.linalg.norm(pb[:3]-pa[:3])
        angle = (Rotation.from_rotvec(pa[3:]).inv()*Rotation.from_rotvec(pb[3:])).magnitude()
        joint_step = np.max(abs(np.array(b['q_rad'])-np.array(a['q_rad'])))
        durations.append(max(2.,2.5*distance/.0015,2.5*angle/.008,2.5*joint_step/.1))
    lines=source.splitlines(keepends=True);index=0;restored=[]
    for i,line in enumerate(lines):
        if line.startswith('    movel('):
            timed=line.replace(', r=', ', t=%.6f, r='%durations[index])
            restored.append(re.sub(r', t=[0-9.]+, r=', ', r=',timed))
            lines[i]=timed;index+=1
        else:restored.append(line)
    if index!=len(points) or ''.join(restored)!=source:
        raise ValueError('timed air compiler altered original watchdog scaffold')
    return ''.join(lines), {'timing_mode':'air-only explicit MoveL t; original writing compiler unchanged',
        'move_duration_s':durations,'total_motion_duration_s':sum(durations),
        'watchdog_scaffold_unchanged':True,'source_compiler_manifest':original,
        'actual_program_source':'program.script','actual_program_identity':'saved exact text is sent directly; source program digest is not the timed-program digest'}


def validate_air_route(points, audit, collision, origin, normal, tip, anchor, baseline_rotation):
    """Reject contact, missing path evidence, and routes leaving this air cell."""
    if not 2 <= len(points) <= 150:
        raise ValueError('bounded air route required')
    if not audit.get('passed_numeric') or not audit.get('samples'):
        raise ValueError('complete numeric audit required')
    if collision.get('supplied_samples') != audit['samples']:
        raise ValueError('collision report differs from numeric route audit')
    if len(collision.get('models', {})) != 2 or not all(
            m.get('passed_20mm') and m['minimum_clearance_mm'] >= 20
            for m in collision['models'].values()):
        raise ValueError('both attachment collision models must pass')
    origin, normal, tip, anchor = [np.asarray(x, float) for x in (origin, normal, tip, anchor)]
    if any(x.shape != (3,) for x in (origin, normal, tip, anchor)) or abs(np.linalg.norm(normal)-1) > 1e-6:
        raise ValueError('invalid air cell geometry')
    poses = np.array([p['pose_m_rad'] for p in points], float)
    if poses.shape != (len(points), 6) or not np.isfinite(poses).all():
        raise ValueError('finite route poses required')
    refs = np.array([p['unloaded_wrench_n_nm'] for p in points], float)
    if refs.shape != (len(points), 6) or not np.isfinite(refs).all() or not np.allclose(refs, refs[0], atol=0, rtol=0):
        raise ValueError('one measured raw session baseline required; no predicted gravity')
    for point in points:
        if point['contact'] is not False or point.get('stroke_id') is not None:
            raise ValueError('air calibration cannot command contact')
        if point['phase'] != 'unloaded_reorientation' or not 0 < point['speed_m_s'] <= .003:
            raise ValueError('invalid air phase or speed')
        if not 0 < point['acceleration_m_s2'] <= .002:
            raise ValueError('invalid air acceleration')
    for a, b in zip(points, points[1:]):
        pa, pb = [np.asarray(p['pose_m_rad']) for p in (a, b)]
        angle = (Rotation.from_rotvec(pa[3:]).inv()*Rotation.from_rotvec(pb[3:])).magnitude()
        duration = np.linalg.norm(pb[:3]-pa[:3])/b['speed_m_s']
        if angle > math.radians(2.01) or duration + 1e-6 < angle/.010:
            raise ValueError('air rotation step/speed exceeds checked envelope')
    samples = audit['samples']
    # Bind audit endpoints to the supplied route, in addition to collision samples.
    last = [r for r in samples if r['sample'] == 0]
    if len(last) != len(points)-1:
        raise ValueError('audit segment count mismatch')
    for row, point in zip(last, points[:-1]):
        if not np.allclose(row['pose_m_rad'], point['pose_m_rad'], atol=1e-8, rtol=0):
            raise ValueError('audit start differs from route')
    if not np.allclose(samples[-1]['pose_m_rad'], poses[-1], atol=1e-8, rtol=0):
        raise ValueError('audit end differs from route')
    for row in samples:
        pose = np.asarray(row['pose_m_rad'])
        rotation = Rotation.from_rotvec(pose[3:])
        natural_tip = pose[:3] + rotation.apply(tip)
        height = (natural_tip-origin) @ normal
        if not .035 <= height <= .045 or np.linalg.norm(natural_tip-anchor) > .0005:
            raise ValueError('route leaves 35..45mm air cell or fixed-tip anchor')
        if (Rotation.from_rotvec(baseline_rotation).inv()*rotation).magnitude() > math.radians(10):
            raise ValueError('rotation exceeds local 10 degree calibration envelope')
        if row['cad_paper_clearance_mm'] < 20 or row['shaft_capsule_clearance_mm'] < 10:
            raise ValueError('tool too close to paper')


class AirReader:
    """Keep raw pose/velocity data and check the live no-contact cell."""
    def __init__(self, reader, origin, normal, tip, anchor, stream, baseline_rotation=(0,0,0)):
        self.reader = reader
        self.origin, self.normal, self.tip, self.anchor = [np.asarray(x) for x in (origin, normal, tip, anchor)]
        self.stream = stream
        self.baseline_rotation = Rotation.from_rotvec(baseline_rotation)

    def check(self, sample):
        self.last_sample = sample
        q, pose, velocity = sample
        self.stream.write(json.dumps({'host_unix_ns':time.time_ns(), 'actual_q_rad':q.tolist(),
            'actual_tool0_pose_m_rad':pose.tolist(), 'tcp_velocity':velocity.tolist()})+'\n')
        tip = pose[:3]+Rotation.from_rotvec(pose[3:]).apply(self.tip)
        height = (tip-self.origin)@self.normal
        if not .035 <= height <= .045 or np.linalg.norm(tip-self.anchor) > .0008:
            raise RuntimeError('live tool left checked air cell')
        if (self.baseline_rotation.inv()*Rotation.from_rotvec(pose[3:])).magnitude() > math.radians(10):
            raise RuntimeError('live orientation left local calibration envelope')
        return sample

    def fresh(self, *args, **kwargs):
        return self.check(self.reader.fresh(*args, **kwargs))

    def stationary(self):
        return self.check(self.reader.stationary())


def execute_air_segment(request, out, support, capture):
    """One finite air movement; a failure aborts the caller's entire sweep."""
    import fcntl
    from brush_write_capture import RecordingRows
    out = Path(out)
    out.mkdir(exist_ok=False)
    points = request['points']
    validate_air_route(points, request['audit'], request['collision'], request['origin'],
        request['normal'], request['tip'], request['anchor'], request['baseline_rotation'])
    budget = request['maximum_runtime_s']
    if not isinstance(budget, (float, int)) or not 0 < budget <= 120:
        raise ValueError('finite air segment runtime required')
    script, manifest = compile_air_program(points)
    if manifest['total_motion_duration_s']+10 > budget:
        raise ValueError('timed air route exceeds its explicit runtime budget; split route before execution')
    (out/'program.script').write_text(script)
    import brush_buffered_motion as compiler
    from write_gong_1mm import sha
    watch = json.loads(Path(request['watchdog_path']).read_text())
    if watch.get('passed') is not True or watch.get('compiler_sha256') != sha(compiler.__file__):
        raise ValueError('original controller watchdog hardware evidence invalid')
    report = {'status':'preflight', 'motion_sent':False, 'manifest':manifest,
        'reference_kind':'raw session-start wrench change, gravity included',
        'contact_force_map_verified':False, 'force_threshold_n':10,
        'reference_id':request['reference_id'], 'ticks':[]}
    def save():
        (out/'execution.json').write_text(json.dumps(report, indent=2)+'\n')
    save()
    reader = force = channel = server = None
    lockpath = Path.home()/'UR10/outputs/paper_writing/brush-execution.lock'
    with lockpath.open('a') as lock, (out/'controller_feedback.jsonl').open('x') as feedback, (out/'raw_force.jsonl').open('x') as raw:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            support.controller_ready('192.168.1.3')
            if not np.allclose(capture('192.168.1.3',30011,1.)['active_tcp_offset_m_rad'],0,atol=1e-8,rtol=0):
                raise RuntimeError('active TCP changed')
            reader = ContinuousState(support, '192.168.1.3')
            air = AirReader(reader, request['origin'], request['normal'], request['tip'], request['anchor'], feedback, request['baseline_rotation'])
            q, start, _ = air.stationary()
            expected = np.array(points[0]['pose_m_rad'])
            if np.linalg.norm(start[:3]-expected[:3]) > .0002 or (Rotation.from_rotvec(start[3:]).inv()*Rotation.from_rotvec(expected[3:])).magnitude() > math.radians(.05):
                raise RuntimeError('start differs from checked route')
            prior_q, prior_pose = q, start
            for point in points:
                support.model_gate(prior_q, prior_pose, np.array(point['pose_m_rad']), np.zeros(6))
                prior_q = np.array(point['q_rad']);prior_pose=np.array(point['pose_m_rad'])
                support.model_gate(prior_q, prior_pose, prior_pose, np.zeros(6))
            force = support.RawForce()
            with force.lock:
                force.rows = RecordingRows(force.rows, raw, request['reference_id'])
            deadline = time.monotonic()+10
            while True:
                try:force.recent();break
                except RuntimeError:
                    if time.monotonic()>deadline:raise
                    time.sleep(.02)
            health_tick(air, force, points)
            _, checked, _ = air.stationary()
            if np.linalg.norm(checked[:3]-start[:3])>.0001:
                raise RuntimeError('robot moved during preflight')
            server=socket.socket();server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            server.bind(('192.168.1.10',50071));server.listen(1);server.settimeout(10.)
            report.update(status='executing',motion_sent=True);save()
            support.command('192.168.1.3',script)
            channel,peer=server.accept()
            if peer[0]!='192.168.1.3':raise RuntimeError('unexpected callback peer')
            channel.setblocking(False)
            previous=None;sequence=0;deadline=time.monotonic()+budget
            while time.monotonic()<deadline:
                previous,pose,level=health_tick(air,force,points,previous)
                try:
                    incoming=channel.recv(16)
                    if incoming==b'':break
                    if incoming:raise RuntimeError('unexpected callback payload')
                except BlockingIOError:pass
                sequence+=1;channel.sendall(struct.pack('>i',sequence))
                if sequence%25==0:report['ticks'].append({'sequence':sequence,'raw_change_n':level,'pose_m_rad':pose.tolist()})
            else:raise RuntimeError('air segment runtime exceeded')
            _,end,_=air.stationary();target=np.array(points[-1]['pose_m_rad'])
            if np.linalg.norm(end[:3]-target[:3])>.0001 or (Rotation.from_rotvec(end[3:]).inv()*Rotation.from_rotvec(target[3:])).magnitude()>math.radians(.1):
                raise RuntimeError('callback closed without final pose')
            # Independently record a stationary window; do not tare or reset the session guard baseline.
            samples=[];deadline=time.monotonic()+3.
            while time.monotonic()<deadline:
                _, static_pose, _ = health_tick(air,force,points)
                if (np.linalg.norm(static_pose[:3]-end[:3])>.0001
                        or (Rotation.from_rotvec(end[3:]).inv()*Rotation.from_rotvec(static_pose[3:])).magnitude()>math.radians(.05)):
                    raise RuntimeError('movement during stationary force reference')
                rows=force.recent()
                samples.extend(rows)
            unique={r[1]:r for r in samples};rows=list(unique.values())
            values=np.array([r[3] for r in rows])
            if len(rows)<200 or np.max(np.std(values[:,:3],axis=0))>.5:
                raise RuntimeError('unstable stationary force reference')
            air.stationary()
            report.update(status='completed_air_reference',final_pose_m_rad=end.tolist(),
                static_reference={'actual_pose_m_rad':end.tolist(),'median_n_nm':np.median(values,axis=0).tolist(),
                    'std_n_nm':values.std(axis=0).tolist(),'rows':len(rows),'source':'raw_force.jsonl',
                    'first_unix_s':min(unique),'last_unix_s':max(unique),'independent_validation_passed':False})
        except BaseException as exc:
            report.update(status='stopped_on_error',error=type(exc).__name__+': '+str(exc))
            if channel is not None:channel.close();channel=None
            if report['motion_sent']:
                try:support.command('192.168.1.3','stopl(0.5)\n')
                except Exception as stop:report['stop_error']=str(stop)
            raise
        finally:
            if channel is not None:channel.close()
            if server is not None:server.close()
            if force is not None:force.close()
            if reader is not None:reader.close()
            support.close_commands();save()
    return report
