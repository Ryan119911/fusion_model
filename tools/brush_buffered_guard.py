"""Host-side live guard for one buffered program; never wait at waypoints.

No CLI and no automatic launch. The caller supplies independently audited
physical poses, unloaded wrench references and the robot-reachable guard IP.
"""
import json
import math
import socket
import struct
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from brush_buffered_motion import compile_program,execution_blockers
from brush_live_state import ContinuousState
from brush_trial_force import trial_force_level


class PreparedRoute(list):
    """Snapshot invariant route geometry once, before arming the live stream."""
    def __init__(self,points):
        super().__init__(points)
        poses=np.array([p['pose_m_rad'] for p in points],float)
        self.references=np.array([p['unloaded_wrench_n_nm'] for p in points],float)
        if self.references.shape!=(len(points),6) or not np.isfinite(self.references).all():
            raise ValueError('complete validated unloaded-wrench references required')
        self.xyz=poses[:,:3].copy();self.xyz_delta=np.diff(self.xyz,axis=0)
        self.xyz_denominator=np.maximum(np.sum(self.xyz_delta*self.xyz_delta,axis=1),1e-20)
        self.features=np.column_stack((self.xyz,Rotation.from_rotvec(poses[:,3:]).as_matrix().reshape(-1,9)*.05))
        self.feature_delta=np.diff(self.features,axis=0)
        self.feature_denominator=np.maximum(np.sum(self.feature_delta*self.feature_delta,axis=1),1e-20)


def reference_at_pose(points,pose):
    # Position plus rotation-matrix coordinates avoid rotvec wrap jumps.
    route=points if isinstance(points,PreparedRoute) else PreparedRoute(points)
    refs=route.references;features=route.features
    query=np.r_[pose[:3],Rotation.from_rotvec(pose[3:]).as_matrix().reshape(-1)*.05]
    delta=route.feature_delta
    weight=np.clip(np.sum((query-features[:-1])*delta,axis=1)/route.feature_denominator,0,1)
    projected=features[:-1]+weight[:,None]*delta
    index=int(np.argmin(np.linalg.norm(projected-query,axis=1)))
    return refs[index]+weight[index]*(refs[index+1]-refs[index])


def health_tick(reader,force,points,previous=None):
    q,pose,velocity=reader.fresh(timeout=.15)
    now=time.monotonic()
    if np.linalg.norm(velocity[:3])>.040 or not np.isfinite(np.r_[q,pose,velocity]).all():
        raise RuntimeError('invalid state or TCP speed above guard envelope')
    route=points if isinstance(points,PreparedRoute) else PreparedRoute(points)
    xyz=route.xyz;segments=route.xyz_delta
    fraction=np.clip(np.sum((pose[:3]-xyz[:-1])*segments,axis=1)/route.xyz_denominator,0,1)
    projected=xyz[:-1]+fraction[:,None]*segments
    if np.min(np.linalg.norm(projected-pose[:3],axis=1))>.0008:
        raise RuntimeError('TCP escaped audited route/blend tracking envelope')
    lower=np.deg2rad([-360,-360,-180,-360,-360,-360]);upper=-lower
    if min(np.min(q-lower),np.min(upper-q))<np.deg2rad(3):
        raise RuntimeError('live joint limit margin below 3 degrees')
    if previous is not None:
        stamp,old_q=previous
        if now-stamp>.15:
            raise RuntimeError('host guard scheduling gap above150ms')
        if np.max(abs(q-old_q))/max(now-stamp,.001)>.60:
            raise RuntimeError('unwrapped live joint speed above0.60rad/s')
    reference=reference_at_pose(route,pose)
    level=trial_force_level(force,reference,10.)
    return (now,q.copy()),pose,level


def execute_buffered(bundle,folder,guard_host,guard_port,output,support,capture,reuse_checked_path=False):
    """All model checks precede the one send. Any guard failure stops the job.

    No model_gate/stationary_state/SSH handshake occurs inside motion.
    A dead host or closed callback also causes controller-local timeout/stop.
    """
    folder=Path(folder);output=Path(output)
    script,manifest=compile_program(bundle['points'],guard_host,guard_port)
    blockers=execution_blockers(bundle,manifest,folder)
    if blockers:
        raise RuntimeError('; '.join(blockers))
    budget=bundle.get('maximum_runtime_s')
    if type(budget) not in (int,float) or not math.isfinite(budget) or not 0<budget<=1800:
        raise ValueError('explicit bounded maximum_runtime_s required')
    points=bundle['points'];host='192.168.1.3'
    prepared=PreparedRoute(points)
    output.parent.mkdir(parents=True,exist_ok=True)
    report={'status':'preflight','motion_sent':False,'manifest':manifest,'ticks':[]}
    with output.open('x') as stream:json.dump(report,stream)
    reader=force=server=channel=None

    def save():
        temporary=output.with_suffix('.tmp');temporary.write_text(json.dumps(report,indent=2)+'\n');temporary.replace(output)

    import fcntl
    lockpath=Path.home()/'UR10/outputs/paper_writing/brush-execution.lock'
    lockpath.parent.mkdir(parents=True,exist_ok=True)
    with lockpath.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            report['controller']=support.controller_ready(host)
            offset=np.asarray(capture(host,30011,1)['active_tcp_offset_m_rad'])
            if not np.allclose(offset,bundle['active_tcp_offset_m_rad'],atol=1e-8,rtol=0):
                raise RuntimeError('active controller TCP changed')
            reader=ContinuousState(support,host)
            q,start,_=reader.stationary();expected=np.asarray(bundle['start_pose_m_rad'])
            if np.linalg.norm(start[:3]-expected[:3])>.0002 or (Rotation.from_rotvec(start[3:]).inv()*Rotation.from_rotvec(expected[3:])).magnitude()>math.radians(.05):
                raise RuntimeError('start differs from independently audited route')
            first=np.asarray(points[0]['pose_m_rad'])
            if np.linalg.norm(start[:3]-first[:3])>.0002 or (Rotation.from_rotvec(start[3:]).inv()*Rotation.from_rotvec(first[3:])).magnitude()>math.radians(.05):
                raise RuntimeError('buffered route must include measured start pose as its first free-space waypoint')
            # Expensive geometry checks run before arming the live stream.
            # No robot program has been sent. A new stationary read after the
            # checks binds the unchanged start to the subsequent 150 ms guard.
            reader.close();reader=None
            prior_q,prior_pose=q,start
            if reuse_checked_path:
                # execution_blockers above already bound the complete path
                # audit to this route and compiled program. Recheck actual FK.
                support.model_gate(q,start,start,offset)
            for point in ([] if reuse_checked_path else points):
                target=np.asarray(point['pose_m_rad'])
                support.model_gate(prior_q,prior_pose,target,offset)
                prior_q=np.asarray(point['q_rad'],float)
                support.model_gate(prior_q,target,target,offset)
                prior_pose=target
            force=support.RawForce()
            # Never compute a new contact tare; references come from the
            # validated complete posture-dependent unloaded-force mapping.
            ready_deadline=time.monotonic()+10
            while True:
                try:
                    force.recent()
                    break
                except RuntimeError:
                    if time.monotonic()>=ready_deadline:raise
                    time.sleep(.02)
            reader=ContinuousState(support,host)
            _,rechecked,_=reader.stationary()
            if np.linalg.norm(rechecked[:3]-start[:3])>.0001 or (Rotation.from_rotvec(start[3:]).inv()*Rotation.from_rotvec(rechecked[3:])).magnitude()>math.radians(.05):
                raise RuntimeError('robot moved during preflight')
            health_tick(reader,force,prepared)
            server=socket.socket();server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            server.bind((guard_host,guard_port));server.listen(1);server.settimeout(10.)
            report.update(status='executing',motion_sent=True);save()
            support.command(host,script)
            channel,peer=server.accept()
            if peer[0]!=host:
                raise RuntimeError('guard callback came from an unexpected peer')
            channel.setblocking(False)
            sequence=0;previous=None;deadline=time.monotonic()+bundle['maximum_runtime_s']
            while time.monotonic()<deadline:
                previous,pose,level=health_tick(reader,force,prepared,previous)
                try:
                    incoming=channel.recv(16)
                    if incoming==b'':break
                    if incoming:raise RuntimeError('unexpected guard callback payload')
                except BlockingIOError:
                    pass
                sequence+=1
                channel.sendall(struct.pack('>i',sequence))
                if sequence%25==0:
                    report['ticks'].append({'sequence':sequence,'pose_m_rad':pose.tolist(),'force_delta_n':level})
            else:
                raise RuntimeError('complete buffered program exceeded runtime budget')
            _,end,_=reader.stationary();target=np.asarray(points[-1]['pose_m_rad'])
            if np.linalg.norm(end[:3]-target[:3])>.0001 or (Rotation.from_rotvec(end[3:]).inv()*Rotation.from_rotvec(target[3:])).magnitude()>math.radians(.1):
                raise RuntimeError('program connection closed without verified final pose')
            report.update(status='completed_buffered_program',final_pose_m_rad=end.tolist())
        except BaseException as exc:
            report.update(status='stopped_on_error',error=str(exc))
            # Callback close removes heartbeat permission even if the
            # secondary stop connection fails.
            if channel is not None:channel.close();channel=None
            if report['motion_sent']:
                try:support.command(host,'stopl(0.5)\n')
                except Exception as stop_error:report['stop_error']=str(stop_error)
            raise
        finally:
            try:
                if force is not None:report['force_trace']=support.force_snapshot(force)
                save()
            finally:
                if channel is not None:channel.close()
                if server is not None:server.close()
                try:
                    if force is not None and force.rclpy.ok():force.close()
                finally:
                    if reader is not None:reader.close()
                    support.close_commands()
