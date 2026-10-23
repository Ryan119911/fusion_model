#!/usr/bin/env python3
"""Continuous three-stroke gong XY trial with explicit commanded depth <=1 mm.

Default: live read-only preflight/export. --execute --paper-unchanged starts
one finite job after path and heartbeat checks. Fixed orientation only: neural
H/alpha/beta/gamma are not substituted or claimed as calibrated physical pose.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import socket
import sys

import numpy as np
from scipy.spatial.transform import Rotation
from brush_buffered_motion import blend_radii,compile_program,route_identity
from brush_buffered_guard import execute_buffered
from brush_live_state import ContinuousState
from brush_trial_force import trial_force_level

ROOT=Path.home()/'UR10'
DEFAULT_GEOMETRY=ROOT/'outputs/gong_xy_water_trial_20261016/gong_geometry_ready.json'
WATCHDOG=ROOT/'outputs/gong_1mm_continuous_20261016/watchdog_validation_v2.json'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_points(data,start,press_mm):
    if not math.isfinite(press_mm) or not 0<press_mm<=1:
        raise ValueError('max-press-mm must be >0 and <=1 mm')
    start=np.asarray(start,float);zero=np.asarray(data['center_zero_controller_pose_m_rad'],float).copy()
    if start.shape!=(6,) or zero.shape!=(6,) or not np.isfinite(np.r_[start,zero]).all():
        raise ValueError('invalid pose')
    angle=(Rotation.from_rotvec(zero[3:]).inv()*Rotation.from_rotvec(start[3:])).magnitude()
    if angle>np.deg2rad(.1):raise ValueError('restore taught fixed brush orientation')
    correction=Rotation.from_rotvec(zero[3:]).apply([0,0,.315])-Rotation.from_rotvec(start[3:]).apply([0,0,.315])
    zero[:3]+=correction;zero[3:]=start[3:]
    u=np.asarray(data['paper_u_base'],float);normal=np.asarray(data['paper_normal_base'],float)
    if not np.isfinite(np.r_[u,normal]).all() or abs(np.linalg.norm(u)-1)>1e-6 or abs(np.linalg.norm(normal)-1)>1e-6 or abs(u@normal)>1e-6:
        raise ValueError('invalid taught paper axes')
    v=np.cross(normal,u);v/=np.linalg.norm(v)
    xy=np.asarray(data['glyph_xy_m'],float);ids=np.asarray(data['glyph_stroke_ids'],int)
    if xy.shape!=(16,2) or ids.shape!=(16,) or not np.isfinite(xy).all() or list(dict.fromkeys(ids))!=[0,1,2]:
        raise ValueError('source must retain16 controls/3 ordered strokes')
    bbox=np.ptp(xy,axis=0)
    if bbox[0]>.1485 or bbox[1]>.105 or np.prod(bbox)>.297*.210*.25:
        raise ValueError('glyph exceeds quarter-A4 bounds')
    if not .002<=start[2]-zero[2]<=.025:
        raise ValueError('start must be suspended2..25mm above taught paper at same orientation')
    safez=max(start[2],max(np.asarray(data['canonical_controller_corners_m'])[:,2])+correction[2]+.010)
    points=[]
    baseline=np.asarray(data['force_reference_baseline_n_nm'],float)
    if baseline.shape!=(6,) or not np.isfinite(baseline).all():raise ValueError('original unloaded reference required')

    def append(phase,pose,contact=False,sid=None,speed=.003,accel=.002):
        points.append({'phase':phase,'pose_m_rad':pose.tolist(),'contact':contact,'stroke_id':sid,
                       'speed_m_s':speed,'acceleration_m_s2':accel,
                       'unloaded_wrench_n_nm':baseline.tolist()})

    append('measured_start',start)
    raised=start.copy();raised[2]=safez
    if np.linalg.norm(raised[:3]-start[:3])>.0001:append('raise',raised)
    controls=zero[:3]+xy[:,0,None]*u+xy[:,1,None]*v
    for sid in (0,1,2):
        stroke=controls[ids==sid];first=np.r_[stroke[0],start[3:]]
        air=first.copy();air[2]=safez
        near=first.copy();near[2]+=.001
        append('air_'+str(sid+1),air,speed=.030,accel=.100)
        append('approach_'+str(sid+1),near,speed=.003)
        for index,point in enumerate(stroke):
            contact=np.r_[point,start[3:]];contact[2]-=press_mm/1000
            append('stroke_%d_point_%d'%(sid+1,index),contact,True,sid,
                   .0003 if index==0 else .030,.002 if index==0 else .100)
        lifted=np.r_[stroke[-1],start[3:]];lifted[2]=safez
        append('lift_'+str(sid+1),lifted)
    return points,{'point_m':zero[:3].tolist(),'normal':normal.tolist(),'commanded_press_limit_mm':press_mm}


def audit_points(points,start_q,offset,support):
    # Support loads the existing robotpkg environment before direct imports.
    support.model_gate(start_q,np.asarray(points[0]['pose_m_rad']),np.asarray(points[0]['pose_m_rad']),offset)
    from ur_pose_ik import UR10IK
    from check_move_plan import check_segment,robot_pose
    from self_collision import SelfCollisionModel
    ik=UR10IK();collision=SelfCollisionModel();q=np.asarray(start_q)
    previous=np.asarray(points[0]['pose_m_rad']);reports=[]

    def checked(a,b,seed,count=30):
        gate=check_segment(ik,robot_pose(a),robot_pose(b),seed,count,collision,.020)
        worst=gate['worst']
        if gate['reasons'] or worst['joint_margin_rad']<np.deg2rad(5) or worst['min_singular']<.02:
            raise RuntimeError('robot path gate failed: '+str(gate['reasons'][:3]))
        return gate

    for point in points:
        target=np.asarray(point['pose_m_rad']);gate=checked(previous,target,q)
        length=float(np.linalg.norm(target[:3]-previous[:3]))
        rate=0 if length<1e-6 else gate['worst']['max_joint_step_rad']*30/length*point['speed_m_s']
        if rate>.60:raise RuntimeError('planned unwrapped joint speed estimate exceeds0.60rad/s')
        point['q_rad']=gate['end_q'];q=np.asarray(gate['end_q'])
        reports.append({'phase':point['phase'],'worst':gate['worst'],'joint_speed_estimate_rad_s':rate})
        previous=target
    # Native blend shape stays within a small local region. Sample its
    # conservative3D neighborhood as well as the straight centerlines.
    radii=blend_radii(points);neighborhoods=0
    for i,radius in enumerate(radii):
        if not radius:continue
        anchor=np.asarray(points[i]['pose_m_rad']);seed=np.asarray(points[i]['q_rad'])
        for direction in ([1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]):
            target=anchor.copy();target[:3]+=radius*np.asarray(direction)
            checked(anchor,target,seed,3);neighborhoods+=1
    return {'passed':True,'scope':'factory FK, continuous IK, joint margin/speed estimate, singularity, camera self-clearance; sampled blend neighborhood',
            'segments':reports,'blend_neighborhood_samples':neighborhoods,
            'environment_clearance_basis':'operator confirms clear work area; taught paper and fixed tool pose; not a complete external obstacle mesh proof'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--geometry',type=Path,default=DEFAULT_GEOMETRY)
    p.add_argument('--max-press-mm',type=float,default=1.)
    p.add_argument('--paper-unchanged',action='store_true')
    p.add_argument('--execute',action='store_true')
    p.add_argument('--guard-host');p.add_argument('--guard-port',type=int,default=50071)
    p.add_argument('--output-dir',type=Path)
    args=p.parse_args()
    if not math.isfinite(args.max_press_mm) or not 0<args.max_press_mm<=1:
        p.error('--max-press-mm must be >0 and <=1; larger values are refused')
    if args.execute and not args.paper_unchanged:
        p.error('--execute requires --paper-unchanged after checking paper/tool/clear work area')
    raw=args.geometry.read_bytes();data=json.loads(raw)
    for r in data['sha_sources']:
        if sha(r['path'])!=r['sha256']:p.error('taught/source evidence changed')
    reference=data['force_reference_evidence']
    if sha(reference['path'])!=reference['sha256']:p.error('original unloaded reference evidence changed')
    watchdog=json.loads(WATCHDOG.read_text())
    if watchdog.get('passed') is not True or watchdog['compiler_sha256']!=sha(Path(__file__).with_name('brush_buffered_motion.py')):
        p.error('controller heartbeat validation absent or compiler changed')
    sys.path[:0]=[str(ROOT/'scripts'),str(Path.home()/'.dsh-analysis')]
    import probe_paper_height as support
    from calibrate_pen_tcp import capture
    import fcntl
    import time
    lockpath=ROOT/'outputs/paper_writing/brush-execution.lock'
    out=args.output_dir or ROOT/'outputs/gong_1mm_continuous_20261016'/('run-'+dt.datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    out.mkdir(parents=True,exist_ok=False)
    reader=force=None
    with lockpath.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            controller=support.controller_ready('192.168.1.3')
            sample=capture('192.168.1.3',30011,1);offset=np.asarray(sample['active_tcp_offset_m_rad'])
            if not np.allclose(offset,0,atol=1e-8,rtol=0):raise RuntimeError('restore taught zero active TCP; no set_tcp is sent')
            reader=ContinuousState(support,'192.168.1.3');q,start,_=reader.stationary()
            points,plane=build_points(data,start,args.max_press_mm)
            audit=audit_points(points,q,offset,support)
            if args.guard_host:guard_host=args.guard_host
            else:
                with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as route:
                    route.connect(('192.168.1.3',30013));guard_host=route.getsockname()[0]
            script,manifest=compile_program(points,guard_host,args.guard_port)
            fingerprint=route_identity(points)
            evidence={}
            records={
              'physical_mapping':{'passed':True,'scope':'operator taught paper, declared315mm small orientation correction, fixed orientation XY-only commanded-depth trial; not neural-H calibration','route_sha256':fingerprint,'model_H_angles_executed':False,'plane':plane,'paper_unchanged_operator_asserted':args.paper_unchanged},
              'force_reference':{'passed':True,'scope':'fixed orientation original unloaded baseline vector;10N increment guard; not varying-posture gravity compensation','route_sha256':fingerprint,'reference_evidence':reference},
              'blended_path_audit':{**audit,'route_sha256':fingerprint,'program_sha256':manifest['program_sha256']}}
            for name,record in records.items():
                path=out/(name+'.json')
                with path.open('x') as f:json.dump(record,f,indent=2)
                evidence[name]={'path':path.name,'sha256':sha(path)}
            bundle={'scope':'operator-authorized fixed orientation gong XY trial, commanded press <=1mm','character':'工',
                    'points':points,'start_pose_m_rad':start.tolist(),'active_tcp_offset_m_rad':offset.tolist(),
                    'maximum_runtime_s':240,'physical_mapping_verified':True,'force_reference_verified':True,
                    'controller_watchdog_hardware_validated':True,'watchdog_evidence':{'path':str(WATCHDOG),'sha256':sha(WATCHDOG)},
                    'evidence':evidence,'model_H_angles_executed':False,'max_commanded_press_mm':args.max_press_mm,
                    'operator_paper_tool_unchanged':args.paper_unchanged,'controller':controller,'source_geometry_sha256':hashlib.sha256(raw).hexdigest()}
            with (out/'bundle.json').open('x') as f:json.dump(bundle,f,ensure_ascii=False,indent=2)
            with (out/'program.script.preview').open('x') as f:f.write(script)
            force=support.RawForce();deadline=time.monotonic()+10
            while True:
                try:force.recent();break
                except RuntimeError:
                    if time.monotonic()>=deadline:raise
                    time.sleep(.02)
            delta=trial_force_level(force,np.asarray(data['force_reference_baseline_n_nm']),10.)
            print(json.dumps({'preflight':'passed','output':str(out),'waypoints':len(points),'max_commanded_press_mm':args.max_press_mm,'force_increment_n':delta,'guard_host':guard_host,'execute_requested':args.execute}),flush=True)
        finally:
            if force is not None and force.rclpy.ok():force.close()
            if reader is not None:reader.close()
            support.close_commands()
    if args.execute:
        execute_buffered(bundle,out,guard_host,args.guard_port,out/'execution.json',support,capture)
        print('COMPLETE: three gong strokes and final lift. REPORT:',out/'execution.json',flush=True)
    else:print('NO MOTION: add --execute --paper-unchanged to start a fresh preflight/job',flush=True)


if __name__=='__main__':main()
