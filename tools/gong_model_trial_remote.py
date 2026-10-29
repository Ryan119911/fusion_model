"""Remote body injected by run_gong_model_trial. No motion unless EXECUTE is true."""
import copy,datetime as dt,json,math,struct,sys,time,types,traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
sys.dont_write_bytecode=True
sys.path[:0]=['/home/jx/.dsh-analysis','/home/jx/UR10/scripts']
import probe_paper_height as support
import write_gong_1mm as existing
import brush_buffered_motion as compiler
import brush_buffered_guard as guard
import brush_live_state as live
from calibrate_pen_tcp import capture

out=Path(REMOTE_OUTPUT);out.mkdir(exist_ok=False)
C=P['candidate'];points=copy.deepcopy(C['points']);acceptance=P['reference_acceptance']
force_namespace={};exec(FORCE_SNAPSHOT_SOURCE,force_namespace)
force_data_max_age_s=force_namespace['FORCE_DATA_MAX_AGE_S']
report={'status':'preflight_started','motion_sent':False,'force_tare_performed':False,'camera_triggered':False,
        'remote_output':str(out),'reference_translation_validated':False,'execute_requested':EXECUTE,
        'force_data_max_age_s':force_data_max_age_s}
reader=force=None

def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def event(value):
    # Full traces stay in execution.json/RESULT.json, rather than flooding the
    # terminal and concealing the actual exception behind thousands of ticks.
    console={k:v for k,v in value.items() if k not in ('ticks','manifest','error_traceback')}
    if 'ticks' in value:console['recorded_guard_ticks']=len(value['ticks'])
    print(json.dumps(console,ensure_ascii=False),flush=True)
def prediction(rotations):
    fit=acceptance['local_fit'];kind=fit.get('basis_kind','quadratic')
    x=(Rotation.from_rotvec(fit['baseline_rotation_rad']).inv()*Rotation.from_rotvec(rotations)).as_rotvec()/np.deg2rad(4)
    a,b,c=np.atleast_2d(x).T
    if kind=='linear':f=np.column_stack((a,b,c))
    elif kind=='affine':f=np.column_stack((np.ones(len(a)),a,b,c))
    elif kind=='quadratic':f=np.column_stack((np.ones(len(a)),a,b,c,a*a,b*b,c*c,a*b,a*c,b*c))
    else:raise RuntimeError('unsupported local reference basis: '+str(kind))
    coefficients=fit['coefficients'] if 'coefficients' in fit else fit['coefficients_10x6']
    return f@np.array(coefficients)+np.array(fit.get('offset',[0.]*6))
try:
    for name,text in P['original_software'].items():
        if Path(sys.modules[name].__file__).read_text()!=text:raise RuntimeError('original production source changed: '+name)
    expected_samples=[(r.get('phase'),r.get('sample',r.get('source_waypoint_index')))
        for r in list(C['audit']['samples'])+list(C.get('native_blend_neighborhood',{}).get('samples',[]))]
    supplied=P['collision'].get('supplied_samples')
    if (not C['audit']['passed_numeric'] or not isinstance(supplied,list) or
            [(r.get('phase'),r.get('sample',r.get('source_waypoint_index'))) for r in supplied]!=expected_samples):
        raise RuntimeError('offline full-route evidence mismatch')
    if P.get('body_collision',{}).get('minimum_clearance_mm',0)<P['body_collision'].get('margin_mm',20):
        raise RuntimeError('arm-body clearance is below the accepted margin')
    watch=json.loads(existing.WATCHDOG.read_text())
    if watch.get('passed') is not True or watch.get('compiler_sha256')!=existing.sha(compiler.__file__):raise RuntimeError('controller watchdog validation invalid')
    support.controller_ready('192.168.1.3')
    offset=np.array(capture('192.168.1.3',30011,1.)['active_tcp_offset_m_rad'])
    if not np.allclose(offset,0,atol=1e-8,rtol=0):raise RuntimeError('active zero tool0 TCP changed')
    reader=live.ContinuousState(support,'192.168.1.3');seed,start,_=reader.stationary()
    expected=np.array(points[0]['pose_m_rad'])
    tip=np.array(C['tcp']['translation_m']);origin=np.array(C['paper']['reconstructed_center_tip_base_m']);normal=np.array(C['paper']['paper_normal_base'])
    connector=types.ModuleType('brush_model_start');exec(CONNECTOR_SOURCE,connector.__dict__)
    points,connection=connector.connect_start(points,start,tip,origin,normal,P['reference_orientation_domain'],acceptance['local_fit']['baseline_rotation_rad'])
    height=connection['initial_tip_height_m'];start_height_upper_m=.300
    save(out/'start_connection.json',connection)
    # Independently check every newly added leg, using both existing attachment
    # models and the actual CAD versus the recorded paper plane.
    support.model_gate(seed,start,start,np.zeros(6))
    from self_collision import SelfCollisionModel
    from robot_attachments import load
    from ur_pose_ik import UR10IK
    from check_move_plan import check_segment,robot_pose
    cad_namespace={'np':np,'Rotation':Rotation,'Path':Path,'UPSTREAM':Path('/home/jx/UR10')}
    exec(CAD_SOURCE,cad_namespace)
    meshes=cad_namespace['cad_meshes']()
    cloud=np.concatenate([m.reshape(-1,3) for m in meshes.values()])
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Slerp
    cloud=cloud[ConvexHull(cloud).vertices]
    prefix_count=connection['prefix_point_count'];connector_points=points[:prefix_count+1]
    connector_checks={'passed':True,'models':{},'minimum_body_paper_clearance_mm':None,'motion_sent':False}
    cad_min=float('inf')
    for a,b in zip(connector_points,connector_points[1:]):
        first,last=np.array(a['pose_m_rad']),np.array(b['pose_m_rad'])
        angle=(Rotation.from_rotvec(first[3:]).inv()*Rotation.from_rotvec(last[3:])).magnitude()
        count=max(1,int(np.ceil(np.linalg.norm(last[:3]-first[:3])/.002)),int(np.ceil(angle/np.deg2rad(.5))))
        fractions=np.linspace(0,1,count+1);rotations=Slerp([0,1],Rotation.from_rotvec([first[3:],last[3:]]))(fractions)
        for f,rotation in zip(fractions,rotations):
            position=first[:3]+f*(last[:3]-first[:3])
            gap=float(np.min((cloud@rotation.as_matrix().T+position-origin)@normal))
            cad_min=min(cad_min,gap)
            tip_height=float((position+rotation.apply(tip)-origin)@normal)
            if gap<.020 or tip_height<min(height,.035)-.0001:raise RuntimeError('start connection CAD/paper or tip clearance failed')
    for name in ('robot_attachments.yaml','robot_attachments.preliminary-cad-20260921.yaml'):
        geometry=load('/home/jx/UR10/config/'+name,calibration_path='/home/jx/UR10/config/handeye_eye_in_hand_20260921.yaml',root='/home/jx/UR10')
        model=SelfCollisionModel(geometry=geometry);ik=UR10IK();q=seed.copy();legs=[]
        for a,b in zip(connector_points,connector_points[1:]):
            checked=check_segment(ik,robot_pose(a['pose_m_rad']),robot_pose(b['pose_m_rad']),q,30,model,.020)
            worst=checked['worst']
            if checked['reasons'] or worst['joint_margin_rad']<np.deg2rad(5) or worst['min_singular']<.02:raise RuntimeError('new start connection rejected by '+name)
            q=np.array(checked['end_q']);legs.append(checked)
        connector_checks['models'][name]={'passed':True,'legs':legs}
    connector_checks['minimum_body_paper_clearance_mm']=cad_min*1000 if np.isfinite(cad_min) else None
    save(out/'start_connection_checks.json',connector_checks)
    references=prediction([p['pose_m_rad'][3:] for p in points])
    for point,reference in zip(points,references):point['unloaded_wrench_n_nm']=reference.tolist()
    cache=P.get('cached_preflight')
    if cache:
        event({'event':'reusing_passed_writing_path_checking_new_start_only','waypoints':len(points),'motion_sent':False})
        joined=copy.deepcopy(points[:prefix_count+1])
        connector_audit=existing.audit_points(joined,seed,np.zeros(6),support)
        tail=copy.deepcopy(cache['tail_points'])
        if len(tail)!=len(C['points']) or not connector_audit['passed']:raise RuntimeError('cached writing path or new connector is incomplete')
        if np.max(abs(np.array(joined[-1]['q_rad'])-np.array(tail[0]['q_rad'])))>np.deg2rad(.1):raise RuntimeError('new connection reaches a different IK branch from the checked writing path')
        points=joined[:-1]+tail
        audit={'passed':True,'scope':'exact previously passed writing path plus newly checked current-start connection',
            'cached_preflight':cache['source_directory'],'source_route_sha256':cache['original_route_sha256'],
            'source_program_sha256':cache['original_program_sha256'],'new_connection_audit':connector_audit}
    else:
        event({'event':'checking_existing_native_movel_route_and_blend_neighborhood','waypoints':len(points),'motion_sent':False})
        audit=existing.audit_points(points,seed,np.zeros(6),support)
    if audit.get('passed') is not True:raise RuntimeError('existing full blended-route audit did not pass')
    _,rechecked,_=reader.stationary()
    if np.linalg.norm(rechecked[:3]-start[:3])>.0001 or (Rotation.from_rotvec(rechecked[3:]).inv()*Rotation.from_rotvec(start[3:])).magnitude()>np.deg2rad(.05):
        raise RuntimeError('robot moved during route checks')
    force=support.RawForce()
    force.recent=types.MethodType(force_namespace['recent_force'],force)
    baseline,stats=force.baseline()
    with force.lock:raw=[{'received_unix_s':r[1],'source_unix_s':r[2],'wrench_n_nm':r[3]} for r in force.rows]
    discrepancy=float(np.linalg.norm(np.array(baseline[:3])-prediction([start[3:]])[0,:3]))
    save(out/'suspended_start_observation.json',{'pose_m_rad':start.tolist(),'nominal_tip_height_m':height,
         'observed_raw_wrench_n_nm':baseline.tolist(),'model_error_n':discrepancy,'statistics':stats,'raw_rows':raw,
         'scope':'fresh suspended start observation; no tare and no claim of translated-path reference validation'})
    if discrepancy>=10:raise RuntimeError('fresh suspended reference error reaches the selected 10 N limit')
    force.close();force=None;reader.close();reader=None
    script,manifest=compiler.compile_program(points,'192.168.1.10',50071);identity=compiler.route_identity(points)
    # Constant-speed distance/v badly under-estimates short segments: a 0.4 mm
    # move at 10 mm/s never reaches v because the acceleration limit dominates.
    # Use the rest-to-rest minimum move time per segment (trapezoid/triangle),
    # then a measured margin: actual/estimate was 1.21 on the 1 mm/s route and
    # 1.63 on the fast route (518 s actual vs 317 s estimate), so 1.8 is used.
    def segment_time(distance,speed,acceleration):
        if distance<=0.:return 0.
        if distance<=speed*speed/acceleration:return 2.*math.sqrt(distance/acceleration)
        return distance/speed+speed/acceleration
    pairs=list(zip(points,points[1:]))
    duration=sum(segment_time(float(np.linalg.norm(np.array(b['pose_m_rad'][:3])-a['pose_m_rad'][:3])),
                              float(b['speed_m_s']),float(b['acceleration_m_s2'])) for a,b in pairs)
    constant=sum(float(np.linalg.norm(np.array(b['pose_m_rad'][:3])-a['pose_m_rad'][:3]))/float(b['speed_m_s'])
                 for a,b in pairs)
    budget=max(120.,float(duration)*1.8+60.)
    ceiling=float(P.get('maximum_runtime_s',1800.))
    if budget>ceiling:raise RuntimeError('finite trial exceeds the accepted %.0f s runtime ceiling'%ceiling)
    reports={
        'physical_mapping':{'passed':True,'route_sha256':identity,'scope':'arithmetic conversion of model fields into explicit physical commands, accepted only for one provisional trial; not calibrated brush compression',
             'conversion':C['conversion'],'zero_source':C['zero'],'paper':C['paper'],'tcp':C['tcp'],'brush_mapping_calibrated':False},
        'force_reference':{'passed':True,'route_sha256':identity,'scope':'operator-selected local small-angle air reference with 10 N error acceptance, applied provisionally to this initial translated trial; translations and contact remain unvalidated',
             'reference_acceptance':acceptance,'fresh_start_error_n':discrepancy,'reference_translation_validated':False,
             'force_data_max_age_s':force_data_max_age_s,'operator_policy':P['operator_policy']},
        'blended_path_audit':{**audit,'route_sha256':identity,'program_sha256':manifest['program_sha256'],
             'offline_two_model_collision':P['collision'],'offline_arm_body_collision':P.get('body_collision'),
             'new_start_connection_checks':connector_checks,
             'paper_numeric':{'passed':C['audit']['passed_numeric'],'source_scope':C['audit']['scope']}}}
    evidence={}
    for name,data in reports.items():
        path=out/(name+'.json');save(path,data);evidence[name]={'path':path.name,'sha256':existing.sha(path)}
    bundle={'sample_id':'gong-full-model-initial-trial','points':points,'start_pose_m_rad':points[0]['pose_m_rad'],
       'active_tcp_offset_m_rad':[0.]*6,'maximum_runtime_s':budget,
       'physical_mapping_verified':True,'force_reference_verified':True,'controller_watchdog_hardware_validated':True,
       'evidence':evidence,'zero_source':C['zero'],'model_H_mm_preserved':True,'physical_compression_mm':None,
       'trial_acceptance_scope':'explicit provisional commands and local reference only; no completed physical calibration claim',
       'reference_translation_validated':False,'conversion':C['conversion'],'force_data_max_age_s':force_data_max_age_s}
    blockers=compiler.execution_blockers(bundle,manifest,out)
    if blockers:raise RuntimeError('; '.join(blockers))
    save(out/'bundle.json',bundle);save(out/'compiler_manifest.json',manifest);(out/'program.script.preview').write_text(script)
    save(out/'request.json',P)
    report.update(status='preflight_passed_no_motion',runtime_budget_s=budget,estimated_translation_motion_s=float(duration),
         estimated_constant_speed_s=float(constant),estimate_model='rest-to-rest minimum move time per segment, 1.8 margin + 60 s',
         start_reference_error_n=discrepancy,waypoints=len(points),conversion=C['conversion'],zero=C['zero'],
         start_tip_height_m=height,start_height_gate_m=[.010,start_height_upper_m],start_connection=connection)
    event(report)
    if EXECUTE:
        recorded=SimpleNamespace(**vars(support));namespace={};exec(RECORDING_SOURCE,namespace)
        with (out/'raw_force.jsonl').open('x') as raw_stream,(out/'controller_feedback.jsonl').open('x') as feedback:
            def raw_force(*args,**kwargs):
                value=support.RawForce(*args,**kwargs)
                with value.lock:value.rows=namespace['RecordingRows'](value.rows,raw_stream,'gong-full-model-initial-trial')
                value.recent=types.MethodType(force_namespace['recent_force'],value)
                return value
            recorded.RawForce=raw_force
            recorded.force_snapshot=lambda f:{'path':str(out/'raw_force.jsonl'),'scope':'complete streaming raw force; bounded live window unchanged'}
            # ContinuousState delegates to support.fresh in its background thread.
            def fresh(r,*args,**kwargs):
                value=support.fresh(r,*args,**kwargs)
                feedback.write(json.dumps({'sample_id':'gong-full-model-initial-trial','host_unix_ns':time.time_ns(),
                    'controller_time_s':struct.unpack_from('>d',r.last_frame,4)[0],'actual_q_rad':value[0].tolist(),
                    'actual_tool0_pose_m_rad':value[1].tolist(),'tcp_velocity':value[2].tolist()})+'\n')
                return value
            recorded.fresh=fresh
            try:guard.execute_buffered(bundle,out,'192.168.1.10',50071,out/'execution.json',recorded,capture,reuse_checked_path=bool(cache))
            finally:
                path=out/'execution.json'
                if path.exists():report.update(json.loads(path.read_text()))
except BaseException as exc:
    report.update(status='stopped_on_error',error=type(exc).__name__+': '+str(exc),error_traceback=traceback.format_exc())
    if hasattr(exc,'diagnostics'):report['failure_diagnostics']=exc.diagnostics
finally:
    if force is not None:force.close()
    if reader is not None:reader.close()
    support.close_commands();report['finished_at']=dt.datetime.now(dt.timezone.utc).isoformat()
    save(out/'RESULT.json',report);event(report)
if report['status']=='stopped_on_error':raise SystemExit(1)
