#!/usr/bin/env python3
"""Default offline preview. Explicit modes restore an audited unloaded posture.

Each bounded segment keeps a fixed suspended tip, uses a fresh measured start
wrench without tare, and records its endpoint. Reference plans additionally save
static training, independent holdout and return observations. No writing mode exists.
"""
import argparse,ast,datetime as dt,json,shlex,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'outputs/brush_mapping_20261027/gong_air_alignment_current_mount/candidate.json'


def air_source_at_height(source,height,time_margin=2.5):
    if not np.isfinite(time_margin) or not 1.5<=time_margin<=2.5:
        raise ValueError('air timing margin must remain inside 1.5..2.5')
    low,high=float(height)-.005,float(height)+.005
    if not np.isfinite(height) or low<.035 or high>.300:
        raise ValueError('10 mm air cell must remain inside 35..300 mm nominal clearance')
    old='.035 <= height <= .045'
    if source.count(old)!=2:raise ValueError('air reader/validator source changed')
    new='AIR_HEIGHT_RANGE_M[0] <= height <= AIR_HEIGHT_RANGE_M[1]'
    source=source.replace(old,new)
    prefix='AIR_HEIGHT_RANGE_M='+repr((low,high))+'\n'
    if time_margin!=2.5:
        if source.count('2.5*')!=3:raise ValueError('air timing source changed')
        source=source.replace('2.5*','AIR_TIME_MARGIN*')
        prefix+='AIR_TIME_MARGIN='+repr(float(time_margin))+'\n'
    return prefix+source


REMOTE=r'''
import sys,types,json,time,traceback
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.dont_write_bytecode=True
sys.path[:0]=['/home/jx/.dsh-analysis','/home/jx/UR10/scripts','/opt/openrobots/lib/python3.10/site-packages']
import probe_paper_height as support
import brush_buffered_motion as compiler
import brush_buffered_guard as guard
import brush_live_state as live
from calibrate_pen_tcp import capture
from write_gong_1mm import sha,WATCHDOG
import self_collision as original_collision
base_collision=original_collision.SelfCollisionModel
exec(COLLISION_SOURCE,globals())
# Scope the current CAD adapter to this child process. The original model_gate
# still enforces its FK, limits, singularity and segment-clearance conditions.
original_collision.SelfCollisionModel=lambda *a,**kw:model_for_current_mount('/home/jx/UR10/config/robot_attachments.yaml',BOXES,base_class=base_collision)
namespace={};exec(FORCE_SOURCE,namespace)
raw_class=support.RawForce
fresh_class=lambda *a,**kw:raw_class(*a,**kw)
def fresh_force(*a,**kw):
 f=fresh_class(*a,**kw);f.recent=types.MethodType(namespace['recent_force'],f);return f
support.RawForce=fresh_force
recording=types.ModuleType('brush_write_capture');exec(RECORDING_SOURCE,recording.__dict__);sys.modules['brush_write_capture']=recording
air=types.ModuleType('brush_air_alignment_runtime');air.__file__='scoped_air_alignment';exec(AIR_SOURCE,air.__dict__)
out=Path(REMOTE_OUTPUT);out.mkdir(exist_ok=False)
report={'status':'started','motion_sent':False,'execute_requested':EXECUTE,'completed_segments':[],
 'mode':'unloaded_alignment_only','force_tare_performed':False,'contact_force_map_verified':False,
 'force_data_max_age_s':1.5,'air_height_range_m':list(air.AIR_HEIGHT_RANGE_M),
 'baseline_scope':'fresh unloaded baseline at each bounded segment start, never reset within a segment',
 'mount_model':P['mount_model']}
reader=force=None
try:
 for name,source in PRODUCTION.items():
  if Path(sys.modules[name].__file__).read_text()!=source:raise RuntimeError('production source changed: '+name)
 if not P['audit']['passed_numeric']:raise RuntimeError('numeric air alignment failed')
 collision={'models':{},'supplied_samples':P['audit']['samples'],'motion_sent':False}
 for name in ['robot_attachments.yaml','robot_attachments.preliminary-cad-20260921.yaml']:
  model=model_for_current_mount('/home/jx/UR10/config/'+name,BOXES,base_class=base_collision)
  minimum=min(model.min_clearance(row['q_rad'])[0] for row in P['audit']['samples'])
  collision['models'][name]={'minimum_clearance_mm':minimum*1000,'passed_20mm':minimum>=.020}
 if not all(x['passed_20mm'] for x in collision['models'].values()):raise RuntimeError('current-mount air collision check failed')
 (out/'self_collision.json').write_text(json.dumps(collision,indent=2)+'\n')
 watch=json.loads(WATCHDOG.read_text())
 if watch.get('passed') is not True or watch.get('compiler_sha256')!=sha(compiler.__file__):raise RuntimeError('controller watchdog evidence invalid')
 support.controller_ready('192.168.1.3')
 if not np.allclose(capture('192.168.1.3',30011,1.)['active_tcp_offset_m_rad'],0,atol=1e-8,rtol=0):raise RuntimeError('active TCP changed')
 reader=live.ContinuousState(support,'192.168.1.3');start_q,start,_=reader.stationary()
 expected=np.array(P['points'][START]['pose_m_rad'])
 if np.linalg.norm(start[:3]-expected[:3])>.0002 or (Rotation.from_rotvec(start[3:]).inv()*Rotation.from_rotvec(expected[3:])).magnitude()>np.deg2rad(.05):raise RuntimeError('start differs from checked alignment; no reposition')
 # Check the current controller/model convention once, then use the full dense
 # current-mount route audit for the remaining static geometry checks.
 support.model_gate(start_q,start,start,np.zeros(6))
 checked_session={'points':P['points'],'audit':P['audit'],'collision':collision}
 if P.get('jobs'):
  jobs=json.loads(json.dumps(P['jobs']))
  for job in jobs:
   job['collision']={'models':collision['models'],'supplied_samples':job['audit']['samples']}
   job['watchdog_path']=str(WATCHDOG)
   job['collect_reference']=True
   air.validate_air_route(job['points'],job['audit'],job['collision'],job['origin'],job['normal'],job['tip'],job['anchor'],job['baseline_rotation'])
   _,timing=air.compile_air_program(job['points'])
   if timing['total_motion_duration_s']+10>job['maximum_runtime_s']:raise RuntimeError('reference exceeds air runtime budget')
  report.update(mode='new_mount_posture_references',completed_references=[])
 else:
  offsets=[i for i,row in enumerate(P['audit']['samples']) if row['sample']==0]+[len(P['audit']['samples'])]
  if len(offsets)!=len(P['points']):raise RuntimeError('dense route segment association mismatch')
  jobs=[]
  end_index=min(len(P['points'])-1,START+COUNT)
  for i in range(START,end_index,GROUP):
   end=min(i+GROUP,end_index)
   points=json.loads(json.dumps(P['points'][i:end+1]))
   rows=P['audit']['samples'][offsets[i]:offsets[end]]
   if sum(row['sample']==0 for row in rows)!=end-i or not np.allclose(rows[0]['pose_m_rad'],points[0]['pose_m_rad'],atol=1e-8):raise RuntimeError('air segment audit association mismatch')
   job={'reference_id':('alignment-%03d'%i if GROUP==1 else 'alignment-%03d-%03d'%(i,end-1)),'points':points,'audit':{'passed_numeric':True,'samples':rows},
    'collision':{'models':collision['models'],'supplied_samples':rows},'maximum_runtime_s':120,
    'origin':P['origin'],'normal':P['normal'],'tip':P['tip'],'anchor':P['anchor'],
    'baseline_rotation':points[0]['pose_m_rad'][3:],'watchdog_path':str(WATCHDOG),
    'collect_reference':end==end_index}
   air.validate_air_route(points,job['audit'],job['collision'],job['origin'],job['normal'],job['tip'],job['anchor'],job['baseline_rotation'])
   _,timing=air.compile_air_program(points)
   if timing['total_motion_duration_s']+10>120:raise RuntimeError('segment exceeds existing air runtime budget')
   jobs.append(job)
 reader.close();reader=None
 (out/'request.json').write_text(json.dumps({'candidate':P,'jobs':jobs},indent=2)+'\n')
 report.update(status='alignment_preflight_passed_no_motion',segments=len(jobs))
 print(json.dumps(report,ensure_ascii=False),flush=True)
 if EXECUTE:
  for job in jobs:
   reader=live.ContinuousState(support,'192.168.1.3');_,pose,_=reader.stationary()
   expected=np.array(job['points'][0]['pose_m_rad'])
   if np.linalg.norm(pose[:3]-expected[:3])>.0002 or (Rotation.from_rotvec(pose[3:]).inv()*Rotation.from_rotvec(expected[3:])).magnitude()>np.deg2rad(.05):raise RuntimeError('inter-segment start changed')
   force=support.RawForce()
   with (out/(job['reference_id']+'-start-force.jsonl')).open('x') as stream:
    with force.lock:force.rows=recording.RecordingRows(force.rows,stream,job['reference_id']+'-start')
    baseline,stats=force.baseline()
    force.close();force=None
   _,after,_=reader.stationary();reader.close();reader=None
   if np.linalg.norm(after[:3]-pose[:3])>.0001 or (Rotation.from_rotvec(pose[3:]).inv()*Rotation.from_rotvec(after[3:])).magnitude()>np.deg2rad(.05):raise RuntimeError('moved during start-force observation')
   for point in job['points']:point['unloaded_wrench_n_nm']=baseline.tolist()
   (out/(job['reference_id']+'-start.json')).write_text(json.dumps({'actual_pose_m_rad':pose.tolist(),'baseline_n_nm':baseline.tolist(),'statistics':stats,'force_tare_performed':False},indent=2)+'\n')
   print(json.dumps({'event':'air_alignment_segment_start','reference_id':job['reference_id']},ensure_ascii=False),flush=True)
   try:result=air.execute_air_segment(job,out/job['reference_id'],support,capture,checked_session=checked_session)
   finally:
    path=out/job['reference_id']/'execution.json'
    if path.exists():report['motion_sent']=report['motion_sent'] or json.loads(path.read_text()).get('motion_sent',False)
   if P.get('jobs'):
    report['completed_references'].append({'reference_id':job['reference_id'],'role':job['role'],'angles_deg':job['angles_deg'],'reference':result['static_reference']})
   report['completed_segments'].append({'reference_id':job['reference_id'],'reference':result['static_reference'],'final_pose_m_rad':result['final_pose_m_rad']})
   print(json.dumps({'event':'air_alignment_segment_completed','reference_id':job['reference_id'],'completed':len(report['completed_segments'])}),flush=True)
  report['status']='air_references_collected_pending_independent_validation' if P.get('jobs') else 'selected_air_alignment_segments_completed'
except BaseException as exc:report.update(status='stopped_on_error',error=type(exc).__name__+': '+str(exc),error_traceback=traceback.format_exc())
finally:
 if force is not None:force.close()
 if reader is not None:reader.close()
 report['finished_at']=time.time()
 if P.get('jobs'):(out/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 (out/'RESULT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k not in ('completed_segments','mount_model','error_traceback')},ensure_ascii=False),flush=True)
if report['status']=='stopped_on_error':raise SystemExit(1)
'''


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,default=DEFAULT)
 mode=p.add_mutually_exclusive_group();mode.add_argument('--preflight',action='store_true');mode.add_argument('--execute',action='store_true')
 p.add_argument('--unloaded-confirmed',action='store_true');p.add_argument('--start-index',type=int,default=0);p.add_argument('--max-segments',type=int,default=1)
 p.add_argument('--group-legs',type=int,choices=[1,2,3,4],default=1)
 p.add_argument('--time-margin',type=float,default=2.5)
 a=p.parse_args();request=json.loads(a.candidate.read_text())
 if request.get('jobs'):request['points']=[point for job in request['jobs'] for point in job['points']]
 if not 0<=a.start_index<len(request['points'])-1 or not 1<=a.max_segments<=len(request['points'])-1:p.error('invalid segment range')
 if a.execute and not a.unloaded_confirmed:p.error('--execute requires --unloaded-confirmed')
 source=air_source_at_height((ROOT/'tools/brush_unloaded_calibration.py').read_text(),request['tip_height_m'],a.time_margin)
 if not(a.execute or a.preflight):print(json.dumps({'mode':'preview_only','motion_sent':False,'total_alignment_segments':len(request['points'])-1,'selected_segments':a.max_segments,'air_height_m':request['tip_height_m']}));return
 from sys import path
 path.insert(0,str(ROOT/'ros2_sim_ws/src/fusion_model_ros2_beta'))
 from fusion_model_ros2_beta.ur10_actual_kinematics import CAD_ATTACHMENT_ENVELOPES,CAD_ASSEMBLY_RPY,transform
 boxes={n:{'centre_m':list(c),'size_m':(2*np.array(h)).tolist(),'rotation':transform(rpy=CAD_ASSEMBLY_RPY)[:3,:3].tolist()} for n,c,h in CAD_ATTACHMENT_ENVELOPES}
 recording=next(n for n in ast.parse((ROOT/'tools/brush_write_capture.py').read_text()).body if isinstance(n,ast.ClassDef) and n.name=='RecordingRows')
 sid='gong-air-alignment-'+('execute-' if a.execute else 'preflight-')+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
 out=ROOT/'outputs/brush_mapping_20261027'/sid;out.mkdir(parents=True,exist_ok=False);remote='/home/jx/.dsh-analysis/'+sid
 values={'P':request,'BOXES':boxes,'START':a.start_index,'COUNT':a.max_segments,'GROUP':a.group_legs,'EXECUTE':a.execute,'REMOTE_OUTPUT':remote,
  'AIR_SOURCE':source,'FORCE_SOURCE':(ROOT/'tools/brush_force_snapshot.py').read_text(),
  'COLLISION_SOURCE':(ROOT/'tools/brush_current_mount_collision.py').read_text(),
  'RECORDING_SOURCE':'import json\nfrom collections import deque\n'+ast.unparse(recording),
  'PRODUCTION':{n:(ROOT/'tools'/f'{n}.py').read_text() for n in ['brush_buffered_motion','brush_buffered_guard','brush_live_state']}}
 code='\n'.join(k+'='+repr(v) for k,v in values.items())+'\n'+REMOTE
 (out/'request.py').write_text(code)
 command='source /opt/ros/humble/setup.bash; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1; exec /usr/bin/python3 -B -'
 with (out/'stdout.txt').open('x') as log,(out/'stderr.txt').open('x') as err:
  process=subprocess.Popen(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','jx','bash -c '+shlex.quote(command)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,text=True)
  process.stdin.write(code);process.stdin.close()
  for line in process.stdout:log.write(line);log.flush();print(line,end='',flush=True)
  status=process.wait()
 cp=subprocess.run(['scp','-q','-r','jx:'+remote,str(out/'remote_evidence')],capture_output=True,text=True,timeout=60)
 (out/'download_stderr.txt').write_text(cp.stderr)
 print(json.dumps({'local_evidence':str(out),'ssh_exit_code':status,'raw_evidence_downloaded':cp.returncode==0}))
 if status or cp.returncode:raise SystemExit(1)


if __name__=='__main__':main()
