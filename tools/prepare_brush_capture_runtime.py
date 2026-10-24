#!/usr/bin/env python3
"""Prepare guarded previews or collect ONE stationary unloaded reference.

No motion entry point. Preparation is offline; --capture-reference only reads
state and raw wrench through the existing jx collectors. Native movel points,
blending, watchdog and force thresholds stay in the existing compiler/guard.
"""
import argparse
import datetime as dt
import json
import math
from pathlib import Path
import shlex
import subprocess

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from brush_buffered_motion import compile_program, execution_blockers


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def pose_matches(a, b, position=.0002, angle=math.radians(.05)):
    a, b = np.asarray(a), np.asarray(b)
    return (np.linalg.norm(a[:3]-b[:3]) <= position and
            (Rotation.from_rotvec(a[3:]).inv()*Rotation.from_rotvec(b[3:])).magnitude() <= angle)


def prepare(sample_file, candidate_file, collision_file, output, guard_host):
    sample, candidate, collision = map(read, (sample_file, candidate_file, collision_file))
    rows = candidate['audit']['samples']
    if not candidate['audit']['passed_numeric']:
        raise ValueError('numeric path did not pass')
    if collision['supplied_samples'] != rows:
        raise ValueError('collision report belongs to a different complete path')
    required_models = {'robot_attachments.yaml', 'robot_attachments.preliminary-cad-20260921.yaml'}
    if set(collision['models']) != required_models or not all(m['passed_20mm'] for m in collision['models'].values()):
        raise ValueError('self collision did not pass both models')
    for key in ('planned_write', 'planned_transfer'):
        if sample[key] != candidate[key]:
            raise ValueError('sample and checked geometry differ: '+key)
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    jobs = []; associations = {'write': [], 'transfer': []}

    def add_reference(pose, role, purpose, stage=None, waypoint=None, offset=None):
        if role == 'train':
            for job in jobs:
                if job['role'] == role and pose_matches(pose, job['target_tool0_pose_m_rad'], 1e-9, 1e-9):
                    return job['reference_id']
        identifier = f'force-{role}-{sum(j["role"] == role for j in jobs):03d}'
        jobs.append({'reference_id': identifier, 'trial_sample_id': sample['sample_id'],
                     'role': role, 'purpose': purpose, 'stage': stage, 'waypoint_index': waypoint,
                     'target_tool0_pose_m_rad': list(pose), 'motion_sent': False,
                     'nominal_offset_from_command_m': offset,
                     'requires_operator_unloaded_confirmation': True,
                     'reference_status': 'not_collected'})
        return identifier

    normal = np.array(sample['paper_frame']['paper_normal_base'])
    # First target is the already checked 40 mm fixed-orientation release.
    first_id = add_reference(sample['planned_write'][0]['pose_m_rad'], 'train',
                             'write-start-air; first manual fixed-orientation lift')
    for stage in associations:
        for index, point in enumerate(sample['planned_'+stage]):
            target = np.array(point['pose_m_rad'])
            offset = np.zeros(3)
            if point['contact']:
                # Observe out of contact; never label a contact wrench unloaded.
                # This spatial substitution needs validation before guard use.
                offset = normal*(.005+sample['commanded_depth_mm']/1000)
                target[:3] += offset
            identifier = add_reference(target.tolist(), 'train',
                'waypoint unloaded observation; contact points observed 5 mm above local zero'
                if point['contact'] else 'exact free-space waypoint unloaded observation',
                stage, index, offset.tolist())
            associations[stage].append(identifier)
    # Start with independent midpoint holdouts for each existing leg. Static
    # observations alone do not certify dynamic protection or all orientations.
    # They never add or alter compiled movel waypoints or become training rows.
    for leg, (a, b) in enumerate(zip(sample['planned_transfer'], sample['planned_transfer'][1:])):
        poses = np.array([a['pose_m_rad'],b['pose_m_rad']])
        middle = poses.mean(axis=0)
        middle[3:] = Slerp([0.,1.], Rotation.from_rotvec(poses[:,3:]))([.5]).as_rotvec()[0]
        add_reference(middle.tolist(), 'holdout', 'initial independent check of existing guard reference interpolation',
                      'transfer', leg)
    midpoint = np.array(sample['planned_write'][1]['pose_m_rad'])
    midpoint[:3] = (np.array(sample['planned_write'][1]['pose_m_rad'][:3])+
                     np.array(sample['planned_write'][2]['pose_m_rad'][:3]))/2
    midpoint[:3] += normal*(.005+sample['commanded_depth_mm']/1000)
    add_reference(midpoint.tolist(), 'holdout', 'unloaded midpoint above the short line', 'write', 1)
    plan = {'schema': 'brush_route_unloaded_reference_plan_v1', 'sample_id': sample['sample_id'],
            'first_reference_id': first_id, 'motion_sent': False, 'camera_triggered': False,
            'automatic_pose_advance': False, 'jobs': jobs, 'waypoint_reference_ids': associations,
            'source_files': sample['source_files'],
            'guard_reference_interpolation': 'unchanged brush_buffered_guard.reference_at_pose',
            'validation_state': 'not_collected; static holdouts and air-motion validation pending',
            'scope': 'static references do not alone validate dynamic contact protection'}
    save(out/'reference_plan.json', plan)
    save(out/'manual_initial_lift.json', {
        'sample_id': sample['sample_id'], 'points': sample['planned_initial_lift'],
        'first_target_reference_id': first_id, 'motion_sent': False,
        'execution_ready': False, 'scope': 'manual release only; no program generated',
        'reason': 'existing buffered executor requires a truly free-space first waypoint; current zero is in contact'})
    stages = {}
    for stage in ('write', 'transfer'):
        folder = out/stage; folder.mkdir()
        points = json.loads(json.dumps(sample['planned_'+stage]))
        for point, identifier in zip(points, associations[stage]):
            matches = [r for r in rows if pose_matches(point['pose_m_rad'], r['pose_m_rad'], 1e-8, 1e-8)]
            if not matches:
                raise ValueError('waypoint has no checked IK endpoint')
            point.update(q_rad=matches[-1]['q_rad'], unloaded_wrench_n_nm=None,
                         pending_unloaded_reference_id=identifier)
        length = sum(np.linalg.norm(np.array(b['pose_m_rad'][:3])-a['pose_m_rad'][:3])/b['speed_m_s']
                     for a, b in zip(points, points[1:]))
        bundle = {'sample_id': sample['sample_id'], 'scope': stage+' template; NOT executable',
                  'points': points, 'start_pose_m_rad': points[0]['pose_m_rad'],
                  'active_tcp_offset_m_rad': [0.]*6,
                  'maximum_runtime_s': min(1800., max(60., length*1.5+30.)),
                  'commanded_press_mm': sample['commanded_depth_mm'],
                  'physical_mapping_verified': False, 'force_reference_verified': False,
                  'controller_watchdog_hardware_validated': False, 'evidence': {},
                  'model_H_mm': None, 'robot_execution_ready': False}
        script, manifest = compile_program(points, guard_host, 50071)
        manifest['execution_blockers'] = execution_blockers(bundle, manifest, folder)
        save(folder/'bundle.TEMPLATE.json', bundle); save(folder/'manifest.PREVIEW.json', manifest)
        with (folder/'program.script.preview').open('x') as f: f.write(script)
        stages[stage] = {'waypoints': len(points), 'native_movel_count': manifest['native_movel_count'],
                         'execution_blockers': manifest['execution_blockers']}
    summary = {'sample_id': sample['sample_id'], 'guard_host': guard_host, 'first_reference_id': first_id,
               'training_jobs': sum(j['role']=='train' for j in jobs),
               'holdout_jobs': sum(j['role']=='holdout' for j in jobs), 'stages': stages,
               'motion_sent': False, 'camera_triggered': False, 'robot_execution_ready': False}
    save(out/'RESULT.json', summary)
    return summary


REMOTE = '''import sys,types,json,time,threading,datetime as dt
import numpy as np
from scipy.spatial.transform import Rotation
sys.dont_write_bytecode=True
sys.path[:0]=['/home/jx/.dsh-analysis','/home/jx/UR10/scripts']
module=types.ModuleType('brush_live_state');exec(compile(LIVE_SOURCE,'brush_live_state.py','exec'),module.__dict__)
import probe_paper_height as support
from calibrate_pen_tcp import capture
request=REQUEST
job=request['job'];target=np.array(job['target_tool0_pose_m_rad'])
report={'sample_id':request['sample_id'],'reference_id':job['reference_id'],'role':job['role'],
 'recorded_at':dt.datetime.now(dt.timezone.utc).isoformat(),'status':'started','motion_sent':False,
 'force_tare_performed':False,'camera_triggered':False,'operator_unloaded_confirmed':True,
 'planned_reference':job,'source_files':request['source_files']}
force=reader=thread=None;done=threading.Event();errors=[];states=[]
try:
 report['controller_before']=support.controller_ready('192.168.1.3')
 initial=capture('192.168.1.3',30011,1.);report['pose_before']=initial
 p=np.array(initial['flange_pose_m_rad'])
 if not np.allclose(initial['active_tcp_offset_m_rad'],0,atol=1e-8,rtol=0):raise RuntimeError('expected unchanged zero active TCP offset')
 if np.linalg.norm(p[:3]-target[:3])>.0002 or (Rotation.from_rotvec(p[3:]).inv()*Rotation.from_rotvec(target[3:])).magnitude()>np.deg2rad(.05):raise RuntimeError('current pose does not match requested reference; no movement performed')
 if initial['max_motion_mm']>.1 or initial['max_motion_deg']>.05:raise RuntimeError('initial pose not stationary')
 reader=module.ContinuousState(support,'192.168.1.3');reader.stationary()
 def watch():
  try:
   while not done.is_set():
    q,p,v=reader.fresh(timeout=.15)
    states.append({'host_unix_ns':time.time_ns(),'actual_q_rad':q.tolist(),'actual_tool0_pose_m_rad':p.tolist(),'tcp_velocity':v.tolist()})
    angle=(Rotation.from_rotvec(target[3:]).inv()*Rotation.from_rotvec(p[3:])).magnitude()
    if np.linalg.norm(p[:3]-target[:3])>.0002 or angle>np.deg2rad(.05):raise RuntimeError('departed static reference pose')
  except BaseException as exc:errors.append(str(exc))
 thread=threading.Thread(target=watch,daemon=True);thread.start()
 force=support.RawForce();median,stats=force.baseline()
 report['pose_sample']=capture('192.168.1.3',30011,1.)
 force.recent();done.set();thread.join(timeout=1.)
 if errors:raise RuntimeError('; '.join(errors))
 final=report['pose_sample']
 if final['max_motion_mm']>.1 or final['max_motion_deg']>.05:raise RuntimeError('final pose not stationary')
 if not np.allclose(final['active_tcp_offset_m_rad'],initial['active_tcp_offset_m_rad'],atol=1e-8,rtol=0):raise RuntimeError('active TCP changed')
 report['raw_wrench_summary']={**stats,'median_n_nm':median.tolist()}
 report['controller_after']=support.controller_ready('192.168.1.3');report['status']='recorded'
except BaseException as exc:report.update(status='failed',error=type(exc).__name__+': '+str(exc))
finally:
 done.set()
 if thread is not None:thread.join(timeout=1.)
 report['controller_feedback']=states;report['monitor_errors']=errors
 if force is not None:
  with force.lock:report['raw_wrench_rows']=list(force.rows)
  force.close()
 if reader is not None:reader.close()
 report['finished_at']=dt.datetime.now(dt.timezone.utc).isoformat()
print(json.dumps(report,ensure_ascii=False,allow_nan=False))
'''


def collect(plan_file, identifier, output, unloaded_confirmed, runner=subprocess.run):
    if not unloaded_confirmed:
        raise ValueError('operator must confirm current pose is fully unloaded; no movement to qualify')
    plan = read(plan_file)
    matches = [j for j in plan['jobs'] if j['reference_id']==identifier]
    if len(matches)!=1:
        raise ValueError('unknown/ambiguous reference id')
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    request = {'sample_id': plan['sample_id'], 'job': matches[0], 'source_files': plan['source_files']}
    code = REMOTE.replace('LIVE_SOURCE', repr(Path(__file__).with_name('brush_live_state.py').read_text())).replace('REQUEST', repr(request))
    with (out/'collector_source.py').open('x') as f: f.write(code)
    command = ('source /opt/ros/humble/setup.bash; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1; '
               'if command -v rtk >/dev/null 2>&1; then exec rtk proxy /usr/bin/python3 -B -; else exec /usr/bin/python3 -B -; fi')
    result = runner(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','jx','bash -c '+shlex.quote(command)],
                    input=code, text=True, capture_output=True, timeout=55)
    (out/'stdout.txt').write_text(result.stdout); (out/'stderr.txt').write_text(result.stderr)
    lines = [line for line in result.stdout.splitlines() if line.startswith('{')]
    if not lines:
        raise RuntimeError('no reference returned; see preserved stdout/stderr')
    record = json.loads(lines[-1]); save(out/'reference.json', record)
    return {'reference_id': identifier, 'status': record['status'], 'error': record.get('error'),
            'evidence': str(out/'reference.json'), 'motion_sent': False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sample',type=Path); p.add_argument('--candidate',type=Path)
    p.add_argument('--collision',type=Path); p.add_argument('--guard-host')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--plan',type=Path); p.add_argument('--capture-reference')
    p.add_argument('--unloaded-confirmed',action='store_true')
    a=p.parse_args()
    if a.capture_reference:
        if not a.plan or not a.unloaded_confirmed: p.error('capture requires --plan and --unloaded-confirmed')
        result=collect(a.plan,a.capture_reference,a.output,True)
    else:
        if not all((a.sample,a.candidate,a.collision,a.guard_host)): p.error('preparation requires --sample, --candidate, --collision, --guard-host')
        result=prepare(a.sample,a.candidate,a.collision,a.output,a.guard_host)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
