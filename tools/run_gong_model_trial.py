#!/usr/bin/env python3
"""One complete model gong trial. Default offline preview; motion requires --execute.

An explicit provisional conversion is accepted only for this initial trial.
The original buffered compiler, native transitions, force stop and watchdog are reused.
"""
import argparse,ast,csv,datetime as dt,json,shlex,subprocess,sys
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from validate_brush_unloaded_sweep import features
from brush_buffered_motion import compile_program,execution_blockers

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'outputs/brush_mapping_20261023/gong_initial_full_model_trial'

def load_cached_preflight(folder,request):
    folder=Path(folder)
    bundle=json.loads((folder/'bundle.json').read_text());old=json.loads((folder/'request.json').read_text())
    _,manifest=compile_program(bundle['points'],'192.168.1.10',50071)
    errors=execution_blockers(bundle,manifest,folder)
    if errors:raise ValueError('cached preflight evidence invalid: '+'; '.join(errors))
    if old['candidate']!=request['candidate'] or old['reference_acceptance']!=request['reference_acceptance']:
        raise ValueError('cached path/conversion/zero/reference differs from current trial')
    if old['original_software']['brush_buffered_motion']!=request['original_software']['brush_buffered_motion']:
        raise ValueError('native compiler changed since cached path audit')
    source=request['candidate']['points'];tail=bundle['points'][-len(source):]
    for actual,expected in zip(tail,source):
        if any(actual[k]!=v for k,v in expected.items() if k not in ('q_rad','unloaded_wrench_n_nm')):
            raise ValueError('cached writing route differs from model proposal')
    return {'source_directory':str(folder.resolve()),'tail_points':tail,
        'original_route_sha256':manifest['route_sha256'],'original_program_sha256':manifest['program_sha256'],
        'source_audit_evidence':bundle['evidence']['blended_path_audit']}

def check_conversion(candidate):
    """Verify every contact command and all sixteen original model endpoints."""
    cfg=candidate['conversion'];paper=candidate['paper'];tip=np.array(candidate['tcp']['translation_m'])
    u,v,n=[np.array(paper[k]) for k in ('paper_u_base','paper_v_base','paper_normal_base')]
    origin=np.array(paper['reconstructed_center_tip_base_m']);shift=np.array(cfg['paper_translation_mm'])/1000
    baseline=Rotation.from_rotvec(cfg['baseline_tool0_rotation_rad']);axis=np.array(cfg['tool_brush_axis'])
    keys=['paper_x_m','paper_y_m','model_H_mm','alpha_rad','beta_rad','gamma_absolute_paper_rad']
    contact=[p for p in candidate['points'] if p['contact']]
    for point in contact:
        row=np.array([point['model_fields'][k] for k in keys]);a,b,g=row[3:]*[cfg['alpha_gain'],cfg['beta_gain'],cfg['gamma_gain']]
        rot=Rotation.from_rotvec(v*b)*Rotation.from_rotvec(u*a)*baseline*Rotation.from_rotvec(axis*g)
        depth=row[2]*cfg['H_command_mm_per_model_mm'];uv=row[:2]*cfg['xy_scale']+shift
        target=origin+uv[0]*u+uv[1]*v-depth/1000*n-rot.apply(tip)
        actual=np.array(point['pose_m_rad'])
        if abs(depth-point['commanded_depth_mm'])>1e-9 or np.linalg.norm(target-actual[:3])>1e-9 or (rot.inv()*Rotation.from_rotvec(actual[3:])).magnitude()>1e-9:
            raise ValueError('physical command differs from the recorded initial model conversion')
    controls=candidate['original_controls']
    if [sum(int(c['stroke_id'])==i for c in controls) for i in range(3)]!=[6,4,6]:
        raise ValueError('all original 6+4+6 controls are required')
    for control in controls:
        source=np.array([float(control[k]) for k in keys])
        if not any(p['stroke_id']==int(control['stroke_id']) and np.allclose([p['model_fields'][k] for k in keys],source,atol=1e-12,rtol=0) for p in contact):
            raise ValueError('an original six-field model control was omitted')

def load_checked_trial(folder=DEFAULT):
    folder=Path(folder)
    candidate=json.loads((folder/'candidate.PREVIEW_ONLY.json').read_text())
    collision=json.loads((folder/'self_collision.json').read_text())
    reference=json.loads((folder/'unloaded_reference_acceptance_10N.json').read_text())
    if not candidate['audit']['passed_numeric'] or collision['supplied_samples']!=candidate['audit']['samples']:
        raise ValueError('numeric/collision reports do not bind this candidate')
    models={'robot_attachments.yaml','robot_attachments.preliminary-cad-20260921.yaml'}
    if set(collision['models'])!=models or not all(x['passed_20mm'] for x in collision['models'].values()):
        raise ValueError('both existing collision models must pass')
    controls=list(csv.DictReader(Path(candidate['source_controls']).open()))
    if candidate['original_controls']!=controls or candidate['control_counts']!=[6,4,6]:
        raise ValueError('original model controls changed')
    check_conversion(candidate)
    if not reference['passed'] or reference['reference_error_limit_n']!=10. or reference['moving_max_force_error_n']>10:
        raise ValueError('explicit 10 N local reference acceptance did not pass')
    run=Path(reference['run']);observations=json.loads((run/'result.json').read_text())['completed_references']
    base=reference['local_fit']['baseline_rotation_rad']
    train=[r['reference']['actual_pose_m_rad'][3:] for r in observations if r['role']=='train']
    domain=ConvexHull(features(train,base)[:,1:4])
    rotations=[p['pose_m_rad'][3:] for p in candidate['points']]
    query=features(rotations,base)[:,1:4]
    if np.max(query@domain.equations[:,:3].T+domain.equations[:,3])>1e-5:
        raise ValueError('trial orientations extrapolate outside measured local air references')
    # H remains a commanded displacement, not a measured brush compression.
    depth=np.array([p['commanded_depth_mm'] for p in candidate['points'] if p['contact']])
    if not np.isfinite(depth).all() or min(depth)<=0 or max(depth)>1.:
        raise ValueError('initial trial must remain in the existing <=1 mm command envelope')
    original={name:(ROOT/'tools'/f'{name}.py').read_text() for name in
              ('brush_buffered_motion','brush_buffered_guard','brush_live_state','brush_trial_force')}
    return {'candidate':candidate,'collision':collision,'reference_acceptance':reference,
            'reference_orientation_domain':domain.equations.tolist(),'original_software':original,
            'operator_policy':{'initial_model_conversion_permitted':True,'reference_error_limit_n':10.,
                'zero_selection':'operator explicitly selected the previous recorded touch zero; timestamp retained',
                'reference_translation_validated':False,
                'reference_scope':'local small-angle air observations; translated writing is an initial trial using this reference',
                'motion_requires_execute_and_explicit_acceptance':True}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',type=Path,default=DEFAULT)
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--preflight',action='store_true');mode.add_argument('--execute',action='store_true')
    parser.add_argument('--accept-initial-trial',action='store_true',help='authorize this single provisional trial using the recorded zero and local air reference')
    parser.add_argument('--reuse-preflight',type=Path,help='reuse an exact, already passed writing-path audit; the current start connection is still checked')
    args=parser.parse_args()
    if args.execute and not args.accept_initial_trial:parser.error('--execute requires --accept-initial-trial')
    request=load_checked_trial(args.folder)
    if args.reuse_preflight:request['cached_preflight']=load_cached_preflight(args.reuse_preflight,request)
    if not(args.preflight or args.execute):
        print(json.dumps({'status':'offline_preview_passed','motion_sent':False,'controls':[6,4,6],
            'waypoints':len(request['candidate']['points']),'conversion':request['candidate']['conversion'],
            'zero':request['candidate']['zero'],'reference_max_error_n':request['reference_acceptance']['moving_max_force_error_n'],
            'reference_translation_validated':False},ensure_ascii=False));return
    sid='gong-model-'+('execute-' if args.execute else 'preflight-')+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=args.folder/'runs'/sid;out.mkdir(parents=True,exist_ok=False)
    remote='/home/jx/.dsh-analysis/'+sid
    recording=next(n for n in ast.parse((ROOT/'tools/brush_write_capture.py').read_text()).body if isinstance(n,ast.ClassDef) and n.name=='RecordingRows')
    source=(ROOT/'tools/gong_model_trial_remote.py').read_text()
    cad=next(n for n in ast.parse((ROOT/'tools/brush_write_capture.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='cad_meshes')
    code='P='+repr(request)+'\nREMOTE_OUTPUT='+repr(remote)+'\nEXECUTE='+repr(args.execute)+'\n'
    code+='RECORDING_SOURCE='+repr('import json\nfrom collections import deque\n'+ast.unparse(recording))+'\n'
    code+='CONNECTOR_SOURCE='+repr((ROOT/'tools/brush_model_start.py').read_text())+'\n'
    code+='CAD_SOURCE='+repr(ast.unparse(cad))+'\n'
    code+='FORCE_SNAPSHOT_SOURCE='+repr((ROOT/'tools/brush_force_snapshot.py').read_text())+'\n'
    code+=source
    (out/'remote_request.py').write_text(code)
    command='source /opt/ros/humble/setup.bash; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1; exec /usr/bin/python3 -B -'
    last=None
    with (out/'stdout.txt').open('x') as log,(out/'stderr.txt').open('x') as err:
        proc=subprocess.Popen(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','jx','bash -c '+shlex.quote(command)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,text=True)
        proc.stdin.write(code);proc.stdin.close()
        for line in proc.stdout:
            log.write(line);log.flush()
            if line.startswith('{'):
                d=json.loads(line);print(json.dumps(d,ensure_ascii=False),flush=True)
                if 'status' in d:last=d
        status=proc.wait()
    # Copies actual raw force and feedback, including failure evidence. Never reruns motion.
    copied=subprocess.run(['scp','-q','-r','jx:'+remote,str(out/'remote_evidence')],capture_output=True,text=True)
    (out/'download_stderr.txt').write_text(copied.stderr)
    result={'local_evidence':str(out),'remote_output':remote,'ssh_exit_code':status,'raw_evidence_downloaded':copied.returncode==0,'result':last}
    (out/'RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False),flush=True)
    if status or last is None or last.get('status') not in ('preflight_passed_no_motion','completed_buffered_program') or copied.returncode:raise SystemExit(1)

if __name__=='__main__':main()
