#!/usr/bin/env python3
"""One complete model gong trial. Default offline preview; motion requires --execute.

An explicit provisional conversion is accepted only for this initial trial.
The original buffered compiler, native transitions, force stop and watchdog are reused.
"""
import argparse,ast,csv,datetime as dt,json,shlex,subprocess,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
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
        depth=row[2]*cfg['H_command_mm_per_model_mm']+cfg.get('H_command_offset_mm',0.);uv=row[:2]*cfg['xy_scale']+shift
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

def _checked_samples(candidate):
    """Route audit samples followed by the native blend-neighbourhood samples."""
    rows = list(candidate['audit']['samples'])
    rows += list(candidate.get('native_blend_neighborhood', {}).get('samples', []))
    return rows


def _bind_samples(report, candidate, label):
    """Bind collision evidence to exactly the route plus blend samples checked here."""
    expected = [(r.get('phase'), r.get('sample', r.get('source_waypoint_index')))
                for r in _checked_samples(candidate)]
    supplied = report.get('supplied_samples')
    if not isinstance(supplied, list) or len(supplied) != len(expected):
        raise ValueError('%s collision report does not cover the complete checked route' % label)
    actual = [(r.get('phase'), r.get('sample', r.get('source_waypoint_index'))) for r in supplied]
    if actual != expected:
        raise ValueError('%s collision report samples do not match this route' % label)
    return len(expected)


def _attachment_evidence(candidate, folder, total):
    """Existing attachment/fixture report; the 20 mm rule here is unchanged."""
    report = json.loads((Path(folder)/'self_collision.json').read_text())
    models = {'robot_attachments.yaml', 'robot_attachments.preliminary-cad-20260921.yaml'}
    if set(report['models']) != models:
        raise ValueError('attachment collision report must contain both existing models')
    if _bind_samples(report, candidate, 'attachment') != total:
        raise ValueError('attachment collision report does not cover the complete checked route')
    if not all(m['passed_20mm'] and m['minimum_clearance_mm'] >= 20 for m in report['models'].values()):
        raise ValueError('attachment collision clearance below 20 mm')
    return report


def _body_evidence(candidate, folder, total, body_margin, accept):
    """Arm-body pairs from the installed UR SRDF; a lower margin needs a recorded decision."""
    path = Path(folder)/'full_self_collision.json'
    report = json.loads(path.read_text())
    models = {'robot_attachments.yaml', 'robot_attachments.preliminary-cad-20260921.yaml'}
    if set(report['models']) != models:
        raise ValueError('body collision report must contain both existing models')
    if _bind_samples(report, candidate, 'body') != total:
        raise ValueError('body collision report does not cover the complete checked route')
    rows = {}
    for name, model in report['models'].items():
        worst = min(model['samples'], key=lambda r: r['minimum_clearance_mm'])
        if not any(x.endswith('_0') for x in worst['pair'].split(' vs ')):
            raise ValueError('reported worst body pair is not an arm-body pair: '+worst['pair'])
        rows[name] = {'minimum_clearance_mm': worst['minimum_clearance_mm'], 'pair': worst['pair'],
                      'passed_20mm': model['passed_20mm']}
    minimum = min(r['minimum_clearance_mm'] for r in rows.values())
    if minimum < 20:
        if not (isinstance(body_margin, float) and 10 <= body_margin <= minimum):
            raise ValueError('arm-body clearance %.2f mm is below 20 mm; pass '
                             '--body-clearance-mm within [10,%.2f] to accept a measured, smaller margin'
                             % (minimum, minimum))
        if not accept:
            raise ValueError('arm-body clearance %.2f mm below 20 mm needs '
                             '--accept-body-clearance with the operator reason' % minimum)
    return {'models': rows, 'minimum_clearance_mm': minimum, 'margin_mm': body_margin,
            'below_default_20mm': bool(minimum < 20), 'accepted_by_operator': bool(minimum < 20),
            'operator_reason': accept, 'source': str(path.resolve())}


def _reference_acceptance(candidate, folder, acceptance_path, accept_extrapolation):
    """Local air reference: first-order on three measured postures, no invented coverage."""
    import hashlib
    reference = json.loads((Path(folder)/acceptance_path).read_text())
    kind = reference.get('model_kind')
    if kind not in ('first_order_linear_relative_rotvec', 'affine_relative_rotvec'):
        raise ValueError('unsupported reference model kind for this candidate: '+str(kind))
    if not reference['passed'] or reference['reference_error_limit_n'] != 10.:
        raise ValueError('explicit 10 N local reference acceptance did not pass')
    if reference['moving_max_force_error_n'] > 10:
        raise ValueError('moving reference force error exceeds the 10 N acceptance limit')
    if reference['trial_candidate_sha256'] != hashlib.sha256(
            (Path(folder)/'candidate.PREVIEW_ONLY.json').read_bytes()).hexdigest():
        raise ValueError('reference acceptance does not bind this candidate file')
    if reference['static_reference_count'] < 3:
        raise ValueError('at least three measured air postures are required')
    from gong_reference_local import coverage
    span = coverage(reference['local_fit'], [p['pose_m_rad'][3:] for p in candidate['points']])
    if span['extrapolation_bound_n'] > 10:
        raise ValueError('recomputed extrapolation bound %.2f N exceeds 10 N'
                         % span['extrapolation_bound_n'])
    declared = reference.get('declared_extrapolation')
    if not span['complete']:
        if not declared or declared.get('accepted') is not True or not declared.get('reason'):
            raise ValueError('writing postures lie outside the measured air span and no recorded '
                             'operator acceptance exists')
        if not accept_extrapolation:
            raise ValueError('writing postures lie %.2f deg outside the measured air span; pass '
                             '--accept-reference-extrapolation with the operator reason'
                             % span['max_distance_deg'])
    features = np.array(reference['local_fit']['reference_features'])
    domain = {'kind': 'box', 'lo': features.min(0).tolist(), 'hi': features.max(0).tolist()}
    return reference, span, domain


def load_checked_trial(folder=DEFAULT, body_margin=20., accept_body_clearance=None,
                       accept_reference_extrapolation=None, max_depth=1., accept_deep_press=None):
    folder = Path(folder)
    candidate = json.loads((folder/'candidate.PREVIEW_ONLY.json').read_text())
    audit_count = len(candidate['audit']['samples'])
    blend_count = len(candidate.get('native_blend_neighborhood', {}).get('samples', []))
    total = audit_count+blend_count
    if not candidate['audit']['passed_numeric']:
        raise ValueError('numeric route audit did not pass')
    attachment = _attachment_evidence(candidate, folder, total)
    body = _body_evidence(candidate, folder, total, body_margin, accept_body_clearance)
    controls = list(csv.DictReader(Path(candidate['source_controls']).open()))
    if candidate['original_controls'] != controls or candidate['control_counts'] != [6, 4, 6]:
        raise ValueError('original model controls changed')
    check_conversion(candidate)
    reference, span, domain = _reference_acceptance(
        candidate, folder, 'unloaded_reference_acceptance_10N.json', accept_reference_extrapolation)
    depth = np.array([p['commanded_depth_mm'] for p in candidate['points'] if p['contact']])
    if not np.isfinite(depth).all() or min(depth) <= 0:
        raise ValueError('commanded press depth must be finite and positive')
    if max(depth) > max_depth:
        raise ValueError('commanded depth %.2f mm exceeds the accepted %.2f mm envelope'
                         % (max(depth), max_depth))
    if max(depth) > 1. and not (isinstance(max_depth, float) and max_depth <= 6. and accept_deep_press):
        raise ValueError('commanded depth %.2f mm is beyond the historical 1 mm trial envelope; pass '
                         '--max-commanded-depth-mm within (1,6] and --accept-deeper-press with the '
                         'operator reason' % max(depth))
    original = {name: (ROOT/'tools'/f'{name}.py').read_text() for name in
                ('brush_buffered_motion', 'brush_buffered_guard', 'brush_live_state', 'brush_trial_force')}
    return {'candidate': candidate, 'collision': attachment, 'body_collision': body,
            'reference_acceptance': reference, 'reference_extrapolation': span,
            'reference_orientation_domain': domain, 'original_software': original,
            'operator_policy': {'initial_model_conversion_permitted': True,
                'reference_error_limit_n': 10.,
                'zero_selection': 'operator explicitly selected the previous recorded touch zero; timestamp retained',
                'reference_translation_validated': False,
                'reference_scope': 'three measured small-angle air postures at this installation; '
                                   'writing postures are a declared first-order extrapolation',
                'reference_coverage_complete': span['complete'],
                'body_clearance_mm': body['minimum_clearance_mm'],
                'body_clearance_accepted_by_operator': body['below_default_20mm'],
                'motion_requires_execute_and_explicit_acceptance': True}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',type=Path,default=DEFAULT)
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--preflight',action='store_true');mode.add_argument('--execute',action='store_true')
    parser.add_argument('--accept-initial-trial',action='store_true',help='authorize this single provisional trial using the recorded zero and local air reference')
    parser.add_argument('--reuse-preflight',type=Path,help='reuse an exact, already passed writing-path audit; the current start connection is still checked')
    parser.add_argument('--body-clearance-mm',type=float,default=20.,help='required arm-body mesh clearance; below the default 20 mm only with --accept-body-clearance')
    parser.add_argument('--accept-body-clearance',help='operator reason recorded when the measured arm-body clearance is accepted below 20 mm')
    parser.add_argument('--accept-reference-extrapolation',help='operator reason recorded when the writing postures lie outside the measured air postures')
    parser.add_argument('--maximum-runtime-s',type=float,default=1800.,help='finite trial runtime ceiling; the prepared connector adds pen-up travel')
    parser.add_argument('--max-commanded-depth-mm',type=float,default=1.,help='accepted commanded press depth envelope; above 1 mm needs --accept-deeper-press')
    parser.add_argument('--accept-deeper-press',help='operator reason recorded when the commanded depth exceeds the historical 1 mm trial envelope')
    args=parser.parse_args()
    if args.body_clearance_mm<10 or args.body_clearance_mm>20:parser.error('--body-clearance-mm must stay within [10,20] mm')
    if args.maximum_runtime_s<300 or args.maximum_runtime_s>3600:parser.error('--maximum-runtime-s must stay within [300,3600] s')
    if args.execute and not args.accept_initial_trial:parser.error('--execute requires --accept-initial-trial')
    if args.preflight or args.execute:
        mount_path=ROOT/'integration/ur10_camera_direct_mount_20261027/assembly.json'
        if mount_path.exists():
            candidate=json.loads((args.folder/'candidate.PREVIEW_ONLY.json').read_text())
            current_mount=json.loads(mount_path.read_text())
            if any(candidate.get('mount_model',{}).get(k)!=current_mount.get(k) for k in
                   ('housing_size_cad_xyz_mm','housing_centre_cad_mm','orientation_source')):
                parser.error('camera installation changed; historical route/force evidence cannot be reused; '
                             'prepare the current-mount candidate with tools/prepare_gong_model_trial.py first')
    if not 1.<=args.max_commanded_depth_mm<=6.:parser.error('--max-commanded-depth-mm must stay within [1,6] mm')
    request=load_checked_trial(args.folder,args.body_clearance_mm,args.accept_body_clearance,
                               args.accept_reference_extrapolation,args.max_commanded_depth_mm,
                               args.accept_deeper_press)
    request['maximum_runtime_s']=args.maximum_runtime_s
    if args.reuse_preflight:request['cached_preflight']=load_cached_preflight(args.reuse_preflight,request)
    if not(args.preflight or args.execute):
        print(json.dumps({'status':'offline_preview_passed','motion_sent':False,'controls':request['candidate']['control_counts'],
            'waypoints':len(request['candidate']['points']),'conversion':request['candidate']['conversion'],
            'zero':request['candidate']['zero'],'reference_max_error_n':request['reference_acceptance']['moving_max_force_error_n'],
            'reference_model':request['reference_acceptance']['model_kind'],
            'reference_postures':request['reference_acceptance']['static_reference_count'],
            'reference_coverage_complete':request['reference_extrapolation']['complete'],
            'reference_extrapolation_deg':request['reference_extrapolation']['max_distance_deg'],
            'reference_extrapolation_bound_n':request['reference_extrapolation']['extrapolation_bound_n'],
            'attachment_clearance_mm':min(m['minimum_clearance_mm'] for m in request['collision']['models'].values()),
            'body_clearance_mm':request['body_collision']['minimum_clearance_mm'],
            'body_clearance_below_20mm':request['body_collision']['below_default_20mm'],
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
