"""Reproducible A/B/C original-image UR10 wrist validation; no motion publish."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import sys
import traceback
import xml.etree.ElementTree as ET
from dataclasses import asdict,replace
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from .canvas_layout import plan_canvas


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def runtime_source_hashes():
    package=Path(__file__).parent
    names=('ur10_actual_kinematics.py','ur10_wrist_cost.py','brush_trajectory_driver.py',
        'wrist_aware_transitions.py','joint_continuity.py','gamma_semantics.py',
        'precise_ur10_validation.py','original_image_metrics.py',
        'evaluate_actual_wrist_tradeoff.py','joint_path_audit.py','robot_context_guard.py',
        'joint_candidate.py')
    return {name:sha(package/name) for name in names}


def test_evidence(root):
    suites={}
    for name in ('ros','model','ssim'):
        path=root/'tests'/f'{name}.xml'
        if not path.exists():
            suites[name]={'passed':False,'reason':'missing JUnit evidence'};continue
        items=ET.parse(path).getroot().findall('testsuite')
        counts={key:sum(int(s.get(key,0)) for s in items)
            for key in ('tests','failures','errors','skipped')}
        suites[name]={**counts,'sha256':sha(path),
            'passed':counts['tests']>0 and not any(counts[k] for k in ('failures','errors','skipped'))}
    return suites


def setup(args):
    root=Path(args.output).resolve(); root.mkdir(parents=True,exist_ok=True)
    records=json.loads(Path(args.manifest).read_text())['references']
    record=next(r for r in records if r['character']==args.character)
    if record.get('seed_folder'): raise ValueError('paired validation must start from original fixed source, no historical seed')
    rows=list(csv.DictReader(open(record['source_trajectory'],encoding='utf-8-sig')))
    xy=np.array([[float(r['x']),float(r['y'])] for r in rows]); spans=tuple(np.ptp(xy,axis=0))
    layout=plan_canvas([spans]*args.slots,args.width,args.height,'horizontal')
    cell=layout[args.slots//2]
    # Same current ROS canvas function, source aspect ratio and fixed units.
    from . import ur10_actual_kinematics as k
    context=dict(calibration_hash=k.CALIBRATION_HASH,kinematics_sha256=sha(k.__file__),
        canvas_center=[64.,64.],metres_per_pixel=.29/96.,
        paper_offset_xy_m=[cell['center_x_m'],-.65+cell['center_y_m']],paper_z_m=0.,
        brush_length_m=.328,minimum_singularity_margin=.08)
    pinned=dict(character=args.character,manifest_sha256=sha(args.manifest),source_sha256=sha(record['source_trajectory']),
        original_target_sha256=sha(record['original_target_image']),canvas=cell,slots=args.slots,
        source_spans=spans,steps=args.steps,order=args.order,robot_context=context,
        foreground_weight=args.foreground_weight,full_budget=True,historical_A_used=False,
        runtime_scaling=False,motion_published=False,production_promoted=False)
    directory=root/f'canvas_{args.slots}_slots'; directory.mkdir(exist_ok=True)
    config_file=directory/'experiment.json'
    if config_file.exists() and json.loads(config_file.read_text())!=json.loads(json.dumps(pinned)):
        raise ValueError('experiment identity changed; use new directory')
    write(config_file,pinned); write(directory/'robot_context.json',context)
    return directory,record,cell,pinned


def generate(args):
    directory,record,cell,pinned=setup(args)
    from .offline_fontsize_inversion import OfflineInversionConfig
    from .original_target_inversion import OriginalTargetFontSizeGenerator
    from .trajectory_catalog import TrajectoryEntry
    tool,wrist={'A':(0.,0.),'B':(10.,0.),'C':(10.,10.)}[args.case]
    root=Path(args.model_root)
    config=OfflineInversionConfig(root,Path(args.python),root/'outputs/paper_bbsmg_general_v16_pose_dense/bbsmg_best.pt',
        directory/args.case/'cache',timeout_s=args.timeout_s,order=args.order,max_steps=args.steps,
        device='cuda',joint_foreground_weight=args.foreground_weight,joint_hard_neural_domain=True,
        joint_tool_absolute_rotation_weight=tool,joint_actual_wrist_weight=wrist,
        joint_robot_context=str(directory/'robot_context.json'))
    entry=TrajectoryEntry(character=record['character'],sample_id=record['sample_id'],status='ready',
        trajectory_csv=Path(record['source_trajectory']))
    result=OriginalTargetFontSizeGenerator(config,[record]).generate(entry,cell['font_size_m'],progress=print)
    folder=Path(result.output_dir); report=json.loads((folder/'inversion_report.json').read_text())
    data=dict(case=args.case,candidate=str(folder),tool_weight=tool,actual_wrist_weight=wrist,
        initial_pose_sha256=sha(folder/'scaled_initial_pose.csv'),target_sha256=sha(folder/'scaled_target.png'),
        checkpoint_sha256=config.expected_checkpoint_sha256,physical_csv_sha256=sha(folder/'physical_trajectory.csv'),
        metrics=report['metrics'],steps_completed=report['lm']['steps'],
        lm=report['lm'],original_target_sha256=pinned['original_target_sha256'],motion_published=False)
    write(directory/f'{args.case}_generated.json',data)


def plan(args):
    import rclpy
    from .brush_trajectory_driver import BrushTrajectoryDriver,_brush_rotation,_flange_target
    from . import brush_trajectory_driver as driver
    from .precise_ur10_validation import solve_exact
    driver.solve_ur10_ik=solve_exact
    def forbidden_motion(*args,**kwargs):
        raise RuntimeError('offline validation forbids mechanical-arm motion publication')
    driver.BrushTrajectoryDriver._publish_once=forbidden_motion
    from .joint_candidate import load_joint_candidate
    from .ur10_wrist_cost import fk_jacobian,ActualWristCost
    from .stroke_residuals import audit_candidate
    sys.path.insert(0,args.model_root)
    from utils.joint_rotation_metrics import joint_rotation_metrics
    from .original_image_metrics import ssim
    directory,record,cell,pinned=setup(args)
    data=json.loads((directory/f'{args.case}_generated.json').read_text()); folder=Path(data['candidate'])
    if sha(folder/'physical_trajectory.csv')!=data['physical_csv_sha256']: raise ValueError('CSV changed')
    node=None; planned=dict(feasible=False,motion_published=False,
        runtime_source_sha256=runtime_source_hashes(),
        wrist_measurement_basis='factory-calibrated simulated joint path, not physical encoder measurements')
    # Instantiation has no executor spin and never calls _publish_once. No
    # controller goal, publisher method or real hardware interface is invoked.
    rclpy.init(args=['--ros-args','-p','enable_input_page:=false','-p','publish_on_start:=false',
                    '-p','strict_ik:=true','-p','use_exact_calibrated_ik:=true'])
    try:
        entry=load_joint_candidate(folder,require_training_domain=True)
        # Context validation is a planning gate, not a change to the frozen
        # inversion/loader snapshot or to any already-computed GPU result.
        from .robot_context_guard import load_pinned_context
        context=load_pinned_context(folder,json.loads((folder/'offline_inversion.json').read_text()))
        entry=replace(entry,metadata={**entry.metadata,'canvas_layout':cell,
            'actual_wrist_robot_context':context})
        node=BrushTrajectoryDriver(); node._build_targets(entries=[entry],layout_mode='horizontal',font_size_m=cell['font_size_m'])
        def metrics():
            t=node._targets; q=np.asarray([p.joints for p in t])
            m=joint_rotation_metrics(q,[p.point.state for p in t],[p.point.stroke_id for p in t],[p.duration_s for p in t])
            from .brush_trajectory_driver import _joint_singularity_margin,JOINT_LOWER_LIMITS,JOINT_UPPER_LIMITS
            m['minimum_singularity_margin']=min(_joint_singularity_margin(row) for row in q)
            m['minimum_joint_limit_margin_rad']=float(np.minimum(q-JOINT_LOWER_LIMITS,JOINT_UPPER_LIMITS-q).min())
            dq=abs(np.diff(q,axis=0)); states=np.array([v.point.state for v in t]); ids=np.array([v.point.stroke_id for v in t])
            contact=(states[:-1]!=3)&(states[1:]!=3)&(ids[:-1]==ids[1:])
            first_contact=int(np.flatnonzero(states!=3)[0])
            approach=np.arange(len(dq))<first_contact
            for label,mask in [('contact',contact),('penup',(~contact)&(~approach)),
                               ('approach',approach),('penup_or_approach',~contact)]:
                m[f'{label}_q4_q5_q6_deg']=np.rad2deg(dq[mask,3:6].sum(axis=0)).tolist()
            m['max_internal_joint_step_rad']=float(dq[1:].max()) if len(dq)>1 else 0.
            return m
        legacy=metrics(); legacy_targets=list(node._targets)
        original_contacts=[(t.point,t.joints) for t in node._targets if t.point.state!=3]
        from .wrist_aware_transitions import optimize_penup
        optimize_penup(node)
        if original_contacts != [(t.point,t.joints) for t in node._targets if t.point.state!=3]:
            raise ValueError('pen-up optimization changed contact controls')
        node._validate_obstacle_clearance(); node._validate_final_joint_continuity(); node._retime_joint_targets()
        if original_contacts != [(t.point,t.joints) for t in node._targets if t.point.state!=3]:
            raise ValueError('collision repair changed contact controls')
        if node._penup_rotation_audit['per_stroke_home_reset']:
            raise ValueError('optimized pen-up retained a home reset')
        optimized=metrics()
        planned.update(feasible=True,legacy=legacy,penup_optimized=optimized,penup_audit=node._penup_rotation_audit,
            inverse_solver='same exact calibrated continuous IK as inversion feedback',
            motion_publication_guard_enabled=True,
            metrics=optimized if args.case=='C' else legacy,
            actual_plan_selected='penup_optimized' if args.case=='C' else 'legacy',
            checks={k:True for k in ['strict_ik','paper_table_collision','camera_tool_attachment_collision',
                'joint_limits','singularity','continuous_unwrapped_joints','joint_speed','contact_poses_unchanged']})
        planned['checks']['frozen_V16_training_domain']=True
        if args.case != 'C': node._targets=legacy_targets
        from .joint_path_audit import audit_path
        planned['independent_final_path_audit']=audit_path(node._targets,
            speed_limit=float(node._param('max_joint_speed_rad_s')))
        planned['checks']['full_chain_jacobian_singularity']=True
        errors=[]
        for t in node._targets:
            if t.point.state != 3:
                target_pose=_flange_target(t.point,.328); actual,_=fk_jacobian(t.joints)
                from scipy.spatial.transform import Rotation
                errors.append([float(np.linalg.norm(target_pose[:3,3]-actual[:3,3])),
                    float(np.linalg.norm(Rotation.from_matrix(target_pose[:3,:3]@actual[:3,:3].T).as_rotvec()))])
        planned['max_contact_fk_residual_m_rad']=np.asarray(errors).max(axis=0).tolist()
        if planned['max_contact_fk_residual_m_rad'][0]>1e-5 or planned['max_contact_fk_residual_m_rad'][1]>1e-4:
            raise ValueError('contact calibrated FK parity failed')
        if args.case=='C':
            controls=list(csv.DictReader((folder/'physical_trajectory.csv').open(encoding='utf-8-sig')))
            local=list(csv.DictReader((folder/'inversion_trajectory.csv').open(encoding='utf-8-sig')))
            xy=np.asarray([[float(r['x']),float(r['y'])] for r in controls])/[.29/96,-.29/96]+64
            posture=np.asarray([[float(r[k]) for k in ('z','alpha','beta')] for r in controls])
            gamma=np.asarray([float(r['gamma']) for r in local]); ids=np.asarray([int(r['stroke_id']) for r in controls])
            feedback=ActualWristCost(directory/'robot_context.json',10.)
            q_feedback,valid=feedback.solve(xy,posture,gamma,[np.flatnonzero(ids==s) for s in np.unique(ids)])
            if not valid: raise ValueError('feedback IK recovery failed')
            contact_targets=[t for t in node._targets if t.point.state!=3]
            q_planned=[]
            for r in controls:
                desired=np.asarray([float(r['x'])+cell['center_x_m'],float(r['y'])-.65,float(r['z'])/1000.])
                matches=[t for t in contact_targets if t.point.stroke_id==int(r['stroke_id'])
                         and np.linalg.norm(np.array([t.point.x,t.point.y,-t.point.z])-desired)<1e-7]
                if not matches: raise ValueError('source control missing from actual robot plan')
                q_planned.append(min(matches,key=lambda t:np.linalg.norm(np.asarray(t.joints)-q_feedback[len(q_planned)])).joints)
            delta_error=float(np.max(abs(np.diff(q_feedback,axis=0)-np.diff(np.asarray(q_planned),axis=0))))
            planned['feedback_vs_planned_source_joint_delta_error_rad']=delta_error
            with (directory/'C'/'feedback_ik.csv').open('w',newline='') as f:
                writer=csv.writer(f);writer.writerow(['control','stroke_id']+[f'q{k}' for k in range(1,7)])
                for i,q in enumerate(q_feedback):writer.writerow([i,int(ids[i]),*q])
            if delta_error>.01: raise ValueError('actual IK feedback differs from planned robot branch')
        # Export both paths and independently audit exact physical target -> IK
        # parity, not simply compare gamma scalars.
        path=directory/args.case/'ur10_planned_joints.csv'
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('w',newline='') as f:
            w=csv.writer(f); w.writerow(['state','stroke_id','duration_s','x','y','z','alpha','beta','gamma']+[f'q{i}' for i in range(1,7)])
            for t in node._targets:
                p=t.point; w.writerow([p.state,p.stroke_id,t.duration_s,p.x,p.y,p.z,p.alpha,p.beta,p.gamma,*t.joints])
        planned.update(planned_csv=str(path),planned_csv_sha256=sha(path))
    except Exception as exc:
        planned.update(feasible=False,status='rejected',error_type=type(exc).__name__,error=str(exc)); traceback.print_exc()
    finally:
        if node is not None: node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
    # Unchanged original target forward image, before any ROS rendering.
    with Image.open(folder/'scaled_target.png') as im: target=1-np.asarray(im.convert('L'),float)/255
    with np.load(folder/'neural_ink.npz') as stream: pred=stream['frames'][-1].copy()
    data['metrics']['ssim_original']=ssim(target,pred)
    stroke_file=directory/args.case/'stroke_residuals.json'
    if stroke_file.exists():
        data['stroke_residuals']=json.loads(stroke_file.read_text())
        if data['stroke_residuals'].get('physical_csv_sha256')!=data['physical_csv_sha256']:
            raise ValueError('stored stroke residuals belong to another CSV')
    else:
        data['stroke_residuals']=audit_candidate(folder,stroke_file)
        data['stroke_residuals']['physical_csv_sha256']=data['physical_csv_sha256']
        write(stroke_file,data['stroke_residuals'])
    data['planning']=planned; write(directory/f'{args.case}_result.json',data)
    summarize(Path(args.output))


def summarize(root):
    summaries=[]; artifacts={}
    for directory in sorted(root.glob('canvas_*_slots')):
        results=[json.loads(p.read_text()) for p in sorted(directory.glob('*_result.json'))]
        if not results: continue
        pinned=json.loads((directory/'experiment.json').read_text())
        baseline=next((r for r in results if r['case']=='A'),None)
        rows=[]
        image=Image.new('RGB',(768,240*len(results)),'white'); draw=ImageDraw.Draw(image)
        for n,r in enumerate(results):
            q=r['metrics']; p=r['planning']; m=p.get('metrics',{})
            relative=baseline is not None and q['iou_at_0.5']>=baseline['metrics']['iou_at_0.5']-.01
            visual=q['iou_at_0.5']>=.95 and q['ssim_original']>=.90
            row=dict(case=r['case'],font_size_m=pinned['canvas']['font_size_m'],iou=q['iou_at_0.5'],
                ssim=q['ssim_original'],mse=q['plain_mse'],wrist_deg=m.get('wrist_total_deg'),
                q4_q5_q6_deg=[m.get('joint_total_deg',{}).get(k) for k in ('wrist_1','wrist_2','wrist_3')],
                contact_deg=m.get('contact_q4_q5_q6_deg'),penup_deg=m.get('penup_or_approach_q4_q5_q6_deg'),
                pure_penup_deg=m.get('penup_q4_q5_q6_deg'),approach_deg=m.get('approach_q4_q5_q6_deg'),
                contact_total_deg=sum(m.get('contact_q4_q5_q6_deg',[])) if m else None,
                pure_penup_total_deg=sum(m.get('penup_q4_q5_q6_deg',[])) if m else None,
                approach_total_deg=sum(m.get('approach_q4_q5_q6_deg',[])) if m else None,
                duration_s=m.get('planned_duration_s'),max_internal_step_rad=m.get('max_internal_joint_step_rad'),
                robot_passed=p['feasible'],relative_iou_passed=relative,absolute_visual_passed=visual)
            rows.append(row)
            draw.text((8,n*240+3),f"{r['case']}  {row['font_size_m']*1000:.3f}mm IoU={row['iou']:.5f} SSIM={row['ssim']:.4f} wrist={row['wrist_deg']}",fill='black')
            for k,name in enumerate(('inversion_target.png','inversion_rendered.png','inversion_diff.png')):
                with Image.open(Path(r['candidate'])/name) as im:
                    bw=im.convert('L').point(lambda v:255-v)
                    image.paste(bw.resize((208,208)),(8+k*250,n*240+25))
        image.save(directory/'comparison.png'); write(directory/'comparison.json',rows)
        with (directory/'comparison.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        summaries.append(dict(canvas=pinned['canvas'],slots=pinned['slots'],results=rows,
            complete=len(results)==3,paired_identity_passed=len({r['initial_pose_sha256'] for r in results})==1
                and len({r['target_sha256'] for r in results})==1
                and len({r['checkpoint_sha256'] for r in results})==1,
            cases=results))
    for p in root.rglob('*'):
        if (p.is_file() and p.name not in ('verification.json','verification.sha256')
                and '__pycache__' not in str(p) and 'metric_reference' not in p.parts):
            if p.suffix in ('.json','.csv','.png','.py','.yaml','.md','.sh','.xml','.log','.npz','.stl','.urdf'):
                artifacts[str(p.relative_to(root))]=sha(p)
    complete=len(summaries)==2 and all(s['complete'] for s in summaries)
    tests=test_evidence(root)
    report=dict(format='v16_actual_ur10_wrist_full_abc_v1',complete=complete,
        motion_published=False,production_promoted=False,coppeliasim_modified=False,
        budget='16 steps, order 11, same per-case target/source/checkpoint/canvas/initialization',
        visual_acceptance=dict(iou_min=.95,ssim_min=.90,relative_iou_drop_max=.01,
            note='engineering visual gate, not a robot calibration certificate'),
        all_robot_checks_passed=complete and all(r['robot_passed'] for s in summaries for r in s['results']),
        all_visual_checks_passed=complete and all(r['absolute_visual_passed'] for s in summaries for r in s['results']),
        tests=tests,tests_passed=all(v['passed'] for v in tests.values()),
        wrist_measurement_basis='factory-calibrated simulated joint path; no hardware motion or encoder measurements',
        collision_validation_scope='current paper/table, calibrated tool/camera attachment and robot link envelopes',
        experiments=summaries,sha256=artifacts)
    rotation_checks=[]
    for s in summaries:
        by_case={r['case']:r for r in s['results']}
        if set(by_case)!=set('ABC'): continue
        a,b,c=(by_case[k] for k in 'ABC')
        rotation_checks.append(dict(slots=s['slots'],
            C_rotation_reduced_vs_A=bool(c['wrist_deg'] is not None and a['wrist_deg'] is not None and c['wrist_deg']<a['wrist_deg']-1.),
            C_rotation_reduced_vs_B=bool(c['wrist_deg'] is not None and b['wrist_deg'] is not None and c['wrist_deg']<b['wrist_deg']-1.),
            C_absolute_visual_passed=c['absolute_visual_passed'],C_relative_quality_passed=c['relative_iou_passed'],
            C_robot_passed=c['robot_passed']))
    report['rotation_acceptance']=rotation_checks
    report['accepted']=bool(complete and report['tests_passed'] and all(s['paired_identity_passed'] for s in summaries)
        and len(rotation_checks)==2 and all(all(v for k,v in r.items() if k!='slots') for r in rotation_checks))
    write(root/'verification.json',report)
    (root/'verification.sha256').write_text(sha(root/'verification.json')+'  verification.json\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['generate','plan','summary'],required=True)
    p.add_argument('--case',choices=['A','B','C'],default='A')
    p.add_argument('--manifest',required=True); p.add_argument('--output',required=True)
    p.add_argument('--model-root',default='/home/robot/coppeliasim/machine_learning/model')
    p.add_argument('--python',default='/home/robot/miniconda3/envs/ddpm/bin/python')
    p.add_argument('--character',default='武'); p.add_argument('--width',type=float,default=.52)
    p.add_argument('--height',type=float,default=.32); p.add_argument('--slots',type=int,default=1)
    p.add_argument('--steps',type=int,default=16); p.add_argument('--order',type=int,default=11)
    p.add_argument('--foreground-weight',type=float,default=1.); p.add_argument('--timeout-s',type=int,default=21600)
    args=p.parse_args()
    if args.slots not in (1,3): p.error('paired validation uses current single/three-cell canvas')
    if args.mode=='generate': generate(args)
    elif args.mode=='plan': plan(args)
    else: summarize(Path(args.output))


if __name__=='__main__': main()
