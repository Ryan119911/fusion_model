#!/usr/bin/env python3
"""MU-only geometry plan from the planned air-alignment end; no robot access."""
import copy,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R,Slerp
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'));sys.path.insert(0,str(ROOT/'ros2_sim_ws/src/fusion_model_ros2_beta'))
import brush_write_capture as capture
from prepare_gong_model_trial import current_meshes
from run_gong_model_trial import check_conversion
from brush_model_start import connect_start
from brush_buffered_motion import blend_radii
from validate_brush_unloaded_sweep import features
from fusion_model_ros2_beta.ur10_wrist_cost import ik,fk_jacobian,LOWER,UPPER


def main():
    base=ROOT/'outputs/brush_mapping_20261027'
    c=json.loads((base/'gong_current_mount_retract50mm/prepare-20261027T071307337181Z/candidate.PREVIEW_ONLY.json').read_text())
    align=json.loads((base/'gong_air_alignment_current_mount/fast_resume/candidate.json').read_text())
    old=json.loads((ROOT/'outputs/brush_mapping_20261023/depth_batch_d006_d009_d012/unloaded_sweep_runtime.json').read_text())
    baseline=c['conversion']['baseline_tool0_rotation_rad']
    train=[j['points'][-1]['pose_m_rad'][3:] for j in old['jobs'] if j['role']=='train']
    domain=ConvexHull(features(train,baseline)[:,1:4])
    start=align['points'][-1]['pose_m_rad'];seed=align['points'][-1]['q_rad']
    points,connection=connect_start(c['points'],start,c['tcp']['translation_m'],c['paper']['reconstructed_center_tip_base_m'],c['paper']['paper_normal_base'],domain.equations,baseline)
    points[0]['phase']='planned_after_air_alignment_not_current_feedback'
    connection['planned_start_pose_m_rad']=connection.pop('measured_start_pose_m_rad')
    connection['orientation_domain_scope']='planned new sampling postures only; old wrench values NOT reused'
    meshes=current_meshes();origin=np.array(c['paper']['reconstructed_center_tip_base_m']);normal=np.array(c['paper']['paper_normal_base']);tip=np.array(c['tcp']['translation_m'])
    audit=capture.dense_audit(points,seed,origin,normal,tip,meshes)
    starts=[r for r in audit['samples'] if r['sample']==0]
    for point,row in zip(points[:-1],starts):point['q_rad']=row['q_rad']
    points[-1]['q_rad']=audit['end_q_rad']
    cloud=np.concatenate([m.reshape(-1,3) for m in meshes.values()]);cloud=cloud[ConvexHull(cloud).vertices]
    samples=[];failures=[]
    for index,radius in enumerate(blend_radii(points)):
        if not radius:continue
        source=points[index];pose=np.array(source['pose_m_rad']);rotation=R.from_rotvec(pose[3:])
        for direction in np.r_[np.eye(3),-np.eye(3)]:
            target=np.eye(4);target[:3,:3]=rotation.as_matrix();target[:3,3]=pose[:3]+radius*direction
            q,ok=ik(target,source['q_rad']);actual,jac=fk_jacobian(q)
            margin=float(np.min(np.minimum(q-LOWER,UPPER-q)));singular=float(np.linalg.svd(jac,compute_uv=False)[-1])
            gap=float(np.min((cloud@target[:3,:3].T+target[:3,3]-origin)@normal))
            row={'phase':'native_blend_neighborhood','source_waypoint_index':index,'direction':direction.tolist(),
                 'radius_m':radius,'pose_m_rad':np.r_[target[:3,3],pose[3:]].tolist(),'q_rad':q.tolist(),
                 'min_singular':singular,'joint_margin_deg':np.rad2deg(margin),'cad_paper_clearance_mm':gap*1000}
            samples.append(row)
            if not ok or margin<np.deg2rad(5) or singular<.02 or gap<.020:failures.append(row)
    c.update(points=points,audit=audit,connection=connection,native_blend_neighborhood={'passed':not failures,'samples':samples,'failures':failures},
        robot_execution_ready=False,force_reference_verified=False,
        source_geometry_status='offline planned alignment end; requires current-state binding before execution')
    out=base/'gong_current_mount_runtime_geometry';out.mkdir(exist_ok=False)
    (out/'candidate.json').write_text(json.dumps(c,indent=2)+'\n')
    (out/'geometry_collision_query.json').write_text(json.dumps({'audit':{'samples':audit['samples']+samples}},indent=2)+'\n')
    summary={'motion_sent':False,'numeric_passed':audit['passed_numeric'],'waypoints':len(points),'centerline_samples':len(audit['samples']),
             'blend_neighborhood_samples':len(samples),'blend_numeric_passed':not failures,'robot_execution_ready':False}
    (out/'RESULT.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
    # Reuse the existing 15 training + 4 independent holdout + return design.
    anchor=np.array(align['anchor']);current=np.array(start);q=np.array(seed);jobs=[];rows=[]
    original_interpolate=capture.interpolate
    capture.interpolate=lambda a,b:original_interpolate(a,b,linear_step=.002,angular_step_deg=.5)
    for oldjob in old['jobs']:
        target=R.from_rotvec(oldjob['points'][-1]['pose_m_rad'][3:]);prior=R.from_rotvec(current[3:]);angle=(prior.inv()*target).magnitude()
        rots=Slerp([0,1],R.from_matrix([prior.as_matrix(),target.as_matrix()]))(np.linspace(0,1,max(1,int(np.ceil(np.rad2deg(angle)/2)))+1))
        jobpoints=[]
        for rot in rots:
            jobpoints.append({'phase':'unloaded_reorientation','pose_m_rad':np.r_[anchor-rot.apply(tip),rot.as_rotvec()].tolist(),
                'contact':False,'stroke_id':None,'speed_m_s':.0015,'acceleration_m_s2':.0005,'unloaded_wrench_n_nm':[0.]*6})
        for a,b in zip(jobpoints,jobpoints[1:]):
            x,y=np.array(a['pose_m_rad']),np.array(b['pose_m_rad']);angle=(R.from_rotvec(x[3:]).inv()*R.from_rotvec(y[3:])).magnitude()
            if angle>1e-10:b['speed_m_s']=max(1e-9,min(.0015,np.linalg.norm(y[:3]-x[:3])*.008/angle))
        checked=capture.dense_audit(jobpoints,q,origin,normal,tip,meshes)
        if not checked['passed_numeric']:raise RuntimeError(checked['failures'][:2])
        starts=[r for r in checked['samples'] if r['sample']==0]
        for point,row in zip(jobpoints[:-1],starts):point['q_rad']=row['q_rad']
        jobpoints[-1]['q_rad']=checked['end_q_rad'];q=np.array(checked['end_q_rad']);current=np.array(jobpoints[-1]['pose_m_rad'])
        job={'reference_id':oldjob['reference_id'],'role':oldjob['role'],'angles_deg':oldjob['angles_deg'],'points':jobpoints,'audit':checked,
            'maximum_runtime_s':120,'origin':origin.tolist(),'normal':normal.tolist(),'tip':tip.tolist(),'anchor':anchor.tolist(),
            'baseline_rotation':baseline,'watchdog_path':oldjob['watchdog_path']}
        jobs.append(job);rows+=checked['samples']
    reference={'jobs':jobs,'audit':{'passed_numeric':True,'samples':rows},'mount_model':c['mount_model'],'tip_height_m':align['tip_height_m'],
               'motion_sent':False,'scope':'old target posture design only; all wrench observations must be new for current installation'}
    (out/'reference_plan.json').write_text(json.dumps(reference,indent=2)+'\n')
    print(json.dumps({'new_reference_jobs':len(jobs),'sampling_geometry_passed':True,'new_reference_force_values_collected':False}))
    if not audit['passed_numeric'] or failures:raise SystemExit(1)

if __name__=='__main__':main()
