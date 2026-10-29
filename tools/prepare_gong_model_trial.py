#!/usr/bin/env python3
"""MU-only preparation of a relocated gong with the current direct camera mount.

No SSH, ROS graph, force acquisition, camera capture or robot command is used.
The output is a new offline candidate, NOT an executable/accepted motion bundle.
"""
import argparse
import copy
import datetime as dt
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from brush_buffered_motion import blend_radii
from run_gong_model_trial import check_conversion, DEFAULT, ROOT
from brush_write_capture import dense_audit
import sys
sys.path.insert(0, str(ROOT/'integration/ur10_camera_direct_mount_20261027'))
from camera_direct_mount import read_stl

MOUNT = ROOT/'integration/ur10_camera_direct_mount_20261027/assembly.json'
DEFAULT_OUTPUT = ROOT/'outputs/brush_mapping_20261027/gong_current_mount_retract50mm'


def relocate(source, retract_mm):
    if not np.isfinite(retract_mm) or retract_mm <= 0:
        raise ValueError('retract-base-x-mm must be finite and positive')
    check_conversion(source)
    c = copy.deepcopy(source)
    u, v, n = [np.asarray(c['paper'][k],float) for k in
               ('paper_u_base','paper_v_base','paper_normal_base')]
    basis = np.column_stack((u,v))
    # Exact requested Base XY displacement; follow the existing paper plane
    # in Z, so moving in X does not silently change the commanded compression.
    uv = np.linalg.solve(basis[:2], [-retract_mm/1000, 0.])
    delta = basis@uv
    if abs(delta@n)>1e-9:raise ValueError('paper basis is not tangential')
    c['conversion']['paper_translation_mm'] = (np.asarray(c['conversion']['paper_translation_mm'])+uv*1000).tolist()
    for point in c['points']:
        point['pose_m_rad'][:3] = (np.asarray(point['pose_m_rad'][:3])+delta).tolist()
        point.pop('q_rad',None)
        point['unloaded_wrench_n_nm'] = None
    c['points'][0]['source_phase'] = c['points'][0]['phase']
    c['points'][0]['phase'] = 'planned_suspended_start_not_current_feedback'
    for row in c.get('mapped_contact_samples',[]):
        row['paper_uv_mm'] = (np.asarray(row['paper_uv_mm'])+uv*1000).tolist()
    for name in ['audit','compiler_manifest','execution_blockers','historical_paper_check']:
        c.pop(name,None)
    c['mode'] = 'current_mount_offline_preparation'
    c['motion_sent'] = False; c['robot_execution_ready'] = False
    c['force_reference_verified'] = False
    c['force_reference_status'] = 'previous_mount_reference_retired; new unloaded posture references required'
    c['previous_mount_force_sources_not_accepted'] = {
        k:c.pop(k) for k in ('force_validation_source','force_acceptance_source') if k in c}
    c['start_state_source'] = 'translated historical offline start; NOT a measured current state'
    c['relocation'] = {'requested_base_x_retraction_mm':float(retract_mm),
        'actual_base_translation_mm':(delta*1000).tolist(),
        'paper_translation_delta_mm':(uv*1000).tolist(),
        'scope':'rigid translation of all writing/approach/lift/transfer points along existing paper plane',
        'model_fields_unchanged':True,'native_transition_parameters_unchanged':True}
    assert c['original_controls'] == source['original_controls']
    for a,b in zip(source['points'],c['points']):
        assert a.get('model_fields')==b.get('model_fields')
        assert a.get('commanded_depth_mm')==b.get('commanded_depth_mm')
        assert a['pose_m_rad'][3:]==b['pose_m_rad'][3:]
        assert a['speed_m_s']==b['speed_m_s'] and a['acceleration_m_s2']==b['acceleration_m_s2']
    np.testing.assert_allclose(blend_radii(c['points']),blend_radii(source['points']),atol=1e-12,rtol=0)
    check_conversion(c)
    return c


def current_meshes():
    pkg=ROOT/'ros2_sim_ws/src/fusion_model_ros2_beta'
    folder=pkg/'meshes/end_effector'
    r=Rotation.from_euler('xyz',[-np.pi/2,0,-np.pi/2]).as_matrix()
    offset=np.array([-.080221162,.327612370,.238737610])
    return {p.name:read_stl(p)@r.T/1000+offset for p in sorted(folder.glob('*.stl'))}


def contact_bounds(candidate):
    tip=np.asarray(candidate['tcp']['translation_m'])
    result={}
    for sid in [None,0,1,2]:
        points=[p for p in candidate['points'] if p['contact'] and (sid is None or p['stroke_id']==sid)]
        xyz=np.array([np.asarray(p['pose_m_rad'][:3])+Rotation.from_rotvec(p['pose_m_rad'][3:]).apply(tip) for p in points])
        result['all' if sid is None else str(sid)]={'tip_base_min_mm':(xyz.min(0)*1000).tolist(),
                                                       'tip_base_max_mm':(xyz.max(0)*1000).tolist()}
    return result


def prepare(source_folder, output, retract_mm):
    source_path=Path(source_folder)/'candidate.PREVIEW_ONLY.json'
    source=json.loads(source_path.read_text())
    c=relocate(source,retract_mm)
    meshes=current_meshes()
    if len(meshes)!=4:raise ValueError('current four CAD meshes required')
    c['mount_model']=json.loads(MOUNT.read_text())
    c['mount_model']['source']=str(MOUNT)
    c['source_candidate']=str(source_path.resolve())
    points=c['points'];seed=source['audit']['samples'][0]['q_rad']
    print(json.dumps({'status':'checking_current_mount_offline','waypoints':len(points),'motion_sent':False}),flush=True)
    audit=dense_audit(points,seed,np.array(c['paper']['reconstructed_center_tip_base_m']),
                      np.array(c['paper']['paper_normal_base']),np.array(c['tcp']['translation_m']),meshes)
    audit['scope']+='; current separated camera and original three CAD bodies, MU only'
    c['audit']=audit
    nominal_uv=[]
    for point in points:
        if point['contact']:
            uv=np.array([point['model_fields']['paper_x_m'],point['model_fields']['paper_y_m']])*c['conversion']['xy_scale']
            uv+=np.array(c['conversion']['paper_translation_mm'])/1000
            nominal_uv.append(uv)
    uv=np.array(nominal_uv)
    # The historical paper frame/absolute plane retain their stated status.
    # This is only the centered A4-domain check, not a fresh surveyed polygon.
    margin=float(np.min(np.array([.297,.210])/2-np.abs(uv)))
    c['paper_domain_check']={'inside_nominal_centered_a4':bool(margin>=0),
        'minimum_centerline_edge_margin_mm':margin*1000,
        'paper_u_v_size_mm':[297,210],
        'scope':'nominal A4 centered on historical reconstructed paper origin; physical paper corners NOT freshly validated'}
    c['contact_bounds']=contact_bounds(c)
    c['native_blend_radii_m']=blend_radii(points)
    c['execution_blockers']=[
        'new-mount unloaded force/posture reference has not been acquired or accepted',
        'current robot suspended start and active TCP have not been read',
        'full robot self-collision and native blended-route check with current runtime geometry pending']
    if not audit['passed_numeric']:c['execution_blockers'].insert(0,'offline continuous IK/CAD-paper audit failed')
    if margin<0:c['execution_blockers'].insert(0,'nominal centered A4-domain check failed')
    c['checked_at']=dt.datetime.now(dt.timezone.utc).isoformat()
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    (out/'candidate.PREVIEW_ONLY.json').write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n')
    summary={'status':'offline_candidate_checked_not_execution_ready' if audit['passed_numeric'] and margin>=0 else 'offline_candidate_check_failed',
        'motion_sent':False,'robot_execution_ready':False,'output':str(out.resolve()),
        'control_counts':c['control_counts'],'waypoints':len(points),'relocation':c['relocation'],
        'numeric_audit_passed':audit['passed_numeric'],'audit_samples':len(audit['samples']),
        'numeric_failure_count':audit['failure_count'],'paper_domain_check':c['paper_domain_check'],
        'minimum_cad_paper_clearance_mm':min(x['cad_paper_clearance_mm'] for x in audit['samples']),
        'contact_bounds':c['contact_bounds'],'execution_blockers':c['execution_blockers']}
    (out/'RESULT.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-folder',type=Path,default=DEFAULT)
    p.add_argument('--retract-base-x-mm',type=float,default=50.)
    p.add_argument('--output-dir',type=Path,help='new output folder; defaults to a unique MU preparation run')
    a=p.parse_args()
    out=a.output_dir or DEFAULT_OUTPUT/('prepare-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    result=prepare(a.source_folder,out,a.retract_base_x_mm)
    if not result['numeric_audit_passed'] or not result['paper_domain_check']['inside_nominal_centered_a4']:raise SystemExit(1)


if __name__=='__main__':main()
