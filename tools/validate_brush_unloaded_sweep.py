"""Fit a local static wrench reference and check independent held-out/air data.

This does not accept brush H mapping, contact dynamics, new cable routing, or
unobserved translations. The production writing gate remains separate.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def features(rotations, baseline):
    x = (Rotation.from_rotvec(baseline).inv()*Rotation.from_rotvec(rotations)).as_rotvec()/np.deg2rad(4)
    x = np.atleast_2d(x)
    a,b,c = x.T
    return np.column_stack((np.ones(len(x)),a,b,c,a*a,b*b,c*c,a*b,a*c,b*c))


def validate(run, force_error_limit_n=1.):
    if not np.isfinite(force_error_limit_n) or not 0 < force_error_limit_n <= 10:
        raise ValueError('force reference error limit must be within (0,10] N')
    run=Path(run)
    result=json.loads((run/'result.json').read_text())
    request=json.loads((run/'request.json').read_text())
    if result['status']!='air_references_collected_pending_independent_validation':
        raise ValueError('complete sweep required')
    references=result['completed_references']
    if len(references)!=20 or sum(r['role']=='holdout' for r in references)!=4:
        raise ValueError('all static training, held-out and return references required')
    base=request['jobs'][0]['baseline_rotation']
    rotations=np.array([r['reference']['actual_pose_m_rad'][3:] for r in references])
    wrench=np.array([r['reference']['median_n_nm'] for r in references])
    train=np.array([r['role']=='train' for r in references]);held=np.array([r['role']=='holdout' for r in references])
    x=features(rotations,base)
    if np.linalg.matrix_rank(x[train])!=10:
        raise ValueError('static wrench fit is rank deficient')
    coefficients=np.linalg.lstsq(x[train],wrench[train],rcond=None)[0]
    error=x@coefficients-wrench
    held_force=np.linalg.norm(error[held,:3],axis=1)
    held_torque=np.linalg.norm(error[held,3:],axis=1)
    drift=np.linalg.norm(wrench[-1,:3]-wrench[0,:3])
    dynamic=[];unpaired=0
    for ref in references:
        folder=Path(ref.get('reference_source_directory',run/ref['reference_id']))
        feedback=[json.loads(s) for s in (folder/'controller_feedback.jsonl').read_text().splitlines()]
        times=np.array([r['host_unix_ns']/1e9 for r in feedback])
        poses=np.array([r['actual_tool0_pose_m_rad'] for r in feedback])
        velocity=np.array([r['tcp_velocity'] for r in feedback])
        raw=[json.loads(s) for s in (folder/'raw_force.jsonl').read_text().splitlines()]
        stamps=np.array([r['received_unix_s'] for r in raw]);w=np.array([r['wrench_n_nm'] for r in raw])
        indices=np.searchsorted(times,stamps).clip(0,len(times)-1)
        left=(indices-1).clip(0,len(times)-1)
        indices=np.where(abs(times[left]-stamps)<abs(times[indices]-stamps),left,indices)
        paired=abs(times[indices]-stamps)<=.025
        moving=(np.linalg.norm(velocity[indices,:3],axis=1)>.0001)|(np.linalg.norm(velocity[indices,3:],axis=1)>.001)
        use=paired&moving
        unpaired+=int(np.sum(~paired))
        if np.any(use):
            residual=features(poses[indices[use],3:],base)@coefficients-w[use]
            dynamic.extend(np.linalg.norm(residual[:,:3],axis=1).tolist())
    if not dynamic:raise ValueError('independent moving-force records required')
    # Accuracy acceptance for this local reference, not a change to the10N stop.
    passed=bool(max(held_force)<=force_error_limit_n and max(held_torque)<=.1 and drift<=.5 and max(dynamic)<=force_error_limit_n)
    return {'status':'passed_local_air_reference_validation' if passed else 'failed_local_air_reference_validation',
        'passed':passed,'training_static_count':int(sum(train)),'heldout_static_count':int(sum(held)),
        'heldout_max_force_error_n':float(max(held_force)),'heldout_max_torque_error_nm':float(max(held_torque)),
        'return_drift_force_n':float(drift),'moving_paired_rows':len(dynamic),'unpaired_raw_rows':unpaired,
        'moving_max_force_error_n':float(max(dynamic)),'moving_p99_force_error_n':float(np.quantile(dynamic,.99)),
        'reference_error_limit_n':float(force_error_limit_n),'force_stop_threshold_n_unchanged':10.,
        'local_fit':{'basis':'quadratic relative actual rotation vector /4deg; order1,x,y,z,xx,yy,zz,xy,xz,yz',
                     'baseline_rotation_rad':base,'coefficients_10x6':coefficients.tolist()},
        'scope':'observed small-angle air trajectories at this tip anchor and cable installation; no extrapolation or contact acceptance',
        'write_force_reference_verified':False,'brush_H_mapping_accepted':False,'run':str(run.resolve())}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--force-error-limit-n',type=float,default=1.,help='explicit acceptance limit; original 10 N motion stop is unchanged')
    a=p.parse_args();d=validate(a.run,a.force_error_limit_n)
    with a.output.open('x') as f:json.dump(d,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in d.items() if k!='local_fit'}))
    if not d['passed']:raise SystemExit(1)


if __name__=='__main__':main()
