"""Export the actual calibrated FK of planned joints, not stale air labels."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .ur10_wrist_cost import fk_jacobian


def export_path(targets,path,brush_length=.328):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    columns=['state','stroke_id','duration_s','time_from_start_s','x','y','z','alpha','beta','gamma']
    columns += [f'q{k}' for k in range(1,7)]
    columns += ['desired_'+k for k in ('x','y','z','alpha','beta','gamma')]
    columns += ['cartesian_target_applicable','position_unit','angle_unit','joint_unit','pose_frame','tool_frame']
    elapsed=0.
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=columns);writer.writeheader()
        for index,target in enumerate(targets):
            point=target.point;pose,_=fk_jacobian(target.joints)
            tip=pose[:3,3]+pose[:3,2]*brush_length
            # Same ROS convention Rz(gamma) Ry(beta) Rx(alpha) Rx(pi).
            angles=Rotation.from_matrix(pose[:3,:3] @ np.diag([1.,-1.,-1.])).as_euler('xyz')
            elapsed+=target.duration_s
            row=dict(state=point.state,stroke_id=point.stroke_id,duration_s=target.duration_s,
                time_from_start_s=elapsed,position_unit='m',angle_unit='rad',joint_unit='rad',
                pose_frame='base',tool_frame='brush_tip',
                cartesian_target_applicable=index!=0)
            row.update(zip(('x','y','z'),map(float,tip)))
            row.update(zip(('alpha','beta','gamma'),map(float,angles)))
            row.update(zip([f'q{k}' for k in range(1,7)],map(float,target.joints)))
            row.update({'desired_'+k:getattr(point,k) for k in ('x','y','z','alpha','beta','gamma')})
            # Actual q1..q6 are untouched and unwrapped; Euler representation
            # is never used to compute or conceal accumulated joint turns.
            writer.writerow(row)
    return path
