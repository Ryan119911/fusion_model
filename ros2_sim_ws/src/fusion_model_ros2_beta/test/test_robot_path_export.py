import csv
from types import SimpleNamespace
import numpy as np
from fusion_model_ros2_beta.export_robot_path import export_path
from fusion_model_ros2_beta.ur10_wrist_cost import fk_jacobian


def target(q,state=3):
    p=SimpleNamespace(x=-999.,y=-999.,z=-999.,alpha=0.,beta=0.,gamma=0.,state=state,stroke_id=0)
    return SimpleNamespace(point=p,joints=q,duration_s=.5)


def test_air_seed_coordinates_use_actual_fk_and_keep_desired_labels(tmp_path):
    q=np.array([1.2,-1.6,2.1,-2.2,-1.5,5.8])
    p=export_path([target(q)],tmp_path/'path.csv')
    row=next(csv.DictReader(p.open()))
    pose,_=fk_jacobian(q)
    np.testing.assert_allclose([float(row[k]) for k in ('x','y','z')],pose[:3,3]+pose[:3,2]*.328)
    assert row['desired_x']=='-999.0' and row['cartesian_target_applicable']=='False'
    assert row['pose_frame']=='base' and row['tool_frame']=='brush_tip'
    assert float(row['q6'])==5.8


def test_export_does_not_wrap_whole_joint_turns_and_preserves_time(tmp_path):
    q=np.array([1.2,-1.6,2.1,-2.2,-1.5,0.]);other=q.copy();other[5]=2*np.pi
    p=export_path([target(q),target(other,1)],tmp_path/'path.csv')
    rows=list(csv.DictReader(p.open()))
    assert abs(float(rows[1]['q6'])-float(rows[0]['q6'])-2*np.pi)<1e-12
    assert float(rows[1]['time_from_start_s'])==1.
    assert rows[1]['cartesian_target_applicable']=='True'
