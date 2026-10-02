import numpy as np
import pytest
from fusion_model_ros2_beta.ur10_wrist_cost import fk_jacobian, ik, ActualWristCost, tool_rotation
from fusion_model_ros2_beta import ur10_actual_kinematics as robot


def context():
    return dict(canvas_center=[64.,64.],metres_per_pixel=.29/96,
        paper_offset_xy_m=[0.,-.65],paper_z_m=0.,brush_length_m=.328,
        calibration_hash=robot.CALIBRATION_HASH)


def test_exact_factory_fk_and_analytic_jacobian():
    q=np.array([1.2,-1.6,2.1,-2.2,-1.5,.8])
    pose,j=fk_jacobian(q)
    np.testing.assert_allclose(pose,robot.forward_kinematics(q),atol=1e-14)
    from scipy.spatial.transform import Rotation
    for k in range(6):
        d=q.copy(); d[k]+=1e-6; p,_=fk_jacobian(d)
        got=np.r_[(p[:3,3]-pose[:3,3])/1e-6,
            Rotation.from_matrix(p[:3,:3]@pose[:3,:3].T).as_rotvec()/1e-6]
        np.testing.assert_allclose(got,j[:,k],atol=5e-7)


def test_continuous_ik_keeps_full_turn_representative():
    q=np.array([1.2,-1.6,2.1,-2.2,-1.5,5.8])
    pose,_=fk_jacobian(q)
    found,ok=ik(pose,q+.002)
    assert ok
    np.testing.assert_allclose(found,q,atol=1e-5)
    assert found[5]>np.pi


def test_canvas_and_tool_transform_exact_ros_convention():
    cost=ActualWristCost(context(),1.)
    xy=np.array([[60.,64.],[61.,65.],[62.,66.]])
    posture=np.tile([15.,.03,.02],(3,1)); gamma=np.array([.1,.12,.13])
    targets=cost.targets(xy,posture,gamma,[np.arange(3)])
    r=tool_rotation(.03,.02,.1-np.pi/4)
    np.testing.assert_allclose(targets[0,:3,:3],r,atol=1e-12)
    np.testing.assert_allclose(targets[0,:3,3]+r[:,2]*.328,[-4*.29/96,-.65,-.015],atol=1e-12)


def test_actual_wrist_feedback_changes_with_all_six_fields():
    torch=pytest.importorskip('torch')
    cost=ActualWristCost(context(),10.)
    xy=torch.tensor([[62.,64.],[64.,64.5],[66.,65.5]],dtype=torch.float64)
    p=torch.tensor([[14.,.04,.025],[15.,.05,.03],[16.,.06,.035]],dtype=torch.float64)
    g=torch.tensor([.1,.12,.13],dtype=torch.float64); members=[np.arange(3)]
    r=cost.residual(xy,p,g,members).numpy()
    assert cost.last_report['valid']
    for f in range(6):
        xx,pp,gg=xy.clone(),p.clone(),g.clone()
        if f<2: xx[1,f]+=.001
        elif f<5: pp[1,f-2]+=.001 if f==2 else .0001
        else: gg[1]+=.0001
        trial=cost.residual(xx,pp,gg,members).numpy()
        assert np.linalg.norm(trial-r)>1e-9, f


def test_calibration_change_fails_closed():
    c=context(); c['kinematics_sha256']='wrong'
    with pytest.raises(ValueError): ActualWristCost(c,10.)


def test_final_path_rejects_unwrapped_fast_full_turn():
    from types import SimpleNamespace
    from fusion_model_ros2_beta.joint_path_audit import audit_path
    q=np.array([1.2,-1.6,2.1,-2.2,-1.5,.1]); other=q.copy();other[5]=5.8
    path=[SimpleNamespace(joints=q,duration_s=10.),SimpleNamespace(joints=other,duration_s=.001)]
    with pytest.raises(ValueError,match='path gate failed'):
        audit_path(path)


def test_final_path_accepts_slow_continuous_calibrated_motion():
    from types import SimpleNamespace
    from fusion_model_ros2_beta.joint_path_audit import audit_path
    from fusion_model_ros2_beta.ur10_wrist_cost import INITIAL
    path=[SimpleNamespace(joints=INITIAL,duration_s=.03),
          SimpleNamespace(joints=INITIAL+np.array([.01,0,0,0,0,.02]),duration_s=.1)]
    result=audit_path(path)
    assert result['passed'] and result['max_joint_speed_rad_s']<.6
