import numpy as np
import pytest
from fusion_model_ros2_beta.wrist_aware_transitions import travel


def test_real_joint_turns_are_not_hidden_by_modulo():
    q=np.zeros((2,6)); q[1,5]=2*np.pi
    assert abs(travel(q)-2*np.pi)<1e-12


def test_triangle_bound_on_shortcut():
    q=np.zeros((3,6)); q[1,3:]=[2.,-2.,3.]; q[2,3:]=[1.,1.,1.]
    assert travel(q[[0,2]]) < travel(q)


def test_penup_optimization_preserves_contact_and_never_resets_home():
    pytest.importorskip('rclpy')
    from types import SimpleNamespace
    from fusion_model_ros2_beta.brush_trajectory_driver import BrushPoint,JointTarget
    from fusion_model_ros2_beta.wrist_aware_transitions import optimize_penup
    from fusion_model_ros2_beta.ur10_wrist_cost import ik,tool_rotation,INITIAL
    target=np.eye(4);target[:3,:3]=tool_rotation(.03,.02,.1)
    target[:3,3]=np.array([0.,-.65,-.015])-target[:3,2]*.328
    q,ok=ik(target,INITIAL);assert ok
    mid=q.copy();mid[5]+=.7
    goal=q.copy();goal[5]+=.1
    p=BrushPoint(0.,-.65,-9.,.03,.02,.1,1,0)
    from dataclasses import replace
    path=[JointTarget(p,tuple(q),.03),JointTarget(replace(p,state=3),tuple(mid),.03),
          JointTarget(replace(p,stroke_id=1),tuple(goal),.03)]
    values={'brush_length_m':.328,'paper_z_m':-10.,'lift_height_m':.025,
        'minimum_singularity_margin':.08,'air_interval_s':.03}
    node=SimpleNamespace(_targets=path,_param=values.__getitem__,_motion_obstacle_collision=lambda a,b:None)
    optimize_penup(node)
    assert node._targets[0] == path[0] and node._targets[-1]==path[-1]
    assert travel([t.joints for t in node._targets]) < travel([t.joints for t in path])
    assert node._penup_rotation_audit['contact_targets_preserved']
    assert not node._penup_rotation_audit['per_stroke_home_reset']
