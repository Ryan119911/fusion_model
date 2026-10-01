import math
import numpy as np
import pytest
import torch
from optim.tool_orientation import (absolute_tool_rotation_residuals, absolute_tool_rotations,
                                    absolute_tool_rotation_report, canvas_paper_headings, brush_rotations)
from utils.joint_rotation_metrics import joint_rotation_metrics


def tensors():
    return (torch.tensor([[0., 0.], [1., 0.], [2., 1.], [3., 2.]], dtype=torch.float64),
            torch.tensor([[15., .04, .02]] * 4, dtype=torch.float64), torch.zeros(4, dtype=torch.float64), [np.arange(4)])


def test_driver_matrix_convention_and_canvas_y_flip():
    xy, pose, gamma, ids = tensors()
    heading = canvas_paper_headings(xy, ids)
    np.testing.assert_allclose(heading, [0., -math.pi/4, -math.pi/4, -math.pi/4])
    a, b, g = .04, .02, -math.pi/4
    rx = np.array([[1,0,0],[0,math.cos(a),-math.sin(a)],[0,math.sin(a),math.cos(a)]])
    ry = np.array([[math.cos(b),0,math.sin(b)],[0,1,0],[-math.sin(b),0,math.cos(b)]])
    rz = np.array([[math.cos(g),-math.sin(g),0],[math.sin(g),math.cos(g),0],[0,0,1]])
    np.testing.assert_allclose(absolute_tool_rotations(xy, pose, gamma, ids)[1], rz @ ry @ rx @ np.diag([1,-1,-1]), atol=1e-12)


def test_xy_and_local_gamma_are_both_in_rotation_cost():
    xy, pose, gamma, ids = tensors()
    xy.requires_grad_(); pose.requires_grad_(); gamma.requires_grad_()
    residual = torch.cat(absolute_tool_rotation_residuals(xy, pose, gamma, ids, 10.))
    (residual.square().sum()).backward()
    assert torch.isfinite(xy.grad).all() and xy.grad.abs().max() > 0
    assert torch.isfinite(gamma.grad).all() and gamma.grad.abs().max() > 0
    assert torch.isfinite(pose.grad).all()
    assert absolute_tool_rotation_report(xy, pose, gamma, ids)["total_rad"] == pytest.approx(math.pi/4)


def test_no_pen_up_or_interstroke_rotation_cost():
    xy, pose, gamma, _ = tensors()
    ids = [np.array([0,1]), np.array([2,3])]
    gamma[2:] = 2.
    assert torch.cat(absolute_tool_rotation_residuals(xy, pose, gamma, ids, 1.)).abs().max() == 0
    assert absolute_tool_rotation_residuals(xy, pose, gamma, ids, 0.) == []


def test_periodic_and_duplicate_point_gradients_are_finite():
    xy, pose, gamma, ids = tensors()
    xy[1] = xy[0]
    xy.requires_grad_()
    gamma = torch.tensor([math.pi-1e-4,-math.pi+1e-4,0.,0.], requires_grad=True, dtype=torch.float64)
    torch.cat(absolute_tool_rotation_residuals(xy, pose, gamma, ids, 1.)).square().sum().backward()
    assert torch.isfinite(xy.grad).all() and torch.isfinite(gamma.grad).all()
    torch.testing.assert_close(brush_rotations(pose[:,1],pose[:,2],gamma), brush_rotations(pose[:,1],pose[:,2],gamma+2*math.pi), atol=1e-12, rtol=1e-12)


def test_planned_wrist_full_turn_is_not_hidden():
    q = np.zeros((4,6)); q[1:,5] = [2*math.pi,2*math.pi+.1,2*math.pi+.3]
    report = joint_rotation_metrics(q, [3,1,1,3], [0,0,0,1], [.25,10.,.5,1.])
    assert report["wrist_total_rad"] == pytest.approx(2*math.pi+.3)
    assert report["wrist_writing_rad"] == pytest.approx(.1)
    assert report["wrist_transfer_rad"] == pytest.approx(2*math.pi+.2)
    with pytest.raises(ValueError):
        joint_rotation_metrics(q[:1], [1], [0])
