import numpy as np
from fusion_model_ros2_beta.ur10_actual_kinematics import attachment_intersects_obstacle


def test_rotated_corner_false_positive_is_removed():
    c = np.sqrt(.5)
    rotation = np.array([[c,-c,0],[c,c,0],[0,0,1]])
    # Long thin diagonal box: its AABB covers this off-diagonal corner.
    assert not attachment_intersects_obstacle([0,0,0], rotation, [1,.01,.1],
                                             [.6,.7,-.7,-.6,.05])
    assert attachment_intersects_obstacle([0,0,0], rotation, [1,.01,.1],
                                         [.6,.7,.6,.7,.05])


def test_clearance_and_containment():
    bounds = [-1,1,-1,1,0]
    assert not attachment_intersects_obstacle([0,0,.11],np.eye(3),[.1,.1,.1],bounds)
    assert attachment_intersects_obstacle([0,0,.09],np.eye(3),[.1,.1,.1],bounds)
    assert attachment_intersects_obstacle([0,0,-2],np.eye(3),[.1,.1,.1],bounds)
