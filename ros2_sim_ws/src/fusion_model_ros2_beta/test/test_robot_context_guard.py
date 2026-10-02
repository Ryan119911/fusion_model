import hashlib
import numpy as np
import pytest
from pathlib import Path
from fusion_model_ros2_beta import ur10_actual_kinematics as robot
from fusion_model_ros2_beta.robot_context_guard import validate_placement


def context():
    return dict(calibration_hash=robot.CALIBRATION_HASH,
        kinematics_sha256=hashlib.sha256(Path(robot.__file__).read_bytes()).hexdigest(),
        paper_offset_xy_m=[0.,-.65],paper_z_m=0.,brush_length_m=.328)


def test_same_calibration_and_placement_passes():
    validate_placement(context(),paper_xy=[0.,-.65],paper_z=0.,brush_length=.328)


@pytest.mark.parametrize('xy,z,tcp', [([.01,-.65],0.,.328),([0.,-.65],.01,.328),([0.,-.65],0.,.30)])
def test_wrong_placement_or_tcp_fails_closed(xy,z,tcp):
    with pytest.raises(ValueError,match='different paper placement or TCP'):
        validate_placement(context(),paper_xy=xy,paper_z=z,brush_length=tcp)


def test_changed_factory_geometry_fails_closed():
    c=context();c['kinematics_sha256']='changed'
    with pytest.raises(ValueError,match='geometry changed'):
        validate_placement(c,paper_xy=[0.,-.65],paper_z=0.,brush_length=.328)
