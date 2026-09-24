from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from fusion_model_ros2_beta.ur10_actual_kinematics import (
    CALIBRATION_HASH,
    FACTORY_JOINT_ORIGINS,
    attachment_envelopes,
    forward_kinematics,
)


PACKAGE = Path(__file__).resolve().parents[1]
JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


def _triplet(value):
    return np.asarray([float(item) for item in value.split()], dtype=float)


def test_static_urdf_uses_the_factory_calibration_and_real_stack():
    root = ET.parse(PACKAGE / "urdf" / "ur10_official.urdf").getroot()
    by_name = {joint.attrib["name"]: joint for joint in root.findall("joint")}
    for name, (xyz, rpy) in zip(JOINTS, FACTORY_JOINT_ORIGINS):
        origin = by_name[name].find("origin")
        np.testing.assert_allclose(_triplet(origin.attrib["xyz"]), xyz, atol=1e-12)
        np.testing.assert_allclose(_triplet(origin.attrib["rpy"]), rpy, atol=1e-12)
    assert by_name["base-paper"].find("parent").attrib["link"] == "base"
    assert by_name["base-writing_table"].find("parent").attrib["link"] == "base"
    assert by_name["gripper-brush"].find("parent").attrib["link"] == "robotiq_gripper_link"
    assert "mecheye_camera_link" in {link.attrib["name"] for link in root.findall("link")}
    assert CALIBRATION_HASH == "calib_15120592593058779304"


def test_safe_initial_pose_has_finite_tool_and_attachment_envelopes():
    joints = (0.0, -1.2, 1.2, -1.5, -np.pi / 2.0, 0.0)
    tool = forward_kinematics(joints)
    assert np.all(np.isfinite(tool))
    np.testing.assert_allclose(tool[:3, :3].T @ tool[:3, :3], np.eye(3), atol=1e-9)
    envelopes = attachment_envelopes(joints)
    assert {item[0] for item in envelopes} == {"ati_and_robotiq", "mecheye_camera"}
    assert all(np.all(item[2] > 0.0) for item in envelopes)
