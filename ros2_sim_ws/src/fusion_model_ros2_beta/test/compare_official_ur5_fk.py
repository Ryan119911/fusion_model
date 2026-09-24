"""Check that the numerical IK model matches the bundled official UR10 frames."""

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


JOINT_CHAIN = [
    "base_link-base_link_inertia",
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
    "wrist_3-flange",
    "flange-tool0",
]
MOVING_JOINTS = JOINT_CHAIN[1:7]


def rotation_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rotation_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rotation_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def homogeneous(rotation: np.ndarray, xyz=(0.0, 0.0, 0.0)) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = xyz
    return result


def urdf_fk(path: Path, values: np.ndarray) -> np.ndarray:
    root = ET.parse(path).getroot()
    joints = {item.attrib["name"]: item for item in root.findall("joint")}
    value_by_name = dict(zip(MOVING_JOINTS, values))
    transform = np.eye(4)
    for name in JOINT_CHAIN:
        joint = joints[name]
        origin = joint.find("origin")
        xyz = tuple(float(v) for v in origin.attrib["xyz"].split())
        roll, pitch, yaw = (float(v) for v in origin.attrib["rpy"].split())
        transform = transform @ homogeneous(
            rotation_z(yaw) @ rotation_y(pitch) @ rotation_x(roll), xyz
        )
        if name in value_by_name:
            transform = transform @ homogeneous(rotation_z(value_by_name[name]))
    return transform


def dh_fk(values: np.ndarray) -> np.ndarray:
    a_values = [0.0, -0.612, -0.5723, 0.0, 0.0, 0.0]
    d_values = [0.1273, 0.0, 0.0, 0.163941, 0.1157, 0.0922]
    alpha_values = [math.pi / 2, 0.0, 0.0, math.pi / 2, -math.pi / 2, 0.0]
    transform = np.eye(4)
    for a, d, alpha, theta in zip(a_values, d_values, alpha_values, values):
        c_theta, s_theta = math.cos(theta), math.sin(theta)
        c_alpha, s_alpha = math.cos(alpha), math.sin(alpha)
        transform = transform @ np.array(
            [
                [c_theta, -s_theta * c_alpha, s_theta * s_alpha, a * c_theta],
                [s_theta, c_theta * c_alpha, -c_theta * s_alpha, a * s_theta],
                [0.0, s_alpha, c_alpha, d],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    return transform


def main() -> None:
    urdf_path = Path(__file__).parents[1] / "urdf" / "ur10_official.urdf"
    base_link_from_dh_base = homogeneous(rotation_z(math.pi))
    samples = [
        np.zeros(6),
        np.array([0.3, -1.1, 1.4, -1.7, -1.2, 0.8]),
    ]
    for values in samples:
        expected = base_link_from_dh_base @ dh_fk(values)
        actual = urdf_fk(urdf_path, values)
        error = float(np.max(np.abs(actual - expected)))
        if error > 1e-8:
            raise AssertionError(f"official URDF/DH frame mismatch: {error}")
    print("official_ur10_urdf_matches_dh_with_base_link_rz_pi")


if __name__ == "__main__":
    main()
