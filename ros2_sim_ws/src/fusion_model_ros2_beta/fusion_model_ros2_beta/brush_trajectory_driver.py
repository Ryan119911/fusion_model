"""ROS 2 native UR10 brush trajectory driver.

This node does not import or modify the CoppeliaSim driver.  A separate
offline layer scales the complete target image and re-inverts all six fields
with V16.  This node only places the resulting
physical, glyph-centred trajectory, converts its paper poses to UR10 flange
poses, solves numerical IK with continuity seeds, and publishes a standard
``trajectory_msgs/JointTrajectory`` for a ROS 2 controller.
"""

from __future__ import annotations

import csv
import bisect
import hashlib
import math
import queue
import threading
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import rclpy
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import Point, Pose, PoseArray
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import ColorRGBA, Float32, String, UInt8
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker, MarkerArray

from .paper_brush_model import (
    DynamicFootprintState,
    FlexibleBrushModel,
    PaperBrushParameters,
    evaluate_masks,
    load_normalized_target,
    normalize_ink_mask,
    parameters_as_dict,
    render_world_footprints,
    save_evaluation_artifacts,
    triangle_fan,
)
from .input_page import CharacterInputPage
from .neural_ink import frame_quads, schedule_frames
from .joint_candidate import is_joint_contract
from .canvas_layout import plan_canvas
from .offline_fontsize_inversion import (
    OfflineFontSizeGenerator,
    OfflineInversionConfig,
)
from .trajectory_catalog import TrajectoryCatalog, TrajectoryEntry, character_directory_name
from .ur10_actual_kinematics import (
    CALIBRATION_HASH,
    ROBOT_MODEL,
    ROBOT_SERIAL,
    attachment_envelopes as actual_attachment_envelopes,
    forward_kinematics as actual_ur10_fk,
    joint_origins as actual_ur10_joint_origins,
)


JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Collision-free folded configurations used as joint-space detours when a
# direct IK transition would sweep an arm link through the paper edge.  They
# are deliberately kept in the Beta package; the original ROS2 driver is
# untouched.
PARKING_JOINT_SEEDS = (
    (0.0, -1.2, 1.2, -1.5, -math.pi / 2.0, 0.0),
    (0.0, -1.4, 1.8, -2.1, -math.pi / 2.0, 0.0),
    (math.pi, -1.2, 1.2, -1.5, -math.pi / 2.0, 0.0),
    (math.pi, -1.2, -1.2, -1.5, math.pi / 2.0, 0.0),
)

# The fake hardware, planner and obstacle validator must start from exactly the
# same posture.  Keeping this pose folded beside the long edge prevents an arm
# link from occupying the table before the first command is received.
SAFE_INITIAL_JOINTS = PARKING_JOINT_SEEDS[0]

# Conservative capsule radii for the six serial-chain segments returned by
# ``_joint_positions_in_base``.  The previous centre-line-only test could
# report a clear path while an official UR10 visual/collision mesh still clipped
# the paper edge.  These radii include the bulky shoulder and wrist housings.
LINK_COLLISION_RADII_M = (0.120, 0.095, 0.085, 0.080, 0.070, 0.065)

# Official UR10 limits used when replacing an IK angle by its nearest
# kinematically equivalent ``q + 2*pi*k`` representation.
JOINT_LOWER_LIMITS = np.array(
    [-2.0 * math.pi, -2.0 * math.pi, -math.pi, -2.0 * math.pi,
     -2.0 * math.pi, -2.0 * math.pi],
    dtype=np.float64,
)
JOINT_UPPER_LIMITS = -JOINT_LOWER_LIMITS

REQUIRED_COLUMNS = {
    "character",
    "sample_id",
    "stroke_id",
    "point_id",
    "x",
    "y",
    "z",
    "alpha",
    "beta",
    "gamma",
    "state",
}


@dataclass(frozen=True)
class SourcePoint:
    character: str
    sample_id: str
    stroke_id: int
    point_id: int
    x: float
    y: float
    z_mm: float
    alpha: float
    beta: float
    gamma: float
    state: int


@dataclass(frozen=True)
class BrushPoint:
    x: float
    y: float
    z: float
    alpha: float
    beta: float
    gamma: float
    state: int
    stroke_id: int
    press_depth_mm: float = 0.0
    footprint_scale: float = 1.0


@dataclass(frozen=True)
class JointTarget:
    point: BrushPoint
    joints: Tuple[float, ...]
    duration_s: float


def _rot_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _wrap_angle(angle: float) -> float:
    """Wrap an angle to the shortest continuous representation."""

    return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


def _dh(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    ca, sa = math.cos(alpha), math.sin(alpha)
    ct, st = math.cos(theta), math.sin(theta)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def ur10_fk(joints: Sequence[float]) -> np.ndarray:
    """Factory-calibrated ``base -> tool0`` FK for the physical UR10 CB3."""
    return actual_ur10_fk(joints)


def _so3_log(rotation: np.ndarray) -> np.ndarray:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle < 1e-8:
        return np.array(
            [(rotation[2, 1] - rotation[1, 2]) * 0.5,
             (rotation[0, 2] - rotation[2, 0]) * 0.5,
             (rotation[1, 0] - rotation[0, 1]) * 0.5]
        )
    sine = max(math.sin(angle), 1e-8)
    axis = np.array(
        [rotation[2, 1] - rotation[1, 2],
         rotation[0, 2] - rotation[2, 0],
         rotation[1, 0] - rotation[0, 1]]
    ) / (2.0 * sine)
    return axis * angle


def _pose_error(joints: np.ndarray, target: np.ndarray) -> np.ndarray:
    current = ur10_fk(joints)
    position_error = target[:3, 3] - current[:3, 3]
    rotation_error = _so3_log(target[:3, :3] @ current[:3, :3].T)
    return np.concatenate((position_error, rotation_error))


def solve_ur10_ik(
    target: np.ndarray,
    seed: Sequence[float],
    *,
    max_iterations: int = 180,
    position_tolerance_m: float = 0.004,
    orientation_tolerance_rad: float = 0.20,
) -> Tuple[np.ndarray, float, float, bool]:
    """Solve a continuous local UR10 IK target without external Coppelia code."""
    joints = np.asarray(seed, dtype=np.float64).copy()
    damping = 2.0e-3
    epsilon = 1.0e-5
    for _ in range(max_iterations):
        error = _pose_error(joints, target)
        if (
            float(np.linalg.norm(error[:3])) <= position_tolerance_m
            and float(np.linalg.norm(error[3:])) <= orientation_tolerance_rad
        ):
            return joints, float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:])), True
        jacobian = np.zeros((6, 6), dtype=np.float64)
        for index in range(6):
            perturbed = joints.copy()
            perturbed[index] += epsilon
            jacobian[:, index] = (
                _pose_error(perturbed, target) - error
            ) / epsilon
        normal = jacobian.T @ jacobian + damping * np.eye(6)
        # ``error`` is target-current, so the Gauss-Newton correction moves
        # against the error Jacobian.
        step = -np.linalg.solve(normal, jacobian.T @ error)
        step = np.clip(step, -0.12, 0.12)
        joints += step
        joints = (joints + math.pi) % (2.0 * math.pi) - math.pi
    error = _pose_error(joints, target)
    return joints, float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:])), False


def load_points(path: Path, character: str, sample_id: str) -> Tuple[List[SourcePoint], str]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"trajectory CSV missing columns: {sorted(missing)}")
        rows = [row for row in reader if row.get("character") == character]
    if not rows:
        raise ValueError(f"character {character!r} not found in {path}")
    selected = [row for row in rows if row.get("sample_id") == sample_id]
    if not selected:
        raise ValueError(f"sample {sample_id!r} not found for {character!r}")
    points = [
        SourcePoint(
            character=character,
            sample_id=sample_id,
            stroke_id=int(float(row["stroke_id"])),
            point_id=int(float(row["point_id"])),
            x=float(row["x"]),
            y=float(row["y"]),
            z_mm=float(row["z"]),
            alpha=float(row["alpha"]),
            beta=float(row["beta"]),
            gamma=float(row["gamma"]),
            state=int(float(row["state"])),
        )
        for row in selected
    ]
    return points, digest


def _brush_rotation(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Map paper-model alpha/beta/gamma to a ROS base-frame brush pose."""
    # Brush local +Z points from the flange toward the paper.  V17's alpha and
    # beta are the two paper-frame tilt components and gamma is the in-plane
    # twist relative to the stroke heading.  Apply all three before flipping
    # the tool toward the paper; ignoring alpha/beta here would make the
    # rendered footprint and the executed UR10 pose describe different brush
    # widths.
    down = _rot_x(math.pi)
    return _rot_z(gamma) @ _rot_y(beta) @ _rot_x(alpha) @ down


def place_physical_source_points(
    source: Sequence[SourcePoint],
    *,
    paper_z: float,
    paper_offset_x: float,
    paper_offset_y: float,
    max_compression: float,
    lift_height: float,
) -> List[BrushPoint]:
    """Translate an offline physical trajectory without resizing its glyph."""

    if not source:
        return []
    result: List[BrushPoint] = []
    for point in source:
        if not all(
            math.isfinite(value)
            for value in (point.x, point.y, point.z_mm, point.alpha, point.beta, point.gamma)
        ):
            raise ValueError("offline physical trajectory contains non-finite values")
        if point.state == 3:
            depth = 0.0
            compression = 0.0
        else:
            depth = point.z_mm
            compression = depth / 1000.0
            if compression < 0.0 or compression > max_compression + 1.0e-12:
                raise ValueError(
                    f"offline press depth {depth:.3f} mm is outside ROS safety "
                    f"range 0..{max_compression * 1000.0:.3f} mm"
                )
        # state=2 is the final *contact* point of a stroke.  Only state=3
        # denotes an airborne point; dropping state=2 truncates the last
        # segment of every stroke.
        z = paper_z + lift_height if point.state == 3 else paper_z - compression
        result.append(
            BrushPoint(
                x=point.x + paper_offset_x,
                y=point.y + paper_offset_y,
                z=z,
                alpha=point.alpha,
                beta=point.beta,
                gamma=_wrap_angle(point.gamma),
                state=point.state,
                stroke_id=point.stroke_id,
                press_depth_mm=depth,
                footprint_scale=1.0,
            )
        )
    return result


def _interpolate(left: BrushPoint, right: BrushPoint, max_step: float) -> List[BrushPoint]:
    distance = math.sqrt(
        (right.x - left.x) ** 2 + (right.y - left.y) ** 2 + (right.z - left.z) ** 2
    )
    count = max(1, int(math.ceil(distance / max(max_step, 1e-5))))
    return [
        BrushPoint(
            x=left.x + (right.x - left.x) * ratio,
            y=left.y + (right.y - left.y) * ratio,
            z=left.z + (right.z - left.z) * ratio,
            alpha=left.alpha + (right.alpha - left.alpha) * ratio,
            beta=left.beta + (right.beta - left.beta) * ratio,
            gamma=left.gamma + _wrap_angle(right.gamma - left.gamma) * ratio,
            state=right.state,
            stroke_id=right.stroke_id,
            press_depth_mm=(
                left.press_depth_mm
                + (right.press_depth_mm - left.press_depth_mm) * ratio
            ),
            footprint_scale=(
                left.footprint_scale
                + (right.footprint_scale - left.footprint_scale) * ratio
            ),
        )
        for index in range(1, count + 1)
        for ratio in (index / count,)
    ]


def densify(points: Sequence[BrushPoint], max_step: float) -> List[BrushPoint]:
    dense: List[BrushPoint] = []
    for left, right in zip(points, points[1:]):
        if not dense:
            dense.append(left)
        dense.extend(_interpolate(left, right, max_step))
    if points and not dense:
        dense.append(points[0])
    return dense


def _offline_pose_summary(points: Sequence[BrushPoint]) -> dict:
    """Summarise the V16 poses consumed unchanged by the ROS driver."""

    contacts = [point for point in points if point.state != 3]
    if not contacts:
        return {
            "method": "V16 weights + v42 fused-pose interface; ROS placement only",
            "contact_point_count": 0,
        }
    return {
        "method": "V16 weights + v42 fused-pose interface; ROS placement only",
        "input_fields": [
            "x_m",
            "y_m",
            "H_mm",
            "alpha_rad",
            "beta_rad",
            "gamma_rad",
        ],
        "contact_point_count": len(contacts),
        "z_range_mm": [
            min(float(point.press_depth_mm) for point in contacts),
            max(float(point.press_depth_mm) for point in contacts),
        ],
        "alpha_range_rad": [
            min(float(point.alpha) for point in contacts),
            max(float(point.alpha) for point in contacts),
        ],
        "beta_range_rad": [
            min(float(point.beta) for point in contacts),
            max(float(point.beta) for point in contacts),
        ],
        "gamma_range_rad": [
            min(float(point.gamma) for point in contacts),
            max(float(point.gamma) for point in contacts),
        ],
    }


def _flange_target(point: BrushPoint, brush_length: float) -> np.ndarray:
    rotation = _brush_rotation(point.alpha, point.beta, point.gamma)
    base_target = np.eye(4, dtype=np.float64)
    base_target[:3, :3] = rotation
    base_target[:3, 3] = (
        np.array([point.x, point.y, point.z]) - rotation[:, 2] * brush_length
    )
    return base_target


def _joint_positions_in_base(joints: Sequence[float]) -> List[np.ndarray]:
    """Return calibrated UR10 joint origins in controller ``base``."""
    return actual_ur10_joint_origins(joints)


def _collision_sample_count(left: np.ndarray, right: np.ndarray) -> int:
    """Use a fine enough joint-space subdivision for table collision checks."""

    return max(1, int(math.ceil(float(np.max(np.abs(right - left))) / 0.04)))


def _joint_singularity_margin(joints: Sequence[float]) -> float:
    """Return a compact UR10 elbow/wrist singularity margin in ``[0, 1]``.

    UR-style elbow and spherical-wrist singularities occur as ``sin(q3)`` or
    ``sin(q5)`` approaches zero.  This inexpensive guard is applied during IK
    branch selection; trajectory time scaling separately handles large but
    valid folded-pose transitions.
    """

    values = np.asarray(joints, dtype=np.float64)
    return min(abs(math.sin(float(values[2]))), abs(math.sin(float(values[4]))))


class BrushTrajectoryDriver(Node):
    """Precompute and publish a smooth UR10 joint trajectory through ROS 2."""

    def __init__(self) -> None:
        super().__init__("brush_trajectory_driver_beta")
        self.declare_parameter("trajectory_csv", "/home/robot/coppeliasim/machine_learning/model/outputs/kaishu_pose_v16_all_fields_footprint_wu/inversion_trajectory.csv")
        self.declare_parameter("character", "武")
        self.declare_parameter("sample_id", "武_fake_sim")
        self.declare_parameter("required_sha256", "")
        self.declare_parameter("paper_width_m", 0.80)
        self.declare_parameter("paper_height_m", 0.40)
        # In Beta the UR10 is mounted on a pedestal beside the table, with its
        # base frame at the paper centre-plane height.  Paper/table Z values
        # are consequently expressed close to zero in controller base.
        self.declare_parameter("paper_z_m", 0.0)
        self.declare_parameter("paper_thickness_m", 0.002)
        # The robot base is centred outside the lower long edge.  These are
        # paper/table centres expressed in controller base.
        self.declare_parameter("paper_offset_x_m", 0.0)
        self.declare_parameter("paper_offset_y_m", -0.65)
        self.declare_parameter("paper_margin_m", 0.018)
        self.declare_parameter("character_gap_m", 0.045)
        self.declare_parameter("max_text_characters", 3)
        self.declare_parameter("default_layout_mode", "horizontal")
        # The page submits a bounded writing area. Character sizes are derived
        # from it and generated by the offline model, never scaled in ROS.
        self.declare_parameter("writing_width_m", 0.52)
        self.declare_parameter("writing_height_m", 0.32)
        self.declare_parameter("default_font_size_m", 0.12)
        self.declare_parameter("offline_reference_font_size_m", 0.29)
        self.declare_parameter(
            "offline_model_root",
            "/home/robot/coppeliasim/machine_learning/model",
        )
        self.declare_parameter(
            "offline_python_executable",
            "/home/robot/miniconda3/envs/ddpm/bin/python",
        )
        self.declare_parameter(
            "offline_bbsmg_checkpoint",
            "/home/robot/coppeliasim/machine_learning/model/outputs/"
            "paper_bbsmg_general_v16_pose_dense/bbsmg_best.pt",
        )
        self.declare_parameter(
            "offline_bbsmg_checkpoint_sha256",
            "30d5e0c37dc7b7913c02fea26930babafa46a4a07b5f698520df277a04d5b717",
        )
        self.declare_parameter(
            "offline_cache_root",
            "/home/robot/coppeliasim/machine_learning/model/outputs/"
            "ros2_v16_fontsize_cache",
        )
        # A 16-step six-field original-target inversion on the MU host can take
        # longer than 30 minutes. Do not kill healthy generation at the old
        # 1800 s boundary.
        self.declare_parameter("offline_inversion_timeout_s", 3600)
        self.declare_parameter("offline_inversion_device", "cuda")
        self.declare_parameter("offline_inversion_backend", "legacy_fused")
        self.declare_parameter("offline_joint_reference_manifest", "")
        self.declare_parameter("offline_original_target_manifest", "")
        self.declare_parameter("offline_joint_max_steps", 16)
        self.declare_parameter("offline_joint_foreground_weight", 1.0)
        self.declare_parameter("offline_joint_target_skeleton_weight", 0.0)
        self.declare_parameter("offline_joint_target_skeleton_max_distance_px", 12.0)
        self.declare_parameter("offline_joint_target_skeleton_threshold", 0.35)
        # The physical table has the same footprint as the paper.  Its full
        # footprint is an obstacle; the arm is not allowed to use an edge
        # overhang as a shortcut.
        self.declare_parameter("table_width_m", 0.80)
        self.declare_parameter("table_depth_m", 0.40)
        self.declare_parameter("table_top_z_m", -0.001)
        self.declare_parameter("obstacle_clearance_m", 0.005)
        self.declare_parameter("max_brush_compression_m", 0.020)
        # Real stack: tool0 -> manually measured closed-gripper grasp centre
        # (220 mm) -> 108 mm brush.  Recalibrate if the physical brush clamp
        # or protruding length changes.
        self.declare_parameter("brush_length_m", 0.328)
        # Joint-space detours handle the paper edge; keep the visible brush
        # lift moderate so all airborne poses remain inside the UR10 workspace.
        self.declare_parameter("lift_height_m", 0.025)
        self.declare_parameter("max_step_m", 0.002)
        self.declare_parameter("contact_interval_s", 0.03)
        self.declare_parameter("air_interval_s", 0.03)
        self.declare_parameter("initial_transition_s", 1.5)
        self.declare_parameter("max_joint_speed_rad_s", 0.60)
        self.declare_parameter("minimum_singularity_margin", 0.08)
        self.declare_parameter(
            "controller_topic",
            "/fusion_controller_manager/joint_trajectory",
        )
        # The wider Beta paper reaches the edge of the numerical IK workspace.
        # Keep the existing 4 mm solver tolerance, but allow the best bounded
        # solution so a multi-character replay is not rejected by a marginal
        # 0.03 mm numerical residual.
        self.declare_parameter("strict_ik", False)
        self.declare_parameter("publish_on_start", False)
        self.declare_parameter("frame_id", "base")
        self.declare_parameter(
            "trajectory_catalog_root",
            "/home/robot/coppeliasim/machine_learning/model/outputs/"
            "kaishu_pose_trajectory_batch_v1",
        )
        self.declare_parameter(
            "allow_v17_ineligible",
            False,
        )
        self.declare_parameter(
            "database_csv",
            "/home/robot/coppeliasim/machine_learning/model/data/raw/data.csv",
        )
        self.declare_parameter("database_style", "楷")
        self.declare_parameter("target_overrides_json", "{}")
        self.declare_parameter("enable_input_page", True)
        self.declare_parameter("input_page_host", "127.0.0.1")
        self.declare_parameter("input_page_port", 18080)
        self.declare_parameter(
            "target_image",
            "/home/robot/coppeliasim/machine_learning/model/data/raw/targets/wu_kaishu_target.png",
        )
        self.declare_parameter(
            "evaluation_output_dir",
            "/home/robot/ros2_ws/evaluation/fusion_model_ros2",
        )
        self.declare_parameter("evaluation_image_size", 128)
        # Paper Table 4 minima for robot-written complete characters.
        self.declare_parameter("minimum_csim", 0.9136)
        self.declare_parameter("minimum_ssim", 0.7042)
        self.declare_parameter("brush_bundle_length_m", 0.048)
        self.declare_parameter("brush_bundle_radius_m", 0.006)
        self.declare_parameter("brush_press_depth_min_mm", 11.0)
        self.declare_parameter("brush_press_depth_max_mm", 20.0)
        self.declare_parameter("brush_tilt_max_rad", math.radians(10.0))
        self.declare_parameter("brush_rotation_max_rad", math.radians(5.0))
        self.declare_parameter("brush_width_inertia", 0.02)
        self.declare_parameter("brush_drag_inertia", 0.02)
        self.declare_parameter("brush_regression_length_unit_m", 0.01)
        self.declare_parameter("brush_bezier_samples", 14)

        self._brush_parameters = PaperBrushParameters(
            bundle_length_m=float(self._param("brush_bundle_length_m")),
            bundle_radius_m=float(self._param("brush_bundle_radius_m")),
            press_depth_min_mm=float(self._param("brush_press_depth_min_mm")),
            press_depth_max_mm=float(self._param("brush_press_depth_max_mm")),
            tilt_max_rad=float(self._param("brush_tilt_max_rad")),
            rotation_max_rad=float(self._param("brush_rotation_max_rad")),
            width_inertia=float(self._param("brush_width_inertia")),
            drag_inertia=float(self._param("brush_drag_inertia")),
            regression_length_unit_m=float(
                self._param("brush_regression_length_unit_m")
            ),
        )
        self._brush_model = FlexibleBrushModel(self._brush_parameters)

        self._trajectory_pub = self.create_publisher(
            JointTrajectory, str(self._param("controller_topic")), 10
        )
        self._state_pub = self.create_publisher(UInt8, "~/brush_state", 10)
        self._progress_pub = self.create_publisher(Float32, "~/progress", 10)
        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._preview_pub = self.create_publisher(MarkerArray, "~/preview", latched_qos)
        self._ink_pub = self.create_publisher(MarkerArray, "~/ink", latched_qos)
        self._pose_pub = self.create_publisher(PoseArray, "~/target_poses", latched_qos)
        self._evaluation_pub = self.create_publisher(String, "~/evaluation", latched_qos)
        self._ink_markers = MarkerArray()
        self._neural_jobs = []
        self._ink_last_index = -1
        self._ink_present = False
        self._dynamic_states: List[Optional[DynamicFootprintState]] = []
        self._footprints: List[Optional[np.ndarray]] = []
        self._evaluation_report: Optional[dict] = None
        self._catalog = TrajectoryCatalog(
            Path(str(self._param("trajectory_catalog_root"))).expanduser(),
            fallback_csv=Path(str(self._param("trajectory_csv"))).expanduser(),
            fallback_target=Path(str(self._param("target_image"))).expanduser(),
            fallback_character=str(self._param("character")),
            database_csv=Path(str(self._param("database_csv"))).expanduser(),
            database_style=str(self._param("database_style")),
            target_overrides=self._parse_target_overrides(),
            allow_v17_ineligible=bool(self._param("allow_v17_ineligible")),
        )
        from .inversion_backend import make_inversion_backend
        self._offline_generator = make_inversion_backend(
            OfflineInversionConfig(
                model_root=Path(str(self._param("offline_model_root"))).expanduser(),
                python_executable=Path(
                    str(self._param("offline_python_executable"))
                ).expanduser(),
                bbsmg_checkpoint=Path(
                    str(self._param("offline_bbsmg_checkpoint"))
                ).expanduser(),
                cache_root=Path(str(self._param("offline_cache_root"))).expanduser(),
                reference_font_size_m=float(
                    self._param("offline_reference_font_size_m")
                ),
                timeout_s=int(self._param("offline_inversion_timeout_s")),
                device=str(self._param("offline_inversion_device")),
                joint_foreground_weight=float(self._param("offline_joint_foreground_weight")),
                joint_xy_target_skeleton_weight=float(
                    self._param("offline_joint_target_skeleton_weight")
                ),
                joint_xy_target_skeleton_max_distance_px=float(
                    self._param("offline_joint_target_skeleton_max_distance_px")
                ),
                joint_xy_target_skeleton_threshold=float(
                    self._param("offline_joint_target_skeleton_threshold")
                ),
                expected_checkpoint_sha256=str(
                    self._param("offline_bbsmg_checkpoint_sha256")
                ),
            ),
            backend=str(self._param("offline_inversion_backend")),
            reference_manifest=str(self._param("offline_joint_reference_manifest")),
            joint_steps=int(self._param("offline_joint_max_steps")),
            original_target_manifest=str(
                self._param("offline_original_target_manifest")
            ),
        )
        self._job_queue: "queue.Queue[tuple[List[TrajectoryEntry], str, float]]" = queue.Queue()
        self._job_lock = threading.Lock()
        self._generation_thread: Optional[threading.Thread] = None
        self._active_layout_mode = self._normalize_layout_mode(
            str(self._param("default_layout_mode"))
        )
        self._job_state = {
            "status": "starting",
            "character": str(self._param("character")),
            "sample_id": str(self._param("sample_id")),
            "layout_mode": self._active_layout_mode,
            "layout_label": self._layout_label(self._active_layout_mode),
            "writing_width_m": float(self._param("writing_width_m")),
            "writing_height_m": float(self._param("writing_height_m")),
            "font_size_m": float(self._param("default_font_size_m")),
            "progress": 0.0,
            "message": "离线字号反演层已就绪",
        }
        self._active_character = str(self._param("character"))
        self._active_sample_id = str(self._param("sample_id"))
        self._active_target_path = Path(str(self._param("target_image"))).expanduser()
        self._active_target_is_normalized = False
        self._active_entry: Optional[TrajectoryEntry] = None
        self._active_evaluation_dir = Path(
            str(self._param("evaluation_output_dir"))
        ).expanduser()
        self._input_page: Optional[CharacterInputPage] = None
        self._publish_timer = self.create_timer(0.5, self._publish_once)
        self._visualization_timer = self.create_timer(
            1.0, self._republish_ink
        )
        self._execution_timer = self.create_timer(0.03, self._publish_execution_state)
        self._job_timer = self.create_timer(0.1, self._process_job_queue)
        self._published = False
        self._targets: List[JointTarget] = []
        self._target_times: List[float] = []
        self._execution_started_ns: Optional[int] = None
        self._last_execution_index = -1
        with self._job_lock:
            self._job_state.update(
                {
                    "status": "ready",
                    "progress": 0.0,
                    "message": "请输入文字和字号；首次使用该字号会执行离线反演",
                }
            )
        if bool(self._param("publish_on_start")):
            self.get_logger().warning(
                "publish_on_start is ignored: offline font-size inversion "
                "requires an explicit web request"
            )
        self.get_logger().info(
            f"physical robot profile: {ROBOT_MODEL}, serial={ROBOT_SERIAL}, "
            f"kinematics={CALIBRATION_HASH}, frame=base, "
            f"tool0_to_brush_tip={float(self._param('brush_length_m')):.3f} m"
        )
        if bool(self._param("enable_input_page")):
            self._input_page = CharacterInputPage(
                self,
                str(self._param("input_page_host")),
                int(self._param("input_page_port")),
            )
            host, port = self._input_page.address
            self.get_logger().info(
                f"character input page: http://{host}:{port}/ "
                "(enter one or more generated 楷书 characters)"
            )

    def _param(self, name: str):
        return self.get_parameter(name).value

    @staticmethod
    def _normalize_layout_mode(value: str) -> str:
        aliases = {
            "horizontal": "horizontal",
            "row": "horizontal",
            "横向": "horizontal",
            "横排": "horizontal",
            "vertical": "vertical",
            "column": "vertical",
            "竖向": "vertical",
            "竖排": "vertical",
        }
        mode = aliases.get(str(value).strip().lower())
        if mode is None:
            raise ValueError("排版方向必须是 horizontal（横向）或 vertical（竖向）。")
        return mode

    @staticmethod
    def _layout_label(layout_mode: str) -> str:
        return "横向（从左到右）" if layout_mode == "horizontal" else "竖向（从上到下）"

    def _fixed_writing_area(self) -> Tuple[float, float]:
        width = float(self._param("writing_width_m"))
        height = float(self._param("writing_height_m"))
        paper_width = float(self._param("paper_width_m"))
        paper_height = float(self._param("paper_height_m"))
        if not math.isfinite(width) or not math.isfinite(height):
            raise ValueError("固定安全书写区域必须是有限数值。")
        if width <= 0.0 or height <= 0.0:
            raise ValueError("固定安全书写区域必须大于 0。")
        if width > paper_width or height > paper_height:
            raise ValueError(
                f"固定安全书写区域不能超过纸面 {paper_width * 1000:.0f}×"
                f"{paper_height * 1000:.0f} mm。"
            )
        return width, height

    def _validated_font_size(
        self,
        font_size_m: Optional[float],
        character_count: int,
        layout_mode: str,
    ) -> float:
        value = (
            float(self._param("default_font_size_m"))
            if font_size_m is None
            else float(font_size_m)
        )
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("字号必须是大于 0 的有限数值。")
        reference = float(self._param("offline_reference_font_size_m"))
        if value > reference + 1.0e-12:
            raise ValueError(f"字号不能超过离线模型标定上限 {reference * 1000:.0f} mm。")
        width, height = self._fixed_writing_area()
        gap = max(0.0, float(self._param("character_gap_m")))
        axis = width if layout_mode == "horizontal" else height
        cross = height if layout_mode == "horizontal" else width
        required = character_count * value + max(0, character_count - 1) * gap
        if value > cross + 1.0e-12 or required > axis + 1.0e-12:
            maximum_axis = (
                axis - max(0, character_count - 1) * gap
            ) / max(character_count, 1)
            maximum = max(0.0, min(cross, maximum_axis, reference))
            raise ValueError(
                f"当前{self._layout_label(layout_mode)}最多可使用 "
                f"{maximum * 1000:.0f} mm 字号。"
            )
        return value

    def _parse_target_overrides(self) -> dict[str, str]:
        raw = str(self._param("target_overrides_json")).strip()
        if not raw:
            return {}
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError("target_overrides_json must be a JSON object") from error
        if not isinstance(value, dict):
            raise ValueError("target_overrides_json must be a JSON object")
        return {str(key): str(path) for key, path in value.items()}

    def input_status(self) -> dict:
        """Return a JSON-safe snapshot for the character input page."""

        with self._job_lock:
            state = dict(self._job_state)
        if self._execution_started_ns is None and state.get("status") == "running":
            state["status"] = "completed"
        state["queue_size"] = self._job_queue.qsize()
        state["offline_generation_active"] = bool(
            self._generation_thread is not None
            and self._generation_thread.is_alive()
        )
        state["inversion_interface"] = self._offline_generator.identity()
        state['paper_width_m'] = float(self._param('paper_width_m'))
        state['paper_height_m'] = float(self._param('paper_height_m'))
        state["robot_profile"] = {
            "model": ROBOT_MODEL,
            "serial": ROBOT_SERIAL,
            "kinematics_hash": CALIBRATION_HASH,
            "control_frame": "base",
            "tool0_to_brush_tip_m": float(self._param("brush_length_m")),
        }
        state["ink_present"] = self._ink_present
        state["can_clear"] = (
            self._ink_present
            and self._execution_started_ns is None
            and self._job_queue.empty()
            and not state["offline_generation_active"]
        )
        return state

    def catalog_summary(self, query: str = "") -> dict:
        return self._catalog.summary(query=query, limit=50)

    def request_character(
        self,
        character: str,
        layout_mode: str = "horizontal",
        font_size_m: Optional[float] = None,
        writing_width_m: Optional[float] = None,
        writing_height_m: Optional[float] = None,
    ) -> dict:
        """Queue offline full-pose inversion, then execute its physical path."""

        text = "".join(value for value in str(character) if not value.isspace())
        try:
            normalized_layout = self._normalize_layout_mode(layout_mode)
        except ValueError as error:
            return {
                "accepted": False,
                "status": "invalid_layout",
                "message": str(error),
            }
        maximum = max(1, int(self._param("max_text_characters")))
        if not text or len(text) > maximum:
            return {
                "accepted": False,
                "status": "invalid",
                "message": f"请输入 1 到 {maximum} 个楷书字。",
            }
        entries: List[TrajectoryEntry] = []
        for value in text:
            entry = self._catalog.resolve(value)
            if entry is None:
                return {
                    "accepted": False,
                    "status": "not_found",
                    "message": f"数据库中没有找到“{value}”的轨迹记录。",
                }
            if not entry.ready:
                status = entry.status or "unavailable"
                return {
                    "accepted": False,
                    "status": status,
                    "message": (
                        f"“{value}”的笔触模型轨迹尚未完成或不可用（{status}）。"
                        "请等待批处理生成完成后刷新页面。"
                    ),
                }
            entries.append(entry)

        try:
            canvas_plan = None
            if writing_width_m is not None or writing_height_m is not None or font_size_m is None:
                width = float(self._param('writing_width_m') if writing_width_m is None else writing_width_m)
                height = float(self._param('writing_height_m') if writing_height_m is None else writing_height_m)
                spans = []
                for entry in entries:
                    points, _ = load_points(entry.trajectory_csv, entry.character, entry.sample_id)
                    spans.append((max(p.x for p in points)-min(p.x for p in points),
                                  max(p.y for p in points)-min(p.y for p in points)))
                canvas_plan = plan_canvas(spans, width, height, normalized_layout,
                    paper_width=float(self._param('paper_width_m')), paper_height=float(self._param('paper_height_m')),
                    gap=float(self._param('character_gap_m')), reference=float(self._param('offline_reference_font_size_m')))
                font_size = max(item['font_size_m'] for item in canvas_plan)
            else:
                font_size = self._validated_font_size(font_size_m, len(text), normalized_layout)
                width, height = self._fixed_writing_area()
        except (TypeError, ValueError, OSError) as error:
            return dict(accepted=False, status='invalid_canvas', message=str(error))

        with self._job_lock:
            generation_active = bool(
                self._generation_thread is not None
                and self._generation_thread.is_alive()
            )
            if (
                self._execution_started_ns is not None
                or not self._job_queue.empty()
                or generation_active
            ):
                return {
                    "accepted": False,
                    "status": "busy",
                    "message": "当前文字仍在书写，请等待完成后再输入。",
                }
            if self._ink_present:
                return {
                    "accepted": False,
                    "status": "clear_required",
                    "message": "请先点击“清除字迹”，再输入下一组文字。",
                }
            self._job_state = {
                "status": "offline_generating",
                "character": text,
                "sample_id": ",".join(entry.sample_id or "" for entry in entries),
                "layout_mode": normalized_layout,
                "layout_label": self._layout_label(normalized_layout),
                "writing_width_m": width,
                "writing_height_m": height,
                "computed_font_sizes_mm": [item['font_size_m']*1000 for item in canvas_plan] if canvas_plan else [font_size*1000]*len(entries),
                "font_size_m": font_size,
                "robot_profile": {
                    "model": ROBOT_MODEL,
                    "serial": ROBOT_SERIAL,
                    "kinematics_hash": CALIBRATION_HASH,
                    "control_frame": "base",
                    "tool0_to_brush_tip_m": float(self._param("brush_length_m")),
                    "attachments": ["ATI Net F/T", "Robotiq 2F", "Mech-Eye PRO XS"],
                },
                "progress": 0.0,
                "message": (
                    "正在准备离线字号反演；ROS 不会调整字格或缩放轨迹"
                ),
            }
            worker = threading.Thread(
                target=self._generate_offline_job,
                args=(entries, normalized_layout, font_size, canvas_plan),
                daemon=True,
                name="v16-fontsize-inversion",
            )
            self._generation_thread = worker
        worker.start()
        return {
            "accepted": True,
            "status": "offline_generating",
            "layout_mode": normalized_layout,
            "font_size_m": font_size,
            "writing_width_m": width,
            "writing_height_m": height,
            "computed_font_sizes_mm": [item['font_size_m']*1000 for item in canvas_plan] if canvas_plan else [font_size*1000]*len(entries),
            "message": (
                f"已接受“{text}”的 {font_size * 1000:.0f} mm 离线反演任务；"
                "完成后会自动开始 ROS 仿真。"
            ),
            "characters": [entry.as_dict() for entry in entries],
        }

    def _update_offline_progress(self, message: str) -> None:
        with self._job_lock:
            self._job_state.update(
                {
                    "status": "offline_generating",
                    "message": str(message),
                }
            )

    def _generate_offline_job(
        self,
        entries: Sequence[TrajectoryEntry],
        layout_mode: str,
        font_size_m: float,
        canvas_plan=None,
    ) -> None:
        try:
            generated: List[TrajectoryEntry] = []
            total = len(entries)
            for index, entry in enumerate(entries):
                self._update_offline_progress(
                    f"离线反演 {index + 1}/{total}：“{entry.character}”"
                )
                generated.append(
                    self._offline_generator.generate(
                        entry,
                        canvas_plan[index]['font_size_m'] if canvas_plan else font_size_m,
                        progress=self._update_offline_progress,
                    )
                )
                if canvas_plan:
                    generated[-1] = replace(generated[-1], metadata={**generated[-1].metadata, 'canvas_layout': canvas_plan[index]})
                with self._job_lock:
                    self._job_state["progress"] = (index + 1) / total
            self._job_queue.put((generated, layout_mode, font_size_m))
            with self._job_lock:
                self._generation_thread = None
                self._job_state.update(
                    {
                        "status": "queued",
                        "progress": 0.0,
                        "message": "离线六维轨迹已生成，等待 ROS 执行器加载",
                    }
                )
        except Exception as error:
            with self._job_lock:
                self._generation_thread = None
                self._job_state.update(
                    {
                        "status": "error",
                        "progress": 0.0,
                        "message": str(error),
                    }
                )
            self.get_logger().error(f"offline font-size inversion failed: {error}")

    def _process_job_queue(self) -> None:
        try:
            entries, layout_mode, font_size = (
                self._job_queue.get_nowait()
            )
        except queue.Empty:
            return
        with self._job_lock:
            self._job_state.update(
                {
                    "status": "preparing",
                    "character": "".join(entry.character for entry in entries),
                    "sample_id": ",".join(entry.sample_id or "" for entry in entries),
                    "layout_mode": layout_mode,
                    "layout_label": self._layout_label(layout_mode),
                    "writing_width_m": entries[0].metadata.get('canvas_layout', {}).get('writing_width_m', float(self._param('writing_width_m'))),
                    "writing_height_m": entries[0].metadata.get('canvas_layout', {}).get('writing_height_m', float(self._param('writing_height_m'))),
                    "font_size_m": font_size,
                    "progress": 0.0,
                    "message": (
                        f"正在放置 {font_size * 1000:.0f} mm 物理轨迹并计算"
                        f"{self._layout_label(layout_mode)}、逆运动学和柔性笔触"
                    ),
                }
            )
        try:
            self._build_targets(
                entries=entries,
                layout_mode=layout_mode,
                font_size_m=font_size,
            )
            with self._job_lock:
                self._job_state.update(
                    {
                        "status": "prepared",
                        "progress": 0.0,
                        "message": "轨迹已准备，等待控制器接收",
                    }
                )
            self._publish_once(force=True)
        except Exception as error:
            with self._job_lock:
                self._job_state.update(
                    {
                        "status": "error",
                        "progress": 0.0,
                        "message": str(error),
                    }
                )
            self.get_logger().error(
                f"multi-character request failed for {''.join(entry.character for entry in entries)!r}: {error}"
            )

    def clear_current_ink(self) -> dict:
        """Clear the displayed ink after the current character has finished."""

        with self._job_lock:
            generation_active = bool(
                self._generation_thread is not None
                and self._generation_thread.is_alive()
            )
            if (
                self._execution_started_ns is not None
                or not self._job_queue.empty()
                or generation_active
            ):
                return {
                    "accepted": False,
                    "status": "busy",
                    "message": "机械臂仍在书写，完成后才能清除字迹。",
                }
            self._clear_ink()
            self._ink_present = False
            self._job_state.update(
                {
                    "status": "cleared",
                    "progress": 0.0,
                    "message": "字迹已清除，可以输入下一个字",
                }
            )
        return {
            "accepted": True,
            "status": "cleared",
            "message": "字迹已清除，可以输入下一个字。",
        }

    def _clear_ink(self) -> None:
        """Remove the previous character before a new replay."""

        previous_markers = list(self._ink_markers.markers)
        self._neural_jobs = []
        self._ink_markers = MarkerArray()
        message = MarkerArray()
        frame_id = str(self._param("frame_id"))
        for previous in previous_markers:
            marker = Marker()
            marker.header.frame_id = previous.header.frame_id or frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = previous.ns
            marker.id = previous.id
            marker.action = Marker.DELETE
            message.markers.append(marker)

        # DELETEALL is retained as a final safety net for RViz versions that
        # received a marker before this node's local marker cache was filled.
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "ros2_flexible_brush_clear"
        marker.id = 0
        marker.action = Marker.DELETEALL
        message.markers.append(marker)
        self._ink_pub.publish(message)

    def _densify_mapped(self, mapped: Sequence[BrushPoint]) -> List[BrushPoint]:
        """Densify strokes and insert airborne moves within one character."""

        dense: List[BrushPoint] = []
        strokes: dict[int, List[BrushPoint]] = {}
        for point in mapped:
            strokes.setdefault(point.stroke_id, []).append(point)
        max_step = float(self._param("max_step_m"))
        lift_height = float(self._param("lift_height_m"))
        for stroke in strokes.values():
            contact = [point for point in stroke if point.state != 3]
            if not contact:
                continue
            contact_dense = densify(contact, max_step)
            hover_z = max(point.z for point in contact) + lift_height
            hover = BrushPoint(
                contact[0].x,
                contact[0].y,
                hover_z,
                contact[0].alpha,
                contact[0].beta,
                contact[0].gamma,
                3,
                contact[0].stroke_id,
            )
            if dense:
                previous = dense[-1]
                lift = BrushPoint(
                    previous.x,
                    previous.y,
                    max(previous.z, hover_z),
                    previous.alpha,
                    previous.beta,
                    previous.gamma,
                    3,
                    previous.stroke_id,
                )
                dense.extend(_interpolate(previous, lift, max_step))
                dense.extend(_interpolate(lift, hover, max_step))
            else:
                dense.append(hover)
            dense.extend(contact_dense)
        return dense

    def _validate_obstacle_clearance(self) -> None:
        """Reject any arm motion that enters the table/paper exclusion zone.

        The brush tip is allowed to contact the paper by design.  The UR10 arm
        links, however, must stay above the table whenever their swept volume
        is over the table footprint.  Joint-space interpolation is sampled as
        well as each Cartesian target so the controller's spline transition
        cannot silently cut through the writing surface.
        """

        clearance = max(0.0, float(self._param("obstacle_clearance_m")))

        # Start from the elbow-up branch.  The previous nominal seed was
        # elbow-down and could put the upper-arm segment under the paper edge
        # near the end of the character.
        # Re-check the final controller list.  If a transition that was
        # selected during IK is still blocked after all preceding detours
        # have been inserted, repair that exact edge and restart the sweep.
        # This makes the collision guarantee apply to the list that will
        # actually be published, including synthetic parking targets.
        for repair_attempt in range(32):
            previous = np.asarray(SAFE_INITIAL_JOINTS, dtype=np.float64)
            initial_collision = self._motion_obstacle_collision(previous, previous)
            if initial_collision is not None:
                obstacle_name, link_index, point = initial_collision
                raise RuntimeError(
                    "Beta safe initial posture is not clear: "
                    f"obstacle={obstacle_name}, link={link_index}, "
                    f"point=({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}) m"
                )
            repaired = False
            for target_index, target in enumerate(self._targets):
                current = np.asarray(target.joints, dtype=np.float64)
                collision = self._motion_obstacle_collision(previous, current)
                if collision is not None:
                    if self._insert_obstacle_detour(
                        target_index, previous, target.point
                    ):
                        repaired = True
                        self.get_logger().info(
                            "Beta obstacle detour: repaired the final "
                            "controller transition"
                        )
                        break
                    path = self._find_joint_path(previous, current)
                    if path is not None and len(path) > 2:
                        waypoints = [
                            JointTarget(
                                BrushPoint(
                                    target.point.x,
                                    target.point.y,
                                    max(
                                        target.point.z,
                                        float(self._param("paper_z_m"))
                                        + float(self._param("lift_height_m")),
                                    ),
                                    target.point.alpha,
                                    target.point.beta,
                                    target.point.gamma,
                                    3,
                                    target.point.stroke_id,
                                ),
                                tuple(float(value) for value in waypoint),
                                float(self._param("air_interval_s")),
                            )
                            for waypoint in path[1:-1]
                        ]
                        self._targets[target_index:target_index] = waypoints
                        repaired = True
                        self.get_logger().info(
                            "Beta obstacle detour: inserted a sampled "
                            f"joint-space path with {len(waypoints)} waypoints"
                        )
                        break
                    obstacle_name, link_index, point = collision
                    raise RuntimeError(
                        "Beta obstacle check failed: "
                        f"obstacle={obstacle_name}, target={target_index}, "
                        f"link={link_index}, "
                        f"point=({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}) m, "
                        "arm link must remain above the obstacle"
                    )
                previous = current
            if not repaired:
                break
        else:
            raise RuntimeError(
                "Beta obstacle check could not repair the controller trajectory "
                "within 32 detour attempts"
            )

        self.get_logger().info(
            "Beta obstacle check passed: arm links clear the "
            f"{float(self._param('paper_width_m')) * 1000:.0f}x"
            f"{float(self._param('paper_height_m')) * 1000:.0f} mm paper and "
            f"{float(self._param('table_width_m')) * 1000:.0f}x"
            f"{float(self._param('table_depth_m')) * 1000:.0f} mm table "
            f"with {clearance * 1000:.0f} mm clearance and link-radius envelopes"
        )

    def _insert_obstacle_detour(
        self,
        target_index: int,
        previous: np.ndarray,
        point: BrushPoint,
    ) -> bool:
        """Insert a folded joint target that makes one blocked edge safe."""

        for parking_values in PARKING_JOINT_SEEDS:
            parking = np.asarray(parking_values, dtype=np.float64)
            if self._motion_obstacle_collision(previous, parking) is not None:
                continue
            current = np.asarray(self._targets[target_index].joints, dtype=np.float64)
            if self._motion_obstacle_collision(parking, current) is not None:
                continue
            park_point = BrushPoint(
                point.x,
                point.y,
                max(
                    point.z,
                    float(self._param("paper_z_m"))
                    + float(self._param("lift_height_m")),
                ),
                point.alpha,
                point.beta,
                point.gamma,
                3,
                point.stroke_id,
            )
            self._targets.insert(
                target_index,
                JointTarget(
                    park_point,
                    tuple(float(value) for value in parking),
                    float(self._param("air_interval_s")),
                ),
            )
            return True
        return False

    def _find_joint_path(
        self, start: np.ndarray, goal: np.ndarray
    ) -> Optional[List[np.ndarray]]:
        """Find a short collision-free joint-space path between two poses."""

        if self._motion_obstacle_collision(start, goal) is None:
            return [start.copy(), goal.copy()]

        rng = np.random.default_rng(20260828 + len(self._targets))
        nodes: List[np.ndarray] = [start.copy()]
        parents: List[int] = [-1]
        step = 0.28

        def normalise(values: np.ndarray) -> np.ndarray:
            return (values + math.pi) % (2.0 * math.pi) - math.pi

        def add_node(values: np.ndarray, parent: int) -> Optional[int]:
            values = normalise(values)
            if self._motion_obstacle_collision(nodes[parent], values) is not None:
                return None
            nodes.append(values)
            parents.append(parent)
            return len(nodes) - 1

        # Seed the roadmap with the known folded configurations whenever their
        # incoming edge is safe.  Random nodes then bridge around the blocked
        # side of the paper instead of repeatedly rediscovering the same arm
        # branch.
        for parking_values in PARKING_JOINT_SEEDS:
            parking = np.asarray(parking_values, dtype=np.float64)
            if self._motion_obstacle_collision(start, parking) is None:
                nodes.append(parking)
                parents.append(0)

        for _ in range(640):
            if rng.random() < 0.20:
                sample = goal.copy()
            else:
                sample = rng.uniform(-math.pi, math.pi, size=6)
            distances = [float(np.linalg.norm(normalise(sample - node))) for node in nodes]
            nearest_index = int(np.argmin(distances))
            nearest = nodes[nearest_index]
            delta = normalise(sample - nearest)
            distance = float(np.linalg.norm(delta))
            if distance < 1.0e-9:
                continue
            candidate = nearest + delta * min(step, distance)
            new_index = add_node(candidate, nearest_index)
            if new_index is None:
                continue
            if self._motion_obstacle_collision(nodes[new_index], goal) is None:
                path: List[np.ndarray] = [goal.copy()]
                cursor = new_index
                while cursor >= 0:
                    path.append(nodes[cursor].copy())
                    cursor = parents[cursor]
                path.reverse()
                return path
        return None

    def _obstacle_limits(
        self,
    ) -> List[Tuple[str, float, float, float, float, float]]:
        """Return paper and table keep-out boxes in the base-link frame.

        The brush tip is intentionally not part of this check: it is the
        writing tool and is allowed to reach the paper.  Every UR10 arm-link
        segment is checked against both thin paper and the support table.
        """

        center_x = float(self._param("paper_offset_x_m"))
        center_y = float(self._param("paper_offset_y_m"))
        clearance = max(0.0, float(self._param("obstacle_clearance_m")))
        paper_half_width = float(self._param("paper_width_m")) / 2.0
        paper_half_depth = float(self._param("paper_height_m")) / 2.0
        table_half_width = float(self._param("table_width_m")) / 2.0
        table_half_depth = float(self._param("table_depth_m")) / 2.0
        paper_top = (
            float(self._param("paper_z_m"))
            + float(self._param("paper_thickness_m")) / 2.0
        )
        return [
            (
                "paper",
                center_x - paper_half_width - clearance,
                center_x + paper_half_width + clearance,
                center_y - paper_half_depth - clearance,
                center_y + paper_half_depth + clearance,
                paper_top + clearance,
            ),
            (
                "table",
                center_x - table_half_width - clearance,
                center_x + table_half_width + clearance,
                center_y - table_half_depth - clearance,
                center_y + table_half_depth + clearance,
                float(self._param("table_top_z_m")) + clearance,
            ),
        ]

    def _motion_obstacle_collision(
        self,
        left: np.ndarray,
        right: np.ndarray,
    ) -> Optional[Tuple[str, int, np.ndarray]]:
        """Return the first paper/table collision along joint-space motion."""

        obstacles = self._obstacle_limits()
        count = _collision_sample_count(left, right)
        for ratio in np.linspace(0.0, 1.0, count + 1)[1:]:
            joints = left + (right - left) * float(ratio)
            positions = _joint_positions_in_base(joints)
            for link_index, (segment_left, segment_right) in enumerate(
                zip(positions, positions[1:])
            ):
                link_radius = LINK_COLLISION_RADII_M[link_index]
                segment_count = max(
                    1,
                    int(
                        math.ceil(
                            float(np.linalg.norm(segment_right - segment_left)) / 0.01
                        )
                    ),
                )
                for segment_ratio in np.linspace(0.0, 1.0, segment_count + 1):
                    point = segment_left + (segment_right - segment_left) * float(
                        segment_ratio
                    )
                    for obstacle_name, x_min, x_max, y_min, y_max, safe_z in obstacles:
                        over_obstacle = (
                            x_min - link_radius <= point[0] <= x_max + link_radius
                            and y_min - link_radius
                            <= point[1]
                            <= y_max + link_radius
                        )
                        if over_obstacle and point[2] - link_radius < safe_z:
                            return obstacle_name, link_index, point
            from .ur10_actual_kinematics import attachment_intersects_obstacle
            for attachment_index, (
                attachment_name,
                centre,
                oriented_box,
            ) in enumerate(actual_attachment_envelopes(joints, oriented=True), start=6):
                for obstacle_name, x_min, x_max, y_min, y_max, safe_z in obstacles:
                    if attachment_intersects_obstacle(
                        centre, *oriented_box, (x_min, x_max, y_min, y_max, safe_z)
                    ):
                        return (
                            f"{obstacle_name}:{attachment_name}",
                            attachment_index,
                            centre,
                        )
        return None

    def _retime_joint_targets(self) -> None:
        """Enforce a joint-speed limit on every controller transition.

        Cartesian brush samples remain at their requested interval when the
        joint change is small.  Folded obstacle detours and IK branch changes
        receive proportionally more time, preventing a valid transition from
        looking like a wrist flip or singular snap in RViz.
        """

        max_speed = max(0.05, float(self._param("max_joint_speed_rad_s")))
        previous = np.asarray(SAFE_INITIAL_JOINTS, dtype=np.float64)
        retimed: List[JointTarget] = []
        largest_delta = 0.0
        minimum_margin = 1.0
        for target in self._targets:
            current = np.asarray(target.joints, dtype=np.float64)
            joint_delta = float(np.max(np.abs(current - previous)))
            duration = max(float(target.duration_s), joint_delta / max_speed)
            retimed.append(JointTarget(target.point, target.joints, duration))
            largest_delta = max(largest_delta, joint_delta)
            minimum_margin = min(
                minimum_margin, _joint_singularity_margin(current)
            )
            previous = current
        self._targets = retimed
        self.get_logger().info(
            "Beta joint timing passed: "
            f"speed_limit={max_speed:.2f} rad/s, "
            f"largest_transition={largest_delta:.3f} rad, "
            f"minimum_singularity_margin={minimum_margin:.3f}"
        )

    def _unwrap_joint_targets(self) -> None:
        """Choose the nearest limit-valid equivalent angle at every point."""

        previous = np.asarray(SAFE_INITIAL_JOINTS, dtype=np.float64)
        unwrapped: List[JointTarget] = []
        changed = 0
        largest_delta = 0.0
        for target in self._targets:
            current = np.asarray(target.joints, dtype=np.float64).copy()
            for joint_index, value in enumerate(current):
                equivalents = [
                    float(value + turns * 2.0 * math.pi)
                    for turns in range(-2, 3)
                    if JOINT_LOWER_LIMITS[joint_index] - 1.0e-9
                    <= value + turns * 2.0 * math.pi
                    <= JOINT_UPPER_LIMITS[joint_index] + 1.0e-9
                ]
                if equivalents:
                    nearest = min(
                        equivalents,
                        key=lambda candidate: abs(candidate - previous[joint_index]),
                    )
                    if abs(nearest - value) > 1.0e-8:
                        changed += 1
                    current[joint_index] = nearest
            largest_delta = max(
                largest_delta, float(np.max(np.abs(current - previous)))
            )
            unwrapped.append(
                JointTarget(
                    target.point,
                    tuple(float(value) for value in current),
                    target.duration_s,
                )
            )
            previous = current
        self._targets = unwrapped
        self.get_logger().info(
            "Beta joint continuity passed: "
            f"equivalent_angles_adjusted={changed}, "
            f"largest_transition={largest_delta:.3f} rad"
        )

    def _build_targets(
        self,
        *,
        entries: Optional[Sequence[TrajectoryEntry]] = None,
        layout_mode: Optional[str] = None,
        font_size_m: Optional[float] = None,
    ) -> None:
        """Place offline physical glyphs and build one joint trajectory."""

        if not self.has_parameter("character_gap_m"):
            self.declare_parameter("character_gap_m", 0.045)
        if not self.has_parameter("max_text_characters"):
            self.declare_parameter("max_text_characters", 3)

        if entries is None:
            raise RuntimeError(
                "ROS runtime scaling is disabled; an offline font-size trajectory is required"
            )
        entries = list(entries)
        if not entries:
            raise RuntimeError("no character trajectory was requested")
        if layout_mode is None:
            layout_mode = str(self._param("default_layout_mode"))
        layout_mode = self._normalize_layout_mode(layout_mode)

        text = "".join(entry.character for entry in entries)
        canvas_plan = [entry.metadata.get('canvas_layout') for entry in entries]
        if any(canvas_plan) and not all(canvas_plan):
            raise ValueError('Incomplete canvas layout')
        if all(canvas_plan):
            writing_width = canvas_plan[0]['writing_width_m']
            writing_height = canvas_plan[0]['writing_height_m']
            font_size = max(item['font_size_m'] for item in canvas_plan)
        else:
            writing_width, writing_height = self._fixed_writing_area()
            font_size = self._validated_font_size(font_size_m, len(entries), layout_mode)
        character_gap = max(0.0, float(self._param("character_gap_m")))
        total_axis_length = (
            len(entries) * font_size + max(0, len(entries) - 1) * character_gap
        )

        self._active_character = text
        self._active_layout_mode = layout_mode
        self._active_sample_id = ",".join(entry.sample_id or "" for entry in entries)
        self._active_target_path = entries[0].target_image or Path(
            str(self._param("target_image"))
        ).expanduser()
        self._active_target_is_normalized = bool(entries[0].target_is_normalized)
        self._active_evaluation_dir = (
            Path(str(self._param("evaluation_output_dir"))).expanduser()
            / character_directory_name(text)
            / f"font_{font_size * 1000.0:.0f}mm"
        )
        self._active_entry = entries[0] if len(entries) == 1 else None
        self._targets = []
        self._target_times = []
        self._dynamic_states = []
        self._footprints = []
        self._published = False
        self._execution_started_ns = None
        self._last_execution_index = -1
        self._ink_last_index = -1
        self._ink_present = False
        self._clear_ink()

        all_source: List[SourcePoint] = []
        all_dense: List[BrushPoint] = []
        digests: List[str] = []
        paper_offset_x = float(self._param("paper_offset_x_m"))
        paper_offset_y = float(self._param("paper_offset_y_m"))
        for index, entry in enumerate(entries):
            if entry.trajectory_csv is None or entry.target_image is None or not entry.sample_id:
                raise RuntimeError(
                    f"trajectory entry for {entry.character!r} is incomplete"
                )
            if (
                entry.metadata.get("coordinate_frame") != "glyph_center_m"
                or entry.metadata.get("xy_unit") != "m"
                or not entry.metadata.get("offline_full_pose_inversion")
                or not (entry.metadata.get("shape_preserving_v42_fused_pose")
                        or is_joint_contract(entry.metadata))
            ):
                raise RuntimeError(
                    f"{entry.character!r} lacks a supported offline physical-pose contract"
                )
            source, digest = load_points(
                entry.trajectory_csv, entry.character, entry.sample_id
            )
            x_span = max(point.x for point in source) - min(point.x for point in source)
            y_span = max(point.y for point in source) - min(point.y for point in source)
            if max(x_span, y_span) > font_size * 1.25:
                raise RuntimeError(
                    f"offline trajectory for {entry.character!r} exceeds its "
                    f"{font_size * 1000:.0f} mm font-size envelope"
                )
            required = str(self._param("required_sha256"))
            if required and len(entries) == 1 and digest.lower() != required.lower():
                raise RuntimeError(
                    f"trajectory SHA256 mismatch: {digest} != {required}"
                )
            if layout_mode == "horizontal":
                cell_center_x = (
                    paper_offset_x
                    - total_axis_length / 2.0
                    + font_size / 2.0
                    + index * (font_size + character_gap)
                )
                cell_center_y = paper_offset_y
            else:
                # World/base-frame Y grows away from the robot and maps to
                # the top of the page.  First input character is therefore
                # placed at the far/top cell, then proceeds downward.
                cell_center_x = paper_offset_x
                cell_center_y = (
                    paper_offset_y
                    + total_axis_length / 2.0
                    - font_size / 2.0
                    - index * (font_size + character_gap)
                )
            if all(canvas_plan):
                cell_center_x = paper_offset_x + canvas_plan[index]['center_x_m']
                cell_center_y = paper_offset_y + canvas_plan[index]['center_y_m']
            mapped = place_physical_source_points(
                source,
                paper_z=float(self._param("paper_z_m")),
                paper_offset_x=cell_center_x,
                paper_offset_y=cell_center_y,
                max_compression=float(self._param("max_brush_compression_m")),
                lift_height=float(self._param("lift_height_m")),
            )
            stroke_offset = index * 100000
            mapped = [replace(point, stroke_id=point.stroke_id + stroke_offset) for point in mapped]
            stream_path = Path(entry.metadata['neural_ink_path'])
            audit = json.loads(Path(entry.metadata['neural_ink_audit']).read_text())
            if hashlib.sha256(stream_path.read_bytes()).hexdigest() != audit['stream_sha256']:
                raise RuntimeError('Neural ink cache hash mismatch')
            if digest != audit['physical_csv_sha256']:
                raise RuntimeError('Neural ink does not match physical trajectory')
            with np.load(stream_path, allow_pickle=False) as stream:
                shift = np.array([cell_center_x, cell_center_y])
                self._neural_jobs.append(dict(
                    frames=stream['frames'], xy=stream['xy_m'] + shift,
                    ids=stream['stroke_ids'], origin=stream['origin_m'] + shift,
                    pixel_size=float(stream['pixel_size_m']), offset=stroke_offset,
                    last_frame=-1, marker=None, audit=audit, character=entry.character))
            character_dense = self._densify_mapped(mapped)
            if all_dense and character_dense:
                previous = all_dense[-1]
                first = character_dense[0]
                lift = BrushPoint(
                    previous.x,
                    previous.y,
                    max(previous.z, first.z) + float(self._param("lift_height_m")),
                    previous.alpha,
                    previous.beta,
                    previous.gamma,
                    3,
                    previous.stroke_id,
                )
                all_dense.extend(_interpolate(previous, lift, float(self._param("max_step_m"))))
                all_dense.extend(_interpolate(lift, first, float(self._param("max_step_m"))))
                all_dense.extend(character_dense[1:])
            else:
                all_dense.extend(character_dense)
            all_source.extend(source)
            digests.append(f"{entry.character}:{digest}")

        # Use the same elbow-up branch as the validator so the full trajectory
        # stays on a continuous, paper-safe configuration branch.
        seed = np.asarray(SAFE_INITIAL_JOINTS, dtype=np.float64)
        minimum_singularity_margin = max(
            0.0, float(self._param("minimum_singularity_margin"))
        )
        for point in all_dense:
            target = _flange_target(point, float(self._param("brush_length_m")))
            # State 3 points already describe the correct pen-up motion:
            # lift vertically, travel above the paper, then descend at the
            # next stroke start.  Do not return to the folded initial posture
            # between ordinary strokes.  The fallback below inserts a folded
            # detour only if that specific pen-up transition is obstructed.
            # Keep the continuous solution whenever it is safe.  If the
            # elbow-down branch would sweep through the table, try nearby
            # elbow/wrist branches before accepting any target.  This keeps
            # the motion continuous while making obstacle avoidance part of
            # IK selection rather than an after-the-fact warning.
            candidates = [
                seed,
                np.array([seed[0], seed[1], -seed[2], seed[3], seed[4], seed[5]]),
                np.array([seed[0], -1.2, -1.2, -1.5, -1.57, 0.0]),
                np.array([0.0, -1.2, 1.2, -1.5, -1.57, 0.0]),
                np.array([0.0, -1.4, 1.8, -2.1, -1.57, 0.0]),
                np.array([math.pi, -1.2, 1.2, -1.5, -1.57, 0.0]),
                np.array([math.pi, -1.2, -1.2, -1.5, 1.57, 0.0]),
                np.array([0.0, -0.6, 2.0, -2.4, -1.57, 0.0]),
                np.array([0.0, -2.0, -1.8, -1.0, 1.57, 0.0]),
                np.zeros(6),
            ]
            solved = None
            first_solved = None
            for candidate in candidates:
                result = solve_ur10_ik(target, candidate)
                if result[3]:
                    candidate_joints = np.asarray(result[0], dtype=np.float64)
                    if (
                        _joint_singularity_margin(candidate_joints)
                        < minimum_singularity_margin
                    ):
                        continue
                    if first_solved is None:
                        first_solved = result
                    if self._motion_obstacle_collision(seed, candidate_joints) is None:
                        solved = result
                        break
            if solved is None:
                # A direct transition can be blocked even when both endpoint
                # poses are valid (typically while crossing from one stroke
                # to the next).  Route through a folded, collision-free joint
                # pose before trying another IK branch.
                routed = False
                for parking_values in PARKING_JOINT_SEEDS:
                    parking = np.asarray(parking_values, dtype=np.float64)
                    if self._motion_obstacle_collision(seed, parking) is not None:
                        continue
                    for candidate in [parking] + candidates:
                        result = solve_ur10_ik(target, candidate)
                        if not result[3]:
                            continue
                        candidate_joints = np.asarray(result[0], dtype=np.float64)
                        if (
                            _joint_singularity_margin(candidate_joints)
                            < minimum_singularity_margin
                        ):
                            continue
                        if self._motion_obstacle_collision(parking, candidate_joints) is not None:
                            continue
                        park_point = BrushPoint(
                            point.x,
                            point.y,
                            max(
                                point.z,
                                float(self._param("paper_z_m"))
                                + float(self._param("lift_height_m")),
                            ),
                            point.alpha,
                            point.beta,
                            point.gamma,
                            3,
                            point.stroke_id,
                        )
                        self._targets.append(
                            JointTarget(
                                park_point,
                                tuple(float(value) for value in parking),
                                float(self._param("air_interval_s")),
                            )
                        )
                        seed = parking
                        solved = result
                        routed = True
                        self.get_logger().info(
                            "Beta obstacle detour: routed one target through "
                            "a folded parking pose"
                        )
                        break
                    if routed:
                        break
            if solved is None:
                # Numerical IK is local.  At the edge of a large paper cell,
                # the usual seeds can all converge to the same elbow branch.
                # Use a deterministic bounded search for another branch, but
                # accept it only when the complete joint transition is clear.
                random = np.random.default_rng(19082026 + len(self._targets))
                for _ in range(48):
                    random_seed = random.uniform(-math.pi, math.pi, size=6)
                    result = solve_ur10_ik(target, random_seed, max_iterations=120)
                    if not result[3]:
                        continue
                    candidate_joints = np.asarray(result[0], dtype=np.float64)
                    if (
                        _joint_singularity_margin(candidate_joints)
                        < minimum_singularity_margin
                    ):
                        continue
                    if self._motion_obstacle_collision(seed, candidate_joints) is None:
                        solved = result
                        break
            if solved is None:
                result = first_solved or solve_ur10_ik(target, seed)
                result_margin = _joint_singularity_margin(result[0])
                if result_margin < minimum_singularity_margin:
                    raise RuntimeError(
                        "Beta singularity guard rejected IK at "
                        f"stroke={point.stroke_id}: margin={result_margin:.4f} "
                        f"< {minimum_singularity_margin:.4f}"
                    )
                if bool(self._param("strict_ik")):
                    if first_solved is not None:
                        collision = self._motion_obstacle_collision(seed, first_solved[0])
                        raise RuntimeError(
                            f"UR10 transition rejected at stroke={point.stroke_id}: "
                            f"IK converged but no collision-free route was found; collision={collision}"
                        )
                    raise RuntimeError(
                        f"strict IK failed at stroke={point.stroke_id}: "
                        f"position={result[1]:.6f} m orientation={result[2]:.6f} rad"
                    )
                solved = result
            seed = solved[0]
            duration = (
                float(self._param("air_interval_s"))
                if point.state == 3
                else float(self._param("contact_interval_s"))
            )
            self._targets.append(
                JointTarget(point, tuple(float(value) for value in seed), duration)
            )

        if self._targets:
            # Keep the fake robot at the known safe posture while the
            # trajectory controller accepts the command, then make the first
            # approach slow enough to be visible and physically plausible.
            first = self._targets[0]
            first_point = first.point
            safe_point = BrushPoint(
                first_point.x,
                first_point.y,
                max(
                    first_point.z,
                    float(self._param("paper_z_m"))
                    + float(self._param("lift_height_m")),
                ),
                first_point.alpha,
                first_point.beta,
                first_point.gamma,
                3,
                first_point.stroke_id,
            )
            self._targets[0] = JointTarget(
                first.point,
                first.joints,
                max(
                    first.duration_s,
                    float(self._param("initial_transition_s")),
                ),
            )
            self._targets.insert(
                0,
                JointTarget(
                    safe_point,
                    tuple(float(value) for value in SAFE_INITIAL_JOINTS),
                    0.25,
                ),
            )

        self._unwrap_joint_targets()
        self._validate_obstacle_clearance()
        self._retime_joint_targets()
        paper_width = float(self._param("paper_width_m"))
        paper_height = float(self._param("paper_height_m"))
        self.get_logger().info(
            f"prepared {len(self._targets)} ROS 2 joint targets for {text!r}; "
            f"layout={layout_mode}, "
            f"paper={paper_width * 1000:.0f}x{paper_height * 1000:.0f} mm, "
            f"writing_area={writing_width * 1000:.0f}x{writing_height * 1000:.0f} mm, "
            f"font_size={font_size * 1000:.1f} mm (offline V16 inversion), "
            f"gap={character_gap * 1000:.0f} mm, "
            f"sha256={'; '.join(digests)}"
        )
        for job in self._neural_jobs:
            job['schedule'] = schedule_frames(job['xy'], job['ids'], self._targets, job['offset'])
        offline_pose = _offline_pose_summary(
            [target.point for target in self._targets]
        )
        if self._neural_jobs:
            self._evaluation_report = {
                "mode": "beta_multi_character",
                "characters": text,
                "character_count": len(entries),
                "layout_mode": layout_mode,
                "paper_width_m": paper_width,
                "paper_height_m": paper_height,
                "writing_width_m": writing_width,
                "writing_height_m": writing_height,
                "character_gap_m": character_gap,
                "font_size_m": font_size,
                "offline_full_pose_inversion": True,
                "ros_runtime_scaling": False,
                "trajectory_sha256": digests,
                "offline_pose": offline_pose,
                "ink_renderer": "shared V16 PaperFusionRenderer",
                "forward_audits": [{"character": job['character'], **job['audit']} for job in self._neural_jobs],
                "evaluation_scope": "Commanded CSV prediction against synthesized shape target; not measured contact or independent target accuracy",
                "message": "Beta 多字布局已生成；每个字使用离线 V16 字号反演轨迹。",
            }
            self._evaluation_pub.publish(
                String(data=json.dumps(self._evaluation_report, ensure_ascii=False))
            )
            self._active_evaluation_dir.mkdir(parents=True, exist_ok=True)
            (self._active_evaluation_dir / 'shared_forward_audit.json').write_text(
                json.dumps(self._evaluation_report, ensure_ascii=False, indent=2), encoding='utf-8')
        # Do not publish the complete planned path.  The only visible path is
        # the ink accumulated by _publish_ink while the controller executes.

    def _build_flexible_footprints(self) -> None:
        """Run the dynamic brush state and virtual B-BSM pose bridge."""

        self._dynamic_states = [None] * len(self._targets)
        self._footprints = [None] * len(self._targets)
        previous: Optional[DynamicFootprintState] = None
        previous_stroke: Optional[int] = None
        samples = int(self._param("brush_bezier_samples"))
        offline_pose_contacts = 0
        for index, target in enumerate(self._targets):
            point = target.point
            if point.state == 3:
                previous = None
                previous_stroke = None
                continue
            if previous is None or previous_stroke != point.stroke_id:
                state = self._brush_model.initial_state(
                    x_m=point.x,
                    y_m=point.y,
                    press_depth_mm=point.press_depth_mm,
                    alpha_rad=point.alpha,
                    beta_rad=point.beta,
                    heading_rad=point.gamma,
                )
            else:
                state = self._brush_model.step(
                    previous,
                    x_m=point.x,
                    y_m=point.y,
                    press_depth_mm=point.press_depth_mm,
                    alpha_rad=point.alpha,
                    beta_rad=point.beta,
                    heading_rad=point.gamma,
                )
            offline_pose_contacts += 1
            self._dynamic_states[index] = state
            if state.dimensions.half_width_m > 1.0e-9:
                polygon = self._brush_model.polygon(
                    state, bezier_samples=samples
                )
                # Geometry comes entirely from the executed offline V16 pose.
                self._footprints[index] = polygon
            previous = state
            previous_stroke = point.stroke_id

        rendered = sum(footprint is not None for footprint in self._footprints)
        self.get_logger().info(
            "flexible brush ready: "
            f"states={rendered}, bundle=48 mm, radius=6 mm, "
            f"Kw=Kd=0.02, offline_pose_contacts={offline_pose_contacts}, "
            "fields=z/alpha/beta/gamma"
        )

    def _evaluate_trajectory(
        self, source: Sequence[SourcePoint], trajectory_sha256: str
    ) -> None:
        """Evaluate the final simulated ink against the independent target glyph."""

        target_path = self._active_target_path
        image_size = int(self._param("evaluation_image_size"))
        executed_points = [target.point for target in self._targets]
        violations = [
            {
                "stroke_id": point.stroke_id,
                "point_id": point.point_id,
                "h_mm": point.press_depth_mm,
                "alpha_rad": point.alpha,
                "beta_rad": point.beta,
            }
            for point in executed_points
            if point.state != 3
            and not self._brush_parameters.point_in_calibrated_domain(
                point.press_depth_mm, point.alpha, point.beta
            )
        ]
        try:
            raw_simulated = render_world_footprints(
                (p for p in self._footprints if p is not None),
                paper_width_m=float(self._param("paper_width_m")),
                paper_height_m=float(self._param("paper_height_m")),
                paper_offset_x_m=float(self._param("paper_offset_x_m")),
                paper_offset_y_m=float(self._param("paper_offset_y_m")),
                image_size=image_size,
                supersampling=2,
            )
            margin_ratio = float(self._param("paper_margin_m")) / float(
                self._param("paper_width_m")
            )
            # The paper evaluates images only after cropping and alignment.
            simulated = normalize_ink_mask(
                raw_simulated,
                image_size=image_size,
                margin_ratio=margin_ratio,
            )
            if self._active_target_is_normalized:
                gray = Image.open(target_path).convert("L")
                target = 1.0 - np.asarray(gray, dtype=np.float32) / 255.0
                if target.shape != (image_size, image_size):
                    raise ValueError(
                        "normalized catalog target must have shape "
                        f"{image_size}x{image_size}, got {target.shape}"
                    )
            else:
                target = load_normalized_target(
                    target_path,
                    image_size=image_size,
                    margin_ratio=margin_ratio,
                )
            report = evaluate_masks(
                simulated,
                target,
                minimum_csim=float(self._param("minimum_csim")),
                minimum_ssim=float(self._param("minimum_ssim")),
                calibrated_domain_valid=not violations,
            )
            report.update(
                {
                    "character": self._active_character,
                    "sample_id": self._active_sample_id,
                    "trajectory_sha256": trajectory_sha256,
                    "target_image": str(target_path),
                    "image_size": image_size,
                    "source_point_count": len(source),
                    "rendered_dynamic_footprint_count": sum(
                        p is not None for p in self._footprints
                    ),
                    "offline_pose": _offline_pose_summary(
                        executed_points
                    ),
                    "ros_runtime_scaling": False,
                    "parameter_domain_violations": violations,
                    "evaluation_alignment": "foreground crop and 128x128 alignment",
                    "brush_parameters": parameters_as_dict(self._brush_parameters),
                    "model_chain": [
                        "complete target image scaled for requested physical font size",
                        "V16 B-BSMG checkpoint (SHA256 verified)",
                        "v42 fused-pose: fixed scaled x/y, optimized H",
                        "alpha/beta/gamma derived from dynamic brush geometry",
                        "physical glyph-centred x/y translated by ROS without resizing",
                        "dynamic w/d/o/theta with inertia and friction-snap",
                        "symmetric cubic-Bezier B-BSM footprint",
                    ],
                }
            )
            report_path, panel_path = save_evaluation_artifacts(
                self._active_evaluation_dir,
                simulated,
                target,
                report,
            )
            report["report_path"] = str(report_path)
            report["comparison_path"] = str(panel_path)
            # Rewrite once so the report contains its own artifact paths.
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self._evaluation_report = report
            self._evaluation_pub.publish(
                String(data=json.dumps(report, ensure_ascii=False))
            )
            self.get_logger().info(
                "brush trajectory evaluation: "
                f"CSIM={report['csim']:.4f}, SSIM={report['ssim']:.4f}, "
                f"feasible={str(report['feasible']).lower()}, report={report_path}"
            )
        except Exception as error:
            self._evaluation_report = {
                "feasible": False,
                "error": str(error),
                "target_image": str(target_path),
            }
            self._evaluation_pub.publish(
                String(data=json.dumps(self._evaluation_report, ensure_ascii=False))
            )
            self.get_logger().error(f"brush trajectory evaluation failed: {error}")

    def _publish_preview(self, points: Sequence[BrushPoint]) -> None:
        poses = PoseArray()
        poses.header.frame_id = str(self._param("frame_id"))
        markers = MarkerArray()
        stroke_points: List[Point] = []

        def append_stroke() -> None:
            if not stroke_points:
                return
            marker = Marker()
            marker.header.frame_id = poses.header.frame_id
            marker.ns = "ros2_brush_preview"
            marker.id = len(markers.markers)
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.0015
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = (
                0.35,
                0.55,
                0.75,
                0.55,
            )
            marker.points = list(stroke_points)
            markers.markers.append(marker)
            stroke_points.clear()

        for point in points:
            if point.state == 3:
                append_stroke()
                continue
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = point.x, point.y, point.z
            poses.poses.append(pose)
            msg_point = Point(
                x=point.x,
                y=point.y,
                z=float(self._param("paper_z_m")) + 0.0015,
            )
            stroke_points.append(msg_point)
        append_stroke()
        self._pose_pub.publish(poses)
        self._preview_pub.publish(markers)

    def _republish_ink(self) -> None:
        """Keep the accumulated real-time ink available to RViz."""
        if self._ink_markers.markers:
            self._ink_pub.publish(self._ink_markers)
        if self._evaluation_report is not None:
            self._evaluation_pub.publish(
                String(data=json.dumps(self._evaluation_report, ensure_ascii=False))
            )

    def _publish_once(self, *, force: bool = False) -> None:
        if self._published or (not force and not bool(self._param("publish_on_start"))):
            return
        if self._trajectory_pub.get_subscription_count() == 0:
            return
        trajectory = JointTrajectory()
        trajectory.joint_names = JOINT_NAMES
        elapsed = 0.0
        self._target_times.clear()
        for target in self._targets:
            elapsed += target.duration_s
            self._target_times.append(elapsed)
            point = JointTrajectoryPoint()
            point.positions = list(target.joints)
            point.time_from_start.sec = int(elapsed)
            point.time_from_start.nanosec = int((elapsed - int(elapsed)) * 1e9)
            trajectory.points.append(point)
        self._trajectory_pub.publish(trajectory)
        self._execution_started_ns = self.get_clock().now().nanoseconds
        self._last_execution_index = -1
        self._published = True
        with self._job_lock:
            self._job_state.update(
                {
                    "status": "running",
                    "character": self._active_character,
                    "sample_id": self._active_sample_id,
                    "progress": 0.0,
                    "message": "机械臂正在书写",
                }
            )
        self.get_logger().info(
            f"published {len(trajectory.points)} points to {self._param('controller_topic')}"
        )

    def _publish_execution_state(self) -> None:
        if self._execution_started_ns is None or not self._target_times:
            return
        elapsed = (self.get_clock().now().nanoseconds - self._execution_started_ns) * 1e-9
        index = min(bisect.bisect_right(self._target_times, elapsed) - 1, len(self._targets) - 1)
        if index != self._last_execution_index:
            target = self._targets[index]
            self._state_pub.publish(UInt8(data=int(target.point.state)))
            progress = (index + 1) / len(self._targets)
            self._progress_pub.publish(Float32(data=progress))
            with self._job_lock:
                self._job_state["progress"] = progress
            self._publish_ink(index)
            self._last_execution_index = index
        if elapsed >= self._target_times[-1]:
            self._progress_pub.publish(Float32(data=1.0))
            self._execution_started_ns = None
            with self._job_lock:
                self._job_state.update(
                    {
                        "status": "completed",
                        "progress": 1.0,
                        "message": "当前字符书写完成",
                    }
                )

    def _publish_ink(self, last_index: int) -> None:
        """Replay unchanged neural transmittance masks at commanded progress."""
        if last_index <= self._ink_last_index:
            return
        changed = MarkerArray()
        for glyph_index, job in enumerate(self._neural_jobs):
            frame_index = int(np.searchsorted(job['schedule'], last_index, side='right')) - 1
            if frame_index <= job['last_frame']:
                continue
            quads, opacity = frame_quads(job['frames'][frame_index], job['origin'], job['pixel_size'])
            if len(opacity) == 0:
                job['last_frame'] = frame_index
                continue
            marker = Marker()
            marker.header.frame_id = str(self._param('frame_id'))
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'shared_v16_ink'
            marker.id = glyph_index
            marker.type = Marker.TRIANGLE_LIST
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = marker.scale.z = 1.0
            marker.color.a = 1.0
            z = float(self._param('paper_z_m')) + 0.0018
            marker.points = [Point(x=float(v[0]), y=float(v[1]), z=z) for quad in quads for v in quad]
            marker.colors = [ColorRGBA(r=0.0, g=0.0, b=0.0, a=float(alpha)) for alpha in opacity for _ in range(6)]
            job.update(last_frame=frame_index, marker=marker)
            changed.markers.append(marker)
        self._ink_last_index = last_index
        self._ink_markers.markers = [job['marker'] for job in self._neural_jobs if job['marker'] is not None]
        if changed.markers:
            self._ink_present = True
            self._ink_pub.publish(changed)

    def _publish_legacy_polygon_ink(self, last_index: int) -> None:
        """Publish only newly deposited dynamic B-BSM footprints."""

        if last_index <= self._ink_last_index:
            return
        paper_surface = float(self._param("paper_z_m")) + 0.0018
        now = self.get_clock().now().to_msg()
        incremental = MarkerArray()
        for index in range(self._ink_last_index + 1, last_index + 1):
            footprint = self._footprints[index]
            if footprint is None:
                continue
            target = self._targets[index]
            marker = Marker()
            marker.header.frame_id = str(self._param("frame_id"))
            marker.header.stamp = now
            marker.ns = f"ros2_flexible_brush_stroke_{target.point.stroke_id}"
            marker.id = index
            marker.type = Marker.TRIANGLE_LIST
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = marker.scale.z = 1.0
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = (
                0.025,
                0.025,
                0.025,
                1.0,
            )
            triangles = triangle_fan(footprint)
            marker.points = [
                Point(x=float(vertex[0]), y=float(vertex[1]), z=paper_surface)
                for triangle in triangles
                for vertex in triangle
            ]
            incremental.markers.append(marker)
            self._ink_markers.markers.append(marker)
        self._ink_last_index = last_index
        if incremental.markers:
            self._ink_present = True
            self._ink_pub.publish(incremental)


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = BrushTrajectoryDriver()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node._input_page is not None:
            node._input_page.close()
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
