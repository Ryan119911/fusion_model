"""Launch the UR10 brush driver with official ros2_control fake hardware."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, EmitEvent, LogInfo
from launch.events import Shutdown
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _after_success(next_action):
    def on_exit(event, context):
        if event.returncode != 0:
            return [LogInfo(msg="ERROR: Controller startup failed; writing is disabled."),
                    EmitEvent(event=Shutdown(reason="Controller startup failed"))]
        return [next_action]
    return on_exit


def generate_launch_description() -> LaunchDescription:
    package_share = Path(get_package_share_directory("fusion_model_ros2_beta"))
    robot_description = (package_share / "urdf" / "ur10_official.urdf").read_text()
    rviz_config = str(package_share / "rviz" / "ur10_brush.rviz")
    controller_manager_config = str(
        package_share / "config" / "controller_manager.yaml"
    )
    joint_state_config = str(
        package_share / "config" / "joint_state_broadcaster.yaml"
    )
    trajectory_controller_config = str(
        package_share / "config" / "joint_trajectory_controller.yaml"
    )
    default_csv = (
        "/home/robot/coppeliasim/machine_learning/model/outputs/"
        "kaishu_pose_v16_all_fields_footprint_wu/inversion_trajectory.csv"
    )
    default_target = (
        "/home/robot/coppeliasim/machine_learning/model/data/raw/targets/wu_kaishu_target.png"
    )
    default_catalog_root = (
        "/home/robot/coppeliasim/machine_learning/model/outputs/"
        "kaishu_pose_trajectory_batch_v1"
    )
    default_database_csv = (
        "/home/robot/coppeliasim/machine_learning/model/data/raw/data.csv"
    )
    default_evaluation_dir = "/home/robot/ros2_ws/evaluation/fusion_model_ros2_beta"
    default_target_overrides = (
        '{"武":"/home/robot/coppeliasim/machine_learning/model/data/raw/targets/'
        'wu_kaishu_target.png"}'
    )
    controller_manager_name = "fusion_beta_controller_manager"
    state_broadcaster_name = "fusion_beta_joint_state_broadcaster"
    trajectory_controller_name = "fusion_beta_joint_trajectory_controller"
    controller_topic = f"/{trajectory_controller_name}/joint_trajectory"
    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        # A global __node remap is inherited by in-process controllers on Humble.
        # Scope the rename to the manager so controller private topics stay distinct.
        remappings=[("controller_manager:__node", controller_manager_name)],
        output="screen",
        parameters=[
            {"robot_description": robot_description},
            controller_manager_config,
        ],
    )
    joint_state_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            state_broadcaster_name,
            "--controller-manager",
            f"/{controller_manager_name}",
            "--param-file",
            joint_state_config,
            "--controller-manager-timeout",
            "120",
        ],
        output="screen",
    )
    trajectory_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            trajectory_controller_name,
            "--controller-manager",
            f"/{controller_manager_name}",
            "--param-file",
            trajectory_controller_config,
            "--controller-manager-timeout",
            "120",
        ],
        output="screen",
    )
    brush_driver = Node(
        package="fusion_model_ros2_beta",
        executable="brush_trajectory_driver",
        name="brush_trajectory_driver_beta",
        output="screen",
        parameters=[
            {"trajectory_csv": LaunchConfiguration("trajectory_csv")},
            {"controller_topic": LaunchConfiguration("controller_topic")},
            {"required_sha256": LaunchConfiguration("required_sha256")},
            {
                "publish_on_start": ParameterValue(
                    LaunchConfiguration("publish_on_start"), value_type=bool
                )
            },
            {"target_image": LaunchConfiguration("target_image")},
            {
                "trajectory_catalog_root": LaunchConfiguration(
                    "trajectory_catalog_root"
                )
            },
            {
                "allow_v17_ineligible": ParameterValue(
                    LaunchConfiguration("allow_v17_ineligible"), value_type=bool
                )
            },
            {"database_csv": LaunchConfiguration("database_csv")},
            {
                "database_style": ParameterValue(
                    LaunchConfiguration("database_style"), value_type=str
                )
            },
            {
                "target_overrides_json": ParameterValue(
                    LaunchConfiguration("target_overrides_json"), value_type=str
                )
            },
            {
                "enable_input_page": LaunchConfiguration("enable_input_page")
            },
            {"input_page_host": LaunchConfiguration("input_page_host")},
            {"input_page_port": LaunchConfiguration("input_page_port")},
            {
                "evaluation_output_dir": LaunchConfiguration(
                    "evaluation_output_dir"
                )
            },
            {"paper_width_m": LaunchConfiguration("paper_width_m")},
            {"paper_height_m": LaunchConfiguration("paper_height_m")},
            {"paper_z_m": LaunchConfiguration("paper_z_m")},
            {"paper_thickness_m": LaunchConfiguration("paper_thickness_m")},
            {"paper_offset_x_m": LaunchConfiguration("paper_offset_x_m")},
            {"paper_offset_y_m": LaunchConfiguration("paper_offset_y_m")},
            {"character_gap_m": LaunchConfiguration("character_gap_m")},
            {"max_text_characters": LaunchConfiguration("max_text_characters")},
            {"default_layout_mode": LaunchConfiguration("default_layout_mode")},
            {"writing_width_m": LaunchConfiguration("writing_width_m")},
            {"writing_height_m": LaunchConfiguration("writing_height_m")},
            {"default_font_size_m": LaunchConfiguration("default_font_size_m")},
            {
                "offline_reference_font_size_m": LaunchConfiguration(
                    "offline_reference_font_size_m"
                )
            },
            {"offline_model_root": LaunchConfiguration("offline_model_root")},
            {
                "offline_python_executable": LaunchConfiguration(
                    "offline_python_executable"
                )
            },
            {
                "offline_bbsmg_checkpoint": LaunchConfiguration(
                    "offline_bbsmg_checkpoint"
                )
            },
            {
                "offline_bbsmg_checkpoint_sha256": LaunchConfiguration(
                    "offline_bbsmg_checkpoint_sha256"
                )
            },
            {"offline_cache_root": LaunchConfiguration("offline_cache_root")},
            {"offline_inversion_backend": LaunchConfiguration("offline_inversion_backend")},
            {"offline_joint_reference_manifest": ParameterValue(LaunchConfiguration("offline_joint_reference_manifest"), value_type=str)},
            {"offline_original_target_manifest": ParameterValue(LaunchConfiguration("offline_original_target_manifest"), value_type=str)},
            {"offline_joint_max_steps": LaunchConfiguration("offline_joint_max_steps")},
            {"offline_joint_foreground_weight": ParameterValue(LaunchConfiguration("offline_joint_foreground_weight"), value_type=float)},
            {"offline_joint_target_skeleton_weight": ParameterValue(LaunchConfiguration("offline_joint_target_skeleton_weight"), value_type=float)},
            {"offline_joint_target_skeleton_max_distance_px": ParameterValue(LaunchConfiguration("offline_joint_target_skeleton_max_distance_px"), value_type=float)},
            {"offline_joint_target_skeleton_threshold": ParameterValue(LaunchConfiguration("offline_joint_target_skeleton_threshold"), value_type=float)},
            {
                "offline_inversion_timeout_s": LaunchConfiguration(
                    "offline_inversion_timeout_s"
                )
            },
            {
                "offline_inversion_device": LaunchConfiguration(
                    "offline_inversion_device"
                )
            },
            {"table_width_m": LaunchConfiguration("table_width_m")},
            {"table_depth_m": LaunchConfiguration("table_depth_m")},
            {"table_top_z_m": LaunchConfiguration("table_top_z_m")},
            {"obstacle_clearance_m": LaunchConfiguration("obstacle_clearance_m")},
            {"brush_length_m": LaunchConfiguration("brush_length_m")},
            {"lift_height_m": LaunchConfiguration("lift_height_m")},
            {
                "initial_transition_s": LaunchConfiguration(
                    "initial_transition_s"
                )
            },
            {"max_joint_speed_rad_s": LaunchConfiguration("max_joint_speed_rad_s")},
            {
                "minimum_singularity_margin": LaunchConfiguration(
                    "minimum_singularity_margin"
                )
            },
            {
                "strict_ik": ParameterValue(
                    LaunchConfiguration("strict_ik"), value_type=bool
                )
            },
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("trajectory_csv", default_value=default_csv),
            DeclareLaunchArgument("target_image", default_value=default_target),
            DeclareLaunchArgument(
                "trajectory_catalog_root", default_value=default_catalog_root
            ),
            DeclareLaunchArgument("allow_v17_ineligible", default_value="false"),
            DeclareLaunchArgument("database_csv", default_value=default_database_csv),
            DeclareLaunchArgument("database_style", default_value="楷"),
            DeclareLaunchArgument(
                "target_overrides_json", default_value=default_target_overrides
            ),
            DeclareLaunchArgument("required_sha256", default_value=""),
            DeclareLaunchArgument(
                "evaluation_output_dir", default_value=default_evaluation_dir
            ),
            DeclareLaunchArgument("paper_width_m", default_value="0.80"),
            DeclareLaunchArgument("paper_height_m", default_value="0.40"),
            DeclareLaunchArgument("paper_z_m", default_value="0.0"),
            DeclareLaunchArgument("paper_thickness_m", default_value="0.002"),
            DeclareLaunchArgument("paper_offset_x_m", default_value="0.0"),
            DeclareLaunchArgument("paper_offset_y_m", default_value="-0.65"),
            DeclareLaunchArgument("character_gap_m", default_value="0.045"),
            DeclareLaunchArgument("max_text_characters", default_value="3"),
            DeclareLaunchArgument("default_layout_mode", default_value="horizontal"),
            DeclareLaunchArgument("writing_width_m", default_value="0.52"),
            DeclareLaunchArgument("writing_height_m", default_value="0.32"),
            DeclareLaunchArgument("default_font_size_m", default_value="0.12"),
            DeclareLaunchArgument(
                "offline_reference_font_size_m", default_value="0.29"
            ),
            DeclareLaunchArgument(
                "offline_model_root",
                default_value="/home/robot/coppeliasim/machine_learning/model",
            ),
            DeclareLaunchArgument(
                "offline_python_executable",
                default_value="/home/robot/miniconda3/envs/ddpm/bin/python",
            ),
            DeclareLaunchArgument(
                "offline_bbsmg_checkpoint",
                default_value=(
                    "/home/robot/coppeliasim/machine_learning/model/outputs/"
                    "paper_bbsmg_general_v16_pose_dense/bbsmg_best.pt"
                ),
            ),
            DeclareLaunchArgument(
                "offline_bbsmg_checkpoint_sha256",
                default_value="30d5e0c37dc7b7913c02fea26930babafa46a4a07b5f698520df277a04d5b717",
            ),
            DeclareLaunchArgument(
                "offline_cache_root",
                default_value=(
                    "/home/robot/coppeliasim/machine_learning/model/outputs/"
                    "ros2_v16_fontsize_cache"
                ),
            ),
            DeclareLaunchArgument("offline_inversion_timeout_s", default_value="3600"),
            DeclareLaunchArgument("offline_inversion_device", default_value="cuda"),
            DeclareLaunchArgument("offline_inversion_backend", default_value="legacy_fused", choices=["legacy_fused", "joint_A", "joint_target"]),
            DeclareLaunchArgument("offline_joint_reference_manifest", default_value=""),
            DeclareLaunchArgument("offline_original_target_manifest", default_value=""),
            DeclareLaunchArgument("offline_joint_max_steps", default_value="16"),
            DeclareLaunchArgument("offline_joint_foreground_weight", default_value="1.0"),
            DeclareLaunchArgument("offline_joint_target_skeleton_weight", default_value="0.0"),
            DeclareLaunchArgument("offline_joint_target_skeleton_max_distance_px", default_value="12.0"),
            DeclareLaunchArgument("offline_joint_target_skeleton_threshold", default_value="0.35"),
            DeclareLaunchArgument("table_width_m", default_value="0.80"),
            DeclareLaunchArgument("table_depth_m", default_value="0.40"),
            DeclareLaunchArgument("table_top_z_m", default_value="-0.001"),
            DeclareLaunchArgument("obstacle_clearance_m", default_value="0.005"),
            DeclareLaunchArgument("brush_length_m", default_value="0.328"),
            DeclareLaunchArgument("lift_height_m", default_value="0.025"),
            DeclareLaunchArgument("initial_transition_s", default_value="1.5"),
            DeclareLaunchArgument("max_joint_speed_rad_s", default_value="0.60"),
            DeclareLaunchArgument(
                "minimum_singularity_margin", default_value="0.08"
            ),
            DeclareLaunchArgument("strict_ik", default_value="false"),
            DeclareLaunchArgument("controller_topic", default_value=controller_topic),
            DeclareLaunchArgument("publish_on_start", default_value="false"),
            DeclareLaunchArgument("enable_input_page", default_value="true"),
            DeclareLaunchArgument("input_page_host", default_value="127.0.0.1"),
            DeclareLaunchArgument("input_page_port", default_value="18082"),
            DeclareLaunchArgument("with_rviz", default_value="true"),
            DeclareLaunchArgument(
                "rviz_display",
                default_value=EnvironmentVariable("DISPLAY", default_value=":0"),
            ),
            DeclareLaunchArgument(
                "rviz_xauthority",
                default_value=EnvironmentVariable(
                    "XAUTHORITY", default_value="/run/user/1000/gdm/Xauthority"
                ),
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher_beta",
                output="screen",
                parameters=[{"robot_description": robot_description}],
                remappings=[
                    (
                        "/joint_states",
                        f"/{state_broadcaster_name}/joint_states",
                    )
                ],
            ),
            ros2_control_node,
            RegisterEventHandler(
                OnProcessExit(
                    target_action=joint_state_spawner,
                    on_exit=_after_success(trajectory_spawner),
                )
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=trajectory_spawner,
                    on_exit=_after_success(brush_driver),
                )
            ),
            joint_state_spawner,
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2_beta",
                output="screen",
                condition=IfCondition(LaunchConfiguration("with_rviz")),
                arguments=["-d", rviz_config],
                remappings=[
                    (
                        "/visualization_marker_array",
                        "/brush_trajectory_driver_beta/ink",
                    )
                ],
                additional_env={
                    "DISPLAY": LaunchConfiguration("rviz_display"),
                    "XAUTHORITY": LaunchConfiguration("rviz_xauthority"),
                    "QT_QPA_PLATFORM_PLUGIN_PATH": "/usr/lib/x86_64-linux-gnu/qt5/plugins",
                },
            ),
        ]
    )
