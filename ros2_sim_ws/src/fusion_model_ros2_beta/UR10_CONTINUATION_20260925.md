# UR10 simulation continuation, 2026-09-25

Target: /home/robot/ros2_ws/src/fusion_model_ros2_beta on MU.
Read the ROS 2 simulation task's recent original messages and handoff before editing.
Real robot reference: https://github.com/jinxiao123580-hub/UR10
Factory hash: calib_15120592593058779304; controller frame: base.

## Changes in this continuation

- Scope the controller_manager node rename to `controller_manager:__node`.
  A global rename on Humble also renamed loaded controller nodes, confirmed by
  runtime duplicate-name warnings and manager-named private topics.
- Publish to /fusion_beta_joint_trajectory_controller/joint_trajectory.
- Read local broadcaster states from /fusion_beta_joint_state_broadcaster/joint_states.
- Stop the launch when either controller spawner fails; do not start writing.
- Add test/test_ur10_launch_contract.py for these contracts and fake-hardware plugin.

## Validation and boundaries

21 tests passed (actual profile, launch contract, canvas layout).
Independent install_actual_ur10 build succeeded and launch arguments parsed.
No real robot connection, motion commands, package upgrades, or CoppeliaSim edits.
Test invocations need PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 because user-installed AnyIO
is incompatible with system pytest; no dependencies were changed. Preserve the
ROS PYTHONPATH when adding the package source path.

Backup and runtime logs: /home/robot/ros2_ws/backups/ur10_resume_20260925/.
Original launch SHA256: d8e690ffc07b0e8de69d4c05d39ef6fefe7c1d9d93b511b5d170398040622b70.

The prior 290 mm full inversion and whole-path writing acceptance are NOT certified
by these startup tests. Existing legacy_fused defaults and timer-based ink progress
remain unchanged; do not describe this simulation as feedback-verified execution or
as a ready-to-run physical robot controller. Attachment envelopes and brush TCP
still require physical measurement/validation before hardware use.
