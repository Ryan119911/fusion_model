# fusion_model_ros2_beta

Current acceptance status and reproduction: [VALIDATION_20260925.md](../../VALIDATION_20260925.md).
The actual-UR10 full-writing cases remain blocked by dense gamma feasibility and
camera-envelope clearance; the startup tests do not certify complete writing.

This package is the staging implementation for the formal
`fusion_model_ros2` package. It replays the existing
V16 paper-model trajectories through a ROS 2 `JointTrajectoryController` and
places multiple database characters in horizontal or vertical cells on a wider
paper. The original CoppeliaSim driver is not modified.

Model identity is explicit: the learned B-BSMG checkpoint is **V16**, while
`paper_psoc_lm_v42_fused_pose` is the inversion/output protocol, not a v42
trained model. Startup verifies the checkpoint SHA-256
`30d5e0c37dc7b7913c02fea26930babafa46a4a07b5f698520df277a04d5b717`.
The web status page exposes the weight version, protocol, checksum, and field
contract for every running process.

## Flexible brush model

The requested physical font size is produced before ROS trajectory planning:

1. Read the complete target image and V16 paper-frame initial pose
   `(x, y, H, alpha, beta, gamma)` from `char_uXXXX`.
2. Uniformly scale the complete target canvas to the requested physical font
   size and scale only the initial `x/y` geometry about the glyph centre.
3. Run the full catalog's `v42_fused_pose` inversion again: the uniformly
   scaled `x/y` centreline is fixed, `H` is optimized, and
   `alpha/beta/gamma` are re-derived from the dynamic brush geometry. This
   avoids the non-identifiable six-free-variable drift while still producing
   a complete new physical pose trajectory. Cache it under
   `outputs/ros2_v16_fontsize_cache`.
   The matching v42 rendered glyph is used as the shape target when available;
   this prevents a mislabeled source image such as simplified `汉` paired with
   traditional `漢` from deforming the trajectory.
4. ROS translates that glyph-centred physical trajectory into its horizontal
   or vertical page position. It does not normalize a glyph into a cell and
   does not alter `H` or any rotation angle.
5. During execution, update the dynamic state `(w, d, o, theta)`. Width and drag use `Kw = Kd =
   0.02`; paper friction holds the previous brush root until the deformation
   limit is exceeded, then the root snaps to the bounded location.
6. Deposit a filled footprint enclosed by the two symmetric cubic Bezier curves
   of the B-BSM model.

The default No. 1 brush parameters are the values reported in the supplied
paper: bundle length `48 mm`, radius `6 mm`, `H = 11..20 mm`, `alpha = 0..10
deg`, `beta = 0..5 deg`, and the Table 1 regression coefficients. The resulting
filled footprints are published incrementally, so ink appears only as the ROS 2
controller executes the stroke.

## Run

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --build-base build_actual_ur10 --install-base install_actual_ur10 \
  --packages-select fusion_model_ros2_beta --symlink-install
source install_actual_ur10/setup.bash
ros2 launch fusion_model_ros2_beta ur10_brush_ros2_beta.launch.py
```

## Physical UR10 profile

This simulation profile is pinned to the UR10 CB3 robot that will run the
trajectory in the physical system:

- robot serial: `2022300602`;
- controller software: `3.15.8.106339`;
- factory calibration hash: `calib_15120592593058779304`;
- official real-driver family: `ur_robot_driver` 2.14.0 on ROS 2 Humble;
- controller pose frame: `base` (not `base_link`, whose UR convention differs
  by a pi-yaw transform);
- simulated end stack: flange, ATI Net F/T sensor, Robotiq 2F gripper, measured
  220 mm `tool0`-to-grasp-centre offset, then the 108 mm brush;
- Mech-Eye PRO XS uses the validated hand-eye transform from the physical
  repository.

The factory joint origins are copied into both the URDF and the planner's
forward kinematics. Startup prints the robot serial and calibration hash, and
the status API exposes them under `robot_profile`. The ATI, gripper, and camera
solids are conservative collision envelopes because the physical repository
does not yet provide final measured CAD geometry for all attachments. They are
used to reject unsafe plans, not as millimetre-accurate visual meshes.

The Beta paper is `800 x 400 mm`. Its long edge is parallel to the X axis and
the UR10 robot is mounted on a pedestal centred outside the lower long edge.
The table's near edge is about `350 mm` from the pedestal envelope. The arm therefore approaches from the
long-edge side without passing through the tabletop or folding too close to the
writing area. Fake hardware starts in
the same folded safe posture used by the planner. The support table is
`800 x 400 mm`, the same size as the paper, so the paper has no overhang.
Between strokes the brush lifts `25 mm` and travels directly to the next
stroke; a folded joint-space detour is used only if that direct path is
obstructed. The first writing approach is
spread over at least `1.5 s`. All transitions are retimed to a maximum joint
speed of `0.60 rad/s`, and IK branches with an elbow/wrist singularity margin
below `0.08` are rejected. The default character
gap is `45 mm`, and the web page accepts one to three characters. The default
UR10-safe writing area is a `520 x 320 mm` region inside the larger paper. The
web page accepts the writing-area width and height; each physical character
size is derived from that area before the offline inversion. Horizontal
layout proceeds from left to right; vertical layout follows Chinese ordering
from the top/far side toward the bottom/robot side. Font-size changes are
therefore real V16 inversions of the complete target, including brush width;
they are not display-only polygon scaling. This also avoids the old `一`
failure, because ROS no longer normalizes an almost-constant Y axis into a
full-height cell.

Before a trajectory is published, Beta samples the interpolated joint motion
and checks every UR10 arm-link capsule against both the paper footprint and the
physical table footprint. The capsule radii approximate the official UR10 mesh,
instead of treating each link as a zero-width centre line. The brush tip may
contact the paper, but a UR10 link is never allowed to enter either surface's
keep-out volume. The default `obstacle_clearance_m` is `5 mm`; a colliding plan
is rejected instead of being sent to the controller. The paper and table centre
are kept coincident in the world model, while the robot pedestal remains
outside the lower long edge.

The accumulated ink is available on
`/brush_trajectory_driver_beta/ink`. The machine-readable evaluation is a
transient local message on `/brush_trajectory_driver_beta/evaluation`.

## Character input page

The launch starts a small standard-library HTTP page on the MU host:

```text
http://127.0.0.1:18082/
```

Open this address in the browser inside the Sunlogin desktop, enter one to
three characters, set the writing-area width and height, and press **开始写字**. The
launch does not write automatically. The first request for a character/size
pair may take several minutes while V16 performs offline inversion; repeated
requests use the cache. Choose
**横向（从左到右）** or **竖向（从上到下）** before starting. After a group
is finished, press **清除字迹** before starting the next group. The page scans
`outputs/kaishu_pose_trajectory_batch_v1` on every request. It merges the root
manifest with every `manifest_shard_*.jsonl` file and accepts a character when
its `char_uXXXX/pose_refined.csv` and `target.png` exist. Characters still being
generated, failed, or missing an input trajectory are reported as
pending/unavailable; no synthetic trajectory is invented. The database file
`data/raw/data.csv` is also indexed, so a database character appears in the
page before its batch directory exists.

The default V16 output uses this contract:

```text
outputs/kaishu_pose_trajectory_batch_v1/
  manifest_shard_0_of_2.jsonl
  manifest_shard_1_of_2.jsonl
  char_u4e00/pose_refined.csv
  char_u4e00/target.png
  ...
```

The default `武` replay uses the latest completed real-target inversion at
`outputs/kaishu_pose_v16_all_fields_footprint_wu/inversion_trajectory.csv`
while its batch entry is not ready. The special target override keeps it paired with
`data/raw/targets/wu_kaishu_target.png`.

The dedicated completed `武` V16 trajectory remains a fallback if the full
batch entry is missing. V17 directories are still supported as an optional
override. Their rank-1 quality gate is respected unless
`allow_v17_ineligible:=true` is supplied explicitly.

Useful launch overrides:

```bash
ros2 launch fusion_model_ros2_beta ur10_brush_ros2_beta.launch.py \
  trajectory_catalog_root:=/home/robot/coppeliasim/machine_learning/model/outputs/kaishu_pose_trajectory_batch_v1 \
  input_page_port:=18082 \
  allow_v17_ineligible:=false \
  character_gap_m:=0.045 \
  default_layout_mode:=horizontal \
  writing_width_m:=0.52 \
  writing_height_m:=0.32 \
  default_font_size_m:=0.12 \
  offline_inversion_device:=cuda
```

The input page binds to `127.0.0.1` by default. Change
`input_page_host:=0.0.0.0` only when the page must be reached from another
machine.

## Consistency evaluation

The simulated and target masks are foreground-cropped and aligned at `128 x
128`, matching the paper's preprocessing. The report contains CSIM, SSIM,
paper-domain violations, input SHA-256, and the exact brush parameters. A
trajectory is feasible only when all contact poses are in the calibrated domain
and both paper-derived robot-writing minima are met:

The ROS paper frame uses an upward world `y` axis, so the evaluator converts it
to the downward image-row axis before comparing with the PNG target. This keeps
the simulated glyph vertically aligned with `wu_kaishu_target.png`.

- `CSIM >= 0.9136`
- `SSIM >= 0.7042`

Default artifacts:

- `~/ros2_ws/evaluation/fusion_model_ros2/char_u6b66/brush_evaluation.json`
- `~/ros2_ws/evaluation/fusion_model_ros2/char_u6b66/brush_evaluation.png`

The PNG columns are target, simulated flexible-brush ink, and absolute
difference.
