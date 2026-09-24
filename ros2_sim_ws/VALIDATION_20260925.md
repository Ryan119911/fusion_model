# Actual UR10 continuation: validation status

**Status: integration fixes verified; full writing acceptance remains blocked.**

Source is the MU package /home/robot/ros2_ws/src/fusion_model_ros2_beta.
Reference hardware: https://github.com/jinxiao123580-hub/UR10 at
2e37519e32561606d1a4123aa238597897529a17. UR10 CB3 factory calibration
calib_15120592593058779304; controller coordinate frame base.

## Completed

- Factory-calibrated FK and URDF, ATI/Robotiq/camera envelopes, 125 Hz fake control.
- Scoped manager rename on Humble, controller-owned command/state topics,
  shutdown on controller-spawner failure.
- Original-target backend now always enables the hard dense neural-domain constraint.
- Gamma initialization uses deterministic alternate starts and reports linear
  relaxation infeasibility instead of recommending longer unconstrained runs.
- Strict planner distinguishes converged IK rejected by collision from numerical IK failure.
- MU test suite: 50 passed (one SciPy optimizer intermediate-bound warning; final
  output checked against exact constraints). Independent colcon build succeeded.
- Localhost ROS_DOMAIN_ID=217 fake-hardware startup: command and joint-state
  topics each had one publisher and one subscriber. All test nodes exited cleanly.

## Full-case findings

1. Prior 290 mm, 16-step original-target inversion did finish; its neural audit
   reports 68/209 gamma inputs outside +/-30 degrees. IoU 0.391748, MSE 0.066668,
   ink ratio 0.698460. It was correctly rejected and never marked complete.
2. Hard-domain reruns at CGL orders 5 and 11, including alternate starts, fail
   initialization. The failing stroke heading range is [-1.422258, 1.467638] rad.
   Linear relaxation with independent per-source-point gamma and the exact
   dense interpolation is infeasible. This conclusion applies to the current
   fixed source XY and interpolation, not every possible optimized XY trajectory.
3. Existing in-domain 137.5 mm candidate was independently planned with strict IK.
   IK converged, but no collision-free route was found at stroke 0. Diagnostic:
   paper:mecheye_camera, envelope point [-0.00250953, -0.87451813, 0.12949571] m.
   The camera collision uses a conservative world AABB; this is not evidence of
   independently measured physical collision.

No motion was published during these full-case checks. No full-case success,
physical readiness or visual acceptance is claimed. Current ordinary launch still
defaults to legacy_fused; explicit joint_target plus a pinned manifest is required
for original-image work. Existing timer-driven ink is commanded prediction.

## Reproduce on MU

```bash
source /opt/ros/humble/setup.bash
cd /home/robot/ros2_ws/src/fusion_model_ros2_beta
PYTHONPATH=$PWD:$PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q test
ROS_DOMAIN_ID=217 ROS_LOCALHOST_ONLY=1 python3 test/verify_actual_ur10_pipeline.py \
  --manifest /home/robot/ros2_ws/evaluation/original_target_20260923/references_wu_verified.json \
  --output /home/robot/ros2_ws/evaluation/actual_ur10_new_run
```

Use a fresh output directory for a failed generation; incomplete caches are
intentionally not reused. Model weights, datasets and generated caches remain
external; install the model dependencies and supply hash-pinned original targets.
Evidence remains under /home/robot/ros2_ws/evaluation/actual_ur10_20260925* and
/home/robot/ros2_ws/backups/ur10_resume_20260925/.
