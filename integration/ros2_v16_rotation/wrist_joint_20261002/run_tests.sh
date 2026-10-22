#!/usr/bin/env bash
# Offline only: this script does not launch a controller or publish motion.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODEL=/home/robot/coppeliasim/machine_learning/model
mkdir -p "$ROOT/tests"
set +u
source /opt/ros/humble/setup.bash
set -u
export PYTHONPATH="${ROS_PACKAGE_SOURCE:-$ROOT/source}:$MODEL${PYTHONPATH:+:$PYTHONPATH}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ROOT"
/usr/bin/python3 -m pytest -q -p no:cacheprovider \
  --junitxml="$ROOT/tests/ros.xml" \
  source/test/test_actual_ur10_profile.py \
  source/test/test_attachment_collision.py \
  source/test/test_canvas_layout.py \
  source/test/test_gamma_semantics.py \
  source/test/test_joint_continuity.py \
  source/test/test_trajectory_mapping.py \
  source/test/test_offline_fontsize_inversion.py \
  source/test/test_ur10_launch_contract.py \
  source/test/test_wrist_aware_transitions.py \
  source/test/test_actual_wrist_report.py \
  source/test/test_robot_context_guard.py \
  source/test/test_wrist_deployment.py \
  source/test/test_wrist_artifacts.py \
  source/test/test_robot_path_export.py \
  source/test/test_frozen_requests.py \
  2>&1 | tee "$ROOT/tests/ros.log"
/home/robot/miniconda3/envs/ddpm/bin/python -m pytest -q -p no:cacheprovider \
  --junitxml="$ROOT/tests/model.xml" \
  source/test/test_actual_wrist_feedback.py \
  source/test/test_tool_rotation_tradeoff.py \
  "$MODEL/tests/test_tool_orientation.py" \
  2>&1 | tee "$ROOT/tests/model.log"
PYTHONPATH="$ROOT/metric_reference:$PYTHONPATH" \
  /home/robot/miniconda3/envs/ddpm/bin/python -m pytest -q -p no:cacheprovider \
  --junitxml="$ROOT/tests/ssim.xml" source/test/test_original_image_metrics.py \
  2>&1 | tee "$ROOT/tests/ssim.log"
