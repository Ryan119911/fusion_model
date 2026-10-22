#!/usr/bin/env bash
# Isolated package build; does not replace the formal ROS install or launch.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
set +u
source /opt/ros/humble/setup.bash
set -u
cd "$ROOT"
colcon --log-base "$ROOT/build_logs" build \
  --base-paths "$ROOT/deployed_source" \
  --build-base "$ROOT/build_optin" --install-base "$ROOT/install_optin" \
  --packages-select fusion_model_ros2_beta --symlink-install
set +u
source "$ROOT/install_optin/setup.bash"
set -u
ros2 launch fusion_model_ros2_beta ur10_brush_ros2_beta.launch.py --show-args
