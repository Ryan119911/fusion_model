#!/usr/bin/env bash
# Only offline planning; generation uses the immutable earlier source snapshot.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODEL=/home/robot/coppeliasim/machine_learning/model
MANIFEST=/home/robot/ros2_ws/evaluation/original_target_20260923_codex_staging_copy/references_wu_verified.json
test -d "$ROOT/deployed_source/fusion_model_ros2_beta"
for slots in 3 1; do
  for case in A B C; do
    test -f "$ROOT/canvas_${slots}_slots/${case}_generated.json"
  done
done
exec 9>"$ROOT/.final_acceptance.lock"
flock -n 9
set +u
source /opt/ros/humble/setup.bash
set -u
export PYTHONPATH="$ROOT/deployed_source:$MODEL${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for slots in 3 1; do
  for case in A B C; do
    /usr/bin/python3 -u -m fusion_model_ros2_beta.evaluate_actual_wrist_tradeoff \
      --mode plan --manifest "$MANIFEST" --output "$ROOT" \
      --case "$case" --slots "$slots" --steps 16 --order 11 --width .52 --height .32
  done
done
ROS_PACKAGE_SOURCE="$ROOT/deployed_source" bash "$ROOT/run_tests.sh"
/usr/bin/python3 "$ROOT/source/verify_frozen_requests.py" --output "$ROOT"
/usr/bin/python3 -m fusion_model_ros2_beta.evaluate_actual_wrist_tradeoff \
  --mode summary --manifest "$MANIFEST" --output "$ROOT"
/usr/bin/python3 "$ROOT/write_wrist_report.py" --output "$ROOT"
# Include the human-readable report in the final integrity inventory.
/usr/bin/python3 -m fusion_model_ros2_beta.evaluate_actual_wrist_tradeoff \
  --mode summary --manifest "$MANIFEST" --output "$ROOT"
cd "$ROOT"
sha256sum -c verification.sha256
/usr/bin/python3 source/verify_wrist_artifacts.py --output "$ROOT"
