#!/usr/bin/env bash
# Attach the standard ROS screening/registration stage to an already-running
# four-shard v17 job.  It never starts or interrupts inversion processes.
set -u

cd /home/robot/coppeliasim/machine_learning/model
PYTHON=/home/robot/miniconda3/envs/ddpm/bin/python
INPUT_ROOT="${INPUT_ROOT:-outputs/kaishu_pose_v17_full_snap}"
ROS_REGISTRY_DIR="${ROS_REGISTRY_DIR:-outputs/ros_trajectory_registry/$(basename "${INPUT_ROOT}")_ur10}"
ROBOT_MODEL="${ROBOT_MODEL:-ur10}"
ROBOT_VALIDATION_DIR="${ROBOT_VALIDATION_DIR:-outputs/ros_robot_validation/$(basename "${INPUT_ROOT}")_${ROBOT_MODEL}}"
POLL_SECONDS="${POLL_SECONDS:-300}"

if [ "${POLL_SECONDS}" -lt 30 ]; then
  echo "POLL_SECONDS must be at least 30"
  exit 2
fi

while true; do
  complete=1
  for shard in 0 1 2 3; do
    summary="${INPUT_ROOT}/batch_summary_shard_${shard}_of_4.json"
    if [ ! -f "${summary}" ]; then
      complete=0
      break
    fi
  done
  if [ "${complete}" -eq 1 ]; then
    break
  fi
  if ! pgrep -f "tools/invert_kaishu_v17_batch.py.*--output_dir ${INPUT_ROOT}" >/dev/null; then
    echo "[WATCH] inversion stopped before all four shard summaries appeared"
    "${PYTHON}" -u tools/register_pose_library_for_ros.py \
      --input_root "${INPUT_ROOT}" \
      --output_root "${ROS_REGISTRY_DIR}" \
      --mode audit --expected_shards 4 \
      --robot_model "${ROBOT_MODEL}" \
      --robot_validation_root "${ROBOT_VALIDATION_DIR}"
    exit 1
  fi
  sleep "${POLL_SECONDS}"
done

"${PYTHON}" -u tools/register_pose_library_for_ros.py \
  --input_root "${INPUT_ROOT}" \
  --output_root "${ROS_REGISTRY_DIR}" \
  --expected_shards 4 \
  --robot_model "${ROBOT_MODEL}" \
  --robot_validation_root "${ROBOT_VALIDATION_DIR}"
