#!/usr/bin/env bash
set -u
cd /home/robot/coppeliasim/machine_learning/model
OUTPUT_DIR="${OUTPUT_DIR:-outputs/kaishu_pose_v17_full_snap}"
PYTHON=/home/robot/miniconda3/envs/ddpm/bin/python
ROS_REGISTRY_DIR="${ROS_REGISTRY_DIR:-outputs/ros_trajectory_registry/$(basename "${OUTPUT_DIR}")_ur10}"
ROBOT_MODEL="${ROBOT_MODEL:-ur10}"
ROBOT_VALIDATION_DIR="${ROBOT_VALIDATION_DIR:-outputs/ros_robot_validation/$(basename "${OUTPUT_DIR}")_${ROBOT_MODEL}}"

run_shard() {
  local shard="$1"
  local code=0
  echo "[SHARD ${shard}] start $(date -Is)"
  "$PYTHON" -u tools/invert_kaishu_v17_batch.py \
    --trajectory_csv data/raw/trajectories.csv \
    --bbsmg_ckpt outputs/paper_bbsmg_general_v16_pose_dense/bbsmg_best.pt \
    --output_dir "${OUTPUT_DIR}" --target_selection best_model_support \
    --model_support_width_px 7 --chirography 楷 \
    --shard_count 4 --shard_index "${shard}" \
    --order 1 --max_steps 3 --optimization_size 64 \
    --pixel_weight 12 --h_smoothness_weight 0.2 \
    --h_point_velocity_weight 5 --h_point_acceleration_weight 10 \
    --perturbation_scales -1 0 1 --top_k 3 --optimize_xy \
    --xy_max_offset_px 3 --xy_smoothness_weight 1 --xy_prior_weight 0.5 \
    --xy_segment_length_weight 0.10 --xy_segment_direction_weight 0.10 \
    --xy_target_skeleton_weight 0.20 --xy_target_skeleton_max_distance_px 8 \
    --device cuda --resume_completed \
    --target_override 武=data/raw/targets/wu_kaishu_target.png \
    --timeout_seconds 1800 || code=$?
  echo "[SHARD ${shard}] exit=${code} $(date -Is)"
  return "${code}"
}

pids=()
for shard in 0 1 2 3; do
  run_shard "${shard}" > "/tmp/kaishu_v17_full_snap_shard_${shard}.log" 2>&1 &
  pids+=("$!")
  echo "[LAUNCHED] shard=${shard} pid=${pids[-1]}"
done
shard_failure=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    shard_failure=1
  fi
done
echo "[PARALLEL] shards 0-3 finished $(date -Is)"
if [ "${shard_failure}" -ne 0 ]; then
  echo "[PARALLEL] at least one shard failed; ROS registration was not started"
  exit 1
fi

"${PYTHON}" -u tools/register_pose_library_for_ros.py \
  --input_root "${OUTPUT_DIR}" \
  --output_root "${ROS_REGISTRY_DIR}" \
  --expected_shards 4 \
  --robot_model "${ROBOT_MODEL}" \
  --robot_validation_root "${ROBOT_VALIDATION_DIR}"
