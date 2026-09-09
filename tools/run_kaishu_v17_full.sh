#!/usr/bin/env bash
set -u
cd /home/robot/coppeliasim/machine_learning/model
OUTPUT_DIR="${OUTPUT_DIR:-outputs/kaishu_pose_v17_full}"

for shard in 0 1 2 3; do
  echo "[SUPERVISOR] starting shard ${shard} at $(date -Is)"
  /home/robot/miniconda3/envs/ddpm/bin/python -u tools/invert_kaishu_v17_batch.py \
    --trajectory_csv data/raw/trajectories.csv \
    --bbsmg_ckpt outputs/paper_bbsmg_general_v16_pose_dense/bbsmg_best.pt \
    --output_dir "${OUTPUT_DIR}" \
    --target_selection best_model_support \
    --model_support_width_px 7 \
    --chirography 楷 \
    --shard_count 4 --shard_index "${shard}" \
    --order 1 --max_steps 3 --optimization_size 64 \
    --pixel_weight 12 --h_smoothness_weight 0.2 \
    --h_point_velocity_weight 5 --h_point_acceleration_weight 10 \
    --perturbation_scales -1 0 1 --top_k 3 \
    --optimize_xy --xy_max_offset_px 3 --xy_smoothness_weight 1 \
    --xy_prior_weight 0.5 --xy_segment_length_weight 0.10 \
    --xy_segment_direction_weight 0.10 --xy_target_skeleton_weight 0.20 \
    --xy_target_skeleton_max_distance_px 8 \
    --device cuda --resume_completed \
    --target_override 武=data/raw/targets/wu_kaishu_target.png \
    --timeout_seconds 1800
  code=$?
  echo "[SUPERVISOR] shard ${shard} exit=${code} at $(date -Is)"
  if [ "${code}" -ne 0 ]; then
    echo "[SUPERVISOR] continuing after shard failure"
  fi
done
echo "[SUPERVISOR] all shards finished at $(date -Is)"
