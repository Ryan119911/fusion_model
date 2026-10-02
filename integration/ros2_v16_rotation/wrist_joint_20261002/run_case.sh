#!/usr/bin/env bash
# Optional independent offline case; never publish mechanical-arm motion.
set -euo pipefail
OUT=/home/robot/ros2_ws/evaluation/wrist_joint_20261002
MODEL=/home/robot/coppeliasim/machine_learning/model
CASE=${1:?A/B/C required}
SLOTS=${2:?1/3 required}
set +u
source /opt/ros/humble/setup.bash
set -u
export PYTHONPATH="$OUT/source:$MODEL${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MANIFEST=/home/robot/ros2_ws/evaluation/original_target_20260923_codex_staging_copy/references_wu_verified.json
common=(--manifest "$MANIFEST" --output "$OUT" --case "$CASE" --slots "$SLOTS" --steps 16 --order 11 --width .52 --height .32)
/home/robot/miniconda3/envs/ddpm/bin/python -u -m fusion_model_ros2_beta.evaluate_actual_wrist_tradeoff --mode generate "${common[@]}"
/usr/bin/python3 -u -m fusion_model_ros2_beta.evaluate_actual_wrist_tradeoff --mode plan "${common[@]}"
