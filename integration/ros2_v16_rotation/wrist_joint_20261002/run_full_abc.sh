#!/usr/bin/env bash
set -euo pipefail
OUT=/home/robot/ros2_ws/evaluation/wrist_joint_20261002
MODEL=/home/robot/coppeliasim/machine_learning/model
set +u
source /opt/ros/humble/setup.bash
set -u
export PYTHONPATH="$OUT/source:$MODEL${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MANIFEST=/home/robot/ros2_ws/evaluation/original_target_20260923_codex_staging_copy/references_wu_verified.json
for slots in 3 1; do
  for variant in C A B; do
    common=(--manifest "$MANIFEST" --output "$OUT" --case "$variant" --slots "$slots" --steps 16 --order 11 --width .52 --height .32)
    printf '\nBEGIN slots=%s case=%s UTC=%s\n' "$slots" "$variant" "$(date -u +%FT%TZ)"
    /home/robot/miniconda3/envs/ddpm/bin/python -u -m fusion_model_ros2_beta.evaluate_actual_wrist_tradeoff --mode generate "${common[@]}"
    /usr/bin/python3 -u -m fusion_model_ros2_beta.evaluate_actual_wrist_tradeoff --mode plan "${common[@]}"
    printf 'DONE slots=%s case=%s UTC=%s\n' "$slots" "$variant" "$(date -u +%FT%TZ)"
  done
done
/usr/bin/python3 -m fusion_model_ros2_beta.evaluate_actual_wrist_tradeoff --mode summary --manifest "$MANIFEST" --output "$OUT"
printf 'FULL_ABC_FINISHED UTC=%s\n' "$(date -u +%FT%TZ)"
