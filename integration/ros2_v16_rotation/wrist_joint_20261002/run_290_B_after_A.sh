#!/usr/bin/env bash
# The independent A worker and primary C worker occupy at most two GPU jobs.
set -euo pipefail
OUT=/home/robot/ros2_ws/evaluation/wrist_joint_20261002
deadline=$(( $(date +%s) + 10800 ))
while [[ ! -f "$OUT/canvas_1_slots/A_result.json" ]]; do
  if (( $(date +%s) > deadline )); then exit 2; fi
  if ! pgrep -f '^/home/robot/miniconda3/envs/ddpm/bin/python -u -m fusion_model_ros2_beta.evaluate_actual_wrist_tradeoff.*--case A --slots 1' >/dev/null; then
    printf 'A worker ended without a complete result; do not start B\n' >&2
    exit 2
  fi
  sleep 15
done
cd "$OUT"
bash run_case.sh B 1
