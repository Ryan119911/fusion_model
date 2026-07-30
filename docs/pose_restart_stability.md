# Pose restart stability without real pose truth

Use independent optimizer settings to test whether a simulation-only pose
solution is locally repeatable. This does not create physical ground truth.

```bash
python -u tools/compare_pose_refinements.py \
  --trajectory_csvs \
    outputs/wu_kaishu_target_v29_gamma_after_xy/wu_trajectory.csv \
    outputs/wu_kaishu_target_v29b_gamma_restart/wu_trajectory.csv \
  --report_jsons \
    outputs/wu_kaishu_target_v29_gamma_after_xy/wu_report.json \
    outputs/wu_kaishu_target_v29b_gamma_restart/wu_report.json \
  --pose_fields gamma \
  --output_json outputs/wu_kaishu_target_v29_gamma_stability.json
```

The audit requires fixed x/y, normalized pose RMS standard deviation at most
0.02, an identifiable joint Jacobian, no more than 5% pose-bound saturation,
and trajectory coverage of at least 0.99. Passing only supports simulation
restart stability; real brush, camera, TCP, paper, and robot calibration are
still required.
