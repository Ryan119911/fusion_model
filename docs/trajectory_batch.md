# Batch trajectory repair

`tools/process_trajectory_batch.py` applies the same stroke-state repair,
per-stroke smoothing, XY densification, Y-axis-correct preview, and safety
audit to every sample in a trajectory CSV database.

```bash
python -u tools/process_trajectory_batch.py \
  --input_csv data/raw/trajectories.csv \
  --output_dir outputs/trajectory_database_processed \
  --smooth_passes 2 \
  --smooth_strength 0.25 \
  --max_step_xy 2.0 \
  --preview_count 24 \
  --report_every 250 \
  --fail_on_unsafe
```

The source CSV is never overwritten. The output directory contains:

```text
trajectories_processed.csv       # all repaired samples
trajectory_reports.jsonl         # one raw/processed report per sample
trajectory_batch_summary.json    # aggregate counts and warnings
previews/                         # only the configured audit subset
```

`--preview_count` limits the number of PNG panels. Add `--preview_every N` to
also audit every Nth sample. Previews flip source-frame Y-up into image-frame
Y-down by default; the exported CSV remains in the source coordinate frame.
When `--max_step_xy` is set, it is also used as the default safety threshold,
so oversized raw segments are explicitly reported as unsafe.
