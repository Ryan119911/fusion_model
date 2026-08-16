"""Augment the existing real Kaishu style set with trajectory features.

The v27 NPZ already contains the 20k real Kaishu crops and grayscale targets.
Reusing it avoids re-reading 631 source images/LabelMe files while preserving
the same target pixels.  We append matched trajectory centerline, direction,
stroke order, z-height pressure proxy, speed proxy, and a real target-derived
footprint-width channel.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.trajectory_dataset import load_trajectory_csv
from utils.kaishu_style_features import (
    STYLE_FEATURE_CHANNEL_NAMES,
    build_style_features,
)


FORMAT = "kaishu_style_dataset_v16"


def main(args: argparse.Namespace) -> None:
    base_path = Path(args.base_npz)
    data = np.load(base_path, allow_pickle=False)
    required = {"targets", "characters", "sources"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"Base style NPZ missing keys: {sorted(missing)}")
    targets = np.asarray(data["targets"])
    characters = np.asarray(data["characters"]).astype(str)
    sources = np.asarray(data["sources"]).astype(str)
    if targets.ndim == 3:
        targets = targets[:, None]
    if targets.ndim != 4 or targets.shape[1] != 1:
        raise ValueError(f"Expected targets [N,1,H,W], got {targets.shape}")
    if targets.shape[0] != len(characters) or len(sources) != len(characters):
        raise ValueError("Base style arrays have inconsistent lengths")
    if targets.max(initial=0) > 1:
        targets_float = targets.astype(np.float32) / 255.0
    else:
        targets_float = targets.astype(np.float32)

    trajectories = {}
    for trajectory in load_trajectory_csv(args.trajectory_csv):
        trajectories.setdefault(trajectory.character, trajectory)

    features = []
    kept_targets = []
    kept_characters = []
    kept_sources = []
    records = []
    skipped_reasons: dict[str, int] = {}
    coverage_values = []
    support_dice_values = []
    for index, character in enumerate(characters):
        trajectory = trajectories.get(character)
        if trajectory is None:
            skipped_reasons["no_matching_trajectory"] = skipped_reasons.get(
                "no_matching_trajectory", 0
            ) + 1
            continue
        try:
            feature, alignment = build_style_features(
                targets_float[index, 0],
                trajectory,
                canvas_size=targets.shape[-1],
                trajectory_padding=args.trajectory_padding,
                trajectory_width=args.trajectory_width,
                structure_threshold=args.structure_threshold,
                footprint_width_scale_px=args.footprint_width_scale_px,
            )
        except (RuntimeError, ValueError) as error:
            key = f"build_failed:{type(error).__name__}"
            skipped_reasons[key] = skipped_reasons.get(key, 0) + 1
            continue
        coverage = float(alignment["trajectory_target_coverage"])
        if coverage < args.min_trajectory_coverage:
            skipped_reasons["trajectory_coverage_below_threshold"] = (
                skipped_reasons.get("trajectory_coverage_below_threshold", 0) + 1
            )
            continue
        features.append(np.rint(np.clip(feature, 0.0, 1.0) * 255).astype(np.uint8))
        kept_targets.append(targets[index])
        kept_characters.append(character)
        kept_sources.append(sources[index])
        coverage_values.append(coverage)
        support_dice_values.append(float(alignment["support_dice"]))
        records.append(
            {
                "base_index": int(index),
                "character": character,
                "source": sources[index],
                "trajectory_sample_id": trajectory.meta.get("sample_id"),
                **alignment,
            }
        )
    if not features:
        raise RuntimeError("No v16 samples survived trajectory alignment filtering")

    metadata = {
        "format": FORMAT,
        "base_npz": str(base_path),
        "trajectory_csv": args.trajectory_csv,
        "chirography": "楷",
        "heldout_character": args.heldout_character,
        "feature_channels": list(STYLE_FEATURE_CHANNEL_NAMES),
        "trajectory_padding": args.trajectory_padding,
        "trajectory_width": args.trajectory_width,
        "structure_threshold": args.structure_threshold,
        "footprint_width_scale_px": args.footprint_width_scale_px,
        "velocity_source": "consecutive_xy_displacement_per_sample_proxy",
        "pressure_source": "trajectory_z_height_proxy",
        "real_footprint_source": "target_distance_transform_width",
        "alignment_filter": {
            "min_trajectory_coverage": args.min_trajectory_coverage,
        },
        "sample_count": len(features),
        "source_count": len(set(kept_sources)),
        "character_count": len(set(kept_characters)),
        "heldout_samples": sum(c == args.heldout_character for c in kept_characters),
        "skipped": int(sum(skipped_reasons.values())),
        "skipped_reasons": skipped_reasons,
        "alignment": {
            "coverage_mean": float(np.mean(coverage_values)),
            "coverage_median": float(np.median(coverage_values)),
            "coverage_min": float(np.min(coverage_values)),
            "support_dice_mean": float(np.mean(support_dice_values)),
        },
        "records": records,
    }
    output = Path(args.output_npz)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        features=np.stack(features),
        targets=np.stack(kept_targets),
        characters=np.asarray(kept_characters),
        sources=np.asarray(kept_sources),
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    output.with_suffix(".summary.json").write_text(
        json.dumps({k: v for k, v in metadata.items() if k != "records"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in metadata.items() if k != "records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_npz", default="data/processed/kaishu_style_v27.npz")
    parser.add_argument("--trajectory_csv", default="data/raw/trajectories.csv")
    parser.add_argument("--output_npz", default="data/processed/kaishu_style_v16.npz")
    parser.add_argument("--heldout_character", default="武")
    parser.add_argument("--trajectory_padding", type=int, default=4)
    parser.add_argument("--trajectory_width", type=int, default=3)
    parser.add_argument("--structure_threshold", type=float, default=0.35)
    parser.add_argument("--footprint_width_scale_px", type=float, default=16.0)
    parser.add_argument("--min_trajectory_coverage", type=float, default=0.30)
    main(parser.parse_args())
