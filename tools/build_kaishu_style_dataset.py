"""Build leakage-controlled geometry-to-brush-appearance supervision."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.calligraphy_image_dataset import CalligraphyImageDataset
from utils.image_preprocessing import letterbox_character_image
from datasets.trajectory_dataset import load_trajectory_csv
from utils.kaishu_style_features import (
    STYLE_FEATURE_CHANNEL_NAMES,
    build_style_features,
    geometry_features,
)


FORMAT = "kaishu_style_dataset_v16"


def source_key(item: dict) -> str:
    return Path(item["image_path"]).as_posix()


def main(args: argparse.Namespace) -> None:
    dataset = CalligraphyImageDataset(
        image_dir=args.image_dir,
        json_dir=args.json_dir,
        image_ext=args.image_ext,
        image_size=None,
        grayscale=True,
        padding=0.0,
        data_csv=args.data_csv,
        chirography_filter=args.chirography,
    )
    rng = np.random.default_rng(args.seed)
    indices = np.arange(len(dataset.index))
    if args.max_samples > 0 and len(indices) > args.max_samples:
        indices = np.sort(rng.choice(indices, args.max_samples, replace=False))
    features, targets, characters, sources, records = [], [], [], [], []
    skipped = 0
    skipped_reasons = {}
    trajectories = {}
    for trajectory in load_trajectory_csv(args.trajectory_csv):
        trajectories.setdefault(trajectory.character, trajectory)
    coverage_values = []
    support_dice_values = []
    for raw_index in indices:
        item = dataset.index[int(raw_index)]
        try:
            sample = dataset._build_sample(item)
            gray, transform = letterbox_character_image(
                sample["image"],
                canvas_size=args.image_size,
                padding=args.padding,
                crop_foreground=True,
            )
            trajectory = trajectories.get(item["character"])
            if trajectory is None:
                skipped += 1
                skipped_reasons["no_matching_trajectory"] = (
                    skipped_reasons.get("no_matching_trajectory", 0) + 1
                )
                continue
            feature, alignment = build_style_features(
                gray,
                trajectory,
                canvas_size=args.image_size,
                trajectory_padding=args.trajectory_padding,
                trajectory_width=args.trajectory_width,
                structure_threshold=args.structure_threshold,
                footprint_width_scale_px=args.footprint_width_scale_px,
            )
            if alignment["trajectory_target_coverage"] < args.min_trajectory_coverage:
                skipped += 1
                skipped_reasons["trajectory_coverage_below_threshold"] = (
                    skipped_reasons.get("trajectory_coverage_below_threshold", 0) + 1
                )
                continue
        except (OSError, RuntimeError, ValueError):
            skipped += 1
            skipped_reasons["build_failed"] = skipped_reasons.get("build_failed", 0) + 1
            continue
        features.append(np.rint(feature * 255).astype(np.uint8))
        targets.append(np.rint(gray[None] * 255).astype(np.uint8))
        characters.append(item["character"])
        sources.append(source_key(item))
        records.append(
            {
                "dataset_index": int(raw_index),
                "character": item["character"],
                "source": source_key(item),
                "bbox": list(item["bbox"]),
                "transform": transform,
                "trajectory_sample_id": trajectory.meta.get("sample_id"),
                "trajectory_target_coverage": alignment["trajectory_target_coverage"],
                "support_dice": alignment["support_dice"],
            }
        )
        coverage_values.append(alignment["trajectory_target_coverage"])
        support_dice_values.append(alignment["support_dice"])
    if not features:
        raise RuntimeError("No valid style samples were built")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        features=np.stack(features),
        targets=np.stack(targets),
        characters=np.asarray(characters),
        sources=np.asarray(sources),
        metadata=np.asarray(
            json.dumps(
                {
                    "format": FORMAT,
                    "chirography": args.chirography,
                    "excluded_from_generic_training": args.heldout_character,
                    "feature_channels": list(STYLE_FEATURE_CHANNEL_NAMES),
                    "trajectory_csv": args.trajectory_csv,
                    "trajectory_padding": args.trajectory_padding,
                    "trajectory_width": args.trajectory_width,
                    "footprint_width_scale_px": args.footprint_width_scale_px,
                    "velocity_source": "consecutive_xy_displacement_per_sample_proxy",
                    "pressure_source": "trajectory_z_height_proxy",
                    "real_footprint_source": "target_distance_transform_width",
                    "alignment_filter": {
                        "min_trajectory_coverage": args.min_trajectory_coverage,
                    },
                    "sample_count": len(features),
                    "skipped": skipped,
                    "skipped_reasons": skipped_reasons,
                    "records": records,
                },
                ensure_ascii=False,
            )
        ),
    )
    summary = {
        "format": FORMAT,
        "output": str(output),
        "samples": len(features),
        "sources": len(set(sources)),
        "characters": len(set(characters)),
        "heldout_character": args.heldout_character,
        "heldout_samples": sum(value == args.heldout_character for value in characters),
        "skipped": skipped,
        "skipped_reasons": skipped_reasons,
        "feature_channels": list(STYLE_FEATURE_CHANNEL_NAMES),
        "trajectory_csv": args.trajectory_csv,
        "trajectory_padding": args.trajectory_padding,
        "trajectory_width": args.trajectory_width,
        "footprint_width_scale_px": args.footprint_width_scale_px,
        "velocity_source": "consecutive_xy_displacement_per_sample_proxy",
        "pressure_source": "trajectory_z_height_proxy",
        "real_footprint_source": "target_distance_transform_width",
        "alignment": {
            "min_trajectory_coverage": args.min_trajectory_coverage,
            "coverage_mean": float(np.mean(coverage_values)) if coverage_values else 0.0,
            "coverage_median": float(np.median(coverage_values)) if coverage_values else 0.0,
            "coverage_min": float(np.min(coverage_values)) if coverage_values else 0.0,
            "support_dice_mean": float(np.mean(support_dice_values)) if support_dice_values else 0.0,
        },
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--image_dir", default="data/raw/images")
    parser.add_argument("--json_dir", default="data/raw/json_files")
    parser.add_argument("--data_csv", default="data/raw/data.csv")
    parser.add_argument("--image_ext", default=".jpg")
    parser.add_argument("--chirography", default="楷")
    parser.add_argument("--heldout_character", default="武")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--padding", type=int, default=4)
    parser.add_argument("--structure_threshold", type=float, default=0.35)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trajectory_csv", default="data/raw/trajectories.csv")
    parser.add_argument("--trajectory_padding", type=int, default=4)
    parser.add_argument("--trajectory_width", type=int, default=3)
    parser.add_argument("--footprint_width_scale_px", type=float, default=16.0)
    parser.add_argument("--min_trajectory_coverage", type=float, default=0.30)
    main(parser.parse_args())
