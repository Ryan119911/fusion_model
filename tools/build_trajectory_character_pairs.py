import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import numpy as np
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ensure_dirs, load_config
from datasets.character_dataset import CHARACTER_DATA_FORMAT
from datasets.trajectory_dataset import load_trajectory_csv
from utils.character_features import (
    SPATIAL_CHANNEL_NAMES,
    extract_character_spatial_maps,
)
from utils.structure_mask import symmetric_structure_metrics
from utils.trajectory_target import (
    TRAJECTORY_RENDER_VERSION,
    TRAJECTORY_TARGET_MODE,
    render_trajectory_target,
)


def save_audit_panel(
    path: Path,
    spatial_maps: np.ndarray,
    target: np.ndarray,
) -> None:
    centerline = np.clip(spatial_maps[0], 0.0, 1.0)
    proximity = np.clip(spatial_maps[1], 0.0, 1.0)
    pressure = np.clip(spatial_maps[2], 0.0, 1.0)
    gray_panels = [
        np.repeat((panel * 255).astype(np.uint8)[..., None], 3, axis=2)
        for panel in (centerline, proximity, pressure, target)
    ]
    overlap = np.zeros((*target.shape, 3), dtype=np.uint8)
    overlap[..., 0] = (target * 255).astype(np.uint8)
    overlap[..., 1] = (centerline * 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(
        np.concatenate([*gray_panels, overlap], axis=1),
        mode="RGB",
    ).save(path)


def main(args) -> None:
    if args.trajectory_width < 1:
        raise ValueError("--trajectory_width must be positive")
    if args.render_min_width < 1.0:
        raise ValueError("--render_min_width must be at least 1")
    if args.render_max_width < args.render_min_width:
        raise ValueError("--render_max_width must be >= --render_min_width")
    if args.pressure_gamma <= 0.0:
        raise ValueError("--pressure_gamma must be positive")
    if not 0.0 <= args.min_trajectory_coverage <= 1.0:
        raise ValueError("--min_trajectory_coverage must be in [0, 1]")
    if args.skeleton_tolerance < 0:
        raise ValueError("--skeleton_tolerance must be non-negative")
    if args.audit_limit < 0:
        raise ValueError("--audit_limit must be non-negative")

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    canvas_size = int(cfg.character_generator.image_size)
    trajectory_padding = (
        int(args.trajectory_padding)
        if args.trajectory_padding is not None
        else int(cfg.data.character_trajectory_padding)
    )
    trajectories = load_trajectory_csv(
        args.trajectory_csv or cfg.data.trajectory_csv
    )
    if args.character:
        trajectories = [
            sample for sample in trajectories
            if sample.character == args.character
        ]
    if not trajectories:
        raise RuntimeError("No trajectory samples matched the requested filter")
    print(f"[INFO] Loaded {len(trajectories)} character trajectories")

    inputs: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    metadata: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    coverages: List[float] = []
    skeleton_scores: List[float] = []
    width_means: List[float] = []
    output_path = Path(args.output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_dir = (
        Path(args.audit_dir)
        if args.audit_dir
        else output_path.parent / f"{output_path.stem}_audit"
    )
    audit_manifest: List[Dict[str, Any]] = []

    for sample_index, trajectory in enumerate(trajectories):
        character = trajectory.character
        if not character:
            rejected.append({
                "trajectory_sample_index": sample_index,
                "reason": "missing_character",
            })
            continue
        try:
            spatial_maps, normalized_strokes = extract_character_spatial_maps(
                trajectory,
                canvas_size=canvas_size,
                padding=trajectory_padding,
                line_width=args.trajectory_width,
            )
            target, render_info = render_trajectory_target(
                trajectory,
                normalized_strokes,
                canvas_size=canvas_size,
                min_width=args.render_min_width,
                max_width=args.render_max_width,
                pressure_gamma=args.pressure_gamma,
                pressure_invert=args.pressure_invert,
            )
        except (ValueError, RuntimeError) as error:
            rejected.append({
                "character": character,
                "sample_id": trajectory.meta.get("sample_id"),
                "trajectory_sample_index": sample_index,
                "reason": "trajectory_render_failed",
                "error": str(error),
            })
            continue

        centerline = spatial_maps[0] >= 0.5
        coverage = float(target[centerline].mean()) if np.any(centerline) else 0.0
        geometry = symmetric_structure_metrics(
            target,
            spatial_maps[0],
            spatial_maps[1],
            skeleton_tolerance=args.skeleton_tolerance,
        )
        if coverage < args.min_trajectory_coverage:
            rejected.append({
                "character": character,
                "sample_id": trajectory.meta.get("sample_id"),
                "trajectory_sample_index": sample_index,
                "reason": "trajectory_coverage_below_threshold",
                "trajectory_target_coverage": coverage,
                "minimum": args.min_trajectory_coverage,
            })
            continue

        inputs.append(spatial_maps.astype(np.float16))
        targets.append(target.astype(np.float16)[None, ...])
        coverages.append(coverage)
        skeleton_scores.append(float(geometry["symmetric_skeleton_score"]))
        width_means.append(float(render_info["rendered_width_mean"]))
        item_meta = {
            "character": character,
            "sample_id": trajectory.meta.get("sample_id"),
            "trajectory_sample_index": sample_index,
            "num_strokes": len(trajectory.sorted_strokes()),
            "trajectory_padding": trajectory_padding,
            "trajectory_width": args.trajectory_width,
            "trajectory_target_coverage": coverage,
            "target_source": "trajectory_renderer",
            "target_mode": TRAJECTORY_TARGET_MODE,
            "target_geometry": "same_source_trajectory",
            "render_info": render_info,
            "alignment_metrics": geometry,
        }
        metadata.append(item_meta)
        if len(audit_manifest) < args.audit_limit:
            codepoints = "-".join(f"U+{ord(value):04X}" for value in character)
            audit_path = audit_dir / f"{len(audit_manifest):04d}_{codepoints}.png"
            save_audit_panel(audit_path, spatial_maps, target)
            audit_manifest.append({
                "character": character,
                "sample_id": trajectory.meta.get("sample_id"),
                "path": str(audit_path),
                "trajectory_target_coverage": coverage,
                "render_info": render_info,
                "alignment_metrics": geometry,
            })

    rejection_path = output_path.with_suffix(".rejected.json")
    with open(rejection_path, "w", encoding="utf-8") as file:
        json.dump(rejected, file, ensure_ascii=False, indent=2)
    if not inputs:
        raise RuntimeError(
            "No trajectory-faithful pairs were built. "
            f"See {rejection_path}"
        )
    if audit_manifest:
        audit_dir.mkdir(parents=True, exist_ok=True)
        with open(
            audit_dir / "audit_manifest.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(audit_manifest, file, ensure_ascii=False, indent=2)

    summary = {
        "format_version": CHARACTER_DATA_FORMAT,
        "preprocessing_version": TRAJECTORY_RENDER_VERSION,
        "target_mode": TRAJECTORY_TARGET_MODE,
        "target_geometry": "same_source_trajectory",
        "accepted": len(inputs),
        "rejected": len(rejected),
        "rejection_reasons": dict(
            Counter(item["reason"] for item in rejected)
        ),
        "trajectory_padding": trajectory_padding,
        "trajectory_width": args.trajectory_width,
        "render_min_width": args.render_min_width,
        "render_max_width": args.render_max_width,
        "render_pressure_gamma": args.pressure_gamma,
        "render_pressure_invert": args.pressure_invert,
        "trajectory_target_coverage": {
            "mean": float(np.mean(coverages)),
            "median": float(np.median(coverages)),
            "min": float(np.min(coverages)),
        },
        "symmetric_skeleton_score": {
            "mean": float(np.mean(skeleton_scores)),
            "median": float(np.median(skeleton_scores)),
            "min": float(np.min(skeleton_scores)),
        },
        "rendered_width_mean": {
            "mean": float(np.mean(width_means)),
            "min": float(np.min(width_means)),
            "max": float(np.max(width_means)),
        },
    }
    with open(
        output_path.with_suffix(".summary.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    input_array = np.stack(inputs)
    target_array = np.stack(targets)
    np.savez_compressed(
        output_path,
        inputs=input_array,
        targets=target_array,
        meta=np.asarray(metadata, dtype=object),
        channel_names=np.asarray(SPATIAL_CHANNEL_NAMES),
        format_version=np.asarray(CHARACTER_DATA_FORMAT),
        trajectory_padding=np.asarray(trajectory_padding, dtype=np.int32),
        trajectory_width=np.asarray(args.trajectory_width, dtype=np.int32),
        preprocessing_version=np.asarray(TRAJECTORY_RENDER_VERSION),
        min_alignment_coverage=np.asarray(
            args.min_trajectory_coverage,
            dtype=np.float32,
        ),
        target_script=np.asarray("trajectory"),
        quality_thresholds=np.asarray("{}"),
        target_mode=np.asarray(TRAJECTORY_TARGET_MODE),
        structure_threshold=np.asarray(0.5, dtype=np.float32),
        min_component_pixels=np.asarray(0, dtype=np.int32),
        opening_iterations=np.asarray(0, dtype=np.int32),
        skeleton_tolerance=np.asarray(
            args.skeleton_tolerance,
            dtype=np.int32,
        ),
        render_min_width=np.asarray(args.render_min_width, dtype=np.float32),
        render_max_width=np.asarray(args.render_max_width, dtype=np.float32),
        render_pressure_gamma=np.asarray(args.pressure_gamma, dtype=np.float32),
        render_pressure_invert=np.asarray(args.pressure_invert, dtype=np.bool_),
    )
    print(f"[DONE] Trajectory-faithful pairs saved to: {output_path}")
    print(f"[DONE] inputs shape: {input_array.shape}")
    print(f"[DONE] targets shape: {target_array.shape}")
    print(f"[DONE] target mode: {TRAJECTORY_TARGET_MODE}")
    print(
        "[DONE] renderer: "
        f"width={args.render_min_width:.2f}-{args.render_max_width:.2f}, "
        f"pressure_gamma={args.pressure_gamma:.3f}, "
        f"pressure_invert={args.pressure_invert}"
    )
    print(f"[DONE] accepted={len(inputs)}, rejected={len(rejected)}")
    print(
        "[CHECK] trajectory/target coverage: "
        f"mean={np.mean(coverages):.6f}, "
        f"median={np.median(coverages):.6f}, "
        f"min={np.min(coverages):.6f}"
    )
    print(
        "[CHECK] symmetric skeleton score: "
        f"mean={np.mean(skeleton_scores):.6f}, "
        f"median={np.median(skeleton_scores):.6f}, "
        f"min={np.min(skeleton_scores):.6f}"
    )
    print(f"[DONE] summary: {output_path.with_suffix('.summary.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Build v8 whole-character targets whose length, position, and "
            "centerline geometry come from the same input trajectory"
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--trajectory_csv", default=None)
    parser.add_argument(
        "--output_npz",
        default="data/processed/character_trajectory_faithful_v8.npz",
    )
    parser.add_argument("--character", default=None)
    parser.add_argument("--trajectory_width", type=int, default=3)
    parser.add_argument("--trajectory_padding", type=int, default=None)
    parser.add_argument("--render_min_width", type=float, default=4.0)
    parser.add_argument("--render_max_width", type=float, default=8.0)
    parser.add_argument("--pressure_gamma", type=float, default=1.0)
    parser.add_argument("--pressure_invert", action="store_true")
    parser.add_argument("--min_trajectory_coverage", type=float, default=0.999)
    parser.add_argument("--skeleton_tolerance", type=int, default=3)
    parser.add_argument("--audit_dir", default=None)
    parser.add_argument("--audit_limit", type=int, default=200)
    main(parser.parse_args())
