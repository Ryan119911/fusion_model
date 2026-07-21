import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any, DefaultDict, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ensure_dirs, load_config
from datasets.calligraphy_image_dataset import CalligraphyImageDataset
from datasets.character_dataset import CHARACTER_DATA_FORMAT
from datasets.trajectory_dataset import load_trajectory_csv
from utils.character_features import SPATIAL_CHANNEL_NAMES, extract_character_spatial_maps
from utils.character_alignment import align_target_to_trajectory, alignment_metrics
from utils.image_preprocessing import letterbox_character_image, load_character_image


TARGET_PADDING = 4


def rasterize_character(
    normalized_strokes,
    canvas_size: int,
    width: int = 5,
) -> np.ndarray:
    image = Image.new("L", (canvas_size, canvas_size), 0)
    draw = ImageDraw.Draw(image)
    for points in normalized_strokes:
        if len(points) >= 2:
            draw.line(points, fill=255, width=width, joint="curve")
        elif len(points) == 1:
            x, y = points[0]
            radius = max(width // 2, 1)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
    return np.asarray(image, dtype=np.float32) / 255.0


def build_image_index(dataset: CalligraphyImageDataset) -> DefaultDict[str, List[Dict[str, Any]]]:
    result: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in dataset.index:
        character = item.get("character")
        if character:
            result[str(character)].append(item)
    return result


def select_best_real_target(
    image_dataset: CalligraphyImageDataset,
    candidates: List[Dict[str, Any]],
    spatial_maps: np.ndarray,
    canvas_size: int,
    max_target_candidates: int,
    max_registered_candidates: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Clean candidates, register the strongest matches, and return the best one."""
    if max_target_candidates > 0:
        candidates = candidates[:max_target_candidates]
    prepared = []
    errors = 0
    for selected in candidates:
        try:
            image_sample = image_dataset._build_sample(selected)
            target, transform = letterbox_character_image(
                image_sample["image"],
                canvas_size=canvas_size,
                padding=TARGET_PADDING,
                crop_foreground=True,
            )
            raw_metrics = alignment_metrics(target, spatial_maps[0], spatial_maps[1])
            prepared.append((raw_metrics["score"], target, transform, selected, raw_metrics))
        except (OSError, ValueError, RuntimeError) as error:
            errors += 1
            print(f"[WARN] Target candidate failed: {selected.get('image_path')}: {error}")
    if not prepared:
        raise ValueError("No readable target candidates remain for this character")

    prepared.sort(key=lambda item: item[0], reverse=True)
    registration_pool = prepared[:max(max_registered_candidates, 1)]
    best = None
    for raw_score, target, transform, selected, raw_metrics in registration_pool:
        aligned, registration = align_target_to_trajectory(
            target,
            centerline=spatial_maps[0],
            proximity=spatial_maps[1],
        )
        score = float(registration["after"]["score"])
        if best is None or score > best[0]:
            best = (score, aligned, transform, selected, raw_metrics, registration)

    _, aligned, transform, selected, raw_metrics, registration = best
    info = {
        "image_path": str(selected.get("image_path")),
        "json_path": str(selected.get("json_path")),
        "bbox": selected.get("bbox"),
        "canvas_transform": transform,
        "registration": registration,
        "target_candidates_considered": len(prepared),
        "target_candidates_registered": len(registration_pool),
        "target_candidate_errors": errors,
        "unregistered_alignment": raw_metrics,
    }
    return aligned, info


def main(args) -> None:
    if not 0.0 <= args.min_alignment_coverage <= 1.0:
        raise ValueError("--min_alignment_coverage must be in [0, 1]")
    if args.max_target_candidates < 0:
        raise ValueError("--max_target_candidates must be non-negative")
    if args.max_registered_candidates < 1:
        raise ValueError("--max_registered_candidates must be positive")
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    if args.chirography is not None:
        cfg.data.chirography_filter = args.chirography

    char_cfg = cfg.character_generator
    canvas_size = int(char_cfg.image_size)
    trajectory_padding = (
        int(args.trajectory_padding)
        if args.trajectory_padding is not None
        else int(cfg.data.character_trajectory_padding)
    )
    if char_cfg.input_channels != len(SPATIAL_CHANNEL_NAMES):
        raise ValueError(
            f"character_generator.input_channels={char_cfg.input_channels}, but the spatial "
            f"schema contains {len(SPATIAL_CHANNEL_NAMES)} channels"
        )
    if args.target_image and not args.target_character:
        raise ValueError("--target_image requires --target_character")

    trajectories = load_trajectory_csv(args.trajectory_csv or cfg.data.trajectory_csv)
    if args.character:
        trajectories = [sample for sample in trajectories if sample.character == args.character]
    if not trajectories:
        raise RuntimeError("No trajectory samples matched the requested character filter")
    print(f"[INFO] Loaded {len(trajectories)} character trajectories")

    image_dataset = None
    image_index: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    if Path(cfg.data.image_dir).exists() and Path(cfg.data.json_dir).exists():
        image_dataset = CalligraphyImageDataset(
            image_dir=cfg.data.image_dir,
            json_dir=cfg.data.json_dir,
            image_ext=cfg.data.image_ext,
            image_size=None,
            grayscale=True,
            padding=0.0,
            data_csv=getattr(cfg.data, "data_csv", None),
            chirography_filter=getattr(cfg.data, "chirography_filter", None),
        )
        image_index = build_image_index(image_dataset)
        print(f"[INFO] Indexed real targets for {len(image_index)} characters")
    else:
        print("[WARN] Calligraphy image folders are absent; synthetic whole-character targets will be used")

    override_target = None
    override_transform = None
    if args.target_image:
        override_target, override_transform = load_character_image(
            args.target_image,
            canvas_size=canvas_size,
            padding=TARGET_PADDING,
        )
        print(
            f"[INFO] External target {args.target_image} will supervise "
            f"character {args.target_character!r}"
        )

    inputs: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    metadata: List[Dict[str, Any]] = []
    source_counts: DefaultDict[str, int] = defaultdict(int)
    alignment_coverages: List[float] = []
    rejected_pairs: List[Dict[str, Any]] = []
    skipped = 0

    for sample_index, trajectory in enumerate(trajectories):
        character = trajectory.character
        if not character:
            skipped += 1
            continue
        try:
            spatial_maps, normalized_strokes = extract_character_spatial_maps(
                trajectory,
                canvas_size=canvas_size,
                padding=trajectory_padding,
                line_width=args.trajectory_width,
            )
        except ValueError as error:
            print(f"[SKIP] sample={trajectory.meta.get('sample_id')}: {error}")
            skipped += 1
            continue

        source_type = "synthetic"
        source_info: Dict[str, Any] = {}
        if override_target is not None and character == args.target_character:
            target, registration = align_target_to_trajectory(
                override_target.copy(),
                centerline=spatial_maps[0],
                proximity=spatial_maps[1],
            )
            source_type = "external"
            source_info = {
                "image_path": str(args.target_image),
                "canvas_transform": override_transform,
                "registration": registration,
            }
        elif image_dataset is not None and image_index.get(character):
            candidates = image_index[character]
            try:
                target, source_info = select_best_real_target(
                    image_dataset,
                    candidates,
                    spatial_maps,
                    canvas_size,
                    max_target_candidates=args.max_target_candidates,
                    max_registered_candidates=args.max_registered_candidates,
                )
            except ValueError as error:
                rejected_pairs.append({
                    "character": character,
                    "sample_id": trajectory.meta.get("sample_id"),
                    "reason": "target_selection_failed",
                    "error": str(error),
                })
                skipped += 1
                continue
            source_type = "real"
        else:
            if args.require_real_target:
                rejected_pairs.append({
                    "character": character,
                    "sample_id": trajectory.meta.get("sample_id"),
                    "reason": "no_real_target_candidate",
                })
                skipped += 1
                continue
            target = rasterize_character(
                normalized_strokes,
                canvas_size=canvas_size,
                width=args.synthetic_width,
            )

        centerline_mask = spatial_maps[0] >= 0.5
        trajectory_target_coverage = float(
            (target[centerline_mask] >= 0.5).mean()
        )
        alignment = alignment_metrics(target, spatial_maps[0], spatial_maps[1])
        if (
            source_type in {"real", "external"}
            and trajectory_target_coverage < args.min_alignment_coverage
        ):
            rejected_pairs.append({
                "character": character,
                "sample_id": trajectory.meta.get("sample_id"),
                "target_source": source_type,
                "image_path": source_info.get("image_path"),
                "coverage": trajectory_target_coverage,
                "support_dice": alignment["support_dice"],
                "reason": "coverage_below_threshold",
                "threshold": args.min_alignment_coverage,
            })
            skipped += 1
            continue
        alignment_coverages.append(trajectory_target_coverage)

        # Float16 is sufficient for normalized conditioning/target maps and
        # keeps a full 128x128 six-channel corpus practical in host memory.
        inputs.append(spatial_maps.astype(np.float16))
        targets.append(np.clip(target, 0.0, 1.0).astype(np.float16)[None, ...])
        metadata.append({
            "character": character,
            "sample_id": trajectory.meta.get("sample_id"),
            "trajectory_sample_index": sample_index,
            "num_strokes": len(trajectory.sorted_strokes()),
            "trajectory_padding": trajectory_padding,
            "trajectory_width": args.trajectory_width,
            "trajectory_target_coverage": trajectory_target_coverage,
            "alignment_metrics": alignment,
            "target_source": source_type,
            **source_info,
        })
        source_counts[source_type] += 1

    output_path = Path(args.output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rejection_path = output_path.with_suffix(".rejected.json")
    with open(rejection_path, "w", encoding="utf-8") as file:
        json.dump(rejected_pairs, file, ensure_ascii=False, indent=2)
    if not inputs:
        raise RuntimeError(
            "No character-level pairs passed target cleanup/alignment. "
            f"See {rejection_path}"
        )
    if override_target is not None and source_counts["external"] == 0:
        raise RuntimeError(
            f"The external target was not used because character "
            f"{args.target_character!r} was absent from the selected trajectories"
        )

    np.savez_compressed(
        output_path,
        inputs=np.stack(inputs),
        targets=np.stack(targets),
        meta=np.asarray(metadata, dtype=object),
        channel_names=np.asarray(SPATIAL_CHANNEL_NAMES),
        format_version=np.asarray(CHARACTER_DATA_FORMAT),
        trajectory_padding=np.asarray(trajectory_padding, dtype=np.int32),
        trajectory_width=np.asarray(args.trajectory_width, dtype=np.int32),
        preprocessing_version=np.asarray("clean_register_v1"),
        min_alignment_coverage=np.asarray(args.min_alignment_coverage, dtype=np.float32),
    )
    print(f"[DONE] Character pairs saved to: {output_path}")
    print(
        f"[DONE] inputs shape: "
        f"({len(inputs)}, {len(SPATIAL_CHANNEL_NAMES)}, {canvas_size}, {canvas_size})"
    )
    print(f"[DONE] channels: {', '.join(SPATIAL_CHANNEL_NAMES)}")
    print(
        f"[DONE] trajectory normalization: padding={trajectory_padding}, "
        f"width={args.trajectory_width}"
    )
    print(f"[DONE] targets shape: ({len(targets)}, 1, {canvas_size}, {canvas_size})")
    print(f"[DONE] target sources: {dict(source_counts)}; skipped={skipped}")
    print(
        f"[DONE] alignment filter: threshold={args.min_alignment_coverage:.4f}, "
        f"rejected={len(rejected_pairs)}, report={rejection_path}"
    )
    print(
        "[CHECK] trajectory/target coverage: "
        f"mean={np.mean(alignment_coverages):.4f}, "
        f"median={np.median(alignment_coverages):.4f}, "
        f"min={np.min(alignment_coverages):.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build six-channel spatial trajectory maps for the whole-character U-Net"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--trajectory_csv", default=None)
    parser.add_argument("--output_npz", default="data/processed/character_train.npz")
    parser.add_argument("--character", default=None, help="Optional exact character filter")
    parser.add_argument("--target_character", default=None, help="Character receiving --target_image")
    parser.add_argument("--target_image", default=None, help="Optional external whole-character target")
    parser.add_argument("--chirography", default=None)
    parser.add_argument("--trajectory_width", type=int, default=3)
    parser.add_argument("--trajectory_padding", type=int, default=None)
    parser.add_argument("--synthetic_width", type=int, default=5)
    parser.add_argument("--require_real_target", action="store_true")
    parser.add_argument("--min_alignment_coverage", type=float, default=0.55)
    parser.add_argument("--max_target_candidates", type=int, default=64)
    parser.add_argument("--max_registered_candidates", type=int, default=8)
    main(parser.parse_args())
