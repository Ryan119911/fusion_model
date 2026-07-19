import argparse
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any, DefaultDict, Dict, List

import numpy as np
from PIL import Image, ImageDraw

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ensure_dirs, load_config
from datasets.calligraphy_image_dataset import CalligraphyImageDataset
from datasets.trajectory_dataset import load_trajectory_csv
from models.dynamic_brush import DynamicBrushModel
from utils.character_features import CHARACTER_FEATURE_NAMES, extract_character_features
from utils.image_preprocessing import letterbox_character_image, load_character_image


NORM_PADDING = 4


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


def main(args) -> None:
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    if args.chirography is not None:
        cfg.data.chirography_filter = args.chirography

    char_cfg = cfg.character_generator
    canvas_size = int(char_cfg.image_size)
    max_strokes = int(args.max_strokes or char_cfg.max_strokes)
    if char_cfg.input_dim != len(CHARACTER_FEATURE_NAMES):
        raise ValueError(
            f"character_generator.input_dim={char_cfg.input_dim}, but the feature schema "
            f"contains {len(CHARACTER_FEATURE_NAMES)} fields"
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
            padding=NORM_PADDING,
        )
        print(
            f"[INFO] External target {args.target_image} will supervise "
            f"character {args.target_character!r}"
        )

    brush = DynamicBrushModel()
    image_pick_counter: DefaultDict[str, int] = defaultdict(int)
    inputs: List[np.ndarray] = []
    stroke_masks: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    metadata: List[Dict[str, Any]] = []
    source_counts: DefaultDict[str, int] = defaultdict(int)
    skipped = 0

    for sample_index, trajectory in enumerate(trajectories):
        character = trajectory.character
        if not character:
            skipped += 1
            continue
        try:
            feature_array, stroke_mask, normalized_strokes = extract_character_features(
                trajectory,
                max_strokes=max_strokes,
                canvas_size=canvas_size,
                padding=NORM_PADDING,
                brush=brush,
            )
        except ValueError as error:
            print(f"[SKIP] sample={trajectory.meta.get('sample_id')}: {error}")
            skipped += 1
            continue

        source_type = "synthetic"
        source_info: Dict[str, Any] = {}
        if override_target is not None and character == args.target_character:
            target = override_target.copy()
            source_type = "external"
            source_info = {
                "image_path": str(args.target_image),
                "canvas_transform": override_transform,
            }
        elif image_dataset is not None and image_index.get(character):
            candidates = image_index[character]
            selected = candidates[image_pick_counter[character] % len(candidates)]
            image_pick_counter[character] += 1
            image_sample = image_dataset._build_sample(selected)
            target, transform = letterbox_character_image(
                image_sample["image"],
                canvas_size=canvas_size,
                padding=NORM_PADDING,
                crop_foreground=True,
            )
            source_type = "real"
            source_info = {
                "image_path": str(selected.get("image_path")),
                "json_path": str(selected.get("json_path")),
                "bbox": selected.get("bbox"),
                "canvas_transform": transform,
            }
        else:
            if args.require_real_target:
                skipped += 1
                continue
            target = rasterize_character(
                normalized_strokes,
                canvas_size=canvas_size,
                width=args.synthetic_width,
            )

        inputs.append(feature_array)
        stroke_masks.append(stroke_mask)
        targets.append(np.clip(target, 0.0, 1.0).astype(np.float32)[None, ...])
        metadata.append({
            "character": character,
            "sample_id": trajectory.meta.get("sample_id"),
            "trajectory_sample_index": sample_index,
            "num_strokes": int(stroke_mask.sum()),
            "target_source": source_type,
            **source_info,
        })
        source_counts[source_type] += 1

    if not inputs:
        raise RuntimeError("No character-level training pairs were generated")
    if override_target is not None and source_counts["external"] == 0:
        raise RuntimeError(
            f"The external target was not used because character "
            f"{args.target_character!r} was absent from the selected trajectories"
        )

    output_path = Path(args.output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        inputs=np.stack(inputs).astype(np.float32),
        stroke_masks=np.stack(stroke_masks).astype(np.bool_),
        targets=np.stack(targets).astype(np.float32),
        meta=np.asarray(metadata, dtype=object),
        feature_names=np.asarray(CHARACTER_FEATURE_NAMES),
        format_version=np.asarray("character_sequence_v1"),
    )
    print(f"[DONE] Character pairs saved to: {output_path}")
    print(f"[DONE] inputs shape: ({len(inputs)}, {max_strokes}, {len(CHARACTER_FEATURE_NAMES)})")
    print(f"[DONE] targets shape: ({len(targets)}, 1, {canvas_size}, {canvas_size})")
    print(f"[DONE] target sources: {dict(source_counts)}; skipped={skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build direct whole-character supervision from complete trajectories"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--trajectory_csv", default=None)
    parser.add_argument("--output_npz", default="data/processed/character_train.npz")
    parser.add_argument("--character", default=None, help="Optional exact character filter")
    parser.add_argument("--target_character", default=None, help="Character receiving --target_image")
    parser.add_argument("--target_image", default=None, help="Optional external whole-character target")
    parser.add_argument("--chirography", default=None)
    parser.add_argument("--max_strokes", type=int, default=None)
    parser.add_argument("--synthetic_width", type=int, default=5)
    parser.add_argument("--require_real_target", action="store_true")
    main(parser.parse_args())
