"""Build leakage-controlled geometry-to-brush-appearance supervision."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.calligraphy_image_dataset import CalligraphyImageDataset
from utils.image_preprocessing import letterbox_character_image
from utils.structure_mask import build_structure_mask, skeletonize_binary


FORMAT = "kaishu_style_dataset_v1"


def geometry_features(gray: np.ndarray, threshold: float = 0.35) -> np.ndarray:
    mask, _ = build_structure_mask(
        gray, threshold=threshold, min_component_pixels=8, opening_iterations=1
    )
    binary = mask >= 0.5
    skeleton = skeletonize_binary(binary).astype(np.float32)
    inside = distance_transform_edt(binary).astype(np.float32)
    if inside.max() > 0:
        inside /= inside.max()
    soft = gaussian_filter(mask.astype(np.float32), sigma=1.2)
    return np.stack([mask, skeleton, inside, soft], axis=0).astype(np.float32)


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
            feature = geometry_features(gray, threshold=args.structure_threshold)
        except (OSError, RuntimeError, ValueError):
            skipped += 1
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
            }
        )
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
                    "feature_channels": [
                        "geometry_mask",
                        "skeleton",
                        "interior_distance",
                        "soft_geometry",
                    ],
                    "sample_count": len(features),
                    "skipped": skipped,
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
    main(parser.parse_args())
