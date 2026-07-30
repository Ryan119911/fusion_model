"""Rank same-character calligraphy crops against a trajectory and target."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.calligraphy_image_dataset import CalligraphyImageDataset
from datasets.trajectory_dataset import load_trajectory_csv
from tools.invert_paper_trajectory import pick_sample
from utils.character_alignment import align_target_to_trajectory
from utils.character_features import extract_character_spatial_maps
from utils.image_preprocessing import (
    letterbox_character_image,
    load_character_image,
)
from utils.structure_mask import build_structure_mask, symmetric_structure_metrics


def binary_overlap(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    left_binary = np.asarray(left) >= 0.5
    right_binary = np.asarray(right) >= 0.5
    intersection = int(np.logical_and(left_binary, right_binary).sum())
    union = int(np.logical_or(left_binary, right_binary).sum())
    return {
        "dice_to_canonical": float(
            (2 * intersection + 1e-6)
            / (left_binary.sum() + right_binary.sum() + 1e-6)
        ),
        "iou_to_canonical": float((intersection + 1e-6) / (union + 1e-6)),
    }


def rgb_gray(array: np.ndarray) -> np.ndarray:
    gray = np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def candidate_tile(
    raw: np.ndarray,
    aligned: np.ndarray,
    structure: np.ndarray,
    centerline: np.ndarray,
    label: str,
) -> Image.Image:
    overlap = np.zeros((*structure.shape, 3), dtype=np.uint8)
    overlap[..., 0] = np.rint(np.clip(structure, 0, 1) * 255).astype(np.uint8)
    overlap[..., 1] = np.rint(np.clip(centerline, 0, 1) * 255).astype(np.uint8)
    panels = np.concatenate(
        [rgb_gray(raw), rgb_gray(aligned), rgb_gray(structure), overlap], axis=1
    )
    tile = Image.new("RGB", (panels.shape[1], panels.shape[0] + 18), "white")
    tile.paste(Image.fromarray(panels, mode="RGB"), (0, 18))
    ImageDraw.Draw(tile).text((3, 2), label, fill="black")
    return tile


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectories = load_trajectory_csv(args.trajectory_csv)
    sample = pick_sample(
        trajectories,
        sample_id=args.sample_id,
        character=args.character,
        index=args.index,
    )
    spatial_maps, _ = extract_character_spatial_maps(
        sample,
        canvas_size=args.image_size,
        padding=args.padding,
        line_width=args.trajectory_width,
    )
    canonical_gray, canonical_transform = load_character_image(
        args.target_image,
        canvas_size=args.image_size,
        padding=args.target_padding,
    )
    canonical_structure, canonical_cleanup = build_structure_mask(
        canonical_gray,
        threshold=args.structure_threshold,
        min_component_pixels=args.min_component_pixels,
        opening_iterations=args.opening_iterations,
    )
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
    candidates = [
        item for item in dataset.index if item["character"] == args.character
    ]
    records = []
    assets = {}
    for index, item in enumerate(candidates):
        try:
            sample_image = dataset._build_sample(item)
            raw, transform = letterbox_character_image(
                sample_image["image"],
                canvas_size=args.image_size,
                padding=args.target_padding,
                crop_foreground=True,
            )
            aligned, registration = align_target_to_trajectory(
                raw,
                centerline=spatial_maps[0],
                proximity=spatial_maps[1],
            )
            structure, cleanup = build_structure_mask(
                aligned,
                threshold=args.structure_threshold,
                min_component_pixels=args.min_component_pixels,
                opening_iterations=args.opening_iterations,
            )
        except (OSError, RuntimeError, ValueError) as error:
            records.append(
                {
                    "candidate_index": index,
                    "image_path": str(item["image_path"]),
                    "bbox": list(item["bbox"]),
                    "error": str(error),
                }
            )
            continue
        geometry = symmetric_structure_metrics(
            structure,
            centerline=spatial_maps[0],
            proximity=spatial_maps[1],
            skeleton_tolerance=args.skeleton_tolerance,
        )
        overlap = binary_overlap(structure, canonical_structure)
        ink_ratio = float(
            (structure.mean() + 1e-8) / (canonical_structure.mean() + 1e-8)
        )
        ink_balance = float(np.exp(-abs(np.log(max(ink_ratio, 1e-8)))))
        score = float(
            0.50 * geometry["symmetric_skeleton_score"]
            + 0.35 * overlap["dice_to_canonical"]
            + 0.15 * ink_balance
        )
        key = f"{index:03d}"
        records.append(
            {
                "candidate_index": index,
                "key": key,
                "image_path": str(item["image_path"]),
                "json_path": str(item["json_path"]),
                "bbox": list(item["bbox"]),
                "group_id": item["group_id"],
                "shape_index": item["shape_index"],
                "registration": registration,
                "canvas_transform": transform,
                "structure_cleanup": cleanup,
                **geometry,
                **overlap,
                "ink_ratio_to_canonical": ink_ratio,
                "ink_balance": ink_balance,
                "selection_score": score,
            }
        )
        assets[key] = (raw, aligned, structure)

    ranked = sorted(
        (record for record in records if "selection_score" in record),
        key=lambda record: record["selection_score"],
        reverse=True,
    )
    top = ranked[: args.top_k]
    tiles = []
    for rank, record in enumerate(top, start=1):
        raw, aligned, structure = assets[record["key"]]
        label = (
            f"#{rank} score={record['selection_score']:.3f} "
            f"skel={record['symmetric_skeleton_score']:.3f} "
            f"dice={record['dice_to_canonical']:.3f}"
        )
        tile = candidate_tile(
            raw, aligned, structure, spatial_maps[0], label
        )
        tile.save(output_dir / f"rank_{rank:02d}_{record['key']}.png")
        Image.fromarray(rgb_gray(raw), mode="RGB").save(
            output_dir / f"rank_{rank:02d}_raw.png"
        )
        Image.fromarray(rgb_gray(aligned), mode="RGB").save(
            output_dir / f"rank_{rank:02d}_aligned.png"
        )
        Image.fromarray(rgb_gray(structure), mode="RGB").save(
            output_dir / f"rank_{rank:02d}_structure.png"
        )
        tiles.append(tile)
    if tiles:
        columns = min(args.montage_columns, len(tiles))
        rows = (len(tiles) + columns - 1) // columns
        width = max(tile.width for tile in tiles)
        height = max(tile.height for tile in tiles)
        montage = Image.new("RGB", (columns * width, rows * height), "white")
        for index, tile in enumerate(tiles):
            montage.paste(
                tile, ((index % columns) * width, (index // columns) * height)
            )
        montage.save(output_dir / "top_candidates.png")

    report = {
        "format": "character_target_variant_audit_v1",
        "character": args.character,
        "chirography": args.chirography,
        "target_image": args.target_image,
        "target_transform": canonical_transform,
        "target_cleanup": canonical_cleanup,
        "candidate_count": len(candidates),
        "valid_candidate_count": len(ranked),
        "ranking_formula": (
            "0.50*symmetric_skeleton_score + "
            "0.35*dice_to_canonical + 0.15*ink_balance"
        ),
        "ranked_candidates": ranked,
        "failed_candidates": [
            record for record in records if "selection_score" not in record
        ],
    }
    (output_dir / "variants.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[DONE] character={args.character}, style={args.chirography}, "
        f"candidates={len(candidates)}, valid={len(ranked)}"
    )
    for rank, record in enumerate(top, start=1):
        print(
            f"[RANK {rank:02d}] score={record['selection_score']:.6f}, "
            f"skeleton={record['symmetric_skeleton_score']:.6f}, "
            f"dice={record['dice_to_canonical']:.6f}, "
            f"image={record['image_path']}"
        )
    print(f"[DONE] report={output_dir / 'variants.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory_csv", required=True)
    parser.add_argument("--target_image", required=True)
    parser.add_argument("--character", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sample_id", default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--image_dir", default="data/raw/images")
    parser.add_argument("--json_dir", default="data/raw/json_files")
    parser.add_argument("--data_csv", default="data/raw/data.csv")
    parser.add_argument("--image_ext", default=".jpg")
    parser.add_argument("--chirography", default="楷")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--padding", type=int, default=16)
    parser.add_argument("--target_padding", type=int, default=4)
    parser.add_argument("--trajectory_width", type=int, default=3)
    parser.add_argument("--structure_threshold", type=float, default=0.35)
    parser.add_argument("--min_component_pixels", type=int, default=8)
    parser.add_argument("--opening_iterations", type=int, default=1)
    parser.add_argument("--skeleton_tolerance", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=12)
    parser.add_argument("--montage_columns", type=int, default=2)
    main(parser.parse_args())
