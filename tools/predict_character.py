import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_config
from datasets.trajectory_dataset import load_trajectory_csv
from models.character_generator import (
    CHARACTER_CHECKPOINT_FORMAT,
    SUPPORTED_CHARACTER_CHECKPOINT_FORMATS,
    build_character_generator,
)
from tools.train_character import binary_boundary_f1
from utils.character_alignment import align_target_to_trajectory
from utils.character_features import SPATIAL_CHANNEL_NAMES, extract_character_spatial_maps
from utils.image_preprocessing import load_character_image
from utils.structure_mask import STRUCTURE_TARGET_MODE, build_structure_mask
from utils.trajectory_target import (
    TRAJECTORY_TARGET_MODE,
    render_trajectory_target,
)


def pick_sample(samples, sample_id=None, character=None, index=0):
    if sample_id is not None:
        for sample in samples:
            if str(sample.meta.get("sample_id")) == str(sample_id):
                return sample
        raise ValueError(f"No trajectory sample has sample_id={sample_id!r}")
    candidates = samples
    if character is not None:
        candidates = [sample for sample in samples if sample.character == character]
    if not candidates:
        raise ValueError(f"No trajectory sample matched character={character!r}")
    if index < 0 or index >= len(candidates):
        raise IndexError(f"index={index} is outside the {len(candidates)} matching samples")
    return candidates[index]


def save_gray(array: np.ndarray, path: Path) -> None:
    Image.fromarray(np.clip(array * 255.0, 0, 255).astype(np.uint8), mode="L").save(path)


def image_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    threshold: float,
) -> dict:
    pred_binary = prediction >= threshold
    target_binary = target >= 0.5
    intersection = float(np.logical_and(pred_binary, target_binary).sum())
    union = float(np.logical_or(pred_binary, target_binary).sum())
    dice_denominator = float(pred_binary.sum() + target_binary.sum())
    prediction_tensor = torch.from_numpy(prediction[None, None]).float()
    target_tensor = torch.from_numpy(target[None, None]).float()
    return {
        "plain_mse": float(np.mean((prediction - target) ** 2)),
        "mae": float(np.mean(np.abs(prediction - target))),
        "binary_threshold": float(threshold),
        "dice_at_threshold": (2.0 * intersection + 1e-6) / (dice_denominator + 1e-6),
        "iou_at_threshold": (intersection + 1e-6) / (union + 1e-6),
        "target_ink": float(target.mean()),
        "prediction_ink": float(prediction.mean()),
        "target_mask_ink": float(target_binary.mean()),
        "prediction_mask_ink": float(pred_binary.mean()),
        "mask_ink_ratio": float(
            pred_binary.mean() / max(target_binary.mean(), 1e-6)
        ),
        "boundary_f1": float(
            binary_boundary_f1(
                prediction_tensor,
                target_tensor,
                threshold=threshold,
            ).item()
        ),
        "uncertain_fraction": float(
            np.logical_and(prediction > 0.1, prediction < 0.9).mean()
        ),
    }


def main(args) -> None:
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must satisfy 0 < value < 1")
    cfg = load_config(args.config)
    device = torch.device(
        cfg.train.device if torch.cuda.is_available() or cfg.train.device == "cpu" else "cpu"
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if checkpoint.get("format") not in SUPPORTED_CHARACTER_CHECKPOINT_FORMATS:
        raise ValueError(
            f"Expected one of {SUPPORTED_CHARACTER_CHECKPOINT_FORMATS}. "
            "Stroke-level B-BSMG checkpoints are incompatible."
        )
    if checkpoint.get("target_mode") not in (
        STRUCTURE_TARGET_MODE,
        TRAJECTORY_TARGET_MODE,
    ):
        raise ValueError("Checkpoint target mode is unsupported")
    model_config = dict(checkpoint["model_config"])
    if (
        checkpoint.get("format") != CHARACTER_CHECKPOINT_FORMAT
        and "geometry_gate_threshold" not in model_config
    ):
        model_config["geometry_gate_threshold"] = 0.0
    if tuple(checkpoint.get("channel_names", ())) != tuple(SPATIAL_CHANNEL_NAMES):
        raise ValueError("Checkpoint spatial channel schema is missing or incompatible")
    model = build_character_generator(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    samples = load_trajectory_csv(args.trajectory_csv or cfg.data.trajectory_csv)
    sample = pick_sample(
        samples,
        sample_id=args.sample_id,
        character=args.character,
        index=args.index,
    )
    spatial_maps, normalized_strokes = extract_character_spatial_maps(
        sample,
        canvas_size=int(model_config["image_size"]),
        padding=(
            int(args.trajectory_padding)
            if args.trajectory_padding is not None
            else int(checkpoint.get("trajectory_padding", cfg.data.character_trajectory_padding))
        ),
        line_width=(
            int(args.trajectory_width)
            if args.trajectory_width is not None
            else int(checkpoint.get("trajectory_width", 3))
        ),
    )
    inputs = torch.from_numpy(spatial_maps[None, ...]).to(device)
    with torch.no_grad():
        prediction = model(inputs)[0, 0].clamp(0.0, 1.0).cpu().numpy()
    prediction_mask = (prediction >= args.threshold).astype(np.float32)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    character = sample.character or args.character or "character"
    stem = args.output_stem or f"{character}_whole_character"
    save_gray(spatial_maps[0], output_dir / f"{stem}_trajectory.png")
    save_gray(spatial_maps[1], output_dir / f"{stem}_proximity.png")
    save_gray(prediction, output_dir / f"{stem}_prediction.png")
    save_gray(prediction_mask, output_dir / f"{stem}_prediction_mask.png")

    report = {
        "character": character,
        "sample_id": sample.meta.get("sample_id"),
        "num_strokes": len(sample.sorted_strokes()),
        "input_channels": list(SPATIAL_CHANNEL_NAMES),
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_val_metrics": checkpoint.get("val_metrics"),
        "trajectory_csv": str(args.trajectory_csv or cfg.data.trajectory_csv),
        "trajectory_padding": int(
            args.trajectory_padding
            if args.trajectory_padding is not None
            else checkpoint.get("trajectory_padding", cfg.data.character_trajectory_padding)
        ),
        "trajectory_width": int(
            args.trajectory_width
            if args.trajectory_width is not None
            else checkpoint.get("trajectory_width", 3)
        ),
        "trajectory_preview": str(output_dir / f"{stem}_trajectory.png"),
        "target_mode": checkpoint.get("target_mode"),
        "structure_threshold": checkpoint.get("structure_threshold"),
        "min_component_pixels": checkpoint.get("min_component_pixels"),
        "opening_iterations": checkpoint.get("opening_iterations"),
        "skeleton_tolerance": checkpoint.get("skeleton_tolerance"),
        "render_min_width": checkpoint.get("render_min_width"),
        "render_max_width": checkpoint.get("render_max_width"),
        "render_pressure_gamma": checkpoint.get("render_pressure_gamma"),
        "render_pressure_invert": checkpoint.get("render_pressure_invert"),
        "prediction_threshold": args.threshold,
    }
    if checkpoint.get("target_mode") == TRAJECTORY_TARGET_MODE:
        target, render_info = render_trajectory_target(
            sample,
            normalized_strokes,
            canvas_size=int(model_config["image_size"]),
            min_width=float(checkpoint["render_min_width"]),
            max_width=float(checkpoint["render_max_width"]),
            pressure_gamma=float(checkpoint["render_pressure_gamma"]),
            pressure_invert=bool(checkpoint.get("render_pressure_invert", False)),
        )
        difference = np.abs(prediction - target)
        mask_difference = np.abs(prediction_mask - target)
        save_gray(target, output_dir / f"{stem}_target.png")
        save_gray(difference, output_dir / f"{stem}_diff.png")
        save_gray(mask_difference, output_dir / f"{stem}_mask_diff.png")
        save_gray(
            np.concatenate(
                [spatial_maps[0], target, prediction, prediction_mask, mask_difference],
                axis=1,
            ),
            output_dir / f"{stem}_comparison.png",
        )
        centerline = spatial_maps[0] > 0.5
        target_binary = target >= 0.5
        prediction_binary = prediction >= args.threshold
        report.update({
            "trajectory_target_render": render_info,
            "panel_order": [
                "trajectory",
                "same_source_trajectory_target",
                "prediction_probability",
                "prediction_mask",
                "mask_absolute_difference",
            ],
            "metrics": {
                **image_metrics(prediction, target, threshold=args.threshold),
                "trajectory_target_coverage": float(target_binary[centerline].mean()),
                "trajectory_prediction_coverage": float(
                    prediction_binary[centerline].mean()
                ),
                "trajectory_target_mean": float(target[centerline].mean()),
                "trajectory_prediction_mean": float(prediction[centerline].mean()),
            },
        })
    elif args.target_image:
        target_gray, target_transform = load_character_image(
            args.target_image,
            canvas_size=int(model_config["image_size"]),
            padding=4,
        )
        aligned_gray, target_registration = align_target_to_trajectory(
            target_gray,
            centerline=spatial_maps[0],
            proximity=spatial_maps[1],
        )
        target, structure_info = build_structure_mask(
            aligned_gray,
            threshold=float(checkpoint.get("structure_threshold", 0.35)),
            min_component_pixels=int(checkpoint.get("min_component_pixels", 8)),
            opening_iterations=int(checkpoint.get("opening_iterations", 1)),
        )
        difference = np.abs(prediction - target)
        mask_difference = np.abs(prediction_mask - target)
        save_gray(aligned_gray, output_dir / f"{stem}_target_gray.png")
        save_gray(target, output_dir / f"{stem}_target.png")
        save_gray(difference, output_dir / f"{stem}_diff.png")
        save_gray(mask_difference, output_dir / f"{stem}_mask_diff.png")
        save_gray(
            np.concatenate(
                [spatial_maps[0], target, prediction, prediction_mask, mask_difference],
                axis=1,
            ),
            output_dir / f"{stem}_comparison.png",
        )
        centerline = spatial_maps[0] > 0.5
        target_binary = target >= 0.5
        prediction_binary = prediction >= args.threshold
        report.update({
            "target_image": str(args.target_image),
            "target_transform": target_transform,
            "target_registration": target_registration,
            "structure_cleanup": structure_info,
            "panel_order": [
                "trajectory",
                "target_structure",
                "prediction_probability",
                "prediction_mask",
                "mask_absolute_difference",
            ],
            "metrics": {
                **image_metrics(prediction, target, threshold=args.threshold),
                "trajectory_target_coverage": float(target_binary[centerline].mean()),
                "trajectory_prediction_coverage": float(prediction_binary[centerline].mean()),
                "trajectory_target_mean": float(target[centerline].mean()),
                "trajectory_prediction_mean": float(prediction[centerline].mean()),
            },
        })

    if (
        checkpoint.get("target_mode") == TRAJECTORY_TARGET_MODE
        and args.target_image
    ):
        external_gray, external_transform = load_character_image(
            args.target_image,
            canvas_size=int(model_config["image_size"]),
            padding=4,
        )
        aligned_external, external_registration = align_target_to_trajectory(
            external_gray,
            centerline=spatial_maps[0],
            proximity=spatial_maps[1],
        )
        external_target, external_cleanup = build_structure_mask(
            aligned_external,
            threshold=0.35,
            min_component_pixels=8,
            opening_iterations=1,
        )
        external_difference = np.abs(prediction_mask - external_target)
        save_gray(
            aligned_external,
            output_dir / f"{stem}_external_reference_gray.png",
        )
        save_gray(
            external_target,
            output_dir / f"{stem}_external_reference.png",
        )
        save_gray(
            np.concatenate(
                [
                    spatial_maps[0],
                    external_target,
                    prediction,
                    prediction_mask,
                    external_difference,
                ],
                axis=1,
            ),
            output_dir / f"{stem}_external_reference_comparison.png",
        )
        report["external_reference"] = {
            "warning": (
                "The external image is evaluation-only and is not a model input. "
                "Its stroke geometry may differ from the input trajectory."
            ),
            "target_image": str(args.target_image),
            "target_transform": external_transform,
            "target_registration": external_registration,
            "structure_cleanup": external_cleanup,
            "metrics": image_metrics(
                prediction,
                external_target,
                threshold=args.threshold,
            ),
        }

    with open(output_dir / f"{stem}_metrics.json", "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(
        f"[DONE] Directly generated complete character {character!r} "
        f"with the spatial U-Net on {device}"
    )
    if report.get("metrics"):
        for name, value in report["metrics"].items():
            print(f"{name}: {value:.6f}")
        print(f"[DONE] Comparison panel: {output_dir / (stem + '_comparison.png')}")
    else:
        print(f"[DONE] Prediction: {output_dir / (stem + '_prediction.png')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict one complete character with the spatial U-Net")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--trajectory_csv", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--character", default=None)
    parser.add_argument("--sample_id", default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--target_image", default=None)
    parser.add_argument("--output_dir", default="outputs/predict_character")
    parser.add_argument("--output_stem", default=None)
    parser.add_argument("--trajectory_width", type=int, default=None)
    parser.add_argument("--trajectory_padding", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    main(parser.parse_args())
