import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Subset

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_config
from datasets.character_dataset import (
    CHARACTER_DATA_FORMAT,
    CharacterTrainDataset,
    collate_character_batch,
)
from models.character_generator import (
    CHARACTER_CHECKPOINT_FORMAT,
    SUPPORTED_CHARACTER_CHECKPOINT_FORMATS,
    build_character_generator,
)
from tools.train_bbsmg import set_seed
from tools.train_character import (
    binary_boundary_f1,
    build_loss,
    compute_batch_metrics,
)


def save_gray(array: np.ndarray, path: Path) -> None:
    Image.fromarray(np.clip(array * 255.0, 0, 255).astype(np.uint8), mode="L").save(path)


def metadata_dict(value) -> Dict:
    if isinstance(value, dict):
        return value
    if hasattr(value, "item"):
        item = value.item()
        return item if isinstance(item, dict) else {}
    return {}


def per_sample_metrics(
    predictions,
    targets,
    inputs,
    threshold: float,
) -> Dict[str, np.ndarray]:
    dims = (1, 2, 3)
    pred_binary = predictions >= threshold
    target_binary = targets >= 0.5
    centerline = inputs[:, 0:1] >= 0.5
    intersection = (pred_binary & target_binary).sum(dim=dims).float()
    union = (pred_binary | target_binary).sum(dim=dims).float()
    dice_denominator = pred_binary.sum(dim=dims).float() + target_binary.sum(dim=dims).float()
    centerline_count = centerline.sum(dim=dims).float().clamp_min(1.0)
    background = 1.0 - targets
    values = {
        "plain_mse": ((predictions - targets) ** 2).mean(dim=dims),
        "mae": torch.abs(predictions - targets).mean(dim=dims),
        "dice_at_threshold": (2.0 * intersection + 1e-6) / (dice_denominator + 1e-6),
        "iou_at_threshold": (intersection + 1e-6) / (union + 1e-6),
        "boundary_f1": binary_boundary_f1(
            predictions,
            targets,
            threshold=threshold,
        ),
        "uncertain_fraction": (
            (predictions > 0.1) & (predictions < 0.9)
        ).float().mean(dim=dims),
        "target_ink": targets.mean(dim=dims),
        "prediction_ink": predictions.mean(dim=dims),
        "target_mask_ink": target_binary.float().mean(dim=dims),
        "prediction_mask_ink": pred_binary.float().mean(dim=dims),
        "background_mean": (predictions * background).sum(dim=dims)
        / background.sum(dim=dims).clamp_min(1.0),
        "trajectory_target_coverage": (target_binary & centerline).sum(dim=dims).float()
        / centerline_count,
        "trajectory_prediction_coverage": (pred_binary & centerline).sum(dim=dims).float()
        / centerline_count,
        "zero_baseline_mse": (targets ** 2).mean(dim=dims),
    }
    values["probability_ink_ratio"] = (
        values["prediction_ink"] / values["target_ink"].clamp_min(1e-6)
    )
    values["mask_ink_ratio"] = (
        values["prediction_mask_ink"] / values["target_mask_ink"].clamp_min(1e-6)
    )
    values["ink_balance_score"] = torch.exp(
        -torch.abs(torch.log(values["mask_ink_ratio"].clamp_min(1e-6)))
    )
    return {name: value.detach().cpu().numpy() for name, value in values.items()}


def parse_thresholds(value: str) -> List[float]:
    thresholds = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not thresholds or any(not 0.0 < threshold < 1.0 for threshold in thresholds):
        raise ValueError("--thresholds must contain comma-separated values in (0, 1)")
    return thresholds


def main(args) -> None:
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must satisfy 0 < value < 1")
    cfg = load_config(args.config)
    set_seed(args.seed)
    device = torch.device(
        cfg.train.device if torch.cuda.is_available() or cfg.train.device == "cpu" else "cpu"
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if checkpoint.get("format") not in SUPPORTED_CHARACTER_CHECKPOINT_FORMATS:
        raise ValueError(
            f"Expected one of {SUPPORTED_CHARACTER_CHECKPOINT_FORMATS}. "
            "Rebuild the trajectory-faithful NPZ and train the U-Net from scratch."
        )
    model_config = checkpoint.get("model_config")
    if not model_config:
        raise ValueError("Checkpoint does not contain model_config")
    model_config = dict(model_config)
    if (
        checkpoint.get("format") != CHARACTER_CHECKPOINT_FORMAT
        and "geometry_gate_threshold" not in model_config
    ):
        # Reproduce legacy v7 output exactly; the hard geometry gate is a v8
        # architectural constraint and was not present in those checkpoints.
        model_config["geometry_gate_threshold"] = 0.0
    model = build_character_generator(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset = CharacterTrainDataset(args.npz_path)
    if checkpoint.get("data_format") != dataset.data_format:
        raise ValueError(
            "Checkpoint and NPZ data formats differ"
        )
    if checkpoint.get("target_mode") != dataset.target_mode:
        raise ValueError("Checkpoint and NPZ target modes differ")
    if abs(
        float(checkpoint.get("structure_threshold", -1.0))
        - dataset.structure_threshold
    ) > 1e-6:
        raise ValueError("Checkpoint and NPZ structure thresholds differ")
    if int(checkpoint.get("min_component_pixels", -1)) != dataset.min_component_pixels:
        raise ValueError("Checkpoint and NPZ component-cleanup settings differ")
    if int(checkpoint.get("opening_iterations", -1)) != dataset.opening_iterations:
        raise ValueError("Checkpoint and NPZ morphology-cleanup settings differ")
    if int(checkpoint.get("skeleton_tolerance", -1)) != dataset.skeleton_tolerance:
        raise ValueError("Checkpoint and NPZ skeleton tolerances differ")
    if dataset.data_format == CHARACTER_DATA_FORMAT:
        for key, dataset_value in (
            ("render_min_width", dataset.render_min_width),
            ("render_max_width", dataset.render_max_width),
            ("render_pressure_gamma", dataset.render_pressure_gamma),
        ):
            if abs(float(checkpoint.get(key, -1.0)) - dataset_value) > 1e-6:
                raise ValueError(f"Checkpoint and NPZ {key} values differ")
        if (
            bool(checkpoint.get("render_pressure_invert", False))
            != dataset.render_pressure_invert
        ):
            raise ValueError("Checkpoint and NPZ pressure directions differ")
    if tuple(checkpoint.get("channel_names", ())) != tuple(dataset.channel_names):
        raise ValueError("Checkpoint and NPZ spatial channel schemas differ")
    if int(checkpoint.get("trajectory_padding", -1)) != dataset.trajectory_padding:
        raise ValueError("Checkpoint and NPZ trajectory padding differ")
    if int(checkpoint.get("trajectory_width", -1)) != dataset.trajectory_width:
        raise ValueError("Checkpoint and NPZ trajectory width differ")
    if args.split == "val":
        indices: List[int] = list(checkpoint.get("val_indices", []))
        if not indices:
            raise ValueError("Checkpoint has no validation indices; use --split all")
    elif args.split == "train":
        indices = list(checkpoint.get("train_indices", []))
    else:
        indices = list(range(len(dataset)))
    if args.character:
        indices = [
            index for index in indices
            if metadata_dict(dataset.metadata[index]).get("character") == args.character
        ]
    if args.max_samples > 0:
        indices = indices[:args.max_samples]
    if not indices:
        raise RuntimeError("No samples matched the requested split and character")

    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_character_batch,
    )
    criterion = build_loss(device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    totals: Dict[str, float] = {}
    character_totals: Dict[str, Dict[str, float]] = {}
    character_counts: Dict[str, int] = {}
    count = 0
    saved = 0
    thresholds = sorted({*parse_thresholds(args.thresholds), float(args.threshold)})
    threshold_totals = {
        threshold: {
            "dice": 0.0,
            "iou": 0.0,
            "boundary_f1": 0.0,
            "mask_ink_ratio": 0.0,
            "ink_balance_score": 0.0,
        }
        for threshold in thresholds
    }

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            targets = batch["targets"].to(device).clamp(0.0, 1.0)
            predictions = model(inputs).clamp(0.0, 1.0)
            values = compute_batch_metrics(
                predictions,
                targets,
                inputs,
                criterion,
                threshold=args.threshold,
            )

            zeros = torch.zeros_like(targets)
            zero_binary = zeros.bool()
            target_binary = targets >= 0.5
            zero_intersection = (zero_binary & target_binary).sum(dim=(1, 2, 3)).float()
            zero_union = (zero_binary | target_binary).sum(dim=(1, 2, 3)).float()
            values.update({
                "zero_baseline_mse": float(torch.mean((targets - zeros) ** 2).item()),
                "zero_baseline_mae": float(torch.mean(torch.abs(targets - zeros)).item()),
                "zero_baseline_iou_at_0.5": float(
                    ((zero_intersection + 1e-6) / (zero_union + 1e-6)).mean().item()
                ),
            })
            sample_values = per_sample_metrics(
                predictions,
                targets,
                inputs,
                threshold=args.threshold,
            )
            batch_size = inputs.shape[0]
            for threshold in thresholds:
                sweep_values = per_sample_metrics(
                    predictions,
                    targets,
                    inputs,
                    threshold=threshold,
                )
                threshold_totals[threshold]["dice"] += float(
                    sweep_values["dice_at_threshold"].sum()
                )
                threshold_totals[threshold]["iou"] += float(
                    sweep_values["iou_at_threshold"].sum()
                )
                threshold_totals[threshold]["boundary_f1"] += float(
                    sweep_values["boundary_f1"].sum()
                )
                threshold_totals[threshold]["mask_ink_ratio"] += float(
                    sweep_values["mask_ink_ratio"].sum()
                )
                threshold_totals[threshold]["ink_balance_score"] += float(
                    sweep_values["ink_balance_score"].sum()
                )
            for name, value in values.items():
                totals[name] = totals.get(name, 0.0) + value * batch_size
            count += batch_size
            for item_index, meta in enumerate(batch["meta"]):
                character = str(meta.get("character") or "unknown")
                character_counts[character] = character_counts.get(character, 0) + 1
                bucket = character_totals.setdefault(character, {})
                for name, array in sample_values.items():
                    bucket[name] = bucket.get(name, 0.0) + float(array[item_index])

            pred_np = predictions.cpu().numpy()
            target_np = targets.cpu().numpy()
            input_np = inputs.cpu().numpy()
            for item_index in range(batch_size):
                if saved >= args.num_images:
                    break
                meta = batch["meta"][item_index]
                character = str(meta.get("character") or "character")
                stem = f"character_{saved:03d}_{character}"
                target = target_np[item_index, 0]
                prediction = pred_np[item_index, 0]
                prediction_mask = (prediction >= args.threshold).astype(np.float32)
                difference = np.abs(target - prediction)
                mask_difference = np.abs(target - prediction_mask)
                trajectory = input_np[item_index, 0]
                proximity = input_np[item_index, 1]
                save_gray(target, output_dir / f"{stem}_target.png")
                save_gray(prediction, output_dir / f"{stem}_prediction.png")
                save_gray(prediction_mask, output_dir / f"{stem}_prediction_mask.png")
                save_gray(difference, output_dir / f"{stem}_diff.png")
                save_gray(mask_difference, output_dir / f"{stem}_mask_diff.png")
                save_gray(trajectory, output_dir / f"{stem}_trajectory.png")
                save_gray(proximity, output_dir / f"{stem}_proximity.png")
                save_gray(
                    np.concatenate(
                        [trajectory, target, prediction, prediction_mask, mask_difference],
                        axis=1,
                    ),
                    output_dir / f"{stem}_comparison.png",
                )
                saved += 1

    metrics = {name: value / count for name, value in totals.items()}
    per_character = {
        character: {
            "samples": character_counts[character],
            **{
                name: value / character_counts[character]
                for name, value in character_totals[character].items()
            },
        }
        for character in sorted(character_totals)
    }
    per_character_metric_names = list(next(iter(character_totals.values())))
    macro_metrics = {
        name: float(np.mean([row[name] for row in per_character.values()]))
        for name in per_character_metric_names
    }
    threshold_sweep = [
        {
            "threshold": threshold,
            "macro_dice": values["dice"] / count,
            "macro_iou": values["iou"] / count,
            "macro_boundary_f1": values["boundary_f1"] / count,
            "macro_mask_ink_ratio": values["mask_ink_ratio"] / count,
            "macro_ink_balance_score": values["ink_balance_score"] / count,
            "balanced_score": (
                0.40 * values["iou"]
                + 0.30 * values["boundary_f1"]
                + 0.20 * values["dice"]
                + 0.10 * values["ink_balance_score"]
            ) / count,
        }
        for threshold, values in threshold_totals.items()
    ]
    best_threshold = max(threshold_sweep, key=lambda row: row["macro_iou"])
    best_balanced_threshold = max(
        threshold_sweep, key=lambda row: row["balanced_score"]
    )
    report = {
        "checkpoint": str(args.checkpoint),
        "npz_path": str(args.npz_path),
        "split": args.split,
        "character": args.character,
        "samples": count,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_val_metrics": checkpoint.get("val_metrics"),
        "trajectory_padding": checkpoint.get("trajectory_padding"),
        "trajectory_width": checkpoint.get("trajectory_width"),
        "target_mode": checkpoint.get("target_mode"),
        "render_min_width": checkpoint.get("render_min_width"),
        "render_max_width": checkpoint.get("render_max_width"),
        "render_pressure_gamma": checkpoint.get("render_pressure_gamma"),
        "render_pressure_invert": checkpoint.get("render_pressure_invert"),
        "operating_threshold": args.threshold,
        "threshold_sweep": threshold_sweep,
        "best_threshold_by_macro_iou": best_threshold,
        "best_threshold_by_balanced_score": best_balanced_threshold,
        "panel_order": [
            "trajectory",
            "target_structure",
            "prediction_probability",
            "prediction_mask",
            "mask_absolute_difference",
        ],
        "metrics": metrics,
        "characters": len(per_character),
        "macro_metrics": macro_metrics,
        "per_character": per_character,
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    with open(output_dir / "metrics.csv", "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)
    with open(
        output_dir / "per_character_metrics.csv", "w", encoding="utf-8-sig", newline=""
    ) as file:
        fieldnames = ["character", "samples", *per_character_metric_names]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for character, row in per_character.items():
            writer.writerow({"character": character, **row})
    print(f"[DONE] Evaluated {count} whole characters on {device}")
    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")
    print(
        f"[GENERALIZATION] characters={len(per_character)}, "
        f"macro_dice={macro_metrics['dice_at_threshold']:.6f}, "
        f"macro_iou={macro_metrics['iou_at_threshold']:.6f}, "
        f"macro_boundary_f1={macro_metrics['boundary_f1']:.6f}, "
        f"macro_probability_ink_ratio={macro_metrics['probability_ink_ratio']:.6f}, "
        f"macro_mask_ink_ratio={macro_metrics['mask_ink_ratio']:.6f}"
    )
    print(
        f"[THRESHOLD] best={best_threshold['threshold']:.2f}, "
        f"macro_iou={best_threshold['macro_iou']:.6f}, "
        f"macro_dice={best_threshold['macro_dice']:.6f}"
    )
    print(
        f"[BALANCED THRESHOLD] best={best_balanced_threshold['threshold']:.2f}, "
        f"score={best_balanced_threshold['balanced_score']:.6f}, "
        f"macro_iou={best_balanced_threshold['macro_iou']:.6f}, "
        f"mask_ink_ratio={best_balanced_threshold['macro_mask_ink_ratio']:.6f}"
    )
    print(f"[DONE] Reports and complete-character comparisons saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate whole-character U-Net predictions")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--npz_path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="outputs/eval_character")
    parser.add_argument("--split", choices=("val", "train", "all"), default="val")
    parser.add_argument("--character", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_images", type=int, default=20)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--thresholds",
        default="0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70",
        help="Comma-separated deployment thresholds evaluated on the selected split",
    )
    main(parser.parse_args())
