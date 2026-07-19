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
from models.character_generator import build_character_generator
from utils.character_features import extract_character_features, normalize_character_features
from utils.image_preprocessing import load_character_image


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


def image_metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    pred_binary = prediction >= 0.5
    target_binary = target >= 0.5
    intersection = float(np.logical_and(pred_binary, target_binary).sum())
    union = float(np.logical_or(pred_binary, target_binary).sum())
    dice_denominator = float(pred_binary.sum() + target_binary.sum())
    return {
        "plain_mse": float(np.mean((prediction - target) ** 2)),
        "mae": float(np.mean(np.abs(prediction - target))),
        "dice_at_0.5": (2.0 * intersection + 1e-6) / (dice_denominator + 1e-6),
        "iou_at_0.5": (intersection + 1e-6) / (union + 1e-6),
        "target_ink": float(target.mean()),
        "prediction_ink": float(prediction.mean()),
    }


def main(args) -> None:
    cfg = load_config(args.config)
    device = torch.device(
        cfg.train.device if torch.cuda.is_available() or cfg.train.device == "cpu" else "cpu"
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if checkpoint.get("format") != "character_generator_v1":
        raise ValueError(
            "Expected character_best.pt/character_last.pt from train_character.py; "
            "a stroke-level B-BSMG checkpoint cannot directly predict a whole character."
        )
    model_config = checkpoint["model_config"]
    normalization = checkpoint.get("input_normalization")
    if normalization is None:
        raise ValueError("Whole-character checkpoint has no input_normalization")
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
    raw_inputs, stroke_mask, _ = extract_character_features(
        sample,
        max_strokes=int(model_config["max_strokes"]),
        canvas_size=int(model_config["image_size"]),
        padding=4,
    )
    normalized = normalize_character_features(
        raw_inputs[None, ...],
        stroke_mask[None, ...],
        normalization,
    )
    inputs = torch.from_numpy(normalized).to(device)
    masks = torch.from_numpy(stroke_mask[None, ...]).to(device)
    with torch.no_grad():
        prediction = model(inputs, masks)[0, 0].clamp(0.0, 1.0).cpu().numpy()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    character = sample.character or args.character or "character"
    stem = args.output_stem or f"{character}_whole_character"
    save_gray(prediction, output_dir / f"{stem}_prediction.png")

    report = {
        "character": character,
        "sample_id": sample.meta.get("sample_id"),
        "num_strokes": int(stroke_mask.sum()),
        "checkpoint": str(args.checkpoint),
        "trajectory_csv": str(args.trajectory_csv or cfg.data.trajectory_csv),
    }
    if args.target_image:
        target, target_transform = load_character_image(
            args.target_image,
            canvas_size=int(model_config["image_size"]),
            padding=4,
        )
        difference = np.abs(prediction - target)
        save_gray(target, output_dir / f"{stem}_target.png")
        save_gray(difference, output_dir / f"{stem}_diff.png")
        save_gray(
            np.concatenate([target, prediction, difference], axis=1),
            output_dir / f"{stem}_comparison.png",
        )
        report.update({
            "target_image": str(args.target_image),
            "target_transform": target_transform,
            "panel_order": ["target", "prediction", "absolute_difference"],
            "metrics": image_metrics(prediction, target),
        })

    with open(output_dir / f"{stem}_metrics.json", "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(
        f"[DONE] Directly generated complete character {character!r} "
        f"from {int(stroke_mask.sum())} stroke tokens on {device}"
    )
    if args.target_image:
        for name, value in report["metrics"].items():
            print(f"{name}: {value:.6f}")
        print(f"[DONE] Comparison panel: {output_dir / (stem + '_comparison.png')}")
    else:
        print(f"[DONE] Prediction: {output_dir / (stem + '_prediction.png')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Directly predict one complete character")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--trajectory_csv", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--character", default=None)
    parser.add_argument("--sample_id", default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--target_image", default=None)
    parser.add_argument("--output_dir", default="outputs/predict_character")
    parser.add_argument("--output_stem", default=None)
    main(parser.parse_args())
