import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from config import ensure_dirs, load_config
from datasets.trajectory_dataset import load_trajectory_csv
from models.dynamic_brush import build_dynamic_brush
from models.fusion_renderer import FusionRenderer
from optim.trajectory_optimizer import load_target_image


def select_sample(
    samples: Sequence[Any],
    sample_id: Optional[str] = None,
    character: Optional[str] = None,
    index: int = 0,
) -> Any:
    if sample_id is not None:
        for sample in samples:
            if str(sample.meta.get("sample_id")) == str(sample_id):
                return sample
        raise ValueError(f"sample_id not found: {sample_id}")
    if character is not None:
        matches = [sample for sample in samples if sample.character == character]
        if not matches:
            raise ValueError(f"character not found: {character}")
        return matches[min(index, len(matches) - 1)]
    if not samples:
        raise RuntimeError("No trajectory samples found")
    return samples[min(index, len(samples) - 1)]


def save_image(array: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(
        np.clip(array * 255.0, 0, 255).astype(np.uint8), mode="L"
    ).save(output_path)


def calculate_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    if target.shape != prediction.shape:
        raise ValueError(
            f"Target and prediction shapes differ: {target.shape} != {prediction.shape}"
        )
    target = np.clip(np.asarray(target, dtype=np.float32), 0.0, 1.0)
    prediction = np.clip(np.asarray(prediction, dtype=np.float32), 0.0, 1.0)
    target_binary = target >= threshold
    prediction_binary = prediction >= threshold
    intersection = int(np.logical_and(target_binary, prediction_binary).sum())
    union = int(np.logical_or(target_binary, prediction_binary).sum())
    binary_total = int(target_binary.sum() + prediction_binary.sum())
    return {
        "mse": float(np.mean((prediction - target) ** 2)),
        "mae": float(np.mean(np.abs(prediction - target))),
        "binary_dice_at_0_5": (
            1.0 if binary_total == 0 else float(2 * intersection / binary_total)
        ),
        "binary_iou_at_0_5": 1.0 if union == 0 else float(intersection / union),
        "target_ink_ratio_at_0_5": float(target_binary.mean()),
        "prediction_ink_ratio_at_0_5": float(prediction_binary.mean()),
    }


def main(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    samples = load_trajectory_csv(
        args.trajectory_csv or cfg.data.trajectory_csv,
        timestamp_column=cfg.data.timestamp_column,
        validate=cfg.data.validate_trajectories,
    )
    sample = select_sample(samples, args.sample_id, args.character, args.index)
    renderer = FusionRenderer(
        image_size=cfg.bbsmg.image_size,
        device=cfg.train.device,
        input_dim=cfg.bbsmg.input_dim,
        feature_schema=cfg.bbsmg.feature_schema,
        latent_dim=cfg.bbsmg.latent_dim,
        base_channels=cfg.bbsmg.base_channels,
        out_channels=cfg.bbsmg.out_channels,
        use_tanh=cfg.bbsmg.use_tanh,
        brush=build_dynamic_brush(cfg.dynamic_brush),
    )
    renderer.load_weights(args.checkpoint, args.normalization_npz)
    prediction = np.asarray(
        renderer.render_character(sample)["character_image"], dtype=np.float32
    )
    target = load_target_image(args.target_image, cfg.bbsmg.image_size)
    difference = np.abs(prediction - target)
    comparison = np.concatenate([target, prediction, difference], axis=1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{sample.character or 'sample'}_{sample.meta.get('sample_id', '0')}"
    save_image(target, output_dir / f"{stem}_target.png")
    save_image(prediction, output_dir / f"{stem}_prediction.png")
    save_image(difference, output_dir / f"{stem}_diff.png")
    save_image(comparison, output_dir / f"{stem}_comparison.png")

    report: Dict[str, Any] = {
        "character": sample.character,
        "sample_id": sample.meta.get("sample_id"),
        "stroke_count": len(sample.strokes),
        "point_count": len(sample.all_points()),
        "feature_schema": cfg.bbsmg.feature_schema,
        "image_size": cfg.bbsmg.image_size,
        "device": str(renderer.device),
        "checkpoint": str(Path(args.checkpoint)),
        "normalization_npz": args.normalization_npz,
        "trajectory_csv": args.trajectory_csv or cfg.data.trajectory_csv,
        "target_image": str(Path(args.target_image)),
        "comparison_layout": "target | prediction | absolute_difference",
        "metrics": calculate_metrics(target, prediction),
    }
    metrics_path = output_dir / f"{stem}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[DONE] Saved character comparison to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render a full character from its trajectory and compare it with a target image."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--trajectory_csv")
    parser.add_argument("--target_image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--normalization_npz",
        help="Required only for legacy checkpoints without input normalization.",
    )
    parser.add_argument("--sample_id")
    parser.add_argument("--character")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output_dir", default="outputs/character_comparisons")
    main(parser.parse_args())
