import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from config import ensure_dirs, load_config
from datasets.trajectory_dataset import load_trajectory_csv
from models.dynamic_brush import build_dynamic_brush
from models.fusion_renderer import FusionRenderer
from optim.trajectory_optimizer import TrajectoryOptimizer, load_target_image


def save_character_trajectory_csv(sample, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "character", "sample_id", "stroke_id", "point_id",
        "x", "y", "z", "alpha", "beta", "gamma", "state", "timestamp",
    ]
    with open(output_path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for point in sample.all_points():
            writer.writerow(
                {
                    "character": sample.character,
                    "sample_id": sample.meta.get("sample_id"),
                    **point.as_dict(),
                }
            )


def save_image(array: np.ndarray, output_path: Path) -> None:
    Image.fromarray(
        np.clip(array * 255.0, 0, 255).astype(np.uint8), mode="L"
    ).save(output_path)


def pick_sample(samples, sample_id=None, character=None, index=0):
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


def main(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    samples = load_trajectory_csv(
        args.trajectory_csv or cfg.data.trajectory_csv,
        timestamp_column=cfg.data.timestamp_column,
        validate=cfg.data.validate_trajectories,
    )
    sample = pick_sample(
        samples, args.sample_id, args.character, args.index
    )
    target = load_target_image(args.target_image, cfg.bbsmg.image_size)
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
    if not args.bbsmg_ckpt:
        raise ValueError("--bbsmg_ckpt is required for meaningful optimization")
    renderer.load_weights(args.bbsmg_ckpt, args.normalization_npz)
    optimizer = TrajectoryOptimizer(
        renderer=renderer,
        render_samples=cfg.optim.render_samples_per_stroke,
        jacobian_epsilon=cfg.optim.jacobian_epsilon,
        xy_margin_ratio=cfg.optim.xy_margin_ratio,
        z_margin=cfg.optim.z_margin,
        angle_margin_radians=cfg.optim.angle_margin_radians,
        xyz_reg_weight=cfg.optim.xyz_reg_weight,
        z_reg_weight=cfg.optim.z_reg_weight,
        angle_reg_weight=cfg.optim.angle_reg_weight,
    )
    optimize_angles = bool(args.use_6d or cfg.optim.optimize_angles)
    order = args.order if args.order is not None else cfg.optim.cheb_order_max
    if not cfg.optim.cheb_order_min <= order <= cfg.optim.cheb_order_max:
        raise ValueError(
            f"order must be in [{cfg.optim.cheb_order_min}, {cfg.optim.cheb_order_max}]"
        )
    result = optimizer.optimize(
        sample,
        target,
        order=order,
        damping=cfg.optim.lm_damping,
        max_steps=cfg.optim.lm_max_steps,
        optimize_angles=optimize_angles,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{sample.character or 'sample'}_{sample.meta.get('sample_id', '0')}"
    save_character_trajectory_csv(
        result.optimized_sample, output_dir / f"{stem}_optimized.csv"
    )
    save_image(result.target_image, output_dir / f"{stem}_target.png")
    save_image(result.initial_image, output_dir / f"{stem}_initial.png")
    save_image(result.rendered_image, output_dir / f"{stem}_rendered.png")
    save_image(
        np.abs(result.rendered_image - result.target_image),
        output_dir / f"{stem}_diff.png",
    )
    report = {
        "sample_id": sample.meta.get("sample_id"),
        "character": sample.character,
        "feature_schema": cfg.bbsmg.feature_schema,
        "dynamic_brush_mode": cfg.dynamic_brush.mode,
        "optimize_angles": result.optimize_angles,
        "order": result.order,
        "lm_success": result.lm_result.success,
        "lm_steps": result.lm_result.num_steps,
        "lm_message": result.lm_result.message,
        "lm_final_cost": result.lm_result.final_cost,
        "initial_score": result.initial_score,
        "final_score": result.final_score,
        "accepted_update": result.final_score < result.initial_score,
    }
    with open(output_dir / f"{stem}_metrics.json", "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[DONE] Saved outputs to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--trajectory_csv")
    parser.add_argument("--target_image", required=True)
    parser.add_argument("--bbsmg_ckpt", required=True)
    parser.add_argument("--normalization_npz")
    parser.add_argument("--sample_id")
    parser.add_argument("--character")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--order", type=int)
    parser.add_argument("--use_6d", action="store_true")
    parser.add_argument("--output_dir", default="outputs/optimize")
    main(parser.parse_args())
