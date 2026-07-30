"""Calibrate soft geometry support under heldout-character constraints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.style_refiner import build_style_refiner
from tools.build_kaishu_style_dataset import geometry_features
from tools.evaluate_style_refined_render import (
    exact_canvas,
    image_metrics,
    paper_image,
)
from tools.train_kaishu_style_refiner import StyleDataset, evaluate


FORMAT = "style_support_calibration_v1"
LOSS_CONFIG = {
    "ink_weight": 0.75,
    "local_ink_weight": 0.75,
    "tone_balance_weight": 0.25,
}


def ink_balance(ratio: float) -> float:
    return float(np.exp(-abs(np.log(max(ratio, 1e-8)))))


def select_support_scale(
    candidates: list[dict[str, Any]],
    *,
    baseline_scale: float = 1.0,
    max_heldout_mse_regression: float = 1e-4,
    max_heldout_ink_balance_drop: float = 0.0,
    max_canonical_iou_drop: float = 0.002,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    baseline = min(
        candidates,
        key=lambda item: abs(float(item["scale"]) - baseline_scale),
    )
    eligible = []
    for candidate in candidates:
        heldout = candidate["heldout"]
        canonical = candidate["canonical"]
        accepted = bool(
            heldout["mse"]
            <= baseline["heldout"]["mse"] + max_heldout_mse_regression
            and heldout["ink_balance_score"]
            >= baseline["heldout"]["ink_balance_score"]
            - max_heldout_ink_balance_drop
            and canonical["mse"] <= baseline["canonical"]["mse"]
            and canonical["iou"]
            >= baseline["canonical"]["iou"] - max_canonical_iou_drop
            and canonical["ink_balance_score"]
            >= baseline["canonical"]["ink_balance_score"]
        )
        candidate["eligible"] = accepted
        eligible.append(candidate)
    selected = min(
        (item for item in eligible if item["eligible"]),
        key=lambda item: (item["canonical"]["mse"], item["scale"]),
        default=baseline,
    )
    return selected, eligible


def save_comparison(
    target: np.ndarray,
    render: np.ndarray,
    prediction: np.ndarray,
    output: Path,
) -> None:
    difference = np.abs(prediction - target)
    paper_image(target).save(output / "target.png")
    paper_image(render).save(output / "render_geometry.png")
    paper_image(prediction).save(output / "render_calibrated_support.png")
    Image.fromarray(
        np.rint(np.clip(difference, 0, 1) * 255).astype(np.uint8),
        mode="L",
    ).save(output / "diff.png")
    panels = [
        paper_image(target),
        paper_image(render),
        paper_image(prediction),
        Image.fromarray(
            np.rint(np.clip(difference, 0, 1) * 255).astype(np.uint8),
            mode="L",
        ),
    ]
    labels = ("target", "geometry", "style support", "abs diff")
    size = target.shape[0]
    comparison = Image.new("L", (size * len(panels), size + 18), 255)
    draw = ImageDraw.Draw(comparison)
    for index, (panel, label) in enumerate(zip(panels, labels)):
        comparison.paste(panel, (index * size, 18))
        draw.text((index * size + 3, 2), label, fill=0)
    comparison.save(output / "comparison.png")


def main(args: argparse.Namespace) -> None:
    if not args.scales or any(scale <= 0 for scale in args.scales):
        raise ValueError("--scales must contain positive values")
    device = torch.device(args.device)
    data = np.load(args.npz, allow_pickle=False)
    indices = np.flatnonzero(data["characters"] == args.heldout_character)
    if not len(indices):
        raise RuntimeError(
            f"No heldout samples found for {args.heldout_character!r}"
        )
    loader = DataLoader(
        StyleDataset(data["features"], data["targets"], indices),
        batch_size=args.batch_size,
        shuffle=False,
    )
    checkpoint = torch.load(
        args.style_ckpt, map_location=device, weights_only=False
    )
    config = dict(checkpoint.get("model_config", {}))
    if config.get("support_mode") != "mask_or_soft":
        raise ValueError(
            "Support calibration requires a mask_or_soft checkpoint"
        )
    model = build_style_refiner(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    render = exact_canvas(args.render_image, args.image_size)
    target = exact_canvas(args.target_image, args.image_size)
    features = torch.from_numpy(
        geometry_features(render)[None]
    ).to(device)
    candidates = []
    predictions = {}
    for scale in sorted(set(args.scales)):
        model.soft_support_scale = float(scale)
        heldout = evaluate(model, loader, device, LOSS_CONFIG)
        with torch.no_grad():
            prediction = model(features)[0, 0].cpu().numpy()
        canonical = image_metrics(
            prediction, target, args.metric_threshold
        )
        canonical["ink_balance_score"] = ink_balance(
            canonical["ink_ratio"]
        )
        candidates.append(
            {
                "scale": float(scale),
                "heldout": heldout,
                "canonical": canonical,
            }
        )
        predictions[float(scale)] = prediction
    selected, candidates = select_support_scale(
        candidates,
        max_heldout_mse_regression=args.max_heldout_mse_regression,
        max_heldout_ink_balance_drop=(
            args.max_heldout_ink_balance_drop
        ),
        max_canonical_iou_drop=args.max_canonical_iou_drop,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected_checkpoint = dict(checkpoint)
    selected_config = dict(config)
    selected_config["soft_support_scale"] = selected["scale"]
    selected_checkpoint["model_config"] = selected_config
    selected_checkpoint["support_calibration"] = {
        "format": FORMAT,
        "selected_scale": selected["scale"],
        "heldout_character": args.heldout_character,
        "heldout_samples": int(len(indices)),
        "simulation_only": True,
    }
    checkpoint_path = output / "style_refiner_support_calibrated.pt"
    torch.save(selected_checkpoint, checkpoint_path)
    report = {
        "format": FORMAT,
        "simulation_only": True,
        "checkpoint_source": args.style_ckpt,
        "checkpoint_output": str(checkpoint_path),
        "canonical_target": args.target_image,
        "render_image": args.render_image,
        "heldout_character": args.heldout_character,
        "heldout_samples": int(len(indices)),
        "selection_rule": {
            "objective": "minimum canonical MSE among eligible candidates",
            "max_heldout_mse_regression": args.max_heldout_mse_regression,
            "max_heldout_ink_balance_drop": (
                args.max_heldout_ink_balance_drop
            ),
            "max_canonical_iou_drop": args.max_canonical_iou_drop,
        },
        "selected_scale": selected["scale"],
        "selected": selected,
        "candidates": candidates,
        "warning": (
            "Appearance-support calibration only; it does not change or "
            "identify x/y/z/alpha/beta/gamma."
        ),
    }
    (output / "calibration.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_comparison(
        target,
        render,
        predictions[selected["scale"]],
        output,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--style_ckpt", required=True)
    parser.add_argument("--render_image", required=True)
    parser.add_argument("--target_image", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--heldout_character", default="武")
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=[1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 2.0, 2.5],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--metric_threshold", type=float, default=0.35)
    parser.add_argument(
        "--max_heldout_mse_regression", type=float, default=1e-4
    )
    parser.add_argument(
        "--max_heldout_ink_balance_drop", type=float, default=0.0
    )
    parser.add_argument(
        "--max_canonical_iou_drop", type=float, default=0.002
    )
    main(parser.parse_args())
