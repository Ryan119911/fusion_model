"""Evaluate geometry before and brush appearance after style refinement."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.style_refiner import build_style_refiner
from datasets.trajectory_dataset import load_trajectory_csv
from utils.kaishu_style_features import build_style_features, geometry_features
from utils.image_preprocessing import normalize_image_polarity
from utils.structure_mask import skeletonize_binary


FORMAT = "style_refined_render_evaluation_v1"


def exact_canvas(path: str, image_size: int) -> np.ndarray:
    with Image.open(path) as image:
        gray = normalize_image_polarity(image.convert("L"))
    if gray.shape != (image_size, image_size):
        gray = np.asarray(
            Image.fromarray(np.rint(gray * 255).astype(np.uint8)).resize(
                (image_size, image_size), Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        ) / 255.0
    return gray.astype(np.float32)


def image_metrics(
    prediction: np.ndarray, target: np.ndarray, threshold: float
) -> dict[str, float]:
    prediction = np.clip(np.asarray(prediction, dtype=np.float32), 0, 1)
    target = np.clip(np.asarray(target, dtype=np.float32), 0, 1)
    pred_mask = prediction >= threshold
    target_mask = target >= threshold
    intersection = np.logical_and(pred_mask, target_mask).sum()
    union = np.logical_or(pred_mask, target_mask).sum()
    pred_count, target_count = pred_mask.sum(), target_mask.sum()
    pred_skeleton = skeletonize_binary(pred_mask)
    target_skeleton = skeletonize_binary(target_mask)
    target_distance = distance_transform_edt(~target_skeleton)
    pred_distance = distance_transform_edt(~pred_skeleton)
    pred_to_target = float(target_distance[pred_skeleton].mean()) if pred_skeleton.any() else float("inf")
    target_to_pred = float(pred_distance[target_skeleton].mean()) if target_skeleton.any() else float("inf")
    return {
        "mse": float(np.mean((prediction - target) ** 2)),
        "mae": float(np.mean(np.abs(prediction - target))),
        "dice": float((2 * intersection + 1e-6) / (pred_count + target_count + 1e-6)),
        "iou": float((intersection + 1e-6) / (union + 1e-6)),
        "prediction_ink": float(prediction.mean()),
        "target_ink": float(target.mean()),
        "ink_ratio": float((prediction.mean() + 1e-8) / (target.mean() + 1e-8)),
        "skeleton_pred_to_target_px": pred_to_target,
        "skeleton_target_to_pred_px": target_to_pred,
        "symmetric_skeleton_distance_px": 0.5 * (pred_to_target + target_to_pred),
    }


def solve_clipped_ink_gain(
    prediction: np.ndarray,
    target_mean: float,
    min_gain: float,
    max_gain: float,
    iterations: int = 48,
) -> float:
    """Solve mean(clip(prediction * gain)) under explicit gain bounds."""
    if min_gain <= 0 or max_gain < min_gain:
        raise ValueError("ink gain bounds must satisfy 0 < min <= max")
    prediction = np.clip(
        np.asarray(prediction, dtype=np.float32), 0.0, 1.0
    )
    target_mean = float(np.clip(target_mean, 0.0, 1.0))
    low_mean = float(np.clip(prediction * min_gain, 0, 1).mean())
    high_mean = float(np.clip(prediction * max_gain, 0, 1).mean())
    if target_mean <= low_mean:
        return float(min_gain)
    if target_mean >= high_mean:
        return float(max_gain)
    low, high = float(min_gain), float(max_gain)
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        value = float(np.clip(prediction * middle, 0, 1).mean())
        if value < target_mean:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def pose_continuity(csv_path: str) -> dict[str, Any]:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    per_field: dict[str, list[float]] = {name: [] for name in ("z", "alpha", "beta", "gamma")}
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["stroke_id"], []).append(row)
    for stroke_rows in grouped.values():
        stroke_rows.sort(key=lambda row: int(row["point_id"]))
        for left, right in zip(stroke_rows, stroke_rows[1:]):
            for field in per_field:
                per_field[field].append(abs(float(right[field]) - float(left[field])))
    return {
        "points": len(rows),
        "strokes": len(grouped),
        "max_adjacent_step": {
            field: max(values, default=0.0) for field, values in per_field.items()
        },
        "rms_adjacent_step": {
            field: float(np.sqrt(np.mean(np.square(values)))) if values else 0.0
            for field, values in per_field.items()
        },
        "units": {"z": "mm", "alpha": "rad", "beta": "rad", "gamma": "rad"},
    }


def find_nested(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for child in value.values():
            found = find_nested(child, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = find_nested(child, key)
            if found is not None:
                return found
    return None


def pose_safety(
    report_path: str,
    trajectory_csv: str,
    posture_report_path: str | None = None,
) -> dict[str, Any]:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    decisions = report.get("field_decisions", {})
    posture_report = (
        json.loads(Path(posture_report_path).read_text(encoding="utf-8"))
        if posture_report_path
        else report
    )
    joint_audit = find_nested(posture_report, "joint_jacobian_audit")
    return {
        "simulation_only": bool(report.get("simulation_only", True)),
        "source_report": report_path,
        "trajectory_csv": trajectory_csv,
        "posture_source_report": posture_report_path or report_path,
        "trajectory_target_coverage_at_5px": report.get(
            "trajectory_target_coverage_at_5px"
        ),
        "field_boundary_fractions": {
            field: values.get("boundary_fraction")
            for field, values in decisions.items()
        },
        "joint_jacobian_audit": joint_audit,
        "posture_field_decisions": posture_report.get("field_decisions", {}),
        "xy_optimization": report.get("xy_optimization"),
        "continuity": pose_continuity(trajectory_csv),
        "warning": report.get("warning"),
    }


def paper_image(array: np.ndarray) -> Image.Image:
    return Image.fromarray(
        np.rint((1.0 - np.clip(array, 0, 1)) * 255).astype(np.uint8), mode="L"
    )


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.style_ckpt, map_location=device, weights_only=False)
    model_config = checkpoint.get("model_config", {})
    model = build_style_refiner(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    render = exact_canvas(args.render_image, args.image_size)
    # Match optim.trajectory_optimizer.load_target_image: preserve the original
    # canvas and resize directly. Re-letterboxing here would enlarge the glyph
    # and make the post-hoc report incomparable with the inversion objective.
    target = exact_canvas(args.target_image, args.image_size)
    alignment = None
    if int(model_config.get("input_channels", 4)) == 4:
        features = geometry_features(render, threshold=args.structure_threshold)
    else:
        trajectories = [
            value
            for value in load_trajectory_csv(args.trajectory_csv)
            if value.character == args.character
        ]
        if not trajectories:
            raise RuntimeError(
                f"No trajectory found for v16 style evaluation character {args.character!r}"
            )
        features, alignment = build_style_features(
            render,
            trajectories[0],
            canvas_size=args.image_size,
            trajectory_padding=args.trajectory_padding,
            trajectory_width=args.trajectory_width,
            structure_threshold=args.structure_threshold,
            footprint_width_scale_px=args.footprint_width_scale_px,
        )
    with torch.no_grad():
        refined = model(
            torch.from_numpy(features[None]).to(device)
        )[0, 0].cpu().numpy()
    geometry = image_metrics(render, target, args.metric_threshold)
    appearance = image_metrics(refined, target, args.metric_threshold)
    calibration_gain = solve_clipped_ink_gain(
        refined,
        float(target.mean()),
        args.min_ink_gain,
        args.max_ink_gain,
    )
    calibrated = np.clip(refined * calibration_gain, 0.0, 1.0)
    calibrated_appearance = image_metrics(
        calibrated, target, args.metric_threshold
    )
    before_ink_balance = float(
        np.exp(-abs(np.log(max(geometry["ink_ratio"], 1e-8))))
    )
    calibrated_ink_balance = float(
        np.exp(-abs(np.log(max(calibrated_appearance["ink_ratio"], 1e-8))))
    )
    appearance_accepted = bool(
        calibrated_appearance["mse"] < geometry["mse"]
        and calibrated_appearance["iou"] >= geometry["iou"] - 0.002
        and calibrated_ink_balance >= before_ink_balance - 0.01
    )
    report = {
        "format": FORMAT,
        "canonical_target": args.target_image,
        "render_image": args.render_image,
        "style_checkpoint": args.style_ckpt,
        "support_mode": model_config.get("support_mode", "mask_only"),
        "device": str(device),
        "metric_threshold": args.metric_threshold,
        "character": args.character,
        "feature_channels": checkpoint.get("feature_channels"),
        "v16_alignment": alignment,
        "geometry_before_refinement": geometry,
        "appearance_after_refinement": appearance,
        "ink_calibrated_appearance": calibrated_appearance,
        "ink_calibration_gain": calibration_gain,
        "ink_calibration_method": (
            "bounded_bisection_after_output_clipping"
        ),
        "appearance_accepted": appearance_accepted,
        "appearance_acceptance_rule": (
            "MSE must improve, IoU may fall by at most 0.002, and ink-balance "
            "may fall by at most 0.01; geometry metrics remain authoritative."
        ),
        "delta_after_minus_before": {
            key: appearance[key] - geometry[key]
            for key in ("mse", "mae", "dice", "iou", "ink_ratio", "symmetric_skeleton_distance_px")
        },
        "geometry_gate": (
            "Bounded support derived only from the unrefined render's structure "
            "mask and soft geometry; the style model cannot serve as a pose label."
            if model_config.get("support_mode") == "mask_or_soft"
            else
            "Hard support from the unrefined render; the style model cannot "
            "create ink outside it or serve as a pose label."
        ),
        "pose_safety": pose_safety(
            args.pose_report,
            args.trajectory_csv,
            posture_report_path=args.posture_report,
        ),
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paper_image(target).save(output / "target.png")
    paper_image(render).save(output / "render_geometry.png")
    paper_image(refined).save(output / "render_refined.png")
    paper_image(calibrated).save(output / "render_refined_calibrated.png")
    difference = np.abs(calibrated - target)
    paper_image(difference).save(
        output / "diff.png"
    )
    panels = [
        paper_image(target),
        paper_image(render),
        paper_image(refined),
        paper_image(calibrated),
    ]
    panels.append(paper_image(difference))
    comparison = Image.new("L", (args.image_size * 5, args.image_size + 18), 255)
    draw = ImageDraw.Draw(comparison)
    for index, (panel, label) in enumerate(
        zip(panels, ("target", "geometry", "refined", "ink calibrated", "abs diff"))
    ):
        comparison.paste(panel, (index * args.image_size, 18))
        draw.text((index * args.image_size + 3, 2), label, fill=0)
    comparison.save(output / "comparison.png")
    (output / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render_image", required=True)
    parser.add_argument("--target_image", required=True)
    parser.add_argument("--style_ckpt", required=True)
    parser.add_argument("--pose_report", required=True)
    parser.add_argument(
        "--posture_report",
        default=None,
        help=(
            "Optional parent posture report when the current stage optimizes "
            "only x/y; its joint Jacobian audit remains authoritative."
        ),
    )
    parser.add_argument("--trajectory_csv", required=True)
    parser.add_argument("--character", default="武")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--structure_threshold", type=float, default=0.35)
    parser.add_argument("--metric_threshold", type=float, default=0.35)
    parser.add_argument("--min_ink_gain", type=float, default=0.8)
    parser.add_argument("--max_ink_gain", type=float, default=1.25)
    parser.add_argument("--trajectory_padding", type=int, default=4)
    parser.add_argument("--trajectory_width", type=int, default=3)
    parser.add_argument("--footprint_width_scale_px", type=float, default=16.0)
    main(parser.parse_args())
