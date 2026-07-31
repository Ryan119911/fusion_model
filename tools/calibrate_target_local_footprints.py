"""Calibrate paper-model alpha/beta from target-image local footprints.

This remains a simulation calibration candidate: image cross sections are not
robot/TCP calibration data.  H and x/y are preserved from the input pose CSV;
gamma is exported as the per-stroke forward x/y heading.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.trajectory_dataset import load_trajectory_csv
from models.paper_bbsm import (
    drag_width_to_angles_given_height_torch,
    regression_matrix_numpy,
)
from models.paper_fusion_renderer import PaperDynamicConfig, PaperFusionRenderer
from optim.trajectory_optimizer import load_target_image
from tools.invert_paper_trajectory import (
    binary_metrics,
    pick_sample,
    save_pose_csv,
    source_xy_to_canvas,
)
from tools.render_paper_trajectory import load_pose_csv


def _sample_bilinear(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    h, w = image.shape
    x = np.clip(points[:, 0], 0.0, w - 1.0)
    y = np.clip(points[:, 1], 0.0, h - 1.0)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    wx, wy = x - x0, y - y0
    return (
        image[y0, x0] * (1 - wx) * (1 - wy)
        + image[y0, x1] * wx * (1 - wy)
        + image[y1, x0] * (1 - wx) * wy
        + image[y1, x1] * wx * wy
    )


def _nearest_ink_run(
    values: np.ndarray, offsets: np.ndarray, threshold: float
) -> tuple[float, float, bool]:
    active = values >= threshold
    indices = np.flatnonzero(active)
    if not len(indices):
        return 0.0, float("inf"), False
    breaks = np.flatnonzero(np.diff(indices) > 1) + 1
    runs = np.split(indices, breaks)
    run = min(runs, key=lambda item: abs(float(offsets[item].mean())))
    step = float(abs(offsets[1] - offsets[0])) if len(offsets) > 1 else 1.0
    extent = float(offsets[run[-1]] - offsets[run[0]] + step)
    center_distance = abs(float((offsets[run[-1]] + offsets[run[0]]) / 2.0))
    clipped = bool(run[0] == 0 or run[-1] == len(offsets) - 1)
    return extent, center_distance, clipped


def measure_local_footprints(
    target: np.ndarray,
    xy_canvas: np.ndarray,
    heading: np.ndarray,
    radius_px: float = 12.0,
    step_px: float = 0.25,
    threshold: float = 0.35,
) -> dict[str, np.ndarray]:
    """Measure tangent length and normal width around every path point."""
    offsets = np.arange(-radius_px, radius_px + step_px * 0.5, step_px)
    width, length, confidence = [], [], []
    width_center, length_center = [], []
    for center, theta in zip(xy_canvas, heading):
        tangent = np.asarray([np.cos(theta), np.sin(theta)], dtype=np.float32)
        normal = np.asarray([-np.sin(theta), np.cos(theta)], dtype=np.float32)
        normal_values = _sample_bilinear(
            target, center[None, :] + offsets[:, None] * normal[None, :]
        )
        tangent_values = _sample_bilinear(
            target, center[None, :] + offsets[:, None] * tangent[None, :]
        )
        w, wc, wclip = _nearest_ink_run(normal_values, offsets, threshold)
        length_value, lc, lclip = _nearest_ink_run(
            tangent_values, offsets, threshold
        )
        center_score = np.exp(-(wc + lc) / max(radius_px * 0.5, 1e-6))
        clip_score = (0.45 if wclip else 1.0) * (0.45 if lclip else 1.0)
        valid = float(w > 0.0 and length_value > 0.0)
        width.append(w)
        length.append(length_value)
        confidence.append(float(valid * center_score * clip_score))
        width_center.append(wc)
        length_center.append(lc)
    return {
        "width_px": np.asarray(width, dtype=np.float32),
        "length_px": np.asarray(length, dtype=np.float32),
        "confidence": np.asarray(confidence, dtype=np.float32),
        "width_center_offset_px": np.asarray(width_center, dtype=np.float32),
        "length_center_offset_px": np.asarray(length_center, dtype=np.float32),
    }


def robust_footprint_scales(
    measured: dict[str, np.ndarray],
    dynamic_drag: np.ndarray,
    dynamic_half_width: np.ndarray,
    pixels_per_model_unit: float,
    base_longitudinal_scale: float,
    base_transverse_scale: float,
    minimum_confidence: float = 0.1,
    relative_bounds: tuple[float, float] = (0.6, 1.6),
) -> tuple[float, float, np.ndarray]:
    """Estimate global axis scales before solving pointwise pose angles."""
    valid = (
        (measured["confidence"] >= minimum_confidence)
        & (measured["width_px"] > 0)
        & (measured["length_px"] > 0)
        & (dynamic_drag > 1e-6)
        & (dynamic_half_width > 1e-6)
    )
    if not np.any(valid):
        return base_longitudinal_scale, base_transverse_scale, valid
    longitudinal = np.median(
        measured["length_px"][valid]
        / (2.0 * pixels_per_model_unit * dynamic_drag[valid])
    )
    transverse = np.median(
        measured["width_px"][valid]
        / (2.0 * pixels_per_model_unit * dynamic_half_width[valid])
    )
    low, high = relative_bounds
    longitudinal = np.clip(
        longitudinal,
        base_longitudinal_scale * low,
        base_longitudinal_scale * high,
    )
    transverse = np.clip(
        transverse,
        base_transverse_scale * low,
        base_transverse_scale * high,
    )
    return float(longitudinal), float(transverse), valid


def _save_overlay(
    target: np.ndarray,
    xy: np.ndarray,
    heading: np.ndarray,
    measured: dict[str, np.ndarray],
    path: Path,
) -> None:
    base = Image.fromarray(np.rint(np.clip(target, 0, 1) * 255).astype(np.uint8))
    image = base.convert("RGB")
    draw = ImageDraw.Draw(image)
    for point, theta, width, length, confidence in zip(
        xy,
        heading,
        measured["width_px"],
        measured["length_px"],
        measured["confidence"],
    ):
        tangent = np.asarray([np.cos(theta), np.sin(theta)])
        normal = np.asarray([-np.sin(theta), np.cos(theta)])
        color = (0, int(80 + 175 * confidence), 255)
        a, b = point - normal * width / 2, point + normal * width / 2
        draw.line((tuple(a), tuple(b)), fill=color, width=1)
        a, b = point - tangent * length / 2, point + tangent * length / 2
        draw.line((tuple(a), tuple(b)), fill=(255, 80, 0), width=1)
    image.save(path)


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    sample = pick_sample(
        load_trajectory_csv(args.trajectory_csv),
        sample_id=args.sample_id,
        character=args.character,
        index=args.index,
    )
    posture, xy_source, _ = load_pose_csv(args.pose_csv, sample)
    xy = source_xy_to_canvas(sample, xy_source, args.image_size, args.padding)
    stroke_ids = np.asarray(
        [point.stroke_id for point in sample.all_points()], dtype=np.int64
    )
    target = load_target_image(args.target_image, args.image_size)
    renderer = PaperFusionRenderer.from_checkpoint(
        args.bbsmg_ckpt,
        device=device,
        image_size=args.image_size,
        dynamic=PaperDynamicConfig(
            pixels_per_model_unit=args.pixels_per_model_unit,
            footprint_longitudinal_scale=args.footprint_longitudinal_scale,
            footprint_transverse_scale=args.footprint_transverse_scale,
            fused_pose_from_height=True,
            inverse_regularization=args.pose_inverse_regularization,
        ),
    )
    xy_t = torch.as_tensor(xy, device=device)
    posture_t = torch.as_tensor(posture, device=device)
    stroke_t = torch.as_tensor(stroke_ids, device=device)
    with torch.no_grad():
        states = renderer.compute_dynamic_states(xy_t, posture_t, stroke_t)
        heading_t = states["forward_trajectory_heading"]
        geometry = states["geometry"]
    heading = heading_t.cpu().numpy()
    measured = measure_local_footprints(
        target,
        xy,
        heading,
        radius_px=args.radius_px,
        step_px=args.step_px,
        threshold=args.ink_threshold,
    )
    # Convert image footprint axes back to B-BSM dimensions. Dynamic geometry
    # supplies a stable fallback at intersections or clipped tangent runs.
    dynamic_drag = (geometry[:, 0] + geometry[:, 1]).cpu().numpy()
    dynamic_width = geometry[:, 2].cpu().numpy()
    calibrated_longitudinal_scale, calibrated_transverse_scale, valid = (
        robust_footprint_scales(
            measured,
            dynamic_drag,
            dynamic_width,
            args.pixels_per_model_unit,
            args.footprint_longitudinal_scale,
            args.footprint_transverse_scale,
            minimum_confidence=args.minimum_confidence,
        )
    )
    target_drag = measured["length_px"] / max(
        2.0 * args.pixels_per_model_unit * calibrated_longitudinal_scale,
        1e-6,
    )
    target_half_width = measured["width_px"] / max(
        2.0 * args.pixels_per_model_unit * calibrated_transverse_scale,
        1e-6,
    )
    confidence = measured["confidence"]
    blend = np.clip(confidence * args.target_blend, 0.0, 1.0)
    calibrated_drag = dynamic_drag * (1.0 - blend) + target_drag * blend
    calibrated_width = dynamic_width * (1.0 - blend) + target_half_width * blend
    with torch.no_grad():
        angles = drag_width_to_angles_given_height_torch(
            torch.as_tensor(calibrated_drag, device=device),
            torch.as_tensor(calibrated_width, device=device),
            posture_t[:, 0],
            reference_angles=states["virtual_posture"][:, 1:],
            regularization=args.angle_regularization,
            angle_basis=renderer.regression_angle_basis,
        )
    calibrated = posture.copy()
    calibrated[:, 1:] = angles.cpu().numpy()
    gamma = heading.astype(np.float32)

    baseline_renderer = PaperFusionRenderer.from_checkpoint(
        args.bbsmg_ckpt,
        device=device,
        image_size=args.image_size,
        dynamic=PaperDynamicConfig(
            pixels_per_model_unit=args.pixels_per_model_unit,
            footprint_longitudinal_scale=args.footprint_longitudinal_scale,
            footprint_transverse_scale=args.footprint_transverse_scale,
            fused_pose_from_height=False,
            inverse_regularization=args.pose_inverse_regularization,
        ),
    )
    with torch.no_grad():
        baseline_rendered = baseline_renderer(
            xy_t,
            posture_t,
            stroke_t,
            torch.zeros(len(xy), device=device),
        )[0, 0].cpu().numpy()

    # Validate through the non-fused forward chain; gamma is absolute robot
    # heading in the CSV, so zero relative footprint rotation is rendered.
    forward_renderer = PaperFusionRenderer.from_checkpoint(
        args.bbsmg_ckpt,
        device=device,
        image_size=args.image_size,
        dynamic=PaperDynamicConfig(
            pixels_per_model_unit=args.pixels_per_model_unit,
            footprint_longitudinal_scale=calibrated_longitudinal_scale,
            footprint_transverse_scale=calibrated_transverse_scale,
            fused_pose_from_height=False,
            inverse_regularization=args.pose_inverse_regularization,
        ),
    )
    with torch.no_grad():
        rendered = forward_renderer(
            xy_t,
            torch.as_tensor(calibrated, device=device),
            stroke_t,
            torch.zeros(len(xy), device=device),
        )[0, 0].cpu().numpy()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    decisions = {
        "x": {"source": "input_pose_csv", "confidence": 1.0},
        "y": {"source": "input_pose_csv", "confidence": 1.0},
        "H": {"source": "input_pose_csv", "confidence": 1.0},
        "alpha": {"source": "target_local_footprint", "confidence": float(confidence.mean())},
        "beta": {"source": "target_local_footprint", "confidence": float(confidence.mean())},
        "gamma": {"source": "forward_xy_heading", "confidence": 1.0},
    }
    csv_path = output / f"{sample.character}_target_footprint_trajectory.csv"
    save_pose_csv(
        sample,
        calibrated,
        csv_path,
        renderer.regression_angle_basis,
        decisions,
        xy_source=xy_source,
        gamma=gamma,
        prototype="paper_target_local_footprint_v44",
    )
    Image.fromarray(np.rint(np.clip(rendered, 0, 1) * 255).astype(np.uint8)).save(
        output / "render_calibrated.png"
    )
    Image.fromarray(
        np.rint(np.clip(baseline_rendered, 0, 1) * 255).astype(np.uint8)
    ).save(output / "render_baseline.png")
    Image.fromarray(np.rint(np.abs(target - rendered) * 255).astype(np.uint8)).save(
        output / "diff.png"
    )
    _save_overlay(target, xy, heading, measured, output / "footprint_overlay.png")
    with (output / "local_footprints.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        fields = [
            "stroke_id", "point_id", "width_px", "length_px", "aspect_ratio",
            "confidence", "target_drag", "target_half_width", "blend",
            "alpha_rad", "beta_rad", "gamma_rad",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for i, point in enumerate(sample.all_points()):
            writer.writerow({
                "stroke_id": point.stroke_id,
                "point_id": point.point_id,
                "width_px": float(measured["width_px"][i]),
                "length_px": float(measured["length_px"][i]),
                "aspect_ratio": float(measured["length_px"][i] / max(measured["width_px"][i], 1e-6)),
                "confidence": float(confidence[i]),
                "target_drag": float(target_drag[i]),
                "target_half_width": float(target_half_width[i]),
                "blend": float(blend[i]),
                "alpha_rad": float(calibrated[i, 1]),
                "beta_rad": float(calibrated[i, 2]),
                "gamma_rad": float(gamma[i]),
            })
    baseline_metrics = binary_metrics(baseline_rendered, target)
    metrics = binary_metrics(rendered, target)
    reduced = regression_matrix_numpy(renderer.regression_angle_basis)
    reduced = np.stack((reduced[0] + reduced[1], reduced[2]))[:, 1:]
    angle_condition_number = float(np.linalg.cond(reduced))
    alpha_lower = float(np.mean(calibrated[:, 1] <= 1e-6))
    alpha_upper = float(
        np.mean(calibrated[:, 1] >= np.deg2rad(10) - 1e-6)
    )
    beta_lower = float(np.mean(calibrated[:, 2] <= 1e-6))
    beta_upper = float(
        np.mean(calibrated[:, 2] >= np.deg2rad(5) - 1e-6)
    )
    accepted = bool(
        metrics["iou_at_0.5"] >= baseline_metrics["iou_at_0.5"]
        and max(alpha_lower, alpha_upper, beta_lower, beta_upper)
        <= args.max_boundary_fraction
        and int(valid.sum()) >= args.minimum_valid_points
    )
    valid_aspect = measured["length_px"][valid] / np.maximum(
        measured["width_px"][valid], 1e-6
    )
    report = {
        "format": "paper_target_local_footprint_v44",
        "simulation_only": True,
        "target_image": args.target_image,
        "pose_csv": args.pose_csv,
        "preserved_fields": ["x", "y", "H"],
        "calibrated_fields": ["alpha", "beta"],
        "gamma_semantics": "per-stroke forward atan2(dy,dx), radians",
        "measurement": {
            "threshold": args.ink_threshold,
            "radius_px": args.radius_px,
            "mean_width_px": float(measured["width_px"].mean()),
            "mean_length_px": float(measured["length_px"].mean()),
            "valid_point_count": int(valid.sum()),
            "mean_aspect_ratio_valid": (
                float(valid_aspect.mean()) if len(valid_aspect) else None
            ),
            "mean_confidence": float(confidence.mean()),
        },
        "axis_scale_calibration": {
            "base_longitudinal": args.footprint_longitudinal_scale,
            "calibrated_longitudinal": calibrated_longitudinal_scale,
            "base_transverse": args.footprint_transverse_scale,
            "calibrated_transverse": calibrated_transverse_scale,
            "method": "confidence-filtered median then bounded to 0.6x-1.6x base",
        },
        "angle_mapping_condition_number": angle_condition_number,
        "pose": {
            "alpha_range_rad": [float(calibrated[:, 1].min()), float(calibrated[:, 1].max())],
            "beta_range_rad": [float(calibrated[:, 2].min()), float(calibrated[:, 2].max())],
            "alpha_lower_bound_fraction": alpha_lower,
            "alpha_upper_bound_fraction": alpha_upper,
            "beta_lower_bound_fraction": beta_lower,
            "beta_upper_bound_fraction": beta_upper,
        },
        "baseline_forward_metrics": baseline_metrics,
        "forward_metrics": metrics,
        "acceptance": {
            "accepted": accepted,
            "requirements": {
                "nondecreasing_iou": True,
                "max_boundary_fraction": args.max_boundary_fraction,
                "minimum_valid_points": args.minimum_valid_points,
            },
            "recommended_pose_csv": str(csv_path) if accepted else None,
        },
        "warning": "Image-derived simulation calibration; not real brush/TCP calibration.",
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory_csv", required=True)
    parser.add_argument("--pose_csv", required=True)
    parser.add_argument("--target_image", required=True)
    parser.add_argument("--bbsmg_ckpt", required=True)
    parser.add_argument("--character", default=None)
    parser.add_argument("--sample_id", default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output_dir", default="outputs/target_footprint_calibration")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--padding", type=int, default=16)
    parser.add_argument("--pixels_per_model_unit", type=float, default=20.0)
    parser.add_argument("--footprint_longitudinal_scale", type=float, default=0.22)
    parser.add_argument("--footprint_transverse_scale", type=float, default=0.262)
    parser.add_argument("--pose_inverse_regularization", type=float, default=1e-5)
    parser.add_argument("--angle_regularization", type=float, default=0.01)
    parser.add_argument("--radius_px", type=float, default=12.0)
    parser.add_argument("--step_px", type=float, default=0.25)
    parser.add_argument("--ink_threshold", type=float, default=0.35)
    parser.add_argument("--target_blend", type=float, default=0.5)
    parser.add_argument("--minimum_confidence", type=float, default=0.1)
    parser.add_argument("--minimum_valid_points", type=int, default=20)
    parser.add_argument("--max_boundary_fraction", type=float, default=0.25)
    main(parser.parse_args())
