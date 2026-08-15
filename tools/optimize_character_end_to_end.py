"""Optimize a complete paper-rendered character against one target image.

This is a simulation-only refinement stage.  The B-BSMG checkpoint remains
frozen; bounded, smooth canvas x/y corrections and bounded H/alpha/beta
corrections are optimized through the complete Dynamic-Brush + B-BSMG
renderer.  Gamma is exported as the forward x/y heading and is converted to
the local B-BSMG gamma convention by ``PaperDynamicConfig.gamma_mode``.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.trajectory_dataset import load_trajectory_csv
from models.paper_bbsm import PAPER_POSTURE_MAX, PAPER_POSTURE_MIN
from models.paper_fusion_renderer import PaperDynamicConfig, PaperFusionRenderer
from optim.trajectory_optimizer import load_target_image
from tools.invert_paper_trajectory import (
    binary_metrics,
    canvas_xy_to_source,
    flatten_canvas_trajectory,
    pick_sample,
    source_xy_to_canvas,
)
from tools.render_paper_trajectory import load_pose_csv
from utils.trajectory_processing import repair_sample_states


def _sobel(x: torch.Tensor) -> torch.Tensor:
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=x.dtype,
        device=x.device,
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        dtype=x.dtype,
        device=x.device,
    ).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def _stroke_groups(stroke_ids: np.ndarray) -> list[np.ndarray]:
    groups: list[np.ndarray] = []
    for stroke_id in np.unique(stroke_ids):
        groups.append(np.flatnonzero(stroke_ids == stroke_id))
    return groups


def _smoothness(
    values: torch.Tensor,
    base: torch.Tensor,
    groups: list[np.ndarray],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return prior, first-difference and second-difference penalties."""
    prior = ((values - base) ** 2).mean()
    first_terms = []
    second_terms = []
    for indices in groups:
        if len(indices) > 1:
            idx = torch.as_tensor(indices, dtype=torch.long, device=values.device)
            delta = values[idx] - base[idx]
            first_terms.append((delta[1:] - delta[:-1]).pow(2).mean())
            if len(indices) > 2:
                second_terms.append((delta[2:] - 2.0 * delta[1:-1] + delta[:-2]).pow(2).mean())
    zero = values.new_zeros(())
    return prior, (torch.stack(first_terms).mean() if first_terms else zero), (
        torch.stack(second_terms).mean() if second_terms else zero
    )


def _image_loss(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    weighted_mse = (((1.0 + 8.0 * target) * (pred - target) ** 2).mean())
    inter = (pred * target).sum()
    dice_loss = 1.0 - (2.0 * inter + 1e-6) / (pred.sum() + target.sum() + 1e-6)
    bce = F.binary_cross_entropy(pred.clamp(1e-5, 1.0 - 1e-5), target)
    edge = F.l1_loss(_sobel(pred), _sobel(target))
    ink = torch.abs(pred.mean() - target.mean())
    loss = weighted_mse + 0.35 * dice_loss + 0.08 * bce + 0.05 * edge + 0.20 * ink
    return loss, {
        "weighted_mse": float(weighted_mse.detach().cpu()),
        "dice_loss": float(dice_loss.detach().cpu()),
        "bce": float(bce.detach().cpu()),
        "edge_loss": float(edge.detach().cpu()),
        "ink_loss": float(ink.detach().cpu()),
    }


def _write_pose_csv(
    source_path: Path,
    output_path: Path,
    sample,
    source_xy: np.ndarray,
    posture: np.ndarray,
    gamma: np.ndarray,
    delta_xy_canvas: np.ndarray,
) -> None:
    rows_by_key: dict[tuple[int, int], dict[str, str]] = {}
    with source_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            rows_by_key[(int(row["stroke_id"]), int(row["point_id"]))] = dict(row)
    fields = list(next(iter(rows_by_key.values())).keys()) if rows_by_key else []
    for field in ("x_original", "y_original", "dx_canvas_px", "dy_canvas_px", "gamma_semantics", "simulation_only"):
        if field not in fields:
            fields.append(field)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for index, point in enumerate(sample.all_points()):
            key = (point.stroke_id, point.point_id)
            if key not in rows_by_key:
                raise ValueError(f"Pose CSV is missing {key}")
            row = rows_by_key[key]
            row.update(
                {
                    "x": repr(float(source_xy[index, 0])),
                    "y": repr(float(source_xy[index, 1])),
                    "z": repr(float(posture[index, 0])),
                    "alpha": repr(float(posture[index, 1])),
                    "beta": repr(float(posture[index, 2])),
                    "gamma": repr(float(gamma[index])),
                    "x_original": row.get("x_original", row.get("x", "")),
                    "y_original": row.get("y_original", row.get("y", "")),
                    "dx_canvas_px": repr(float(delta_xy_canvas[index, 0])),
                    "dy_canvas_px": repr(float(delta_xy_canvas[index, 1])),
                    "gamma_semantics": "absolute_forward_xy_heading",
                    "simulation_only": "true",
                }
            )
            writer.writerow(row)


def _save_images(output: Path, target: np.ndarray, prediction: np.ndarray) -> None:
    # Metrics stay in ink=1/background=0 space.  Export all panels as
    # black ink on a white background, matching the target PNG convention.
    target_u8 = np.rint(255.0 - np.clip(target, 0.0, 1.0) * 255.0).astype(np.uint8)
    pred_u8 = np.rint(255.0 - np.clip(prediction, 0.0, 1.0) * 255.0).astype(np.uint8)
    diff_u8 = np.rint(255.0 - np.clip(np.abs(prediction - target), 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(target_u8, mode="L").save(output / "target.png")
    Image.fromarray(pred_u8, mode="L").save(output / "render_refined.png")
    Image.fromarray(diff_u8, mode="L").save(output / "diff.png")
    comparison = Image.new("L", (target.shape[1] * 3, target.shape[0]), 255)
    comparison.paste(Image.fromarray(target_u8, mode="L"), (0, 0))
    comparison.paste(Image.fromarray(pred_u8, mode="L"), (target.shape[1], 0))
    comparison.paste(Image.fromarray(diff_u8, mode="L"), (target.shape[1] * 2, 0))
    comparison.save(output / "comparison.png")


def optimize(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    sample = pick_sample(
        load_trajectory_csv(args.trajectory_csv),
        sample_id=args.sample_id,
        character=args.character,
        index=args.index,
    )
    sample = repair_sample_states(sample)
    base_posture, base_source_xy, base_gamma = load_pose_csv(args.pose_csv, sample)
    base_xy_canvas = source_xy_to_canvas(sample, base_source_xy, args.image_size, args.padding)
    stroke_ids_np = np.asarray([point.stroke_id for point in sample.all_points()], dtype=np.int64)
    groups = _stroke_groups(stroke_ids_np)
    target_np = load_target_image(args.target_image, image_size=args.image_size)
    target = torch.as_tensor(target_np, dtype=torch.float32, device=device)[None, None]

    renderer = PaperFusionRenderer.from_checkpoint(
        args.bbsmg_ckpt,
        device=device,
        image_size=args.image_size,
        point_batch_size=args.point_batch_size,
        dynamic=PaperDynamicConfig(
            calibration_profile=args.dynamic_profile,
            pixels_per_model_unit=args.pixels_per_model_unit,
            footprint_longitudinal_scale=args.footprint_longitudinal_scale,
            footprint_transverse_scale=args.footprint_transverse_scale,
            render_max_step_px=args.render_max_step_px,
            patch_floor=args.patch_floor,
            gamma_mode="relative_to_heading",
        ),
    )
    renderer.eval()
    base_xy = torch.as_tensor(base_xy_canvas, dtype=torch.float32, device=device)
    base_pose = torch.as_tensor(base_posture, dtype=torch.float32, device=device)
    stroke_ids = torch.as_tensor(stroke_ids_np, dtype=torch.long, device=device)
    xy_raw = torch.zeros_like(base_xy, requires_grad=True)
    pose_raw = torch.zeros_like(base_pose, requires_grad=True)
    xy_scale = torch.tensor([args.xy_max_delta_px, args.xy_max_delta_px], dtype=torch.float32, device=device)
    pose_scale = torch.tensor([args.h_max_delta_mm, np.deg2rad(args.alpha_max_delta_deg), np.deg2rad(args.beta_max_delta_deg)], dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([xy_raw, pose_raw], lr=args.learning_rate)
    best = None
    trace: list[dict[str, float]] = []

    def current_values() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xy = base_xy + xy_scale[None, :] * torch.tanh(xy_raw)
        pose = torch.maximum(torch.minimum(base_pose + pose_scale[None, :] * torch.tanh(pose_raw), torch.as_tensor(PAPER_POSTURE_MAX, device=device)), torch.as_tensor(PAPER_POSTURE_MIN, device=device))
        gamma = PaperFusionRenderer.forward_trajectory_heading(xy, stroke_ids)
        return xy, pose, gamma

    with torch.no_grad():
        xy0, pose0, gamma0 = current_values()
        pred0 = renderer(xy0, pose0, stroke_ids, gamma0).clamp(0.0, 1.0)
        base_metrics = binary_metrics(pred0[0, 0].cpu().numpy(), target_np)

    for iteration in range(args.iterations):
        optimizer.zero_grad(set_to_none=True)
        xy, pose, gamma = current_values()
        prediction = renderer(xy, pose, stroke_ids, gamma).clamp(0.0, 1.0)
        image_loss, image_parts = _image_loss(prediction, target)
        xy_prior, xy_first, xy_second = _smoothness(xy, base_xy, groups)
        pose_prior, pose_first, pose_second = _smoothness(pose, base_pose, groups)
        anchor_terms = []
        for indices in groups:
            first = torch.as_tensor(indices[0], dtype=torch.long, device=device)
            anchor_terms.append(((xy[first] - base_xy[first]) ** 2).mean())
        anchor_loss = torch.stack(anchor_terms).mean() if anchor_terms else xy_prior.new_zeros(())
        regularization = (
            args.xy_prior_weight * xy_prior
            + args.xy_first_weight * xy_first
            + args.xy_second_weight * xy_second
            + args.endpoint_weight * anchor_loss
            + args.pose_prior_weight * pose_prior
            + args.pose_first_weight * pose_first
            + args.pose_second_weight * pose_second
        )
        loss = image_loss + regularization
        loss.backward()
        torch.nn.utils.clip_grad_norm_([xy_raw, pose_raw], args.gradient_clip)
        optimizer.step()
        item = {
            "iteration": float(iteration),
            "loss": float(loss.detach().cpu()),
            "image_loss": float(image_loss.detach().cpu()),
            "regularization": float(regularization.detach().cpu()),
            "xy_rms_delta_px": float(torch.sqrt(((xy - base_xy) ** 2).sum(dim=1).mean()).detach().cpu()),
            "xy_max_delta_px": float(torch.linalg.vector_norm(xy - base_xy, dim=1).max().detach().cpu()),
            "pose_rms_delta": float(torch.sqrt(((pose - base_pose) ** 2).mean()).detach().cpu()),
            **image_parts,
        }
        trace.append(item)
        score = float(image_loss.detach().cpu())
        if best is None or score < best["score"]:
            best = {"score": score, "xy": xy.detach().clone(), "pose": pose.detach().clone(), "gamma": gamma.detach().clone(), "prediction": prediction.detach().clone(), "trace": item}
        if args.log_every > 0 and (iteration % args.log_every == 0 or iteration == args.iterations - 1):
            print(json.dumps(item, ensure_ascii=False))

    assert best is not None
    refined_xy = best["xy"].cpu().numpy()
    refined_pose = best["pose"].cpu().numpy()
    refined_gamma = best["gamma"].cpu().numpy()
    refined_prediction = best["prediction"][0, 0].cpu().numpy()
    refined_source_xy = canvas_xy_to_source(sample, refined_xy, args.image_size, args.padding)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _save_images(output, target_np, refined_prediction)
    _write_pose_csv(
        Path(args.pose_csv),
        output / "pose_refined.csv",
        sample,
        refined_source_xy,
        refined_pose,
        refined_gamma,
        refined_xy - base_xy_canvas,
    )
    (output / "optimization_trace.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    refined_metrics = binary_metrics(refined_prediction, target_np)
    report = {
        "format": "paper_character_end_to_end_refinement_v1",
        "simulation_only": True,
        "target_image": args.target_image,
        "pose_input": args.pose_csv,
        "pose_output": str(output / "pose_refined.csv"),
        "gamma_mode": "relative_to_heading",
        "iterations": args.iterations,
        "point_count": int(len(refined_xy)),
        "stroke_count": int(len(groups)),
        "base_metrics": base_metrics,
        "refined_metrics": refined_metrics,
        "improvement": {
            "mse_delta": float(refined_metrics["plain_mse"] - base_metrics["plain_mse"]),
            "iou_delta": float(refined_metrics["iou_at_0.5"] - base_metrics["iou_at_0.5"]),
            "dice_delta": float(refined_metrics["dice_at_0.5"] - base_metrics["dice_at_0.5"]),
        },
        "xy_delta_canvas_px": {
            "max": float(np.linalg.norm(refined_xy - base_xy_canvas, axis=1).max()),
            "rms": float(np.sqrt(np.mean((refined_xy - base_xy_canvas) ** 2))),
        },
        "posture_delta": {
            "H_mm_max": float(np.abs(refined_pose[:, 0] - base_posture[:, 0]).max()),
            "alpha_rad_max": float(np.abs(refined_pose[:, 1] - base_posture[:, 1]).max()),
            "beta_rad_max": float(np.abs(refined_pose[:, 2] - base_posture[:, 2]).max()),
        },
        "best_iteration": best["trace"],
        "notes": [
            "B-BSMG weights are frozen; this stage optimizes only bounded smooth whole-character coordinates and pose.",
            "Gamma is exported as absolute forward x/y heading and converted to local gamma inside the renderer.",
            "Results are simulation candidates, not robot calibration.",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory_csv", required=True)
    parser.add_argument("--pose_csv", required=True)
    parser.add_argument("--bbsmg_ckpt", required=True)
    parser.add_argument("--target_image", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--character", default="武")
    parser.add_argument("--sample_id", default="武_fake_sim")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--padding", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--learning_rate", type=float, default=0.03)
    parser.add_argument("--xy_max_delta_px", type=float, default=4.0)
    parser.add_argument("--h_max_delta_mm", type=float, default=2.0)
    parser.add_argument("--alpha_max_delta_deg", type=float, default=2.0)
    parser.add_argument("--beta_max_delta_deg", type=float, default=1.5)
    parser.add_argument("--point_batch_size", type=int, default=8)
    parser.add_argument("--render_max_step_px", type=float, default=2.0)
    parser.add_argument("--pixels_per_model_unit", type=float, default=20.0)
    parser.add_argument("--footprint_longitudinal_scale", type=float, default=0.2302875519)
    parser.add_argument("--footprint_transverse_scale", type=float, default=0.3296116590)
    parser.add_argument("--patch_floor", type=float, default=0.05)
    parser.add_argument("--dynamic_profile", default="wang2020_figure4_digitized_v1")
    parser.add_argument("--xy_prior_weight", type=float, default=0.006)
    parser.add_argument("--xy_first_weight", type=float, default=0.03)
    parser.add_argument("--xy_second_weight", type=float, default=0.04)
    parser.add_argument("--endpoint_weight", type=float, default=0.02)
    parser.add_argument("--pose_prior_weight", type=float, default=0.012)
    parser.add_argument("--pose_first_weight", type=float, default=0.04)
    parser.add_argument("--pose_second_weight", type=float, default=0.05)
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=10)
    args = parser.parse_args()
    optimize(args)


if __name__ == "__main__":
    main()
