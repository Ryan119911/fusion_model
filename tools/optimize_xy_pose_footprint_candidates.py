"""Jointly refine bounded x/y and paper pose against a target image footprint.

SVG/input x/y is retained as the reference trajectory.  A small canvas-space
offset is optimized together with H/alpha/beta using differentiable target
cross-sections, with smoothness, endpoint, and displacement priors.  Gamma is
recomputed from the refined x/y and safely limited within each stroke.  This
is a paper/image simulation stage only; it is not a robot calibration.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.trajectory_dataset import load_trajectory_csv  # noqa: E402
from models.paper_bbsm import PAPER_POSTURE_MAX, PAPER_POSTURE_MIN  # noqa: E402
from models.paper_fusion_renderer import PaperDynamicConfig, PaperFusionRenderer  # noqa: E402
from optim.trajectory_optimizer import load_target_image  # noqa: E402
from tools.invert_paper_trajectory import (  # noqa: E402
    canvas_xy_to_source,
    pick_sample,
    source_xy_to_canvas,
)
from tools.optimize_footprint_pose_candidates import (  # noqa: E402
    _f,
    _limit_gamma_steps,
    _limit_pose_steps,
    _read_rows,
    _targets,
)
from tools.render_paper_trajectory import load_pose_csv  # noqa: E402


def _group_smooth(values: torch.Tensor, stroke: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    pieces = []
    for sid in torch.unique(stroke):
        local = values[stroke == sid]
        if local.shape[0] > 1:
            pieces.append(((local[1:] - local[:-1]) / scale).pow(2).mean())
    return torch.stack(pieces).mean() if pieces else values.new_zeros(())


def _target_cross_sections(
    image: torch.Tensor,
    xy: torch.Tensor,
    stroke: torch.Tensor,
    renderer: PaperFusionRenderer,
    radius_px: float,
    samples: int,
    threshold: float,
    temperature: float,
    pixels_per_model_unit: float,
    longitudinal_scale: float,
    transverse_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiably measure target tangent length and normal half-width."""
    heading = renderer.forward_trajectory_heading(xy, stroke)
    offsets = torch.linspace(-radius_px, radius_px, samples, device=xy.device, dtype=xy.dtype)
    tangent = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
    normal = torch.stack([-torch.sin(heading), torch.cos(heading)], dim=-1)
    tangent_points = xy[:, None, :] + offsets[None, :, None] * tangent[:, None, :]
    normal_points = xy[:, None, :] + offsets[None, :, None] * normal[:, None, :]
    h, w = image.shape[-2:]

    def sample(points: torch.Tensor) -> torch.Tensor:
        grid = torch.stack(
            [2.0 * points[..., 0] / max(w - 1, 1) - 1.0, 2.0 * points[..., 1] / max(h - 1, 1) - 1.0],
            dim=-1,
        )
        grid = grid.reshape(1, len(points), samples, 2)
        values = F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        return values[0, 0].reshape(len(points), samples)

    tangent_values = sample(tangent_points)
    normal_values = sample(normal_points)
    occupancy_t = torch.sigmoid((tangent_values - threshold) / temperature)
    occupancy_n = torch.sigmoid((normal_values - threshold) / temperature)
    step = 2.0 * radius_px / max(samples - 1, 1)
    length_px = occupancy_t.sum(dim=-1) * step
    width_px = occupancy_n.sum(dim=-1) * step
    target_drag = length_px / max(2.0 * pixels_per_model_unit * longitudinal_scale, 1e-6)
    target_half_width = width_px / max(2.0 * pixels_per_model_unit * transverse_scale, 1e-6)
    return target_drag, target_half_width


def _write_pose(
    path: Path,
    rows: list[dict[str, str]],
    source_xy: np.ndarray,
    base_source_xy: np.ndarray,
    posture: np.ndarray,
    gamma: np.ndarray,
    target_drag: np.ndarray,
    target_width: np.ndarray,
    predicted_drag: np.ndarray,
    predicted_width: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) + [
        "x_original", "y_original", "dx_canvas_px", "dy_canvas_px",
        "target_drag_dynamic", "target_half_width_dynamic",
        "predicted_drag_dynamic", "predicted_width_dynamic",
        "drag_relative_error", "width_relative_error", "simulation_only",
    ]
    fields = list(dict.fromkeys(fields))
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for i, source in enumerate(rows):
            row = dict(source)
            row.update({
                "x": repr(float(source_xy[i, 0])), "y": repr(float(source_xy[i, 1])),
                "x_original": repr(float(base_source_xy[i, 0])), "y_original": repr(float(base_source_xy[i, 1])),
                "dx_canvas_px": repr(float(source_xy[i, 0] - base_source_xy[i, 0])),
                "dy_canvas_px": repr(float(source_xy[i, 1] - base_source_xy[i, 1])),
                "z": repr(float(posture[i, 0])), "alpha": repr(float(posture[i, 1])),
                "beta": repr(float(posture[i, 2])), "gamma": repr(float(gamma[i])),
                "target_drag_dynamic": repr(float(target_drag[i])), "target_half_width_dynamic": repr(float(target_width[i])),
                "predicted_drag_dynamic": repr(float(predicted_drag[i])), "predicted_width_dynamic": repr(float(predicted_width[i])),
                "drag_relative_error": repr(float(abs(predicted_drag[i] - target_drag[i]) / max(abs(target_drag[i]), 0.25))),
                "width_relative_error": repr(float(abs(predicted_width[i] - target_width[i]) / max(abs(target_width[i]), 0.08))),
                "simulation_only": 1,
            })
            writer.writerow(row)


def _optimize(
    renderer: PaperFusionRenderer,
    image: torch.Tensor,
    xy_base: torch.Tensor,
    stroke: torch.Tensor,
    posture_init: np.ndarray,
    target_confidence: np.ndarray,
    args: argparse.Namespace,
    h_shift: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    device = xy_base.device
    posture = torch.nn.Parameter(torch.as_tensor(posture_init, dtype=torch.float32, device=device))
    xy_offset = torch.nn.Parameter(torch.zeros_like(xy_base))
    source_posture = posture.detach().clone()
    confidence = torch.as_tensor(target_confidence, dtype=posture.dtype, device=device)
    valid = confidence >= args.minimum_confidence
    optimizer = torch.optim.Adam([posture, xy_offset], lr=args.lr)
    low_pose = torch.as_tensor(PAPER_POSTURE_MIN, dtype=posture.dtype, device=device)
    high_pose = torch.as_tensor(PAPER_POSTURE_MAX, dtype=posture.dtype, device=device)
    scale_pose = posture.new_tensor([3.0, 0.12, 0.06])
    scale_xy = posture.new_tensor([args.xy_smooth_scale, args.xy_smooth_scale])
    best = (float("inf"), posture.detach().clone(), xy_offset.detach().clone(), None)
    for _ in range(args.iterations):
        optimizer.zero_grad(set_to_none=True)
        xy = xy_base + xy_offset
        target_drag, target_width = _target_cross_sections(
            image, xy, stroke, renderer, args.radius_px, args.cross_section_samples,
            args.ink_threshold, args.ink_temperature, args.pixels_per_model_unit,
            args.footprint_longitudinal_scale, args.footprint_transverse_scale,
        )
        geometry = renderer.compute_dynamic_states(xy, posture, stroke)["geometry"]
        predicted_drag = geometry[:, 0] + geometry[:, 1]
        predicted_width = geometry[:, 2]
        residual_d = (predicted_drag - target_drag) / target_drag.abs().clamp_min(0.25)
        residual_w = (predicted_width - target_width) / target_width.abs().clamp_min(0.08)
        weights = confidence.clamp_min(0.0)
        fit = (weights[valid] * (residual_d[valid].pow(2) + residual_w[valid].pow(2))).sum() / weights[valid].sum().clamp_min(1e-6)
        posture_smooth = _group_smooth(posture, stroke, scale_pose)
        xy_smooth = _group_smooth(xy_offset, stroke, scale_xy)
        xy_accel = _group_smooth(
            torch.cat([xy_offset[:1], xy_offset], dim=0),
            torch.cat([stroke[:1], stroke], dim=0),
            scale_xy,
        )
        posture_prior = ((posture - source_posture) / scale_pose).pow(2).mean()
        xy_prior = (xy_offset / max(args.xy_max_delta_px, 1e-6)).pow(2).mean()
        endpoint = xy_offset[torch.arange(len(stroke), device=device) == 0].pow(2).mean()
        boundary = F.relu(low_pose + posture.new_tensor([0.35, 0.008, 0.004]) - posture).pow(2).mean() + F.relu(posture - (high_pose - posture.new_tensor([0.35, 0.008, 0.004]))).pow(2).mean()
        loss = fit + args.posture_smooth_weight * posture_smooth + args.xy_smooth_weight * xy_smooth + args.xy_accel_weight * xy_accel + args.posture_prior_weight * posture_prior + args.xy_prior_weight * xy_prior + args.endpoint_weight * endpoint + args.boundary_weight * boundary
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite joint xy/pose loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([posture, xy_offset], max_norm=2.0)
        optimizer.step()
        with torch.no_grad():
            posture.clamp_(low_pose, high_pose)
            xy_offset.clamp_(-args.xy_max_delta_px, args.xy_max_delta_px)
            for sid in torch.unique(stroke):
                indices = torch.nonzero(stroke == sid, as_tuple=False).flatten()
                if len(indices):
                    xy_offset[indices[0]].zero_()
                    xy_offset[indices[-1]].zero_()
        value = float(loss.detach().cpu())
        if value < best[0]:
            best = (value, posture.detach().clone(), xy_offset.detach().clone(), (target_drag.detach().clone(), target_width.detach().clone(), predicted_drag.detach().clone(), predicted_width.detach().clone()))
    _, best_posture, best_offset, cached = best
    xy = xy_base + best_offset
    with torch.no_grad():
        geometry = renderer.compute_dynamic_states(xy, best_posture, stroke)["geometry"]
        predicted_drag = (geometry[:, 0] + geometry[:, 1]).cpu().numpy()
        predicted_width = geometry[:, 2].cpu().numpy()
        target_drag, target_width = _target_cross_sections(
            image, xy, stroke, renderer, args.radius_px, args.cross_section_samples,
            args.ink_threshold, args.ink_temperature, args.pixels_per_model_unit,
            args.footprint_longitudinal_scale, args.footprint_transverse_scale,
        )
    report = {
        "best_loss": float(best[0]),
        "target_drag": target_drag.cpu().numpy(), "target_width": target_width.cpu().numpy(),
        "predicted_drag": predicted_drag, "predicted_width": predicted_width,
        "xy_offset_canvas": best_offset.cpu().numpy(),
        "valid_count": int(valid.sum().cpu()),
    }
    valid_np = valid.cpu().numpy()
    d = np.abs(predicted_drag[valid_np] - report["target_drag"][valid_np]) / np.maximum(np.abs(report["target_drag"][valid_np]), 0.25)
    w = np.abs(predicted_width[valid_np] - report["target_width"][valid_np]) / np.maximum(np.abs(report["target_width"][valid_np]), 0.08)
    report["drag_relative_rmse"] = float(np.sqrt(np.mean(d * d))) if len(d) else float("inf")
    report["width_relative_rmse"] = float(np.sqrt(np.mean(w * w))) if len(w) else float("inf")
    return best_posture.cpu().numpy(), xy.cpu().numpy(), report


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    rows = _read_rows(Path(args.pose_csv))
    sample = pick_sample(load_trajectory_csv(args.trajectory_csv), sample_id=args.sample_id, character=args.character, index=args.index)
    base_posture, base_source_xy, base_gamma = load_pose_csv(args.pose_csv, sample, clip_pose_limits=False)
    base_xy_canvas = source_xy_to_canvas(sample, base_source_xy, args.image_size, args.padding)
    stroke_ids = np.asarray([p.stroke_id for p in sample.all_points()], dtype=np.int64)
    _, _, confidence = _targets(Path(args.footprint_csv), rows)
    image = torch.as_tensor(load_target_image(args.target_image, args.image_size), dtype=torch.float32, device=device)[None, None]
    renderer = PaperFusionRenderer.from_checkpoint(
        args.bbsmg_ckpt, device=device, image_size=args.image_size,
        dynamic=PaperDynamicConfig(
            pixels_per_model_unit=args.pixels_per_model_unit,
            footprint_longitudinal_scale=args.footprint_longitudinal_scale,
            footprint_transverse_scale=args.footprint_transverse_scale,
            fused_pose_from_height=False,
        ),
    )
    xy_base = torch.as_tensor(base_xy_canvas, dtype=torch.float32, device=device)
    stroke = torch.as_tensor(stroke_ids, dtype=torch.long, device=device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    specs = [("xy_pose_base", 0.0, 0.0), ("xy_pose_low_h", -0.75, 0.0), ("xy_pose_high_h", 0.75, 0.0), ("xy_pose_heading_plus5", 0.0, 5.0)]
    summaries: list[dict[str, Any]] = []
    for index, (name, h_shift, gamma_offset) in enumerate(specs[: args.candidate_count]):
        init = base_posture.copy()
        init[:, 0] = np.clip(init[:, 0] + h_shift, PAPER_POSTURE_MIN[0], PAPER_POSTURE_MAX[0])
        posture, xy_canvas, report = _optimize(renderer, image, xy_base, stroke, init, confidence, args, h_shift)
        posture, projection = _limit_pose_steps(posture, stroke_ids, np.asarray([args.max_h_step_mm, args.max_alpha_step_rad, args.max_beta_step_rad], np.float32))
        xy_delta = xy_canvas - base_xy_canvas
        xy_source = canvas_xy_to_source(sample, xy_canvas, args.image_size, args.padding)
        gamma_raw = renderer.forward_trajectory_heading(torch.as_tensor(xy_canvas, device=device), stroke).detach().cpu().numpy() + np.deg2rad(gamma_offset)
        gamma, gamma_adjustment = _limit_gamma_steps(gamma_raw, stroke_ids, args.gamma_max_step_rad)
        candidate_id = f"candidate_{index:02d}_{name}"
        candidate_dir = output / candidate_id
        _write_pose(candidate_dir / "pose.csv", rows, xy_source, base_source_xy, posture, gamma, report["target_drag"], report["target_width"], report["predicted_drag"], report["predicted_width"])
        xy_norm = np.linalg.norm(xy_delta, axis=1)
        item = {
            "candidate_id": candidate_id, "pose_csv": str(candidate_dir / "pose.csv"),
            "simulation_only": True, "real_brush_calibration_used": False,
            "xy_max_delta_canvas_px": float(xy_norm.max()), "xy_rms_delta_canvas_px": float(np.sqrt(np.mean(xy_norm ** 2))),
            "xy_mean_delta_canvas_px": float(xy_norm.mean()), "posture_projection_max_adjustment": float(projection),
            "gamma_max_adjustment_rad": float(gamma_adjustment), "gamma_source": "forward_heading_from_refined_xy",
            "max_h_step_mm": float(max(np.max(np.abs(np.diff(posture[stroke_ids == sid, 0]))) if np.sum(stroke_ids == sid) > 1 else 0.0 for sid in np.unique(stroke_ids))),
            "max_alpha_step_rad": float(max(np.max(np.abs(np.diff(posture[stroke_ids == sid, 1]))) if np.sum(stroke_ids == sid) > 1 else 0.0 for sid in np.unique(stroke_ids))),
            "max_beta_step_rad": float(max(np.max(np.abs(np.diff(posture[stroke_ids == sid, 2]))) if np.sum(stroke_ids == sid) > 1 else 0.0 for sid in np.unique(stroke_ids))),
            "max_gamma_step_rad": float(max(np.max(np.abs(np.arctan2(np.sin(np.diff(gamma[stroke_ids == sid])), np.cos(np.diff(gamma[stroke_ids == sid]))))) if np.sum(stroke_ids == sid) > 1 else 0.0 for sid in np.unique(stroke_ids))),
            "boundary_fraction": float(np.mean(np.any((posture <= PAPER_POSTURE_MIN[None, :] + 1e-4) | (posture >= PAPER_POSTURE_MAX[None, :] - 1e-4), axis=1))),
            "footprint_points": int(report["valid_count"]), "drag_relative_rmse": float(report["drag_relative_rmse"]), "width_relative_rmse": float(report["width_relative_rmse"]),
            "target_semantics": "differentiable target-image cross-sections at refined x/y",
        }
        (candidate_dir / "report.json").write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")
        summaries.append(item)
    summary = {"format": "paper_joint_xy_pose_footprint_candidates_v1", "target_image": args.target_image, "footprint_csv": args.footprint_csv, "simulation_only": True, "real_brush_calibration_used": False, "candidates": summaries}
    (output / "candidate_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory_csv", required=True)
    parser.add_argument("--bbsmg_ckpt", required=True)
    parser.add_argument("--pose_csv", required=True)
    parser.add_argument("--footprint_csv", required=True)
    parser.add_argument("--target_image", required=True)
    parser.add_argument("--character", default="武")
    parser.add_argument("--sample_id", default="武_fake_sim")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output_dir", default="outputs/wu_joint_xy_pose_footprint_candidates_v1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--padding", type=int, default=16)
    parser.add_argument("--pixels_per_model_unit", type=float, default=24.0)
    parser.add_argument("--footprint_longitudinal_scale", type=float, default=0.2302875519)
    parser.add_argument("--footprint_transverse_scale", type=float, default=0.3296116590)
    parser.add_argument("--xy_max_delta_px", type=float, default=3.0)
    parser.add_argument("--xy_smooth_scale", type=float, default=1.0)
    parser.add_argument("--radius_px", type=float, default=10.0)
    parser.add_argument("--cross_section_samples", type=int, default=33)
    parser.add_argument("--ink_threshold", type=float, default=0.35)
    parser.add_argument("--ink_temperature", type=float, default=0.08)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--posture_smooth_weight", type=float, default=0.20)
    parser.add_argument("--xy_smooth_weight", type=float, default=0.30)
    parser.add_argument("--xy_accel_weight", type=float, default=0.10)
    parser.add_argument("--posture_prior_weight", type=float, default=0.03)
    parser.add_argument("--xy_prior_weight", type=float, default=0.20)
    parser.add_argument("--endpoint_weight", type=float, default=0.50)
    parser.add_argument("--boundary_weight", type=float, default=0.20)
    parser.add_argument("--minimum_confidence", type=float, default=0.1)
    parser.add_argument("--candidate_count", type=int, default=4)
    parser.add_argument("--gamma_max_step_rad", type=float, default=0.75)
    parser.add_argument("--max_h_step_mm", type=float, default=1.0)
    parser.add_argument("--max_alpha_step_rad", type=float, default=float(np.deg2rad(5.0)))
    parser.add_argument("--max_beta_step_rad", type=float, default=float(np.deg2rad(3.0)))
    main(parser.parse_args())
