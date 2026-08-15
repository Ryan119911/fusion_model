"""Optimize bounded paper poses against local target footprints.

This stage uses the *dynamic* Wang brush state rather than the static
regression geometry alone.  The target image supplies soft local drag and
half-width observations; H/alpha/beta are optimized with bounded, smooth
per-stroke trajectories and gamma remains the forward x/y heading (plus an
explicit diagnostic offset).  The result is a simulation screening artifact,
not a robot or real-brush calibration claim.
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
from models.paper_fusion_renderer import (  # noqa: E402
    PaperDynamicConfig,
    PaperFusionRenderer,
)
from tools.invert_paper_trajectory import pick_sample, source_xy_to_canvas  # noqa: E402
from tools.render_paper_trajectory import load_pose_csv  # noqa: E402


def _f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    return float(default) if value in (None, "") else float(value)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def _targets(path: Path, rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ref = {(int(_f(r, "stroke_id")), int(_f(r, "point_id"))): r for r in _read_rows(path)}
    drag, width, confidence = [], [], []
    for row in rows:
        r = ref.get((int(_f(row, "stroke_id")), int(_f(row, "point_id"))))
        drag.append(_f(r or {}, "target_drag", np.nan))
        width.append(_f(r or {}, "target_half_width", np.nan))
        confidence.append(_f(r or {}, "confidence", 0.0))
    return np.asarray(drag, np.float32), np.asarray(width, np.float32), np.asarray(confidence, np.float32)


def _nearest_continuous_gamma(path: Path, rows: list[dict[str, str]]) -> np.ndarray:
    """Map gamma from a dense repaired trajectory onto sparse pose points.

    The repaired trajectory may contain many resampled points and therefore
    cannot be joined by ``point_id``.  Nearest same-stroke (x,y) matching keeps
    the original sparse coordinates while importing its continuous heading.
    """
    dense = _read_rows(path)
    by_stroke: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for sid in sorted({int(_f(r, "stroke_id")) for r in dense}):
        local = [r for r in dense if int(_f(r, "stroke_id")) == sid]
        xy = np.asarray([[_f(r, "x"), _f(r, "y")] for r in local], np.float32)
        gamma = np.asarray([_f(r, "gamma", 0.0) for r in local], np.float32)
        by_stroke[sid] = (xy, gamma)
    result = []
    for row in rows:
        sid = int(_f(row, "stroke_id"))
        xy, gamma = by_stroke.get(sid, (np.empty((0, 2)), np.empty((0,))))
        if len(xy) == 0:
            result.append(_f(row, "gamma", 0.0))
            continue
        point = np.asarray([_f(row, "x"), _f(row, "y")], np.float32)
        result.append(float(gamma[int(np.argmin(np.sum((xy - point) ** 2, axis=1)))]))
    return np.asarray(result, np.float32)


def _wrapped(values: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(values), np.cos(values))


def _max_step(values: np.ndarray, stroke: np.ndarray, angular: bool = False) -> float:
    result: list[float] = []
    for sid in np.unique(stroke):
        local = values[stroke == sid]
        if len(local) > 1:
            delta = np.diff(local, axis=0)
            if angular:
                delta = _wrapped(delta)
            result.extend(np.abs(delta).reshape(len(delta), -1).max(axis=1).tolist())
    return float(max(result)) if result else 0.0


def _limit_gamma_steps(values: np.ndarray, stroke: np.ndarray, max_step: float) -> tuple[np.ndarray, float]:
    """Limit within-stroke heading jumps while preserving pen-up boundaries."""
    output = np.asarray(values, dtype=np.float32).copy()
    max_adjustment = 0.0
    for sid in np.unique(stroke):
        indices = np.flatnonzero(stroke == sid)
        if len(indices) < 2:
            continue
        previous = float(output[indices[0]])
        for index in indices[1:]:
            raw = float(output[index])
            delta = float(np.arctan2(np.sin(raw - previous), np.cos(raw - previous)))
            clipped = float(np.clip(delta, -max_step, max_step))
            output[index] = np.float32(previous + clipped)
            max_adjustment = max(max_adjustment, abs(clipped - delta))
            previous = float(output[index])
    return _wrapped(output).astype(np.float32), float(max_adjustment)


def _limit_pose_steps(posture: np.ndarray, stroke: np.ndarray, limits: np.ndarray) -> tuple[np.ndarray, float]:
    """Project H/alpha/beta onto conservative within-stroke step limits."""
    output = np.asarray(posture, dtype=np.float32).copy()
    max_adjustment = 0.0
    for sid in np.unique(stroke):
        indices = np.flatnonzero(stroke == sid)
        for column, limit in enumerate(limits):
            if len(indices) < 2:
                continue
            previous = float(output[indices[0], column])
            for index in indices[1:]:
                raw = float(output[index, column])
                clipped = float(np.clip(raw, previous - limit, previous + limit))
                max_adjustment = max(max_adjustment, abs(raw - clipped))
                output[index, column] = np.float32(clipped)
                previous = clipped
    return output, float(max_adjustment)


def _smooth_penalty(posture: torch.Tensor, stroke: torch.Tensor) -> torch.Tensor:
    pieces = []
    scales = posture.new_tensor([3.0, 0.12, 0.06])
    for sid in torch.unique(stroke):
        local = posture[stroke == sid]
        if local.shape[0] > 1:
            pieces.append(((local[1:] - local[:-1]) / scales).pow(2).mean())
    return torch.stack(pieces).mean() if pieces else posture.new_zeros(())


def _boundary_penalty(posture: torch.Tensor) -> torch.Tensor:
    low = torch.as_tensor(PAPER_POSTURE_MIN, dtype=posture.dtype, device=posture.device)
    high = torch.as_tensor(PAPER_POSTURE_MAX, dtype=posture.dtype, device=posture.device)
    margin = posture.new_tensor([0.35, 0.008, 0.004])
    return (
        F.relu(low + margin - posture).pow(2).mean()
        + F.relu(posture - (high - margin)).pow(2).mean()
    )


def _optimize(
    renderer: PaperFusionRenderer,
    xy: torch.Tensor,
    stroke: torch.Tensor,
    initial: np.ndarray,
    target_drag: np.ndarray,
    target_width: np.ndarray,
    confidence: np.ndarray,
    iterations: int,
    lr: float,
    smooth_weight: float,
    prior_weight: float,
    boundary_weight: float,
    minimum_confidence: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    device = xy.device
    posture = torch.nn.Parameter(torch.as_tensor(initial, dtype=torch.float32, device=device))
    target_d = torch.as_tensor(target_drag, dtype=posture.dtype, device=device)
    target_w = torch.as_tensor(target_width, dtype=posture.dtype, device=device)
    conf = torch.as_tensor(confidence, dtype=posture.dtype, device=device)
    valid = torch.isfinite(target_d) & torch.isfinite(target_w) & (conf >= minimum_confidence)
    denom_d = target_d.abs().clamp_min(0.25)
    denom_w = target_w.abs().clamp_min(0.08)
    optimizer = torch.optim.Adam([posture], lr=lr)
    best_loss = float("inf")
    best_posture = posture.detach().clone()
    source = torch.as_tensor(initial, dtype=posture.dtype, device=device)
    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        states = renderer.compute_dynamic_states(xy, posture, stroke)
        geometry = states["geometry"]
        pred_d = geometry[:, 0] + geometry[:, 1]
        pred_w = geometry[:, 2]
        residual_d = (pred_d - target_d) / denom_d
        residual_w = (pred_w - target_w) / denom_w
        weights = conf.clamp_min(0.0)
        fit = (weights[valid] * (residual_d[valid].pow(2) + residual_w[valid].pow(2))).sum() / weights[valid].sum().clamp_min(1e-6)
        smooth = _smooth_penalty(posture, stroke)
        prior = ((posture - source) / posture.new_tensor([3.0, 0.12, 0.06])).pow(2).mean()
        boundary = _boundary_penalty(posture)
        loss = fit + smooth_weight * smooth + prior_weight * prior + boundary_weight * boundary
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite footprint optimization loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([posture], max_norm=2.0)
        optimizer.step()
        with torch.no_grad():
            posture.clamp_(
                torch.as_tensor(PAPER_POSTURE_MIN, dtype=posture.dtype, device=device),
                torch.as_tensor(PAPER_POSTURE_MAX, dtype=posture.dtype, device=device),
            )
        value = float(loss.detach().cpu())
        if value < best_loss:
            best_loss = value
            best_posture = posture.detach().clone()
    with torch.no_grad():
        states = renderer.compute_dynamic_states(xy, best_posture, stroke)
        geometry = states["geometry"]
        pred_d = (geometry[:, 0] + geometry[:, 1]).cpu().numpy()
        pred_w = geometry[:, 2].cpu().numpy()
    valid_np = valid.cpu().numpy()
    d = np.abs(pred_d[valid_np] - target_drag[valid_np]) / np.maximum(np.abs(target_drag[valid_np]), 0.25)
    w = np.abs(pred_w[valid_np] - target_width[valid_np]) / np.maximum(np.abs(target_width[valid_np]), 0.08)
    return best_posture.cpu().numpy(), {
        "best_loss": best_loss,
        "predicted_drag": pred_d,
        "predicted_width": pred_w,
        "valid_count": int(valid_np.sum()),
        "drag_relative_rmse": float(np.sqrt(np.mean(d * d))) if len(d) else float("inf"),
        "width_relative_rmse": float(np.sqrt(np.mean(w * w))) if len(w) else float("inf"),
    }


def _write_pose(path: Path, rows: list[dict[str, str]], posture: np.ndarray, gamma: np.ndarray, target_d: np.ndarray, target_w: np.ndarray, pred_d: np.ndarray, pred_w: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) + ["target_drag", "target_half_width", "predicted_drag", "predicted_width", "drag_relative_error", "width_relative_error", "simulation_only"]
    fields = list(dict.fromkeys(fields))
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for i, source in enumerate(rows):
            row = dict(source)
            row.update({
                "z": repr(float(posture[i, 0])), "alpha": repr(float(posture[i, 1])), "beta": repr(float(posture[i, 2])), "gamma": repr(float(gamma[i])),
                "target_drag": repr(float(target_d[i])), "target_half_width": repr(float(target_w[i])), "predicted_drag": repr(float(pred_d[i])), "predicted_width": repr(float(pred_w[i])),
                "drag_relative_error": repr(float(abs(pred_d[i] - target_d[i]) / max(abs(target_d[i]), 0.25))) if np.isfinite(target_d[i]) else "nan",
                "width_relative_error": repr(float(abs(pred_w[i] - target_w[i]) / max(abs(target_w[i]), 0.08))) if np.isfinite(target_w[i]) else "nan",
                "simulation_only": 1,
            })
            writer.writerow(row)


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    rows = _read_rows(Path(args.pose_csv))
    samples = load_trajectory_csv(args.trajectory_csv)
    sample = pick_sample(samples, sample_id=args.sample_id, character=args.character, index=args.index)
    base_posture, xy_source, base_gamma = load_pose_csv(args.pose_csv, sample, clip_pose_limits=False)
    gamma_source = "pose_csv"
    if args.continuity_pose_csv:
        base_gamma = _nearest_continuous_gamma(Path(args.continuity_pose_csv), rows)
        gamma_source = "nearest_same_stroke_repaired_trajectory"
    xy_canvas = source_xy_to_canvas(sample, xy_source, args.image_size, args.padding)
    stroke_ids = np.asarray([p.stroke_id for p in sample.all_points()], dtype=np.int64)
    target_d, target_w, confidence = _targets(Path(args.footprint_csv), rows)
    xy = torch.as_tensor(xy_canvas, dtype=torch.float32, device=device)
    stroke = torch.as_tensor(stroke_ids, dtype=torch.long, device=device)
    renderer = PaperFusionRenderer.from_checkpoint(
        args.bbsmg_ckpt,
        device=device,
        image_size=args.image_size,
        dynamic=PaperDynamicConfig(
            pixels_per_model_unit=args.pixels_per_model_unit,
            footprint_longitudinal_scale=args.footprint_longitudinal_scale,
            footprint_transverse_scale=args.footprint_transverse_scale,
            fused_pose_from_height=False,
        ),
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    specs = [
        ("dynamic_base", 0.0, 0.0), ("dynamic_low_h", -1.0, 0.0), ("dynamic_high_h", 1.0, 0.0),
        ("dynamic_smooth", 0.0, 4.0), ("dynamic_heading_plus5", 0.0, 5.0), ("dynamic_heading_minus5", 0.0, -5.0),
    ]
    for index, (name, h_shift, gamma_deg) in enumerate(specs[: args.candidate_count]):
        initial = base_posture.copy()
        initial[:, 0] = np.clip(initial[:, 0] + h_shift, PAPER_POSTURE_MIN[0], PAPER_POSTURE_MAX[0])
        posture, report = _optimize(
            renderer, xy, stroke, initial, target_d, target_w, confidence,
            args.iterations, args.lr, args.smooth_weight, args.prior_weight,
            args.boundary_weight, args.minimum_confidence,
        )
        posture, posture_projection_adjustment = _limit_pose_steps(
            posture,
            stroke_ids,
            np.asarray([args.max_h_step_mm, args.max_alpha_step_rad, args.max_beta_step_rad], dtype=np.float32),
        )
        # Re-evaluate the dynamic footprint after the safety projection so the
        # CSV and report always describe the exported, not pre-projection, pose.
        with torch.no_grad():
            projected = torch.as_tensor(posture, dtype=torch.float32, device=device)
            projected_states = renderer.compute_dynamic_states(xy, projected, stroke)
            projected_geometry = projected_states["geometry"]
            report["predicted_drag"] = (projected_geometry[:, 0] + projected_geometry[:, 1]).cpu().numpy()
            report["predicted_width"] = projected_geometry[:, 2].cpu().numpy()
        valid_np = np.isfinite(target_d) & np.isfinite(target_w) & (confidence >= args.minimum_confidence)
        d = np.abs(report["predicted_drag"][valid_np] - target_d[valid_np]) / np.maximum(np.abs(target_d[valid_np]), 0.25)
        w = np.abs(report["predicted_width"][valid_np] - target_w[valid_np]) / np.maximum(np.abs(target_w[valid_np]), 0.08)
        report["drag_relative_rmse"] = float(np.sqrt(np.mean(d * d))) if len(d) else float("inf")
        report["width_relative_rmse"] = float(np.sqrt(np.mean(w * w))) if len(w) else float("inf")
        gamma_raw = _wrapped(base_gamma + np.deg2rad(gamma_deg)).astype(np.float32)
        gamma, gamma_adjustment = _limit_gamma_steps(gamma_raw, stroke_ids, args.gamma_max_step_rad)
        candidate_id = f"candidate_{index:02d}_{name}"
        candidate_dir = output / candidate_id
        _write_pose(candidate_dir / "pose.csv", rows, posture, gamma, target_d, target_w, report["predicted_drag"], report["predicted_width"])
        boundary = float(np.mean(np.any((posture <= PAPER_POSTURE_MIN[None, :] + 1e-4) | (posture >= PAPER_POSTURE_MAX[None, :] - 1e-4), axis=1)))
        item = {
            "candidate_id": candidate_id, "pose_csv": str(candidate_dir / "pose.csv"), "simulation_only": True,
            "real_brush_calibration_used": False, "h_shift_mm": h_shift, "gamma_offset_deg": gamma_deg,
            **{k: v for k, v in report.items() if k not in ("predicted_drag", "predicted_width")},
            "boundary_fraction": boundary,
            "max_h_step_mm": _max_step(posture[:, 0], stroke_ids),
            "max_alpha_step_rad": _max_step(posture[:, 1], stroke_ids),
            "max_beta_step_rad": _max_step(posture[:, 2], stroke_ids),
            "max_gamma_step_rad": _max_step(gamma, stroke_ids, angular=True),
            "trajectory_xy_source": "input_pose_csv",
            "gamma_source": gamma_source,
            "gamma_max_step_rad": args.gamma_max_step_rad,
            "gamma_max_adjustment_rad": gamma_adjustment,
            "posture_projection_max_adjustment": posture_projection_adjustment,
        }
        (candidate_dir / "report.json").write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")
        summaries.append(item)
    summary = {
        "format": "paper_dynamic_local_footprint_pose_candidates_v1",
        "pose_csv": args.pose_csv, "footprint_csv": args.footprint_csv,
        "target_semantics": "soft local image footprint; dynamic geometry is optimized with continuity/prior penalties",
        "simulation_only": True, "real_brush_calibration_used": False,
        "candidates": summaries,
    }
    (output / "candidate_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory_csv", required=True)
    parser.add_argument("--bbsmg_ckpt", required=True)
    parser.add_argument("--pose_csv", required=True)
    parser.add_argument("--footprint_csv", required=True)
    parser.add_argument("--continuity_pose_csv", default=None, help="Optional dense repaired trajectory used only to import continuous gamma by nearest same-stroke x/y.")
    parser.add_argument("--character", default="武")
    parser.add_argument("--sample_id", default="武_fake_sim")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output_dir", default="outputs/wu_dynamic_footprint_pose_candidates_v1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--padding", type=int, default=16)
    parser.add_argument("--pixels_per_model_unit", type=float, default=24.0)
    parser.add_argument("--footprint_longitudinal_scale", type=float, default=0.2302875519)
    parser.add_argument("--footprint_transverse_scale", type=float, default=0.3296116590)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--smooth_weight", type=float, default=0.20)
    parser.add_argument("--prior_weight", type=float, default=0.03)
    parser.add_argument("--boundary_weight", type=float, default=0.20)
    parser.add_argument("--minimum_confidence", type=float, default=0.1)
    parser.add_argument("--candidate_count", type=int, default=6)
    parser.add_argument("--gamma_max_step_rad", type=float, default=0.75, help="Within-stroke gamma step limit for the exported safe candidate.")
    parser.add_argument("--max_h_step_mm", type=float, default=1.0)
    parser.add_argument("--max_alpha_step_rad", type=float, default=0.0872664626)
    parser.add_argument("--max_beta_step_rad", type=float, default=0.0523598776)
    main(parser.parse_args())
