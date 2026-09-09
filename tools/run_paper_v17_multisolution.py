"""Run the v17 multi-solution paper inverse audit.

The command deliberately separates three things that were previously folded
into one pose CSV:

* a seeded synthetic truth/target round trip (no real calibration claim),
* several local PSOC/LM starts, and
* a transparent Top-K ranking with per-field confidence and ambiguity.

Each candidate keeps the normal ``target/rendered/diff/comparison`` artefacts
from :mod:`tools.invert_paper_trajectory`.  The ranking combines image IoU,
local target footprint width/length, per-stroke continuity, boundary safety,
and cross-start stability.  Alpha/beta are explicitly marked low confidence
when their Jacobian/SNR is weak or the solution is boundary saturated; a high
pixel score is never allowed to turn them into a unique physical answer.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.trajectory_dataset import load_trajectory_csv
from models.paper_bbsm import PAPER_POSTURE_MAX, PAPER_POSTURE_MIN
from models.paper_fusion_renderer import PaperDynamicConfig, PaperFusionRenderer
from optim.paper_psoc_lm import target_local_footprint_geometry
from optim.trajectory_optimizer import load_target_image
from tools.build_paper_roundtrip_probe import build_probe
from tools.build_paper_v17_truth import build_random_truth
from tools.invert_paper_trajectory import (
    binary_metrics,
    flatten_canvas_trajectory,
    pick_sample,
    source_xy_to_canvas,
)
from utils.trajectory_processing import repair_sample_states


PHYSICAL_RANGES = {
    "z": 9.0,
    "alpha": float(np.deg2rad(10.0)),
    "beta": float(np.deg2rad(5.0)),
    "gamma": float(np.deg2rad(60.0)),
}


def scale_label(scale: float) -> str:
    sign = "p" if scale >= 0 else "m"
    magnitude = f"{abs(scale):g}".replace(".", "p")
    return f"{sign}{magnitude}"


def _read_pose_rows(path: str) -> dict[tuple[int, int], dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    result: dict[tuple[int, int], dict[str, str]] = {}
    for row in rows:
        key = (int(row["stroke_id"]), int(row["point_id"]))
        if key in result:
            raise ValueError(f"Duplicate stroke/point key {key} in {path}")
        result[key] = row
    return result


def _arrays_for_pose(
    pose_csv: str,
    sample,
    image_size: int,
    padding: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = _read_pose_rows(pose_csv)
    points = sample.all_points()
    values = []
    xy_source = []
    gamma = []
    for point in points:
        key = (point.stroke_id, point.point_id)
        if key not in rows:
            raise ValueError(f"Pose CSV {pose_csv} is missing {key}")
        row = rows[key]
        values.append([float(row["z"]), float(row["alpha"]), float(row["beta"])])
        xy_source.append([float(row["x"]), float(row["y"])])
        gamma.append(float(row.get("gamma", 0.0) or 0.0))
    posture = np.asarray(values, dtype=np.float32)
    if np.any(posture < PAPER_POSTURE_MIN - 1e-5) or np.any(
        posture > PAPER_POSTURE_MAX + 1e-5
    ):
        raise ValueError(f"Pose CSV exceeds paper limits: {pose_csv}")
    xy_canvas = source_xy_to_canvas(
        sample, np.asarray(xy_source, dtype=np.float32), image_size, padding
    )
    return xy_canvas, posture, np.asarray(gamma, dtype=np.float32)


def _per_stroke_differences(
    values: np.ndarray, stroke_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    first: list[np.ndarray] = []
    second: list[np.ndarray] = []
    for stroke_id in np.unique(stroke_ids):
        indices = np.flatnonzero(stroke_ids == stroke_id)
        if len(indices) < 2:
            continue
        series = values[indices]
        first.append(np.diff(series, axis=0))
        if len(series) >= 3:
            second.append(np.diff(series, n=2, axis=0))
    first_array = np.concatenate(first, axis=0) if first else np.zeros((0, values.shape[1]))
    second_array = np.concatenate(second, axis=0) if second else np.zeros((0, values.shape[1]))
    return first_array, second_array


def continuity_metrics(
    posture: np.ndarray, gamma: np.ndarray, stroke_ids: np.ndarray
) -> dict[str, Any]:
    all_values = np.column_stack((posture, gamma))
    first, second = _per_stroke_differences(all_values, stroke_ids)
    ranges = np.asarray(
        [PHYSICAL_RANGES["z"], PHYSICAL_RANGES["alpha"], PHYSICAL_RANGES["beta"], PHYSICAL_RANGES["gamma"]],
        dtype=np.float64,
    )
    first_norm = first / ranges if len(first) else first
    second_norm = second / ranges if len(second) else second
    return {
        "first_difference_normalized_rms": float(
            np.sqrt(np.mean(first_norm**2)) if len(first_norm) else 0.0
        ),
        "second_difference_normalized_rms": float(
            np.sqrt(np.mean(second_norm**2)) if len(second_norm) else 0.0
        ),
        "first_difference_normalized_max": float(
            np.max(np.abs(first_norm)) if len(first_norm) else 0.0
        ),
        "second_difference_normalized_max": float(
            np.max(np.abs(second_norm)) if len(second_norm) else 0.0
        ),
        "field_first_difference_normalized_rms": {
            field: float(
                np.sqrt(np.mean(first_norm[:, index] ** 2))
                if len(first_norm)
                else 0.0
            )
            for index, field in enumerate(("z", "alpha", "beta", "gamma"))
        },
    }


def boundary_fractions(
    posture: np.ndarray, gamma: np.ndarray, tolerance_fraction: float = 0.01
) -> dict[str, float]:
    arrays = {"z": posture[:, 0], "alpha": posture[:, 1], "beta": posture[:, 2], "gamma": gamma}
    limits = {
        "z": (11.0, 20.0),
        "alpha": (0.0, float(np.deg2rad(10.0))),
        "beta": (0.0, float(np.deg2rad(5.0))),
        "gamma": (-float(np.pi), float(np.pi)),
    }
    fractions: dict[str, float] = {}
    for field, values in arrays.items():
        low, high = limits[field]
        width = max(high - low, 1e-9)
        near = (values <= low + tolerance_fraction * width) | (
            values >= high - tolerance_fraction * width
        )
        fractions[field] = float(np.mean(near))
    return fractions


def xy_heading_from_canvas(xy: np.ndarray, stroke_ids: np.ndarray) -> np.ndarray:
    """Return the forward heading implied only by adjacent x/y points."""
    with torch.no_grad():
        heading = PaperFusionRenderer.forward_trajectory_heading(
            torch.as_tensor(xy, dtype=torch.float32),
            torch.as_tensor(stroke_ids, dtype=torch.long),
        )
    return heading.cpu().numpy().astype(np.float64)


def gamma_heading_metrics(
    gamma: np.ndarray, xy: np.ndarray, stroke_ids: np.ndarray
) -> dict[str, Any]:
    """Compare exported gamma with the geometric adjacent-point heading."""
    derived = xy_heading_from_canvas(xy, stroke_ids)
    gamma64 = gamma.astype(np.float64)
    error = np.arctan2(np.sin(gamma64 - derived), np.cos(gamma64 - derived))
    return {
        "source": "adjacent_xy_forward_heading",
        "rmse_rad": float(np.sqrt(np.mean(error**2))),
        "mae_rad": float(np.mean(np.abs(error))),
        "max_abs_rad": float(np.max(np.abs(error))),
        "derived_range_rad": [float(derived.min()), float(derived.max())],
    }


def _safe_exp_penalty(value: float, scale: float = 1.0) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.exp(-max(float(value), 0.0) * scale))


def candidate_confidence(report: dict[str, Any], field: str) -> dict[str, Any]:
    decision = report.get("field_decisions", {}).get(field, {})
    boundary = decision.get("boundary_fraction")
    snr = decision.get("initial_median_snr")
    source = decision.get("source")
    low_reasons = []
    if boundary is not None and float(boundary) > 0.05:
        low_reasons.append("boundary_saturation")
    if snr is not None and float(snr) < 1.0:
        low_reasons.append("weak_image_jacobian")
    if field in {"alpha", "beta"} and report.get("identifiability", {}).get(
        "joint_pose_unique_from_single_image"
    ) is False:
        low_reasons.append("single_image_non_unique")
    if field == "gamma" and source == "initial_pose_csv":
        low_reasons.append("not_optimized")
    if low_reasons:
        level = "low"
    elif field == "z":
        level = "high_simulation"
    else:
        level = "medium_simulation"
    return {
        "level": level,
        "reasons": low_reasons,
        "source": source,
        "boundary_fraction": boundary,
        "median_snr": snr,
    }


def build_inversion_command(args: argparse.Namespace, initial_csv: Path, run_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "tools/invert_paper_trajectory.py",
        "--trajectory_csv",
        args.trajectory_csv,
        "--initial_pose_csv",
        str(initial_csv),
        "--target_image",
        str(args.target_image),
        "--bbsmg_ckpt",
        args.bbsmg_ckpt,
        "--character",
        args.character,
        "--output_dir",
        str(run_dir),
        "--output_stem",
        args.output_stem,
        "--device",
        args.device,
        "--order",
        str(args.order),
        "--optimization_size",
        str(args.optimization_size),
        "--max_steps",
        str(args.max_steps),
        "--damping",
        str(args.damping),
        "--jacobian_mode",
        "finite_difference",
        "--finite_difference_eps",
        str(args.finite_difference_eps),
        "--field_mode",
        "auto",
        "--observability_gate_mode",
        "node_snr",
        "--min_observability_snr",
        str(args.min_observability_snr),
        "--joint_gate_action",
        "prune",
        "--optimize_gamma",
        "--gamma_max_abs_deg",
        str(args.gamma_max_abs_deg),
        "--dynamic_profile",
        "wang2020_figure4_digitized_v1",
        "--offset_transfer_scale",
        str(args.offset_transfer_scale),
        "--pixel_weight",
        str(args.pixel_weight),
        "--h_smoothness_weight",
        str(args.h_smoothness_weight),
        "--alpha_smoothness_weight",
        str(args.alpha_smoothness_weight),
        "--beta_smoothness_weight",
        str(args.beta_smoothness_weight),
        "--gamma_smoothness_weight",
        str(args.gamma_smoothness_weight),
        "--h_prior_weight",
        str(args.h_prior_weight),
        "--alpha_prior_weight",
        str(args.alpha_prior_weight),
        "--beta_prior_weight",
        str(args.beta_prior_weight),
        "--gamma_prior_weight",
        str(args.gamma_prior_weight),
        "--footprint_longitudinal_scale",
        str(args.footprint_longitudinal_scale),
        "--footprint_transverse_scale",
        str(args.footprint_transverse_scale),
        "--render_max_step_px",
        str(args.render_max_step_px),
        "--point_batch_size",
        str(args.point_batch_size),
    ]
    return command


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def _render_truth_target(args: argparse.Namespace, truth_csv: Path, output_image: Path) -> None:
    command = [
        sys.executable,
        "-u",
        "tools/render_paper_trajectory.py",
        "--trajectory_csv",
        args.trajectory_csv,
        "--pose_csv",
        str(truth_csv),
        "--bbsmg_ckpt",
        args.bbsmg_ckpt,
        "--character",
        args.character,
        "--output_image",
        str(output_image),
        "--device",
        args.device,
        "--dynamic_profile",
        "wang2020_figure4_digitized_v1",
        "--offset_transfer_scale",
        str(args.offset_transfer_scale),
        "--footprint_longitudinal_scale",
        str(args.footprint_longitudinal_scale),
        "--footprint_transverse_scale",
        str(args.footprint_transverse_scale),
        "--render_max_step_px",
        str(args.render_max_step_px),
        "--point_batch_size",
        str(args.point_batch_size),
        "--smooth_passes",
        "0",
    ]
    run_logged(command, output_image.with_suffix(".log"))


def local_footprint_metrics(
    renderer: PaperFusionRenderer,
    target: np.ndarray,
    base_xy: np.ndarray,
    candidate_xy: np.ndarray,
    posture: np.ndarray,
    gamma: np.ndarray,
    stroke_ids: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, float]:
    device = next(renderer.parameters()).device
    target_tensor = torch.as_tensor(target, dtype=torch.float32, device=device)
    base_tensor = torch.as_tensor(base_xy, dtype=torch.float32, device=device)
    candidate_tensor = torch.as_tensor(candidate_xy, dtype=torch.float32, device=device)
    posture_tensor = torch.as_tensor(posture, dtype=torch.float32, device=device)
    ids_tensor = torch.as_tensor(stroke_ids, dtype=torch.long, device=device)
    with torch.no_grad():
        target_drag, target_width = target_local_footprint_geometry(
            target_tensor,
            base_tensor,
            ids_tensor,
            radius_px=args.footprint_radius_px,
            samples=args.footprint_samples,
            threshold=args.footprint_threshold,
            temperature=args.footprint_temperature,
            pixels_per_model_unit=args.pixels_per_model_unit,
            longitudinal_scale=args.footprint_longitudinal_scale,
            transverse_scale=args.footprint_transverse_scale,
        )
        states = renderer.compute_dynamic_states(
            candidate_tensor, posture_tensor, ids_tensor
        )
        geometry = states["geometry"]
        predicted_drag = geometry[:, 0] + geometry[:, 1]
        predicted_width = geometry[:, 2]
        drag_scale = target_drag.abs().clamp_min(0.25)
        width_scale = target_width.abs().clamp_min(0.08)
        drag_residual = (predicted_drag - target_drag) / drag_scale
        width_residual = (predicted_width - target_width) / width_scale
        target_aspect = target_width / target_drag.abs().clamp_min(0.25)
        predicted_aspect = predicted_width / predicted_drag.abs().clamp_min(0.25)
        aspect_residual = (predicted_aspect - target_aspect) / target_aspect.abs().clamp_min(0.08)
        combined_residual = torch.cat((drag_residual, width_residual, aspect_residual))
    return {
        "drag_normalized_rmse": float(torch.sqrt(torch.mean(drag_residual**2)).cpu()),
        "width_normalized_rmse": float(torch.sqrt(torch.mean(width_residual**2)).cpu()),
        "aspect_ratio_normalized_rmse": float(torch.sqrt(torch.mean(aspect_residual**2)).cpu()),
        "combined_normalized_rmse": float(torch.sqrt(torch.mean(combined_residual**2)).cpu()),
        "target_drag_mean_model_unit": float(target_drag.mean().cpu()),
        "target_half_width_mean_model_unit": float(target_width.mean().cpu()),
        "predicted_drag_mean_model_unit": float(predicted_drag.mean().cpu()),
        "predicted_half_width_mean_model_unit": float(predicted_width.mean().cpu()),
        "target_aspect_ratio_mean": float(target_aspect.mean().cpu()),
        "predicted_aspect_ratio_mean": float(predicted_aspect.mean().cpu()),
    }


def pose_recovery_metrics(
    truth_pose_csv: str,
    estimate_posture: np.ndarray,
    estimate_gamma: np.ndarray,
    sample,
) -> dict[str, dict[str, float]]:
    """Compare one candidate with synthetic truth using paper ranges.

    This is intentionally kept separate from the ranking score.  A candidate
    may have a good image score while alpha/beta are not identifiable; the
    report must expose that fact instead of hiding it in a composite number.
    """
    truth_rows = _read_pose_rows(truth_pose_csv)
    truth_posture = []
    truth_gamma = []
    for point in sample.all_points():
        row = truth_rows[(point.stroke_id, point.point_id)]
        truth_posture.append(
            [float(row["z"]), float(row["alpha"]), float(row["beta"])]
        )
        truth_gamma.append(float(row.get("gamma", 0.0) or 0.0))
    truth_posture_array = np.asarray(truth_posture, dtype=np.float64)
    truth_gamma_array = np.asarray(truth_gamma, dtype=np.float64)
    estimate = {
        "z": np.asarray(estimate_posture[:, 0], dtype=np.float64),
        "alpha": np.asarray(estimate_posture[:, 1], dtype=np.float64),
        "beta": np.asarray(estimate_posture[:, 2], dtype=np.float64),
        "gamma": np.asarray(estimate_gamma, dtype=np.float64),
    }
    truth = {
        "z": truth_posture_array[:, 0],
        "alpha": truth_posture_array[:, 1],
        "beta": truth_posture_array[:, 2],
        "gamma": truth_gamma_array,
    }
    result: dict[str, dict[str, float]] = {}
    for field in PHYSICAL_RANGES:
        error = estimate[field] - truth[field]
        if field == "gamma":
            error = np.arctan2(np.sin(error), np.cos(error))
        result[field] = {
            "rmse": float(np.sqrt(np.mean(error**2))),
            "mae": float(np.mean(np.abs(error))),
            "normalized_rmse": float(
                np.sqrt(np.mean(error**2)) / PHYSICAL_RANGES[field]
            ),
        }
    return result


def _load_candidate_images(run_dir: Path, stem: str, image_size: int) -> np.ndarray:
    path = run_dir / f"{stem}_rendered.png"
    if not path.exists():
        raise FileNotFoundError(path)
    return load_target_image(str(path), image_size=image_size)


def aggregate_stability(
    candidates: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, float]]:
    arrays: dict[str, list[np.ndarray]] = {field: [] for field in PHYSICAL_RANGES}
    for item in candidates.values():
        for field in PHYSICAL_RANGES:
            arrays[field].append(item["field_arrays"][field])
    field_stability: dict[str, Any] = {}
    per_candidate_distance: dict[str, float] = {label: 0.0 for label in candidates}
    for field, values in arrays.items():
        stack = np.stack(values, axis=0)
        std = stack.std(axis=0)
        normalized = float(np.sqrt(np.mean(std**2)) / PHYSICAL_RANGES[field])
        field_stability[field] = {
            "normalized_cross_start_std_rmse": normalized,
            "cross_start_std_rmse": float(np.sqrt(np.mean(std**2))),
            "range": [float(stack.mean(axis=0).min()), float(stack.mean(axis=0).max())],
            "passed_v17_stability": normalized <= 0.02,
        }
        for index, label in enumerate(candidates):
            per_candidate_distance[label] += float(
                np.sqrt(np.mean(((stack[index] - stack.mean(axis=0)) / PHYSICAL_RANGES[field]) ** 2))
            ) / len(PHYSICAL_RANGES)
    return field_stability, per_candidate_distance


def score_candidate(
    image_metrics: dict[str, Any],
    footprint: dict[str, float],
    continuity: dict[str, Any],
    maximum_boundary: float,
    stability_distance: float,
    weights: dict[str, float],
) -> dict[str, Any]:
    image_score = float(np.clip(image_metrics.get("iou_at_0.5", 0.0), 0.0, 1.0))
    footprint_score = _safe_exp_penalty(footprint.get("combined_normalized_rmse", math.inf), 1.0)
    continuity_score = _safe_exp_penalty(
        continuity.get("first_difference_normalized_rms", math.inf), 4.0
    )
    boundary_score = float(np.clip(1.0 - maximum_boundary, 0.0, 1.0))
    stability_score = _safe_exp_penalty(stability_distance, 8.0)
    components = {
        "image_iou": image_score,
        "local_footprint": footprint_score,
        "continuity": continuity_score,
        "boundary_safety": boundary_score,
        "cross_start_stability": stability_score,
    }
    total = sum(weights[key] * components[key] for key in components)
    return {"total": float(total), "components": components, "weights": weights}


def _copy_ranked_artifacts(item: dict[str, Any], rank_dir: Path) -> None:
    rank_dir.mkdir(parents=True, exist_ok=True)
    for key in ("estimate_csv", "report_json", "rendered_png", "diff_png", "comparison_png", "target_png"):
        source = Path(item[key])
        if source.exists():
            shutil.copy2(source, rank_dir / source.name)
    (rank_dir / "candidate.json").write_text(
        json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    # In heading mode gamma is the absolute forward tangent in the canvas
    # frame, so a stroke may legitimately point anywhere in [-pi, pi].  The
    # historical ±30° default is appropriate only for a bounded relative
    # brush twist and silently clips vertical/diagonal strokes.  Expand the
    # default-like value for heading mode while still allowing an explicit
    # wider bound to pass through unchanged.
    if args.gamma_truth_mode == "heading" and args.gamma_max_abs_deg < 90.0:
        args.gamma_max_abs_deg = 180.0
        print(
            "[GAMMA] heading truth uses absolute canvas tangent; "
            "expanded gamma_max_abs_deg to 180",
            flush=True,
        )
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    truth_csv = Path(args.truth_pose_csv) if args.truth_pose_csv else root / "synthetic_truth.csv"
    if not truth_csv.exists():
        build_random_truth(
            args.trajectory_csv,
            str(truth_csv),
            character=args.character,
            sample_id=args.sample_id,
            seed=args.seed,
            gamma_mode=args.gamma_truth_mode,
        )
    target_image = Path(args.target_image) if args.target_image else root / "synthetic_target.png"
    args.target_image = str(target_image)
    if not target_image.exists():
        _render_truth_target(args, truth_csv, target_image)

    samples = load_trajectory_csv(args.trajectory_csv)
    sample = repair_sample_states(
        pick_sample(
            samples,
            sample_id=args.sample_id,
            character=args.character,
            index=args.index,
        )
    )
    base_xy, stroke_ids = flatten_canvas_trajectory(
        sample, args.image_size, args.padding
    )
    target = load_target_image(str(target_image), image_size=args.image_size)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    renderer = PaperFusionRenderer.from_checkpoint(
        args.bbsmg_ckpt,
        device=device,
        image_size=args.image_size,
        dynamic=PaperDynamicConfig(
            offset_transfer_scale=args.offset_transfer_scale,
            pixels_per_model_unit=args.pixels_per_model_unit,
            footprint_longitudinal_scale=args.footprint_longitudinal_scale,
            footprint_transverse_scale=args.footprint_transverse_scale,
            render_max_step_px=args.render_max_step_px,
            gamma_mode="relative_to_heading",
        ),
        point_batch_size=args.point_batch_size,
    )
    labels: dict[str, str] = {}
    candidates: dict[str, dict[str, Any]] = {}
    manifest: dict[str, Any] = {
        "format": "paper_pose_multisolution_runner_v17",
        "simulation_only": True,
        "trajectory_csv": args.trajectory_csv,
        "truth_pose_csv": str(truth_csv),
        "target_image": str(target_image),
        "bbsmg_ckpt": args.bbsmg_ckpt,
        "character": args.character,
        "seed": args.seed,
        "gamma_truth_mode": args.gamma_truth_mode,
        "gamma_max_abs_deg": args.gamma_max_abs_deg,
        "perturbation_scales": args.perturbation_scales,
        "runs": {},
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for scale in args.perturbation_scales:
        label = scale_label(scale)
        run_dir = root / label
        initial_csv = run_dir / "initial_pose.csv"
        estimate_csv = run_dir / f"{args.output_stem}_trajectory.csv"
        report_json = run_dir / f"{args.output_stem}_report.json"
        build_probe(
            str(truth_csv),
            str(initial_csv),
            profile="perturbed_initial",
            perturbation_scale=scale,
        )
        command = build_inversion_command(args, initial_csv, run_dir)
        manifest["runs"][label] = {
            "scale": scale,
            "initial_pose_csv": str(initial_csv),
            "estimate_csv": str(estimate_csv),
            "command": command,
        }
        (root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if not (args.resume_completed and estimate_csv.exists() and report_json.exists()):
            print(f"[V17] optimizing start {label} (scale={scale:+g})", flush=True)
            run_logged(command, run_dir / "run.log")
        if not estimate_csv.exists() or not report_json.exists():
            raise RuntimeError(f"Missing inversion artefacts for {label}")
        report = json.loads(report_json.read_text(encoding="utf-8"))
        xy, posture, gamma = _arrays_for_pose(
            str(estimate_csv), sample, args.image_size, args.padding
        )
        metrics = report.get("metrics", {})
        footprint = local_footprint_metrics(
            renderer,
            target,
            base_xy,
            xy,
            posture,
            gamma,
            stroke_ids,
            args,
        )
        continuity = continuity_metrics(posture, gamma, stroke_ids)
        gamma_heading = gamma_heading_metrics(gamma, xy, stroke_ids)
        recovery = pose_recovery_metrics(str(truth_csv), posture, gamma, sample)
        bounds = boundary_fractions(posture, gamma)
        item = {
            "label": label,
            "scale": float(scale),
            "estimate_csv": str(estimate_csv),
            "report_json": str(report_json),
            "rendered_png": str(run_dir / f"{args.output_stem}_rendered.png"),
            "diff_png": str(run_dir / f"{args.output_stem}_diff.png"),
            "comparison_png": str(run_dir / f"{args.output_stem}_comparison.png"),
            "target_png": str(run_dir / f"{args.output_stem}_target.png"),
            "image_metrics": metrics,
            "footprint_metrics": footprint,
            "pose_recovery": recovery,
            "continuity_metrics": continuity,
            "gamma_heading_metrics": gamma_heading,
            "boundary_fractions": bounds,
            "maximum_boundary_fraction": float(max(bounds.values(), default=0.0)),
            "field_confidence": {
                field: candidate_confidence(report, field)
                for field in ("H", "alpha", "beta", "gamma")
            },
            "field_arrays": {
                "z": posture[:, 0],
                "alpha": posture[:, 1],
                "beta": posture[:, 2],
                "gamma": gamma,
            },
            "report_format": report.get("format"),
            "joint_identifiability": report.get("identifiability", {}).get(
                "joint_pose_unique_from_single_image"
            ),
        }
        candidates[label] = item

    field_stability, distances = aggregate_stability(candidates)
    field_confidence_intervals = {}
    for field in PHYSICAL_RANGES:
        stack = np.stack([item["field_arrays"][field] for item in candidates.values()], axis=0)
        field_confidence_intervals[field] = {
            "pointwise_p10": np.quantile(stack, 0.10, axis=0).tolist(),
            "pointwise_p90": np.quantile(stack, 0.90, axis=0).tolist(),
            "candidate_min": float(stack.min()),
            "candidate_max": float(stack.max()),
            "interpretation": "empirical multi-start interval; not a calibrated physical confidence interval",
        }
    weights = {
        "image_iou": args.image_weight,
        "local_footprint": args.footprint_weight,
        "continuity": args.continuity_weight,
        "boundary_safety": args.boundary_weight,
        "cross_start_stability": args.stability_weight,
    }
    for label, item in candidates.items():
        item["cross_start_stability_distance"] = distances[label]
        item["score"] = score_candidate(
            item["image_metrics"],
            item["footprint_metrics"],
            item["continuity_metrics"],
            item["maximum_boundary_fraction"],
            distances[label],
            weights,
        )
        # Numpy arrays are for computation only and must not leak into JSON.
        item.pop("field_arrays")
    ranked = sorted(candidates.values(), key=lambda item: item["score"]["total"], reverse=True)
    top_k = max(1, min(args.top_k, len(ranked)))
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
        item["export_eligible"] = bool(
            item["image_metrics"].get("iou_at_0.5", 0.0) >= args.min_iou
            and item["maximum_boundary_fraction"] <= args.max_boundary_fraction
            and item["continuity_metrics"].get("second_difference_normalized_max", 0.0)
            <= args.max_continuity_jump
        )
        if not item["export_eligible"]:
            reasons = []
            if item["image_metrics"].get("iou_at_0.5", 0.0) < args.min_iou:
                reasons.append("image_iou_below_threshold")
            if item["maximum_boundary_fraction"] > args.max_boundary_fraction:
                reasons.append("boundary_saturated")
            if item["continuity_metrics"].get("second_difference_normalized_max", 0.0) > args.max_continuity_jump:
                reasons.append("continuity_jump")
            item["withheld_reasons"] = reasons
    rank_root = root / "top_k"
    if args.copy_top_k:
        for rank, item in enumerate(ranked[:top_k], start=1):
            _copy_ranked_artifacts(item, rank_root / f"rank_{rank:02d}")

    # Convert all candidate data to JSON-safe values after ranking.
    serializable_candidates = []
    for item in ranked:
        serializable = dict(item)
        serializable_candidates.append(serializable)
    acceptance = {
        "synthetic_target_iou_min": float(
            min(item["image_metrics"].get("iou_at_0.5", 0.0) for item in ranked)
        ),
        "synthetic_target_iou_threshold": float(args.min_iou),
        "h_normalized_rmse_threshold": 0.05,
        "h_normalized_rmse_worst": float(
            max(item["pose_recovery"]["z"]["normalized_rmse"] for item in ranked)
        ),
        "h_recovery_passed": all(
            item["pose_recovery"]["z"]["normalized_rmse"] <= 0.05
            for item in ranked
        ),
        "no_boundary_saturation_threshold": float(args.max_boundary_fraction),
        "different_initial_images_consistent": all(
            value["passed_v17_stability"] for value in field_stability.values()
        ),
        "alpha_beta_unique_from_single_image": False,
    }
    summary = {
        "format": "paper_pose_multisolution_v17",
        "simulation_only": True,
        "truth_pose_csv": str(truth_csv),
        "target_image": str(target_image),
        "candidate_count": len(ranked),
        "top_k": top_k,
        "scoring_weights": weights,
        "candidates": serializable_candidates,
        "field_stability": field_stability,
        "field_confidence_intervals": field_confidence_intervals,
        "acceptance": acceptance,
        "ambiguity": {
            "single_image_joint_pose_unique": False,
            "alpha": "low_confidence_without_local_footprint_or_pressure_observation",
            "beta": "low_confidence_without_local_footprint_or_pressure_observation",
            "gamma": (
                "derived_from_adjacent_xy_heading_when_gamma_truth_mode_heading; compare candidate gamma_heading_metrics"
                if args.gamma_truth_mode == "heading"
                else "candidate_is_bounded; adjacent_xy_heading_is_reported_for_comparison"
            ),
            "export_policy": "keep_top_k_candidates; do_not collapse alpha/beta to one physical answer",
        },
        "warning": (
            "All poses and confidence values are simulation candidates. "
            "Real robot alpha/beta/z calibration requires local footprint, "
            "pressure/speed, multi-view, or hardware measurements."
        ),
    }
    (root / "candidate_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # A compact CSV makes Top-K candidates convenient to inspect without
    # treating one row as ground truth.
    with (root / "top_k_candidates.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["rank", "label", "score", "iou_at_0.5", "footprint_rmse", "aspect_ratio_rmse", "gamma_heading_rmse", "max_boundary_fraction", "stability_distance", "export_eligible"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in ranked[:top_k]:
            writer.writerow(
                {
                    "rank": item["rank"],
                    "label": item["label"],
                    "score": item["score"]["total"],
                    "iou_at_0.5": item["image_metrics"].get("iou_at_0.5"),
                    "footprint_rmse": item["footprint_metrics"].get("combined_normalized_rmse"),
                    "aspect_ratio_rmse": item["footprint_metrics"].get("aspect_ratio_normalized_rmse"),
                    "gamma_heading_rmse": item["gamma_heading_metrics"].get("rmse_rad"),
                    "max_boundary_fraction": item["maximum_boundary_fraction"],
                    "stability_distance": item["cross_start_stability_distance"],
                    "export_eligible": item["export_eligible"],
                }
            )
    print(
        f"[DONE] v17 multi-solution audit: candidates={len(ranked)}, "
        f"top_k={top_k}, output={root}",
        flush=True,
    )
    for item in ranked[:top_k]:
        print(
            f"[TOP {item['rank']}] {item['label']} score={item['score']['total']:.6f} "
            f"IoU={item['image_metrics'].get('iou_at_0.5', 0.0):.6f} "
            f"footprint={item['footprint_metrics']['combined_normalized_rmse']:.6f} "
            f"boundary={item['maximum_boundary_fraction']:.6f} "
            f"eligible={item['export_eligible']}",
            flush=True,
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory_csv", required=True)
    parser.add_argument("--truth_pose_csv", default=None)
    parser.add_argument("--target_image", default=None)
    parser.add_argument("--bbsmg_ckpt", required=True)
    parser.add_argument("--character", default="武")
    parser.add_argument("--sample_id", default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_stem", default="wu")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--padding", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17017)
    parser.add_argument("--gamma_truth_mode", choices=("random", "heading"), default="random")
    parser.add_argument("--perturbation_scales", type=float, nargs="+", default=[-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--resume_completed", action="store_true")
    parser.add_argument("--copy_top_k", action="store_true")
    parser.add_argument("--order", type=int, default=1)
    parser.add_argument("--optimization_size", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=8)
    parser.add_argument("--damping", type=float, default=0.1)
    parser.add_argument("--finite_difference_eps", type=float, default=0.01)
    parser.add_argument("--min_observability_snr", type=float, default=1.0)
    parser.add_argument("--gamma_max_abs_deg", type=float, default=30.0)
    parser.add_argument("--offset_transfer_scale", type=float, default=1.0)
    parser.add_argument("--pixel_weight", type=float, default=5.0)
    parser.add_argument("--h_smoothness_weight", type=float, default=0.01)
    parser.add_argument("--alpha_smoothness_weight", type=float, default=0.10)
    parser.add_argument("--beta_smoothness_weight", type=float, default=0.10)
    parser.add_argument("--gamma_smoothness_weight", type=float, default=0.10)
    parser.add_argument("--h_prior_weight", type=float, default=0.001)
    parser.add_argument("--alpha_prior_weight", type=float, default=0.05)
    parser.add_argument("--beta_prior_weight", type=float, default=0.05)
    parser.add_argument("--gamma_prior_weight", type=float, default=0.05)
    parser.add_argument("--footprint_longitudinal_scale", type=float, default=0.22)
    parser.add_argument("--footprint_transverse_scale", type=float, default=0.258)
    parser.add_argument("--pixels_per_model_unit", type=float, default=20.0)
    parser.add_argument("--render_max_step_px", type=float, default=2.0)
    parser.add_argument("--point_batch_size", type=int, default=64)
    parser.add_argument("--footprint_radius_px", type=float, default=10.0)
    parser.add_argument("--footprint_samples", type=int, default=33)
    parser.add_argument("--footprint_threshold", type=float, default=0.35)
    parser.add_argument("--footprint_temperature", type=float, default=0.08)
    parser.add_argument("--image_weight", type=float, default=0.45)
    parser.add_argument("--footprint_weight", type=float, default=0.25)
    parser.add_argument("--continuity_weight", type=float, default=0.15)
    parser.add_argument("--boundary_weight", type=float, default=0.10)
    parser.add_argument("--stability_weight", type=float, default=0.05)
    parser.add_argument("--min_iou", type=float, default=0.95)
    parser.add_argument("--max_boundary_fraction", type=float, default=0.05)
    parser.add_argument("--max_continuity_jump", type=float, default=0.25)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
