"""Invert a target image into bounded planar trajectory and paper posture."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image, ImageOps, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.trajectory_dataset import load_trajectory_csv
from models.geometry import (
    CanvasTransform,
    normalize_trajectory_xy,
    trajectory_bounds,
)
from models.paper_calibration import (
    DYNAMIC_PROFILES,
    WANG2020_PROFILE,
    paper_calibration_metadata,
)
from models.paper_bbsm import PAPER_POSTURE_MAX, PAPER_POSTURE_MIN
from models.paper_fusion_renderer import PaperDynamicConfig, PaperFusionRenderer
from optim.paper_psoc_lm import PaperPSOCLM
from optim.trajectory_optimizer import load_target_image


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


def flatten_canvas_trajectory(sample, image_size: int, padding: int):
    normalized = normalize_trajectory_xy(
        sample, canvas_size=image_size, padding=padding
    )
    xy, stroke_ids = [], []
    for stroke, points in zip(sample.sorted_strokes(), normalized):
        xy.extend(points)
        stroke_ids.extend([stroke.stroke_id] * len(points))
    return np.asarray(xy, dtype=np.float32), np.asarray(stroke_ids, dtype=np.int64)


def trajectory_canvas_transform(
    sample, image_size: int, padding: int
) -> CanvasTransform:
    min_x, max_x, min_y, max_y = trajectory_bounds(sample)
    return CanvasTransform(
        min_x,
        max_x,
        min_y,
        max_y,
        dst_size=image_size,
        padding=padding,
    )


def canvas_xy_to_source(
    sample,
    xy_canvas: np.ndarray,
    image_size: int,
    padding: int,
) -> np.ndarray:
    transform = trajectory_canvas_transform(sample, image_size, padding)
    return np.asarray(
        [transform.unmap_point(float(x), float(y)) for x, y in xy_canvas],
        dtype=np.float32,
    )


def source_xy_to_canvas(
    sample,
    xy_source: np.ndarray,
    image_size: int,
    padding: int,
) -> np.ndarray:
    transform = trajectory_canvas_transform(sample, image_size, padding)
    return np.asarray(
        [transform.map_point(float(x), float(y)) for x, y in xy_source],
        dtype=np.float32,
    )


def load_initial_pose_csv(
    path: str, sample
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load staged x/y/H/alpha/beta/gamma keyed by stroke and point."""
    expected_keys = {
        (point.stroke_id, point.point_id) for point in sample.all_points()
    }
    rows_by_key = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            key = (int(row["stroke_id"]), int(row["point_id"]))
            if key in rows_by_key:
                raise ValueError(
                    f"Initial pose CSV contains duplicate stroke/point {key}"
                )
            row_character = row.get("character")
            if (
                row_character
                and sample.character
                and row_character != sample.character
            ):
                raise ValueError(
                    "Initial pose CSV character does not match trajectory "
                    f"character: {row_character!r} != {sample.character!r}"
                )
            if row.get("z_unit") not in (None, "", "mm"):
                raise ValueError("Initial pose CSV z_unit must be mm")
            if row.get("angle_unit") not in (None, "", "rad"):
                raise ValueError("Initial pose CSV angle_unit must be rad")
            gamma = float(row.get("gamma", 0.0) or 0.0)
            if not np.isfinite(gamma) or abs(gamma) > np.pi + 1e-6:
                raise ValueError("Initial pose CSV gamma must be in [-pi,pi]")
            rows_by_key[key] = {
                "xy": [float(row["x"]), float(row["y"])],
                "posture": [
                    float(row["z"]),
                    float(row["alpha"]),
                    float(row["beta"]),
                ],
                "gamma": gamma,
            }
    actual_keys = set(rows_by_key)
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    if missing or extra:
        raise ValueError(
            "Initial pose CSV stroke/point keys do not match the selected "
            f"trajectory; missing={missing[:5]}, extra={extra[:5]}"
        )
    posture = []
    xy_source = []
    gamma = []
    for point in sample.all_points():
        value = rows_by_key[(point.stroke_id, point.point_id)]
        posture.append(value["posture"])
        xy_source.append(value["xy"])
        gamma.append(value["gamma"])
    posture_array = np.asarray(posture, dtype=np.float32)
    tolerance = 1e-6
    if np.any(posture_array < PAPER_POSTURE_MIN - tolerance) or np.any(
        posture_array > PAPER_POSTURE_MAX + tolerance
    ):
        raise ValueError(
            "Initial pose CSV exceeds H=11-20 mm, alpha=0-10 deg, "
            "beta=0-5 deg"
        )
    return (
        np.clip(posture_array, PAPER_POSTURE_MIN, PAPER_POSTURE_MAX),
        np.asarray(xy_source, dtype=np.float32),
        np.asarray(gamma, dtype=np.float32),
    )


def save_pose_csv(
    sample,
    posture: np.ndarray,
    output_path: Path,
    regression_angle_basis: str,
    field_decisions: dict,
    xy_source: np.ndarray | None = None,
    gamma: np.ndarray | None = None,
    prototype: str = "paper_psoc_lm_v9_bounded_xy",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "character",
        "sample_id",
        "stroke_id",
        "point_id",
        "x",
        "y",
        "x_source",
        "x_confidence",
        "y_source",
        "y_confidence",
        "z",
        "alpha",
        "beta",
        "gamma",
        "state",
        "z_unit",
        "angle_unit",
        "pose_frame",
        "prototype",
        "regression_angle_basis",
        "z_source",
        "z_confidence",
        "alpha_source",
        "alpha_confidence",
        "beta_source",
        "beta_confidence",
        "gamma_source",
        "gamma_confidence",
    ]
    points = sample.all_points()
    if len(points) != len(posture):
        raise ValueError("Posture count does not match trajectory point count")
    if xy_source is None:
        xy_source = np.asarray(
            [[point.x, point.y] for point in points], dtype=np.float32
        )
    if len(xy_source) != len(points):
        raise ValueError("x/y count does not match trajectory point count")
    if gamma is None:
        gamma = np.zeros(len(points), dtype=np.float32)
    if len(gamma) != len(points):
        raise ValueError("gamma count does not match trajectory point count")
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for point, pose, optimized_xy, gamma_value in zip(
            points, posture, xy_source, gamma
        ):
            writer.writerow(
                {
                    "character": sample.character,
                    "sample_id": sample.meta.get("sample_id"),
                    "stroke_id": point.stroke_id,
                    "point_id": point.point_id,
                    "x": repr(float(optimized_xy[0])),
                    "y": repr(float(optimized_xy[1])),
                    "x_source": field_decisions["x"]["source"],
                    "x_confidence": field_decisions["x"]["confidence"],
                    "y_source": field_decisions["y"]["source"],
                    "y_confidence": field_decisions["y"]["confidence"],
                    # Prototype contract: CSV z is paper-model H in millimetres.
                    "z": repr(float(pose[0])),
                    "alpha": repr(float(pose[1])),
                    "beta": repr(float(pose[2])),
                    "gamma": repr(float(gamma_value)),
                    "state": int(point.state),
                    "z_unit": "mm",
                    "angle_unit": "rad",
                    "pose_frame": "paper_model",
                    "prototype": prototype,
                    "regression_angle_basis": regression_angle_basis,
                    "z_source": field_decisions["H"]["source"],
                    "z_confidence": field_decisions["H"]["confidence"],
                    "alpha_source": field_decisions["alpha"]["source"],
                    "alpha_confidence": field_decisions["alpha"]["confidence"],
                    "beta_source": field_decisions["beta"]["source"],
                    "beta_confidence": field_decisions["beta"]["confidence"],
                    "gamma_source": field_decisions["gamma"]["source"],
                    "gamma_confidence": field_decisions["gamma"]["confidence"],
                }
            )


def save_gray(array: np.ndarray, path: Path) -> None:
    image = Image.fromarray(
        np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8),
        mode="L",
    )
    image.save(path)


def comparison_panel(
    target: np.ndarray,
    initial: np.ndarray,
    optimized: np.ndarray,
    output_path: Path,
) -> None:
    arrays = [target, initial, optimized, np.abs(target - optimized)]
    labels = ["Target", "Initial render", "Optimized render", "Absolute diff"]
    panels = []
    for array, label in zip(arrays, labels):
        panel = Image.fromarray(
            np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8),
            mode="L",
        ).convert("RGB")
        panel = ImageOps.expand(panel, border=(0, 24, 0, 0), fill="white")
        ImageDraw.Draw(panel).text((4, 4), label, fill="black")
        panels.append(panel)
    canvas = Image.new(
        "RGB", (sum(panel.width for panel in panels), panels[0].height), "white"
    )
    left = 0
    for panel in panels:
        canvas.paste(panel, (left, 0))
        left += panel.width
    canvas.save(output_path)


def binary_metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    pred = prediction >= 0.5
    truth = target >= 0.5
    intersection = int(np.logical_and(pred, truth).sum())
    union = int(np.logical_or(pred, truth).sum())
    target_ink = float(truth.mean())
    prediction_ink = float(pred.mean())
    soft_target_ink = float(target.mean())
    soft_prediction_ink = float(prediction.mean())
    return {
        "plain_mse": float(np.mean((prediction - target) ** 2)),
        "mae": float(np.mean(np.abs(prediction - target))),
        "dice_at_0.5": float(
            (2 * intersection + 1e-6) / (pred.sum() + truth.sum() + 1e-6)
        ),
        "iou_at_0.5": float((intersection + 1e-6) / (union + 1e-6)),
        "target_ink_at_0.5": target_ink,
        "prediction_ink_at_0.5": prediction_ink,
        "ink_ratio_at_0.5": float(
            (prediction_ink + 1e-8) / (target_ink + 1e-8)
        ),
        "soft_target_ink": soft_target_ink,
        "soft_prediction_ink": soft_prediction_ink,
        "soft_ink_ratio": float(
            (soft_prediction_ink + 1e-8) / (soft_target_ink + 1e-8)
        ),
    }


def trajectory_target_coverage(
    xy_canvas: np.ndarray,
    target: np.ndarray,
    tolerance_px: int = 5,
) -> float:
    mask = Image.fromarray(
        ((target >= 0.5).astype(np.uint8) * 255), mode="L"
    )
    kernel = max(2 * int(tolerance_px) + 1, 3)
    if kernel % 2 == 0:
        kernel += 1
    support = np.asarray(mask.filter(ImageFilter.MaxFilter(kernel))) > 0
    x = np.clip(np.rint(xy_canvas[:, 0]).astype(np.int64), 0, target.shape[1] - 1)
    y = np.clip(np.rint(xy_canvas[:, 1]).astype(np.int64), 0, target.shape[0] - 1)
    return float(support[y, x].mean())


def main(args: argparse.Namespace) -> None:
    if args.optimize_xy and args.xy_max_offset_px > args.padding:
        raise ValueError(
            "xy_max_offset_px must not exceed padding, so corrected points "
            "remain inside the render canvas"
        )
    if (
        args.observability_gate_mode == "node_snr"
        and args.optimization_size < 64
    ):
        print(
            "[WARN] node_snr pose inversion below 64x64 can reduce the "
            "low-resolution LM objective while worsening the 128x128 image. "
            "Use --optimization_size 64 for pose-recovery validation.",
            flush=True,
        )
    v10_continuity_enabled = (
        args.cap_order_to_points
        or args.h_point_velocity_weight > 0
        or args.h_point_acceleration_weight > 0
    )
    output_format = (
        "paper_psoc_lm_v14_node_snr_gate"
        if args.observability_gate_mode == "node_snr"
        else (
            "paper_psoc_lm_v12_nonaxisymmetric_gamma"
            if args.optimize_gamma
            else (
                "paper_psoc_lm_v11_staged_pose"
                if args.initial_pose_csv
                else (
                    "paper_psoc_lm_v10_point_continuity"
                    if v10_continuity_enabled
                    else "paper_psoc_lm_v9_bounded_xy"
                )
            )
        )
    )
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    samples = load_trajectory_csv(args.trajectory_csv)
    sample = pick_sample(
        samples,
        sample_id=args.sample_id,
        character=args.character,
        index=args.index,
    )
    xy_canvas, stroke_ids = flatten_canvas_trajectory(
        sample, args.image_size, args.padding
    )
    if args.initial_pose_csv:
        initial_pose, initial_xy_source, initial_gamma = load_initial_pose_csv(
            args.initial_pose_csv, sample
        )
        xy_canvas = source_xy_to_canvas(
            sample,
            initial_xy_source,
            args.image_size,
            args.padding,
        )
    else:
        initial_xy_source = np.asarray(
            [[point.x, point.y] for point in sample.all_points()],
            dtype=np.float32,
        )
        initial_pose = np.tile(
            np.asarray(
                [
                    args.initial_h_mm,
                    np.deg2rad(args.initial_alpha_deg),
                    np.deg2rad(args.initial_beta_deg),
                ],
                dtype=np.float32,
            ),
            (len(xy_canvas), 1),
        )
        initial_gamma = np.full(
            len(xy_canvas),
            np.deg2rad(args.initial_gamma_deg),
            dtype=np.float32,
        )
    target = load_target_image(args.target_image, image_size=args.image_size)
    dynamic = PaperDynamicConfig(
        width_inertia=args.width_inertia,
        drag_inertia=args.drag_inertia,
        calibration_profile=args.dynamic_profile,
        offset_transfer_scale=args.offset_transfer_scale,
        offset_fraction=args.offset_fraction,
        pixels_per_model_unit=args.pixels_per_model_unit,
        patch_floor=args.patch_floor,
        footprint_scale=args.footprint_scale,
        footprint_longitudinal_scale=args.footprint_longitudinal_scale,
        footprint_transverse_scale=args.footprint_transverse_scale,
        render_max_step_px=args.render_max_step_px,
    )
    renderer = PaperFusionRenderer.from_checkpoint(
        args.bbsmg_ckpt,
        device=device,
        image_size=args.image_size,
        dynamic=dynamic,
        point_batch_size=args.point_batch_size,
    )
    with torch.no_grad():
        initial_render = renderer(
            torch.as_tensor(xy_canvas, device=device),
            torch.as_tensor(initial_pose, device=device),
            torch.as_tensor(stroke_ids, device=device),
            torch.as_tensor(initial_gamma, device=device),
        )[0, 0].cpu().numpy()

    orders = (
        list(range(args.order_min, args.order_max + 1))
        if args.search_orders
        else [args.order]
    )
    if not orders or min(orders) < 1:
        raise ValueError("PSOC orders must be positive")
    if args.search_orders and (args.order_min < 3 or args.order_max > 8):
        raise ValueError("Paper order search must stay within orders 3 through 8")
    if args.order_min > args.order_max:
        raise ValueError("order_min must not exceed order_max")
    candidates = []
    candidate_results = []
    result = None
    selected_order = None
    selected_mse = float("inf")
    for order in orders:
        print(f"[ORDER SEARCH] optimizing CGL order={order}", flush=True)
        solver = PaperPSOCLM(
            renderer,
            order=order,
            optimization_size=args.optimization_size,
            smoothness_weights=(
                args.h_smoothness_weight,
                args.alpha_smoothness_weight,
                args.beta_smoothness_weight,
            ),
            posture_prior_weights=(
                args.h_prior_weight,
                args.alpha_prior_weight,
                args.beta_prior_weight,
            ),
            render_stride=args.render_stride,
            jacobian_mode=args.jacobian_mode,
            finite_difference_eps=args.finite_difference_eps,
            field_mode=args.field_mode,
            min_relative_median_sensitivity=(
                args.min_relative_median_sensitivity
            ),
            terminal_lift_weight=args.terminal_lift_weight,
            terminal_lift_nodes=args.terminal_lift_nodes,
            optimize_xy=args.optimize_xy,
            xy_max_offset_px=args.xy_max_offset_px,
            xy_smoothness_weight=args.xy_smoothness_weight,
            xy_prior_weight=args.xy_prior_weight,
            h_point_velocity_weight=args.h_point_velocity_weight,
            h_point_acceleration_weight=args.h_point_acceleration_weight,
            cap_order_to_points=args.cap_order_to_points,
            optimize_gamma=args.optimize_gamma,
            gamma_max_abs_rad=float(np.deg2rad(args.gamma_max_abs_deg)),
            gamma_smoothness_weight=args.gamma_smoothness_weight,
            gamma_prior_weight=args.gamma_prior_weight,
            observability_gate_mode=args.observability_gate_mode,
            observability_noise_rmse=args.observability_noise_rmse,
            min_observability_snr=args.min_observability_snr,
        )
        candidate = solver.optimize(
            xy_canvas,
            stroke_ids,
            target,
            initial_h_mm=args.initial_h_mm,
            initial_alpha_rad=float(np.deg2rad(args.initial_alpha_deg)),
            initial_beta_rad=float(np.deg2rad(args.initial_beta_deg)),
            initial_gamma_rad=float(np.deg2rad(args.initial_gamma_deg)),
            damping=args.damping,
            max_steps=args.max_steps,
            pixel_weight=args.pixel_weight,
            initial_posture=initial_pose,
            initial_gamma=initial_gamma,
        )
        candidate_metrics = binary_metrics(candidate.rendered_image, target)
        candidates.append(
            {
                "order": order,
                "success": candidate.success,
                "steps": candidate.steps,
                "initial_cost": candidate.initial_cost,
                "final_cost": candidate.final_cost,
                "plain_mse": candidate_metrics["plain_mse"],
                "dice_at_0.5": candidate_metrics["dice_at_0.5"],
                "iou_at_0.5": candidate_metrics["iou_at_0.5"],
                "xy_max_abs_change_px": candidate.diagnostics[
                    "xy_optimization"
                ]["max_abs_change_px"],
            }
        )
        candidate_results.append((order, candidate))
        if candidate_metrics["plain_mse"] < selected_mse:
            result = candidate
            selected_order = order
            selected_mse = candidate_metrics["plain_mse"]
    if result is None or selected_order is None:
        raise RuntimeError("PSOC order search produced no candidate result")
    print(
        f"[ORDER SEARCH] selected order={selected_order}, "
        f"plain_mse={selected_mse:.6f}, cost={result.final_cost:.6f}",
        flush=True,
    )
    optimized_xy_source = canvas_xy_to_source(
        sample,
        result.xy_canvas,
        args.image_size,
        args.padding,
    )
    original_xy_source = np.asarray(
        [[point.x, point.y] for point in sample.all_points()],
        dtype=np.float32,
    )
    xy_delta_canvas = result.xy_canvas - xy_canvas
    xy_delta_source = optimized_xy_source - original_xy_source

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem or f"{sample.character or 'sample'}_paper_inverse"
    if args.search_orders:
        candidate_dir = output_dir / "order_candidates"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        for order, candidate in candidate_results:
            candidate_stem = f"{stem}_order_{order}"
            save_gray(
                candidate.rendered_image,
                candidate_dir / f"{candidate_stem}_rendered.png",
            )
            save_pose_csv(
                sample,
                candidate.posture,
                candidate_dir / f"{candidate_stem}_trajectory.csv",
                renderer.regression_angle_basis,
                candidate.diagnostics["field_decisions"],
                xy_source=canvas_xy_to_source(
                    sample,
                    candidate.xy_canvas,
                    args.image_size,
                    args.padding,
                ),
                prototype=output_format,
                gamma=candidate.gamma,
            )
    save_pose_csv(
        sample,
        result.posture,
        output_dir / f"{stem}_trajectory.csv",
        renderer.regression_angle_basis,
        result.diagnostics["field_decisions"],
        xy_source=optimized_xy_source,
        prototype=output_format,
        gamma=result.gamma,
    )
    save_gray(target, output_dir / f"{stem}_target.png")
    save_gray(initial_render, output_dir / f"{stem}_initial.png")
    save_gray(result.rendered_image, output_dir / f"{stem}_rendered.png")
    save_gray(
        np.abs(result.rendered_image - target), output_dir / f"{stem}_diff.png"
    )
    comparison_panel(
        target,
        initial_render,
        result.rendered_image,
        output_dir / f"{stem}_comparison.png",
    )
    report = {
        "format": output_format,
        "simulation_only": True,
        "character": sample.character,
        "sample_id": sample.meta.get("sample_id"),
        "initialization": {
            "pose_source": (
                "initial_pose_csv"
                if args.initial_pose_csv
                else "command_line_defaults"
            ),
            "initial_pose_csv": args.initial_pose_csv,
            "initial_xy_source_frame": "input_trajectory_coordinates",
            "initial_posture_ranges": {
                "H_mm": [
                    float(initial_pose[:, 0].min()),
                    float(initial_pose[:, 0].max()),
                ],
                "alpha_rad": [
                    float(initial_pose[:, 1].min()),
                    float(initial_pose[:, 1].max()),
                ],
                "beta_rad": [
                    float(initial_pose[:, 2].min()),
                    float(initial_pose[:, 2].max()),
                ],
                "gamma_rad": [
                    float(initial_gamma.min()),
                    float(initial_gamma.max()),
                ],
            },
        },
        "fixed_xy": not args.optimize_xy,
        "xy_max_abs_change": float(np.abs(xy_delta_canvas).max()),
        "xy_optimization": {
            **result.diagnostics["xy_optimization"],
            "coordinate_frame": "normalized_image_canvas",
            "source_xy_max_abs_change": float(
                np.abs(xy_delta_source).max()
            ),
            "source_coordinate_note": (
                "CSV x/y are mapped back into the input trajectory frame; "
                "they are not calibrated robot coordinates."
            ),
        },
        "optimized_fields": result.diagnostics["observability_gate"][
            "optimized_fields"
        ],
        "fixed_fields": result.diagnostics["observability_gate"][
            "fixed_fields"
        ]
        + ([] if args.optimize_gamma else ["gamma"]),
        "field_decisions": result.diagnostics["field_decisions"],
        "gamma_rad": {
            "min": float(result.gamma.min()),
            "max": float(result.gamma.max()),
        },
        "pose_frame": "paper_model",
        "regression_angle_basis": renderer.regression_angle_basis,
        "dynamic_profile": args.dynamic_profile,
        "offset_transfer_scale": args.offset_transfer_scale,
        "paper_calibration": paper_calibration_metadata(args.dynamic_profile),
        "regression_unit_note": (
            "External and CSV angles are radians. The internal regression "
            f"basis is checkpoint-defined as {renderer.regression_angle_basis!r}."
        ),
        "forward_calibration": {
            "dynamic_profile": args.dynamic_profile,
            "offset_transfer_scale": args.offset_transfer_scale,
            "width_inertia_Kw": args.width_inertia,
            "drag_inertia_Kd": args.drag_inertia,
            "pixels_per_model_unit": args.pixels_per_model_unit,
            "footprint_scale": args.footprint_scale,
            "footprint_longitudinal_scale": (
                renderer.dynamic.longitudinal_scale
            ),
            "footprint_transverse_scale": renderer.dynamic.transverse_scale,
            "effective_pixels_per_model_unit": (
                args.pixels_per_model_unit
                * renderer.dynamic.longitudinal_scale
            ),
            "patch_floor": args.patch_floor,
            "render_max_step_px": args.render_max_step_px,
            "nonaxisymmetric_gamma_enabled": args.optimize_gamma,
        },
        "limits": {
            "H_mm": [11.0, 20.0],
            "alpha_rad": [0.0, float(np.deg2rad(10.0))],
            "beta_rad": [0.0, float(np.deg2rad(5.0))],
            "gamma_rad": (
                [
                    -float(np.deg2rad(args.gamma_max_abs_deg)),
                    float(np.deg2rad(args.gamma_max_abs_deg)),
                ]
                if args.optimize_gamma
                else [
                    float(initial_gamma.min()),
                    float(initial_gamma.max()),
                ]
            ),
        },
        "psoc_order": selected_order,
        "psoc_order_search": {
            "enabled": args.search_orders,
            "orders": orders,
            "selection_metric": "plain_full_resolution_mse",
            "selected_order": selected_order,
            "selected_value": selected_mse,
            "scope": (
                "character_global_approximation"
                if args.search_orders
                else "fixed_order"
            ),
            "candidates": candidates,
            "paper_note": (
                "Wang et al. search orders 3-8 per decomposed stroke. "
                "This tool uses a complete-character target, so it compares "
                "one shared order at a time. Cross-order selection uses only "
                "full-resolution image MSE because regularization residual "
                "counts change with CGL node count."
            ),
        },
        "optimized_range": {
            "H_mm": [
                float(result.posture[:, 0].min()),
                float(result.posture[:, 0].max()),
            ],
            "alpha_rad": [
                float(result.posture[:, 1].min()),
                float(result.posture[:, 1].max()),
            ],
            "beta_rad": [
                float(result.posture[:, 2].min()),
                float(result.posture[:, 2].max()),
            ],
            "gamma_rad": [
                float(result.gamma.min()),
                float(result.gamma.max()),
            ],
        },
        "identifiability": {
            "node_gate_scope": (
                "local single-column response above image-noise RMSE"
            ),
            "joint_pose_unique_from_single_image": False,
            "optimization_size": args.optimization_size,
            "recommended_minimum_optimization_size_for_pose_recovery": 64,
            "external_observation_required_for_robot_ground_truth": True,
            "note": (
                "Image similarity and per-node SNR do not prove unique "
                "H/alpha/beta/gamma recovery because fields can compensate "
                "for one another."
            ),
        },
        "lm": {
            "success": result.success,
            "steps": result.steps,
            "message": result.message,
            "initial_cost": result.initial_cost,
            "final_cost": result.final_cost,
            "history": result.history,
            "jacobian_mode": args.jacobian_mode,
            "finite_difference_eps": args.finite_difference_eps,
            "diagnostics": result.diagnostics,
        },
        "metrics": binary_metrics(result.rendered_image, target),
        "trajectory_target_coverage_at_5px": trajectory_target_coverage(
            result.xy_canvas, target, tolerance_px=5
        ),
        "initial_trajectory_target_coverage_at_5px": (
            trajectory_target_coverage(xy_canvas, target, tolerance_px=5)
        ),
        "optimized_trajectory_target_coverage_at_5px": (
            trajectory_target_coverage(
                result.xy_canvas, target, tolerance_px=5
            )
        ),
        "warning": (
            "Prototype paper-frame pose only; do not command a real robot before "
            "brush/camera/TCP/frame calibration and safety validation."
        ),
    }
    (output_dir / f"{stem}_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    optimized = report["optimized_fields"]
    fixed = report["fixed_fields"]
    xy_mode = "bounded x/y correction" if args.optimize_xy else "fixed x/y"
    print(f"[DONE] {xy_mode}; optimized={optimized}; fixed={fixed} on {device}")
    print(
        f"[LM] success={result.success}, steps={result.steps}, "
        f"cost={result.initial_cost:.6f}->{result.final_cost:.6f}"
    )
    for key, value in report["metrics"].items():
        print(f"{key}: {value:.6f}")
    print(
        "trajectory_target_coverage_at_5px: "
        f"{report['trajectory_target_coverage_at_5px']:.6f}"
    )
    if args.optimize_xy:
        xy_summary = report["xy_optimization"]
        print(
            "[XY] max_abs_change_px="
            f"{xy_summary['max_abs_change_px']:.6f}, "
            "mean_displacement_px="
            f"{xy_summary['mean_point_displacement_px']:.6f}, "
            "bound_fraction="
            f"{xy_summary['component_bound_fraction_within_1pct']:.6f}"
        )
    sensitivity = result.diagnostics["observability_gate"].get(
        "initial_image_jacobian_sensitivity", {}
    ) or result.diagnostics.get("image_jacobian_sensitivity", {})
    for field_name in ("H", "alpha", "beta", "gamma"):
        if field_name in sensitivity:
            print(
                f"[SENSITIVITY] {field_name}: "
                f"relative_mean={sensitivity[field_name]['relative_mean']:.6f}, "
                "relative_median="
                f"{sensitivity[field_name]['relative_median']:.6f}"
            )
    node_gate = result.diagnostics["observability_gate"].get(
        "selected_node_columns", {}
    )
    for field_name, selection in node_gate.items():
        print(
            f"[NODE GATE] {field_name}: "
            f"selected={selection['selected_nodes']}/"
            f"{selection['evaluated_nodes']}, "
            f"median_snr={selection['median_snr']:.6f}, "
            f"max_snr={selection['max_snr']:.6f}"
        )
    for field_name, fractions in result.diagnostics[
        "bound_fraction_within_1pct"
    ].items():
        print(
            f"[BOUNDS] {field_name}: lower={fractions['lower']:.6f}, "
            f"upper={fractions['upper']:.6f}"
        )
    continuity = result.diagnostics["trajectory_continuity"]
    print(
        "[H CONTINUITY] max_step_mm="
        f"{continuity['first_difference']['max_abs_mm']:.6f}, "
        "max_second_difference_mm="
        f"{continuity['second_difference']['max_abs_mm']:.6f}"
    )
    layout = result.diagnostics["cgl_layout"]
    print(
        "[CGL LAYOUT] requested_order="
        f"{layout['requested_order']}, effective_orders="
        f"{layout['effective_orders_per_stroke']}, "
        f"active_nodes={layout['active_node_count']}/"
        f"{layout['allocated_node_count']}"
    )
    print(f"[DONE] outputs: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory_csv", required=True)
    parser.add_argument(
        "--initial_pose_csv",
        default=None,
        help=(
            "staged x/y/z/alpha/beta/gamma CSV keyed by stroke_id and point_id; "
            "uses mm/rad and must match the selected base trajectory exactly"
        ),
    )
    parser.add_argument("--target_image", required=True)
    parser.add_argument("--bbsmg_ckpt", required=True)
    parser.add_argument("--character", default=None)
    parser.add_argument("--sample_id", default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output_dir", default="outputs/paper_inverse")
    parser.add_argument("--output_stem", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--padding", type=int, default=16)
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument(
        "--search_orders",
        action="store_true",
        help=(
            "compare the paper-reported CGL orders from order_min through "
            "order_max and retain the lowest full-resolution plain MSE"
        ),
    )
    parser.add_argument("--order_min", type=int, default=3)
    parser.add_argument("--order_max", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=15)
    parser.add_argument("--damping", type=float, default=0.05)
    parser.add_argument("--optimization_size", type=int, default=16)
    parser.add_argument("--render_stride", type=int, default=1)
    parser.add_argument("--point_batch_size", type=int, default=128)
    parser.add_argument("--pixel_weight", type=float, default=3.0)
    parser.add_argument(
        "--optimize_xy",
        action="store_true",
        help=(
            "jointly optimize bounded per-stroke x/y CGL offsets; disabled "
            "by default for backward compatibility"
        ),
    )
    parser.add_argument(
        "--xy_max_offset_px",
        type=float,
        default=6.0,
        help="maximum absolute x or y correction in normalized canvas pixels",
    )
    parser.add_argument(
        "--xy_smoothness_weight",
        type=float,
        default=0.10,
        help="smoothness penalty for normalized x/y CGL offsets",
    )
    parser.add_argument(
        "--xy_prior_weight",
        type=float,
        default=0.05,
        help="zero-offset prior for normalized x/y CGL offsets",
    )
    parser.add_argument(
        "--cap_order_to_points",
        action="store_true",
        help=(
            "cap each stroke's effective CGL order at point_count-1 while "
            "retaining the requested allocation for backward compatibility"
        ),
    )
    parser.add_argument(
        "--h_point_velocity_weight",
        type=float,
        default=0.0,
        help=(
            "first-difference penalty on decoded normalized H at successive "
            "trajectory samples; zero preserves v9 behavior"
        ),
    )
    parser.add_argument(
        "--h_point_acceleration_weight",
        type=float,
        default=0.0,
        help=(
            "second-difference penalty on decoded normalized H at successive "
            "trajectory samples; zero preserves v9 behavior"
        ),
    )
    parser.add_argument("--h_smoothness_weight", type=float, default=0.02)
    parser.add_argument("--alpha_smoothness_weight", type=float, default=0.10)
    parser.add_argument("--beta_smoothness_weight", type=float, default=0.10)
    parser.add_argument("--gamma_smoothness_weight", type=float, default=0.10)
    parser.add_argument("--h_prior_weight", type=float, default=0.001)
    parser.add_argument("--alpha_prior_weight", type=float, default=0.05)
    parser.add_argument("--beta_prior_weight", type=float, default=0.05)
    parser.add_argument("--gamma_prior_weight", type=float, default=0.05)
    parser.add_argument(
        "--terminal_lift_weight",
        type=float,
        default=0.0,
        help=(
            "optional Wang Eq. (19) terminal H regularizer; default is zero "
            "because the paper does not report beta_k"
        ),
    )
    parser.add_argument("--terminal_lift_nodes", type=int, default=1)
    parser.add_argument(
        "--jacobian_mode",
        choices=["finite_difference", "autograd"],
        default="finite_difference",
    )
    parser.add_argument("--finite_difference_eps", type=float, default=0.01)
    parser.add_argument(
        "--field_mode",
        choices=["auto", "all", "h_only", "xy_only"],
        default="auto",
        help=(
            "auto audits enabled posture fields once and optimizes observable "
            "ones; all reproduces unconstrained A/B runs; h_only skips audit; "
            "xy_only fixes H/alpha/beta for a planar-geometry ablation"
        ),
    )
    parser.add_argument(
        "--min_relative_median_sensitivity",
        type=float,
        default=0.45,
    )
    parser.add_argument(
        "--observability_gate_mode",
        choices=["field_relative", "node_snr"],
        default="field_relative",
        help=(
            "legacy whole-field relative gate or v14 per-CGL-node "
            "signal-to-validation-noise gate"
        ),
    )
    parser.add_argument(
        "--observability_noise_rmse",
        type=float,
        default=None,
        help=(
            "pixel RMSE noise floor for node_snr; defaults to sqrt(checkpoint "
            "validation plain_mse)"
        ),
    )
    parser.add_argument(
        "--min_observability_snr",
        type=float,
        default=1.0,
    )
    parser.add_argument("--initial_h_mm", type=float, default=15.5)
    parser.add_argument("--initial_alpha_deg", type=float, default=0.0)
    parser.add_argument("--initial_beta_deg", type=float, default=0.0)
    parser.add_argument("--initial_gamma_deg", type=float, default=0.0)
    parser.add_argument(
        "--optimize_gamma",
        action="store_true",
        help=(
            "enable bounded axial-angle CGL variables; requires unequal "
            "longitudinal/transverse footprint scales and remains subject "
            "to the observability gate in field_mode=auto"
        ),
    )
    parser.add_argument(
        "--gamma_max_abs_deg",
        type=float,
        default=180.0,
        help="symmetric axial-angle bound used only with --optimize_gamma",
    )
    parser.add_argument("--width_inertia", type=float, default=0.02)
    parser.add_argument("--drag_inertia", type=float, default=0.02)
    parser.add_argument(
        "--dynamic_profile",
        choices=DYNAMIC_PROFILES,
        default=WANG2020_PROFILE,
    )
    parser.add_argument(
        "--offset_transfer_scale",
        type=float,
        default=1.0,
        help=(
            "cross-paper scale applied to the digitized Offset/Drag ratio; "
            "use the value selected by forward A/B scanning"
        ),
    )
    parser.add_argument(
        "--offset_fraction",
        type=float,
        default=0.25,
        help="used only when dynamic_profile=legacy_fraction_v1",
    )
    parser.add_argument("--pixels_per_model_unit", type=float, default=20.0)
    parser.add_argument("--patch_floor", type=float, default=0.05)
    parser.add_argument("--footprint_scale", type=float, default=0.22)
    parser.add_argument(
        "--footprint_longitudinal_scale",
        type=float,
        default=None,
        help="local along-stroke scale; defaults to footprint_scale",
    )
    parser.add_argument(
        "--footprint_transverse_scale",
        type=float,
        default=None,
        help="local cross-stroke width scale; defaults to footprint_scale",
    )
    parser.add_argument("--render_max_step_px", type=float, default=2.0)
    main(parser.parse_args())
