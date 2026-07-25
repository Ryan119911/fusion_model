"""Invert a target image into fixed-x/y paper posture using PSOC + LM."""
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
from models.geometry import normalize_trajectory_xy
from models.paper_calibration import (
    DYNAMIC_PROFILES,
    WANG2020_PROFILE,
    paper_calibration_metadata,
)
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


def save_pose_csv(
    sample,
    posture: np.ndarray,
    output_path: Path,
    regression_angle_basis: str,
    field_decisions: dict,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "character",
        "sample_id",
        "stroke_id",
        "point_id",
        "x",
        "y",
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
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for point, pose in zip(points, posture):
            writer.writerow(
                {
                    "character": sample.character,
                    "sample_id": sample.meta.get("sample_id"),
                    "stroke_id": point.stroke_id,
                    "point_id": point.point_id,
                    "x": repr(float(point.x)),
                    "y": repr(float(point.y)),
                    # Prototype contract: CSV z is paper-model H in millimetres.
                    "z": repr(float(pose[0])),
                    "alpha": repr(float(pose[1])),
                    "beta": repr(float(pose[2])),
                    "gamma": "0",
                    "state": int(point.state),
                    "z_unit": "mm",
                    "angle_unit": "rad",
                    "pose_frame": "paper_model",
                    "prototype": "paper_psoc_lm_v8_fullres_checkpoint",
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
    return {
        "plain_mse": float(np.mean((prediction - target) ** 2)),
        "mae": float(np.mean(np.abs(prediction - target))),
        "dice_at_0.5": float(
            (2 * intersection + 1e-6) / (pred.sum() + truth.sum() + 1e-6)
        ),
        "iou_at_0.5": float((intersection + 1e-6) / (union + 1e-6)),
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
        render_max_step_px=args.render_max_step_px,
    )
    renderer = PaperFusionRenderer.from_checkpoint(
        args.bbsmg_ckpt,
        device=device,
        image_size=args.image_size,
        dynamic=dynamic,
        point_batch_size=args.point_batch_size,
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
    with torch.no_grad():
        initial_render = renderer(
            torch.as_tensor(xy_canvas, device=device),
            torch.as_tensor(initial_pose, device=device),
            torch.as_tensor(stroke_ids, device=device),
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
        )
        candidate = solver.optimize(
            xy_canvas,
            stroke_ids,
            target,
            initial_h_mm=args.initial_h_mm,
            initial_alpha_rad=float(np.deg2rad(args.initial_alpha_deg)),
            initial_beta_rad=float(np.deg2rad(args.initial_beta_deg)),
            damping=args.damping,
            max_steps=args.max_steps,
            pixel_weight=args.pixel_weight,
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
            )
    save_pose_csv(
        sample,
        result.posture,
        output_dir / f"{stem}_trajectory.csv",
        renderer.regression_angle_basis,
        result.diagnostics["field_decisions"],
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
        "format": "paper_psoc_lm_v8_fullres_checkpoint",
        "simulation_only": True,
        "character": sample.character,
        "sample_id": sample.meta.get("sample_id"),
        "fixed_xy": True,
        "xy_max_abs_change": 0.0,
        "optimized_fields": result.diagnostics["observability_gate"][
            "optimized_fields"
        ],
        "fixed_fields": result.diagnostics["observability_gate"][
            "fixed_fields"
        ]
        + ["gamma"],
        "field_decisions": result.diagnostics["field_decisions"],
        "gamma_rad": 0.0,
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
            "effective_pixels_per_model_unit": (
                args.pixels_per_model_unit * args.footprint_scale
            ),
            "patch_floor": args.patch_floor,
            "render_max_step_px": args.render_max_step_px,
        },
        "limits": {
            "H_mm": [11.0, 20.0],
            "alpha_rad": [0.0, float(np.deg2rad(10.0))],
            "beta_rad": [0.0, float(np.deg2rad(5.0))],
            "gamma_rad": [0.0, 0.0],
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
            "gamma_rad": [0.0, 0.0],
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
            xy_canvas, target, tolerance_px=5
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
    print(
        f"[DONE] fixed x/y; optimized={optimized}; fixed={fixed} on {device}"
    )
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
    sensitivity = result.diagnostics["observability_gate"].get(
        "initial_image_jacobian_sensitivity", {}
    ) or result.diagnostics.get("image_jacobian_sensitivity", {})
    for field_name in ("H", "alpha", "beta"):
        if field_name in sensitivity:
            print(
                f"[SENSITIVITY] {field_name}: "
                f"relative_mean={sensitivity[field_name]['relative_mean']:.6f}, "
                "relative_median="
                f"{sensitivity[field_name]['relative_median']:.6f}"
            )
    for field_name, fractions in result.diagnostics[
        "bound_fraction_within_1pct"
    ].items():
        print(
            f"[BOUNDS] {field_name}: lower={fractions['lower']:.6f}, "
            f"upper={fractions['upper']:.6f}"
        )
    print(f"[DONE] outputs: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory_csv", required=True)
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
            "order_max and retain the lowest regularized LM cost"
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
    parser.add_argument("--h_smoothness_weight", type=float, default=0.02)
    parser.add_argument("--alpha_smoothness_weight", type=float, default=0.10)
    parser.add_argument("--beta_smoothness_weight", type=float, default=0.10)
    parser.add_argument("--h_prior_weight", type=float, default=0.001)
    parser.add_argument("--alpha_prior_weight", type=float, default=0.05)
    parser.add_argument("--beta_prior_weight", type=float, default=0.05)
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
        choices=["auto", "all", "h_only"],
        default="auto",
        help=(
            "auto audits all posture fields once and optimizes only observable "
            "ones; all reproduces unconstrained A/B runs; h_only skips audit"
        ),
    )
    parser.add_argument(
        "--min_relative_median_sensitivity",
        type=float,
        default=0.45,
    )
    parser.add_argument("--initial_h_mm", type=float, default=15.5)
    parser.add_argument("--initial_alpha_deg", type=float, default=0.0)
    parser.add_argument("--initial_beta_deg", type=float, default=0.0)
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
    parser.add_argument("--render_max_step_px", type=float, default=2.0)
    main(parser.parse_args())
