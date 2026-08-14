"""Run the paper Dynamic-Brush + B-BSMG forward rendering chain."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.trajectory_dataset import load_trajectory_csv
from models.paper_bbsm import PAPER_POSTURE_MAX, PAPER_POSTURE_MIN
from models.paper_calibration import (
    DYNAMIC_PROFILES,
    WANG2020_PROFILE,
    paper_calibration_metadata,
)
from models.paper_fusion_renderer import PaperDynamicConfig, PaperFusionRenderer
from optim.trajectory_optimizer import load_target_image
from tools.invert_paper_trajectory import (
    binary_metrics,
    flatten_canvas_trajectory,
    pick_sample,
    source_xy_to_canvas,
)
from utils.trajectory_processing import (
    TrajectorySafetyLimits,
    repair_sample_states,
    smooth_sample,
    validate_trajectory,
)


def load_pose_csv(
    path: str, sample, clip_pose_limits: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_key = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            key = (int(row["stroke_id"]), int(row["point_id"]))
            gamma = float(row.get("gamma", 0.0) or 0.0)
            if not np.isfinite(gamma) or abs(gamma) > np.pi + 1e-6:
                raise ValueError("Pose CSV gamma must be finite and in [-pi,pi]")
            by_key[key] = {
                "posture": [
                    float(row["z"]),
                    float(row["alpha"]),
                    float(row["beta"]),
                ],
                "xy": [float(row["x"]), float(row["y"])],
                "gamma": gamma,
            }
    values, xy_values, gamma_values = [], [], []
    for point in sample.all_points():
        key = (point.stroke_id, point.point_id)
        if key not in by_key:
            raise ValueError(f"Pose CSV is missing stroke/point {key}")
        values.append(by_key[key]["posture"])
        xy_values.append(by_key[key]["xy"])
        gamma_values.append(by_key[key]["gamma"])
    posture = np.asarray(values, dtype=np.float32)
    tolerance = 1e-6
    invalid = np.any(posture < PAPER_POSTURE_MIN - tolerance) or np.any(
        posture > PAPER_POSTURE_MAX + tolerance
    )
    if invalid and not clip_pose_limits:
        ranges = [
            (float(posture[:, index].min()), float(posture[:, index].max()))
            for index in range(3)
        ]
        raise ValueError(
            "Pose CSV exceeds H=11-20 mm, alpha=0-10 deg, beta=0-5 deg; "
            f"actual ranges={ranges}. Re-run inversion with paper_psoc_lm_v2 "
            "or pass --clip_pose_limits to inspect this legacy result."
        )
    if invalid:
        print(
            "[WARN] Clipping legacy pose CSV to prototype limits. "
            "Use this only for visual inspection; re-run inversion for a "
            "physically valid result."
        )
    return (
        np.clip(posture, PAPER_POSTURE_MIN, PAPER_POSTURE_MAX),
        np.asarray(xy_values, dtype=np.float32),
        np.asarray(gamma_values, dtype=np.float32),
    )


def save_dynamic_states(sample, xy, posture, gamma, states, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    virtual = states["virtual_posture"].cpu().numpy()
    geometry = states["geometry"].cpu().numpy()
    heading = states["heading"].cpu().numpy()
    trajectory_heading = states["trajectory_heading"].cpu().numpy()
    forward_heading = states["forward_trajectory_heading"].cpu().numpy()
    offset_ratio = states["offset_ratio"].cpu().numpy()
    free_offset = states["free_offset_model_unit"].cpu().numpy()
    held_offset = states["held_offset_model_unit"].cpu().numpy()
    offset = states["offset_model_unit"].cpu().numpy()
    contact = states["contact_xy"].cpu().numpy()
    fields = [
        "stroke_id",
        "point_id",
        "x_canvas_px",
        "y_canvas_px",
        "H_input_mm",
        "alpha_input_rad",
        "beta_input_rad",
        "gamma_input_rad",
        "H_virtual_mm",
        "alpha_virtual_rad",
        "beta_virtual_rad",
        "Lt",
        "Lh",
        "Lr",
        "offset_drag_ratio",
        "free_offset_model_unit",
        "held_offset_model_unit",
        "offset_model_unit",
        "trajectory_theta_rad",
        "gamma_forward_xy_rad",
        "brush_theta_rad",
        "contact_x_px",
        "contact_y_px",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for index, point in enumerate(sample.all_points()):
            writer.writerow(
                {
                    "stroke_id": point.stroke_id,
                    "point_id": point.point_id,
                    "x_canvas_px": repr(float(xy[index, 0])),
                    "y_canvas_px": repr(float(xy[index, 1])),
                    "H_input_mm": repr(float(posture[index, 0])),
                    "alpha_input_rad": repr(float(posture[index, 1])),
                    "beta_input_rad": repr(float(posture[index, 2])),
                    "gamma_input_rad": repr(float(gamma[index])),
                    "H_virtual_mm": repr(float(virtual[index, 0])),
                    "alpha_virtual_rad": repr(float(virtual[index, 1])),
                    "beta_virtual_rad": repr(float(virtual[index, 2])),
                    "Lt": repr(float(geometry[index, 0])),
                    "Lh": repr(float(geometry[index, 1])),
                    "Lr": repr(float(geometry[index, 2])),
                    "offset_drag_ratio": repr(float(offset_ratio[index])),
                    "free_offset_model_unit": repr(float(free_offset[index])),
                    "held_offset_model_unit": repr(float(held_offset[index])),
                    "offset_model_unit": repr(float(offset[index])),
                    "trajectory_theta_rad": repr(
                        float(trajectory_heading[index])
                    ),
                    "gamma_forward_xy_rad": repr(
                        float(forward_heading[index])
                    ),
                    "brush_theta_rad": repr(float(heading[index])),
                    "contact_x_px": repr(float(contact[index, 0])),
                    "contact_y_px": repr(float(contact[index, 1])),
                }
            )


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device '{args.device}' was requested, but PyTorch cannot "
            "initialize CUDA. Check nvidia-smi and the NVIDIA driver instead "
            "of silently rendering on CPU."
        )
    sample = pick_sample(
        load_trajectory_csv(args.trajectory_csv),
        sample_id=args.sample_id,
        character=args.character,
        index=args.index,
    )
    # Repair states without changing point IDs so an accompanying pose CSV
    # remains key-compatible. Smoothing is per-stroke and never bridges a
    # pen-up boundary.
    sample = repair_sample_states(sample)
    if args.smooth_passes > 0:
        sample = smooth_sample(
            sample,
            passes=args.smooth_passes,
            strength=args.smooth_strength,
        )
    safety_report = validate_trajectory(
        sample,
        TrajectorySafetyLimits(
            max_step_xy=args.safety_max_step_xy,
            max_step_z=args.safety_max_step_z,
            max_angle_step_rad=args.safety_max_angle_step_rad,
        ),
    )
    if not safety_report["safe"] and args.fail_on_unsafe:
        raise ValueError(
            "Trajectory safety checks failed: "
            + ", ".join(safety_report["errors"])
        )
    xy, stroke_ids = flatten_canvas_trajectory(
        sample, args.image_size, args.padding
    )
    original_xy = xy.copy()
    if args.pose_csv:
        posture, pose_xy_source, gamma = load_pose_csv(
            args.pose_csv,
            sample,
            clip_pose_limits=args.clip_pose_limits,
        )
        xy = source_xy_to_canvas(
            sample,
            pose_xy_source,
            args.image_size,
            args.padding,
        )
        pose_source = args.pose_csv
    else:
        posture = np.tile(
            np.asarray(
                [
                    args.h_mm,
                    np.deg2rad(args.alpha_deg),
                    np.deg2rad(args.beta_deg),
                ],
                dtype=np.float32,
            ),
            (len(xy), 1),
        )
        if np.any(posture < PAPER_POSTURE_MIN) or np.any(
            posture > PAPER_POSTURE_MAX
        ):
            raise ValueError(
                "Default pose exceeds H=11-20 mm, alpha=0-10 deg, beta=0-5 deg"
            )
        gamma = np.full(
            len(xy), np.deg2rad(args.gamma_deg), dtype=np.float32
        )
        pose_source = "command_line_default"
    renderer = PaperFusionRenderer.from_checkpoint(
        args.bbsmg_ckpt,
        device=device,
        image_size=args.image_size,
        dynamic=PaperDynamicConfig(
            width_inertia=args.width_inertia,
            drag_inertia=args.drag_inertia,
            calibration_profile=args.dynamic_profile,
            offset_transfer_scale=args.offset_transfer_scale,
            offset_fraction=args.offset_fraction,
            pixels_per_model_unit=args.pixels_per_model_unit,
            patch_floor=args.patch_floor,
            footprint_scale=args.footprint_scale,
            footprint_longitudinal_scale=(
                args.footprint_longitudinal_scale
            ),
            footprint_transverse_scale=args.footprint_transverse_scale,
            render_max_step_px=args.render_max_step_px,
            fused_pose_from_height=args.fused_pose_from_height,
            inverse_regularization=args.pose_inverse_regularization,
        ),
        point_batch_size=args.point_batch_size,
    )
    with torch.no_grad():
        xy_tensor = torch.as_tensor(xy, device=device)
        posture_tensor = torch.as_tensor(posture, device=device)
        gamma_tensor = torch.as_tensor(gamma, device=device)
        stroke_tensor = torch.as_tensor(stroke_ids, device=device)
        states = renderer.compute_dynamic_states(
            xy_tensor, posture_tensor, stroke_tensor
        )
        dense_xy, _, _ = renderer.densify_for_rendering(
            xy_tensor, posture_tensor, stroke_tensor
        )
        rendered = renderer(
            xy_tensor,
            posture_tensor,
            stroke_tensor,
            gamma_tensor,
        )[0, 0].cpu().numpy()
        contact_shift = torch.linalg.vector_norm(
            states["contact_xy"] - xy_tensor, dim=-1
        )
        derived_angle_error = torch.abs(
            states["virtual_posture"][:, 1:] - posture_tensor[:, 1:]
        )
        derived_gamma = states["forward_trajectory_heading"]
        wrapped_gamma_error = torch.abs(
            torch.atan2(
                torch.sin(derived_gamma - gamma_tensor),
                torch.cos(derived_gamma - gamma_tensor),
            )
        )
    output = Path(args.output_image)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(
        np.rint(np.clip(rendered, 0.0, 1.0) * 255.0).astype(np.uint8),
        mode="L",
    ).save(output)
    save_dynamic_states(
        sample, xy, posture, gamma, states, output.with_suffix(".states.csv")
    )
    report = {
        "format": "paper_forward_renderer_v3_wang2020",
        "simulation_only": True,
        "character": sample.character,
        "sample_id": sample.meta.get("sample_id"),
        "point_count": int(len(xy)),
        "fixed_xy": bool(np.allclose(xy, original_xy, atol=1e-6)),
        "xy_max_abs_change_canvas_px": float(
            np.abs(xy - original_xy).max()
        ),
        "pose_source": pose_source,
        "angle_unit": "rad",
        "z_semantics": "H_mm",
        "gamma_rad": {
            "min": float(gamma.min()),
            "max": float(gamma.max()),
        },
        "regression_angle_basis": renderer.regression_angle_basis,
        "dynamic_profile": args.dynamic_profile,
        "offset_transfer_scale": args.offset_transfer_scale,
        "paper_calibration": paper_calibration_metadata(args.dynamic_profile),
        "footprint_scale": args.footprint_scale,
        "footprint_longitudinal_scale": (
            renderer.dynamic.longitudinal_scale
        ),
        "footprint_transverse_scale": renderer.dynamic.transverse_scale,
        "input_point_count": int(len(xy)),
        "render_sample_count": int(len(dense_xy)),
        "render_max_step_px": args.render_max_step_px,
        "dynamic_state_metrics": {
            "contact_shift_px_mean": float(contact_shift.mean().cpu()),
            "contact_shift_px_max": float(contact_shift.max().cpu()),
            "offset_model_unit_mean": float(
                states["offset_model_unit"].mean().cpu()
            ),
            "offset_model_unit_max": float(
                states["offset_model_unit"].max().cpu()
            ),
        },
        "trajectory_safety": safety_report,
        "trajectory_processing": {
            "state_repair": True,
            "smooth_passes": args.smooth_passes,
            "smooth_strength": args.smooth_strength,
        },
    }
    if args.fused_pose_from_height:
        report["fused_pose_validation"] = {
            "alpha_beta_recomputed_from_z": True,
            "gamma_recomputed_from_xy": True,
            "gamma_double_application_prevented": True,
            "max_abs_alpha_beta_csv_error_rad": float(
                derived_angle_error.max().cpu()
            ),
            "mean_abs_alpha_beta_csv_error_rad": float(
                derived_angle_error.mean().cpu()
            ),
            "max_abs_gamma_csv_error_rad": float(
                wrapped_gamma_error.max().cpu()
            ),
            "mean_abs_gamma_csv_error_rad": float(
                wrapped_gamma_error.mean().cpu()
            ),
        }
    if args.target_image:
        target = load_target_image(args.target_image, image_size=args.image_size)
        report["target_image"] = args.target_image
        report["target_metrics"] = {
            **binary_metrics(rendered, target),
            "target_ink": float((target >= 0.5).mean()),
            "prediction_ink": float((rendered >= 0.5).mean()),
        }
    output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[DONE] Forward-rendered {sample.character or 'sample'} on {device}: "
        f"{output}"
    )
    print(
        "[DYNAMIC] contact_shift_px="
        f"{report['dynamic_state_metrics']['contact_shift_px_mean']:.6f} "
        "(mean), "
        f"{report['dynamic_state_metrics']['contact_shift_px_max']:.6f} (max)"
    )
    for key, value in report.get("target_metrics", {}).items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory_csv", required=True)
    parser.add_argument("--bbsmg_ckpt", required=True)
    parser.add_argument("--target_image", default=None)
    parser.add_argument("--pose_csv", default=None)
    parser.add_argument("--clip_pose_limits", action="store_true")
    parser.add_argument("--character", default=None)
    parser.add_argument("--sample_id", default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--output_image", default="outputs/paper_forward/rendered.png"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--padding", type=int, default=16)
    parser.add_argument("--h_mm", type=float, default=15.5)
    parser.add_argument("--alpha_deg", type=float, default=0.0)
    parser.add_argument("--beta_deg", type=float, default=0.0)
    parser.add_argument("--gamma_deg", type=float, default=0.0)
    parser.add_argument(
        "--fused_pose_from_height",
        action="store_true",
        help=(
            "recompute B-BSMG alpha/beta from Wang geometry at CSV z and "
            "treat CSV gamma as the per-stroke x/y direction"
        ),
    )
    parser.add_argument("--width_inertia", type=float, default=0.02)
    parser.add_argument("--drag_inertia", type=float, default=0.02)
    parser.add_argument(
        "--pose_inverse_regularization", type=float, default=1e-4
    )
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
            "scan this simulation bridge before LM"
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
    parser.add_argument("--point_batch_size", type=int, default=128)
    parser.add_argument(
        "--smooth_passes",
        type=int,
        default=0,
        help="per-stroke Laplacian smoothing passes; 0 preserves XY exactly",
    )
    parser.add_argument("--smooth_strength", type=float, default=0.25)
    parser.add_argument("--safety_max_step_xy", type=float, default=None)
    parser.add_argument("--safety_max_step_z", type=float, default=None)
    parser.add_argument(
        "--safety_max_angle_step_rad", type=float, default=np.pi
    )
    parser.add_argument("--fail_on_unsafe", action="store_true")
    main(parser.parse_args())
