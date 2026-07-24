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
from models.paper_fusion_renderer import PaperDynamicConfig, PaperFusionRenderer
from tools.invert_paper_trajectory import flatten_canvas_trajectory, pick_sample


def load_pose_csv(path: str, sample) -> np.ndarray:
    by_key = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            key = (int(row["stroke_id"]), int(row["point_id"]))
            gamma = float(row.get("gamma", 0.0) or 0.0)
            if abs(gamma) > 1e-9:
                raise ValueError("Prototype requires gamma=0 rad for every point")
            by_key[key] = [
                float(row["z"]),
                float(row["alpha"]),
                float(row["beta"]),
            ]
    values = []
    for point in sample.all_points():
        key = (point.stroke_id, point.point_id)
        if key not in by_key:
            raise ValueError(f"Pose CSV is missing stroke/point {key}")
        values.append(by_key[key])
    posture = np.asarray(values, dtype=np.float32)
    if np.any(posture < PAPER_POSTURE_MIN) or np.any(
        posture > PAPER_POSTURE_MAX
    ):
        raise ValueError(
            "Pose CSV exceeds H=11-20 mm, alpha=0-10 deg, beta=0-5 deg"
        )
    return posture


def save_dynamic_states(sample, xy, posture, states, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    virtual = states["virtual_posture"].cpu().numpy()
    geometry = states["geometry"].cpu().numpy()
    heading = states["heading"].cpu().numpy()
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
        "H_virtual_mm",
        "alpha_virtual_rad",
        "beta_virtual_rad",
        "Lt",
        "Lh",
        "Lr",
        "offset_model_unit",
        "theta_xy_rad",
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
                    "H_virtual_mm": repr(float(virtual[index, 0])),
                    "alpha_virtual_rad": repr(float(virtual[index, 1])),
                    "beta_virtual_rad": repr(float(virtual[index, 2])),
                    "Lt": repr(float(geometry[index, 0])),
                    "Lh": repr(float(geometry[index, 1])),
                    "Lr": repr(float(geometry[index, 2])),
                    "offset_model_unit": repr(float(offset[index])),
                    "theta_xy_rad": repr(float(heading[index])),
                    "contact_x_px": repr(float(contact[index, 0])),
                    "contact_y_px": repr(float(contact[index, 1])),
                }
            )


def main(args: argparse.Namespace) -> None:
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    sample = pick_sample(
        load_trajectory_csv(args.trajectory_csv),
        sample_id=args.sample_id,
        character=args.character,
        index=args.index,
    )
    xy, stroke_ids = flatten_canvas_trajectory(
        sample, args.image_size, args.padding
    )
    if args.pose_csv:
        posture = load_pose_csv(args.pose_csv, sample)
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
        pose_source = "command_line_default"
    renderer = PaperFusionRenderer.from_checkpoint(
        args.bbsmg_ckpt,
        device=device,
        image_size=args.image_size,
        dynamic=PaperDynamicConfig(
            width_inertia=args.width_inertia,
            drag_inertia=args.drag_inertia,
            offset_fraction=args.offset_fraction,
            pixels_per_model_unit=args.pixels_per_model_unit,
            patch_floor=args.patch_floor,
        ),
        point_batch_size=args.point_batch_size,
    )
    with torch.no_grad():
        xy_tensor = torch.as_tensor(xy, device=device)
        posture_tensor = torch.as_tensor(posture, device=device)
        stroke_tensor = torch.as_tensor(stroke_ids, device=device)
        states = renderer.compute_dynamic_states(
            xy_tensor, posture_tensor, stroke_tensor
        )
        rendered = renderer(
            xy_tensor,
            posture_tensor,
            stroke_tensor,
        )[0, 0].cpu().numpy()
    output = Path(args.output_image)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(
        np.rint(np.clip(rendered, 0.0, 1.0) * 255.0).astype(np.uint8),
        mode="L",
    ).save(output)
    save_dynamic_states(
        sample, xy, posture, states, output.with_suffix(".states.csv")
    )
    report = {
        "format": "paper_forward_renderer_v1",
        "simulation_only": True,
        "character": sample.character,
        "sample_id": sample.meta.get("sample_id"),
        "point_count": int(len(xy)),
        "fixed_xy": True,
        "pose_source": pose_source,
        "angle_unit": "rad",
        "z_semantics": "H_mm",
        "gamma_rad": 0.0,
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[DONE] Forward-rendered {sample.character or 'sample'} on {device}: "
        f"{output}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory_csv", required=True)
    parser.add_argument("--bbsmg_ckpt", required=True)
    parser.add_argument("--pose_csv", default=None)
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
    parser.add_argument("--width_inertia", type=float, default=0.02)
    parser.add_argument("--drag_inertia", type=float, default=0.02)
    parser.add_argument("--offset_fraction", type=float, default=0.25)
    parser.add_argument("--pixels_per_model_unit", type=float, default=20.0)
    parser.add_argument("--patch_floor", type=float, default=0.05)
    parser.add_argument("--point_batch_size", type=int, default=128)
    main(parser.parse_args())
