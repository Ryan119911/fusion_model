"""Build a paper-parameterized B-BSMG simulation dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.paper_bbsm import (
    PAPER_ANGLE_BASES,
    PAPER_ANGLE_BASIS_DEGREE_FITTED,
    PAPER_ANGLE_BASIS_RADIAN,
    PAPER_POSTURE_MAX,
    PAPER_POSTURE_MIN,
    render_bbsm_mask,
)


FEATURE_NAMES = ["H_mm", "alpha_rad", "beta_rad", "x0_px", "y0_px"]
GAMMA_FEATURE_NAMES = [
    "H_mm",
    "alpha_rad",
    "beta_rad",
    "gamma_rad",
    "x0_px",
    "y0_px",
]
GAMMA_FORMAT = "paper_bbsmg_gamma_v13"
FORMAT_BY_ANGLE_BASIS = {
    PAPER_ANGLE_BASIS_RADIAN: "paper_bbsmg_v1",
    PAPER_ANGLE_BASIS_DEGREE_FITTED: "paper_bbsmg_degree_fitted_v2",
}


def build_dataset(
    count: int,
    image_size: int,
    pixels_per_model_unit: float,
    supersample: int,
    seed: int,
    anchor_margin: float = 4.0,
    regression_angle_basis: str = PAPER_ANGLE_BASIS_RADIAN,
    include_gamma: bool = False,
    gamma_max_abs_rad: float = np.deg2rad(30.0),
    sampling_mode: str = "random",
) -> tuple[np.ndarray, np.ndarray, dict]:
    if count < 1:
        raise ValueError("count must be positive")
    if gamma_max_abs_rad <= 0 or gamma_max_abs_rad > np.pi:
        raise ValueError("gamma_max_abs_rad must be in (0,pi]")
    if sampling_mode not in {"random", "latin_hypercube"}:
        raise ValueError("sampling_mode must be random or latin_hypercube")
    rng = np.random.default_rng(seed)
    dimensions = 6 if include_gamma else 5
    if sampling_mode == "latin_hypercube":
        unit = np.empty((count, dimensions), dtype=np.float64)
        for dimension in range(dimensions):
            unit[:, dimension] = (
                rng.permutation(count) + rng.random(count)
            ) / float(count)
    else:
        unit = rng.random((count, dimensions))
    posture = PAPER_POSTURE_MIN + unit[:, :3] * (
        PAPER_POSTURE_MAX - PAPER_POSTURE_MIN
    )
    x_range = (float(anchor_margin), image_size - 1.0 - float(anchor_margin))
    y_range = x_range
    if x_range[0] >= x_range[1]:
        raise ValueError("anchor_margin leaves no usable canvas")
    if include_gamma:
        gamma = (
            -gamma_max_abs_rad
            + 2.0 * gamma_max_abs_rad * unit[:, 3:4]
        )
        anchor_unit = unit[:, 4:6]
    else:
        gamma = None
        anchor_unit = unit[:, 3:5]
    anchors = x_range[0] + anchor_unit * (x_range[1] - x_range[0])
    inputs = np.concatenate(
        [posture, gamma, anchors] if include_gamma else [posture, anchors],
        axis=1,
    ).astype(np.float32)
    targets = np.empty((count, 1, image_size, image_size), dtype=np.uint8)
    for index, row in enumerate(inputs):
        gamma_value = float(row[3]) if include_gamma else 0.0
        anchor_offset = 4 if include_gamma else 3
        mask = render_bbsm_mask(
            row[:3],
            float(row[anchor_offset]),
            float(row[anchor_offset + 1]),
            image_size=image_size,
            pixels_per_model_unit=pixels_per_model_unit,
            supersample=supersample,
            angle_basis=regression_angle_basis,
            gamma_rad=gamma_value,
        )
        targets[index, 0] = np.rint(mask * 255.0).astype(np.uint8)
    format_name = (
        GAMMA_FORMAT
        if include_gamma
        else FORMAT_BY_ANGLE_BASIS[regression_angle_basis]
    )
    feature_names = GAMMA_FEATURE_NAMES if include_gamma else FEATURE_NAMES
    scales = [
        float(PAPER_POSTURE_MAX[0]),
        float(PAPER_POSTURE_MAX[1]),
        float(PAPER_POSTURE_MAX[2]),
    ]
    if include_gamma:
        # Symmetric gamma must keep its sign; division by a positive range
        # maps it to [-1,1].
        scales.append(float(gamma_max_abs_rad))
    scales.extend([float(image_size), float(image_size)])
    metadata = {
        "format": format_name,
        "feature_names": feature_names,
        "regression_angle_basis": regression_angle_basis,
        "input_normalization": {
            "version": 2,
            "input_dim": len(feature_names),
            "scales": scales,
            "feature_names": feature_names,
            "checkpoint_format": format_name,
            "regression_angle_basis": regression_angle_basis,
        },
        "units": {
            "H": "mm",
            "alpha": "rad",
            "beta": "rad",
            "gamma": "rad",
            "x0": "pixel",
            "y0": "pixel",
        },
        "limits": {
            "H_mm": [float(PAPER_POSTURE_MIN[0]), float(PAPER_POSTURE_MAX[0])],
            "alpha_rad": [
                float(PAPER_POSTURE_MIN[1]),
                float(PAPER_POSTURE_MAX[1]),
            ],
            "beta_rad": [
                float(PAPER_POSTURE_MIN[2]),
                float(PAPER_POSTURE_MAX[2]),
            ],
            "gamma_rad": (
                [-float(gamma_max_abs_rad), float(gamma_max_abs_rad)]
                if include_gamma
                else [0.0, 0.0]
            ),
        },
        "image_size": int(image_size),
        "pixels_per_model_unit": float(pixels_per_model_unit),
        "supersample": int(supersample),
        "anchor_margin": float(anchor_margin),
        "count": int(count),
        "seed": int(seed),
        "simulation_only": True,
        "sampling_mode": sampling_mode,
        "gamma_conditioned": include_gamma,
    }
    return inputs, targets, metadata


def main(args: argparse.Namespace) -> None:
    inputs, targets, metadata = build_dataset(
        count=args.count,
        image_size=args.image_size,
        pixels_per_model_unit=args.pixels_per_model_unit,
        supersample=args.supersample,
        seed=args.seed,
        anchor_margin=args.anchor_margin,
        regression_angle_basis=args.regression_angle_basis,
        include_gamma=args.include_gamma,
        gamma_max_abs_rad=float(np.deg2rad(args.gamma_max_abs_deg)),
        sampling_mode=args.sampling_mode,
    )
    path = Path(args.output_npz)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        inputs=inputs,
        targets=targets,
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    summary_path = path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[DONE] B-BSMG simulation data: {path}")
    print(f"[DONE] inputs={inputs.shape}, targets={targets.shape}")
    print(
        "[RANGE] H=11-20 mm, alpha=0-0.174533 rad, "
        f"beta=0-0.087266 rad, gamma="
        f"{'-' + str(args.gamma_max_abs_deg) + '..' + str(args.gamma_max_abs_deg) + ' deg' if args.include_gamma else '0 rad'}"
    )
    print(
        "[SEMANTICS] external angles=rad, regression_angle_basis="
        f"{args.regression_angle_basis}"
    )
    print(f"[DONE] summary: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_npz", default="data/processed/paper_bbsmg_v1.npz"
    )
    parser.add_argument(
        "--include_gamma",
        action="store_true",
        help="build the 6D H/alpha/beta/gamma/x/y v13 dataset",
    )
    parser.add_argument("--gamma_max_abs_deg", type=float, default=30.0)
    parser.add_argument(
        "--sampling_mode",
        choices=["random", "latin_hypercube"],
        default="random",
    )
    parser.add_argument("--count", type=int, default=50000)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--pixels_per_model_unit", type=float, default=20.0)
    parser.add_argument("--supersample", type=int, default=4)
    parser.add_argument("--anchor_margin", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--regression_angle_basis",
        choices=PAPER_ANGLE_BASES,
        default=PAPER_ANGLE_BASIS_RADIAN,
    )
    main(parser.parse_args())
