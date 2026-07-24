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
    PAPER_POSTURE_MAX,
    PAPER_POSTURE_MIN,
    render_bbsm_mask,
)


FEATURE_NAMES = ["H_mm", "alpha_rad", "beta_rad", "x0_px", "y0_px"]
FORMAT_NAME = "paper_bbsmg_v1"


def build_dataset(
    count: int,
    image_size: int,
    pixels_per_model_unit: float,
    supersample: int,
    seed: int,
    anchor_margin: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if count < 1:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(seed)
    posture = rng.uniform(PAPER_POSTURE_MIN, PAPER_POSTURE_MAX, size=(count, 3))
    x_range = (float(anchor_margin), image_size - 1.0 - float(anchor_margin))
    y_range = x_range
    if x_range[0] >= x_range[1]:
        raise ValueError("anchor_margin leaves no usable canvas")
    anchors = np.column_stack(
        [rng.uniform(*x_range, size=count), rng.uniform(*y_range, size=count)]
    )
    inputs = np.concatenate([posture, anchors], axis=1).astype(np.float32)
    targets = np.empty((count, 1, image_size, image_size), dtype=np.uint8)
    for index, row in enumerate(inputs):
        mask = render_bbsm_mask(
            row[:3],
            float(row[3]),
            float(row[4]),
            image_size=image_size,
            pixels_per_model_unit=pixels_per_model_unit,
            supersample=supersample,
        )
        targets[index, 0] = np.rint(mask * 255.0).astype(np.uint8)
    metadata = {
        "format": FORMAT_NAME,
        "feature_names": FEATURE_NAMES,
        "input_normalization": {
            "version": 2,
            "input_dim": 5,
            "scales": [
                float(PAPER_POSTURE_MAX[0]),
                float(PAPER_POSTURE_MAX[1]),
                float(PAPER_POSTURE_MAX[2]),
                float(image_size),
                float(image_size),
            ],
            "feature_names": FEATURE_NAMES,
        },
        "units": {
            "H": "mm",
            "alpha": "rad",
            "beta": "rad",
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
            "gamma_rad": [0.0, 0.0],
        },
        "image_size": int(image_size),
        "pixels_per_model_unit": float(pixels_per_model_unit),
        "supersample": int(supersample),
        "anchor_margin": float(anchor_margin),
        "count": int(count),
        "seed": int(seed),
        "simulation_only": True,
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
        "beta=0-0.087266 rad, gamma=0 rad"
    )
    print(f"[DONE] summary: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_npz", default="data/processed/paper_bbsmg_v1.npz"
    )
    parser.add_argument("--count", type=int, default=50000)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--pixels_per_model_unit", type=float, default=20.0)
    parser.add_argument("--supersample", type=int, default=4)
    parser.add_argument("--anchor_margin", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
