"""Build a character-grouped, gamma-consistent B-BSMG simulation dataset.

The analytic B-BSM mask is intentionally character agnostic.  We nevertheless
sample it once per trajectory character and persist group ids so validation can
hold out complete characters rather than leaking nearly identical random
patches into train and validation.  This is a geometry-general v15 baseline;
real brush/style supervision remains a separate stage.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.paper_bbsm import (
    PAPER_ANGLE_BASIS_RADIAN,
    PAPER_POSTURE_MAX,
    PAPER_POSTURE_MIN,
    render_bbsm_mask,
)
from datasets.calligraphy_image_dataset import CalligraphyImageDataset


FEATURE_NAMES = [
    "H_mm",
    "alpha_rad",
    "beta_rad",
    "gamma_relative_rad",
    "x0_px",
    "y0_px",
]


def trajectory_characters(path: str) -> list[str]:
    values: set[str] = set()
    with open(path, "r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            value = str(row.get("character", ""))
            if value:
                values.add(value)
    return sorted(values)


def database_characters(
    image_dir: str,
    json_dir: str,
    data_csv: str,
    image_ext: str,
    chirography: str,
) -> set[str]:
    """Return characters present in the requested database writing style.

    The trajectory CSV does not carry a writing-style field, so filtering it
    directly would silently mix styles.  Reuse the project's LabelMe index,
    which joins each image to ``data.csv`` and applies the exact ``楷`` filter.
    """
    dataset = CalligraphyImageDataset(
        image_dir=image_dir,
        json_dir=json_dir,
        image_ext=image_ext,
        image_size=None,
        grayscale=True,
        padding=0.0,
        data_csv=data_csv,
        chirography_filter=chirography,
    )
    return {str(item["character"]) for item in dataset.index if item.get("character")}


def _latin_hypercube(rng: np.random.Generator, count: int, dims: int) -> np.ndarray:
    unit = np.empty((count, dims), dtype=np.float64)
    for dim in range(dims):
        unit[:, dim] = (rng.permutation(count) + rng.random(count)) / float(count)
    return unit


def build_dataset(
    trajectory_csv: str,
    output_npz: str,
    samples_per_character: int,
    holdout_characters: list[str],
    image_size: int,
    pixels_per_model_unit: float,
    supersample: int,
    anchor_margin: float,
    gamma_max_abs_deg: float,
    seed: int,
    style_image_dir: str,
    style_json_dir: str,
    style_data_csv: str,
    style_image_ext: str,
    chirography: str,
) -> dict:
    all_characters = trajectory_characters(trajectory_csv)
    style_characters = database_characters(
        style_image_dir,
        style_json_dir,
        style_data_csv,
        style_image_ext,
        chirography,
    )
    holdout = set(holdout_characters)
    characters = [
        value
        for value in all_characters
        if value in style_characters and value not in holdout
    ]
    if not characters:
        raise ValueError("No training characters remain after holdout filtering")
    if samples_per_character < 1:
        raise ValueError("samples_per_character must be positive")
    gamma_max = float(np.deg2rad(gamma_max_abs_deg))
    if not 0.0 < gamma_max <= np.pi:
        raise ValueError("gamma_max_abs_deg must be in (0, 180]")
    rng = np.random.default_rng(seed)
    dimensions = 6
    x_min = float(anchor_margin)
    x_max = float(image_size - 1.0 - anchor_margin)
    if x_min >= x_max:
        raise ValueError("anchor_margin leaves no usable anchor range")
    total = len(characters) * samples_per_character
    inputs = np.empty((total, dimensions), dtype=np.float32)
    targets = np.empty((total, 1, image_size, image_size), dtype=np.uint8)
    group_ids = np.empty((total,), dtype=np.int32)
    cursor = 0
    for group_id, character in enumerate(characters):
        unit = _latin_hypercube(rng, samples_per_character, dimensions)
        posture = PAPER_POSTURE_MIN + unit[:, :3] * (PAPER_POSTURE_MAX - PAPER_POSTURE_MIN)
        gamma = -gamma_max + 2.0 * gamma_max * unit[:, 3]
        anchors = x_min + unit[:, 4:6] * (x_max - x_min)
        rows = np.concatenate([posture, gamma[:, None], anchors], axis=1).astype(np.float32)
        inputs[cursor : cursor + samples_per_character] = rows
        group_ids[cursor : cursor + samples_per_character] = group_id
        for local_index, row in enumerate(rows):
            targets[cursor + local_index, 0] = np.rint(
                render_bbsm_mask(
                    row[:3],
                    float(row[4]),
                    float(row[5]),
                    image_size=image_size,
                    pixels_per_model_unit=pixels_per_model_unit,
                    supersample=supersample,
                    angle_basis=PAPER_ANGLE_BASIS_RADIAN,
                    gamma_rad=float(row[3]),
                )
                * 255.0
            ).astype(np.uint8)
        cursor += samples_per_character
    scales = [
        float(PAPER_POSTURE_MAX[0]),
        float(PAPER_POSTURE_MAX[1]),
        float(PAPER_POSTURE_MAX[2]),
        gamma_max,
        float(image_size),
        float(image_size),
    ]
    metadata = {
        "format": "paper_bbsmg_general_v15",
        "feature_names": FEATURE_NAMES,
        "regression_angle_basis": PAPER_ANGLE_BASIS_RADIAN,
        "gamma_semantics": "relative_to_trajectory_heading",
        "input_normalization": {
            "version": 3,
            "input_dim": 6,
            "scales": scales,
            "feature_names": FEATURE_NAMES,
            "checkpoint_format": "paper_bbsmg_general_v15",
            "regression_angle_basis": PAPER_ANGLE_BASIS_RADIAN,
            "gamma_semantics": "relative_to_trajectory_heading",
        },
        "units": {"H": "mm", "alpha": "rad", "beta": "rad", "gamma": "rad", "x0": "pixel", "y0": "pixel"},
        "limits": {
            "H_mm": [float(PAPER_POSTURE_MIN[0]), float(PAPER_POSTURE_MAX[0])],
            "alpha_rad": [float(PAPER_POSTURE_MIN[1]), float(PAPER_POSTURE_MAX[1])],
            "beta_rad": [float(PAPER_POSTURE_MIN[2]), float(PAPER_POSTURE_MAX[2])],
            "gamma_relative_rad": [-gamma_max, gamma_max],
        },
        "image_size": int(image_size),
        "pixels_per_model_unit": float(pixels_per_model_unit),
        "supersample": int(supersample),
        "anchor_margin": float(anchor_margin),
        "samples_per_character": int(samples_per_character),
        "sample_count": int(total),
        "characters": characters,
        "character_count": len(characters),
        "chirography": chirography,
        "style_database": {
            "data_csv": style_data_csv,
            "image_dir": style_image_dir,
            "json_dir": style_json_dir,
            "image_ext": style_image_ext,
            "allowed_character_count": len(style_characters),
        },
        "holdout_characters": sorted(holdout),
        "group_split": "character_grouped_holdout",
        "simulation_only": True,
        "sampling_mode": "latin_hypercube_per_character",
        "style_supervision": "analytic_b_bsmg_only",
    }
    output = Path(output_npz)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        inputs=inputs,
        targets=targets,
        group_ids=group_ids,
        group_names=np.asarray(characters),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    output.with_suffix(".summary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] general v15 dataset: {output}")
    print(f"[DONE] characters={len(characters)}, samples={total}, holdout={sorted(holdout)}")
    print(f"[DONE] inputs={inputs.shape}, targets={targets.shape}, gamma=relative_to_trajectory_heading")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory_csv", default="data/raw/trajectories.csv")
    parser.add_argument("--output_npz", default="data/processed/paper_bbsmg_general_v15.npz")
    parser.add_argument("--samples_per_character", type=int, default=100)
    parser.add_argument("--holdout_character", action="append", default=["武"])
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--pixels_per_model_unit", type=float, default=20.0)
    parser.add_argument("--supersample", type=int, default=4)
    parser.add_argument("--anchor_margin", type=float, default=4.0)
    parser.add_argument("--gamma_max_abs_deg", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=1501)
    parser.add_argument("--style_image_dir", default="data/raw/images")
    parser.add_argument("--style_json_dir", default="data/raw/json_files")
    parser.add_argument("--style_data_csv", default="data/raw/data.csv")
    parser.add_argument("--style_image_ext", default=".jpg")
    parser.add_argument("--chirography", default="楷")
    args = parser.parse_args()
    build_dataset(
        trajectory_csv=args.trajectory_csv,
        output_npz=args.output_npz,
        samples_per_character=args.samples_per_character,
        holdout_characters=args.holdout_character,
        image_size=args.image_size,
        pixels_per_model_unit=args.pixels_per_model_unit,
        supersample=args.supersample,
        anchor_margin=args.anchor_margin,
        gamma_max_abs_deg=args.gamma_max_abs_deg,
        seed=args.seed,
        style_image_dir=args.style_image_dir,
        style_json_dir=args.style_json_dir,
        style_data_csv=args.style_data_csv,
        style_image_ext=args.style_image_ext,
        chirography=args.chirography,
    )


if __name__ == "__main__":
    main()
