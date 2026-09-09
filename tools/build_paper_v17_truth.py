"""Build a seeded synthetic pose truth for the v17 multi-solution audit.

This utility deliberately produces *simulation-only* labels.  It keeps the
input x/y trajectory and assigns a smooth, per-stroke affine profile to
``H_mm``, ``alpha_rad``, ``beta_rad`` and ``gamma_rad``.  The affine profile is
exactly representable by the order-1 CGL parameterisation used by the inverse
solver, so failures in the recovery test are attributable to optimisation or
observability rather than an out-of-model target.

The ``heading`` gamma mode is useful for the paper convention in which gamma
is the forward direction inferred from adjacent x/y points.  ``random`` is
the stronger all-fields probe and gives gamma an independent (but bounded)
synthetic value.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


POSTURE_LIMITS = {
    "z": (11.0, 20.0),
    "alpha": (0.0, float(np.deg2rad(10.0))),
    "beta": (0.0, float(np.deg2rad(5.0))),
    "gamma": (-float(np.pi), float(np.pi)),
}


def _wrap_angle(value: float) -> float:
    return float(np.arctan2(np.sin(value), np.cos(value)))


def _heading_by_key(rows: Iterable[dict]) -> dict[tuple[int, int], float]:
    """Return forward headings in the renderer's canvas frame.

    ``CanvasTransform.map_point`` preserves x but flips y (source trajectory
    coordinates use an upward-positive axis while PIL/canvas pixels use a
    downward-positive axis).  The dynamic renderer consumes the latter frame,
    so the synthetic heading truth must apply the same flip before atan2.
    """
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row["stroke_id"])].append(row)
    headings: dict[tuple[int, int], float] = {}
    for stroke_rows in grouped.values():
        stroke_rows.sort(key=lambda row: int(row["point_id"]))
        if len(stroke_rows) == 1:
            headings[(int(stroke_rows[0]["stroke_id"]), int(stroke_rows[0]["point_id"]))] = 0.0
            continue
        points = np.asarray(
            [[float(row["x"]), float(row["y"])] for row in stroke_rows],
            dtype=np.float64,
        )
        delta = np.diff(points, axis=0)
        # Match models.geometry.CanvasTransform.map_point: ny is based on
        # (src_max_y - y), hence dy_canvas = -dy_source.
        angles = np.arctan2(-delta[:, 1], delta[:, 0])
        # Use the forward segment at each point and the last segment at the
        # terminal point.  Zero-length segments inherit the previous heading.
        previous = float(angles[0])
        for index, row in enumerate(stroke_rows):
            if index < len(angles):
                angle = float(angles[index])
                if np.linalg.norm(delta[index]) <= 1e-9:
                    angle = previous
            else:
                angle = previous
            previous = angle
            headings[(int(row["stroke_id"]), int(row["point_id"]))] = _wrap_angle(angle)
    return headings


def _bounded_affine(rng: np.random.Generator, low: float, high: float) -> tuple[float, float]:
    # Leave a fixed margin from physical boundaries.  Boundary contact is an
    # explicit failure signal in the inverse audit, not a way to make a probe
    # artificially easy or hard.
    margin = 0.12 * (high - low)
    inner_low, inner_high = low + margin, high - margin
    first = float(rng.uniform(inner_low, inner_high))
    second = float(rng.uniform(inner_low, inner_high))
    return first, second


def _select_rows(rows: list[dict], character: str | None, sample_id: str | None) -> list[dict]:
    if character is not None:
        rows = [row for row in rows if row.get("character") == character]
    if sample_id is not None:
        rows = [row for row in rows if row.get("sample_id") == sample_id]
    if not rows:
        raise ValueError("No trajectory rows remain after character/sample filtering")
    sample_ids = sorted({row.get("sample_id", "") for row in rows})
    if len(sample_ids) > 1:
        raise ValueError(
            "Input contains multiple sample_id values; pass --sample_id explicitly: "
            + ", ".join(sample_ids[:8])
        )
    rows.sort(key=lambda row: (int(row["stroke_id"]), int(row["point_id"])))
    return rows


def build_random_truth(
    input_pose_csv: str,
    output_pose_csv: str,
    character: str | None = None,
    sample_id: str | None = None,
    seed: int = 17017,
    gamma_mode: str = "random",
) -> dict:
    if gamma_mode not in {"random", "heading"}:
        raise ValueError("gamma_mode must be random or heading")
    with open(input_pose_csv, "r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        rows = _select_rows(list(reader), character, sample_id)
    required = {"stroke_id", "point_id", "x", "y"}
    if not required.issubset(fieldnames):
        raise ValueError(f"Pose CSV must contain {sorted(required)}")
    for field in (
        "z",
        "alpha",
        "beta",
        "gamma",
        "z_unit",
        "angle_unit",
        "pose_frame",
        "prototype",
        "z_source",
        "alpha_source",
        "beta_source",
        "gamma_source",
    ):
        if field not in fieldnames:
            fieldnames.append(field)

    rng = np.random.default_rng(seed)
    headings = _heading_by_key(rows)
    profile_by_stroke: dict[int, dict[str, tuple[float, float]]] = {}
    for stroke_id in sorted({int(row["stroke_id"]) for row in rows}):
        profile_by_stroke[stroke_id] = {
            field: _bounded_affine(rng, *limits)
            for field, limits in POSTURE_LIMITS.items()
            if field != "gamma"
        }
        if gamma_mode == "random":
            profile_by_stroke[stroke_id]["gamma"] = _bounded_affine(
                rng, float(np.deg2rad(-35.0)), float(np.deg2rad(35.0))
            )

    for row in rows:
        stroke_id = int(row["stroke_id"])
        stroke_rows = [item for item in rows if int(item["stroke_id"]) == stroke_id]
        index = next(i for i, item in enumerate(stroke_rows) if item is row)
        t = index / max(len(stroke_rows) - 1, 1)
        profile = profile_by_stroke[stroke_id]
        row["z"] = repr(float(np.interp(t, [0.0, 1.0], profile["z"])))
        row["alpha"] = repr(float(np.interp(t, [0.0, 1.0], profile["alpha"])))
        row["beta"] = repr(float(np.interp(t, [0.0, 1.0], profile["beta"])))
        row["gamma"] = repr(
            headings[(stroke_id, int(row["point_id"]))]
            if gamma_mode == "heading"
            else float(np.interp(t, [0.0, 1.0], profile["gamma"]))
        )
        row["z_unit"] = "mm"
        row["angle_unit"] = "rad"
        row["pose_frame"] = "paper_model"
        row["prototype"] = "paper_pose_v17_random_truth"
        for field in ("z_source", "alpha_source", "beta_source", "gamma_source"):
            row[field] = "synthetic_known_ground_truth"

    output = Path(output_pose_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    values = {
        field: np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        for field in ("z", "alpha", "beta", "gamma")
    }
    report = {
        "format": "paper_pose_random_truth_v17",
        "simulation_only": True,
        "input_pose_csv": input_pose_csv,
        "output_pose_csv": str(output),
        "character": rows[0].get("character"),
        "sample_id": rows[0].get("sample_id"),
        "seed": int(seed),
        "gamma_mode": gamma_mode,
        "point_count": len(rows),
        "stroke_count": len(profile_by_stroke),
        "angle_unit": "rad",
        "pose_frame": "paper_model",
        "gamma_frame": "canvas_xy_after_source_y_flip",
        "profiles_are_stroke_affine": True,
        "ranges": {
            field: [float(array.min()), float(array.max())]
            for field, array in values.items()
        },
        "warning": (
            "Synthetic observability probe only; values are not calibrated "
            "robot commands or real brush measurements."
        ),
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_pose_csv", required=True)
    parser.add_argument("--output_pose_csv", required=True)
    parser.add_argument("--character", default=None)
    parser.add_argument("--sample_id", default=None)
    parser.add_argument("--seed", type=int, default=17017)
    parser.add_argument("--gamma_mode", choices=("random", "heading"), default="random")
    args = parser.parse_args()
    report = build_random_truth(
        args.input_pose_csv,
        args.output_pose_csv,
        character=args.character,
        sample_id=args.sample_id,
        seed=args.seed,
        gamma_mode=args.gamma_mode,
    )
    print(
        f"[DONE] v17 random synthetic truth: points={report['point_count']}, "
        f"strokes={report['stroke_count']}, output={args.output_pose_csv}"
    )
    for field, limits in report["ranges"].items():
        print(f"{field}: {limits[0]:.6f} .. {limits[1]:.6f}")


if __name__ == "__main__":
    main()
