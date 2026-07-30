"""Initialize planar trajectory points from a target character skeleton."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.trajectory_dataset import load_trajectory_csv
from optim.trajectory_optimizer import load_target_image
from tools.invert_paper_trajectory import (
    canvas_xy_to_source,
    load_initial_pose_csv,
    pick_sample,
    save_pose_csv,
    source_xy_to_canvas,
)


def thin_binary(mask: np.ndarray, max_iterations: int = 128) -> np.ndarray:
    """Zhang-Suen thinning without an optional image-processing dependency."""
    image = np.asarray(mask, dtype=np.uint8).copy()
    image[[0, -1], :] = 0
    image[:, [0, -1]] = 0
    for _ in range(max_iterations):
        changed = False
        for phase in (0, 1):
            p2 = np.roll(image, 1, axis=0)
            p3 = np.roll(p2, -1, axis=1)
            p4 = np.roll(image, -1, axis=1)
            p5 = np.roll(p4, -1, axis=0)
            p6 = np.roll(image, -1, axis=0)
            p7 = np.roll(p6, 1, axis=1)
            p8 = np.roll(image, 1, axis=1)
            p9 = np.roll(p8, 1, axis=0)
            neighbors = (p2, p3, p4, p5, p6, p7, p8, p9)
            count = sum(neighbors)
            transitions = sum(
                (neighbors[index] == 0)
                & (neighbors[(index + 1) % 8] == 1)
                for index in range(8)
            )
            remove = (
                (image == 1)
                & (count >= 2)
                & (count <= 6)
                & (transitions == 1)
            )
            if phase == 0:
                remove &= (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                remove &= (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            remove[[0, -1], :] = False
            remove[:, [0, -1]] = False
            if np.any(remove):
                image[remove] = 0
                changed = True
        if not changed:
            break
    return image.astype(bool)


def nearest_skeleton_displacements(
    xy: np.ndarray,
    skeleton: np.ndarray,
    max_snap_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    skeleton_yx = np.argwhere(skeleton)
    if len(skeleton_yx) == 0:
        raise ValueError("Target skeleton is empty")
    skeleton_xy = skeleton_yx[:, ::-1].astype(np.float32)
    delta = skeleton_xy[None, :, :] - xy[:, None, :]
    squared = np.sum(delta * delta, axis=2)
    nearest = np.argmin(squared, axis=1)
    distances = np.sqrt(squared[np.arange(len(xy)), nearest])
    displacements = delta[np.arange(len(xy)), nearest]
    scale = np.minimum(1.0, max_snap_px / np.maximum(distances, 1e-6))
    return displacements * scale[:, None], distances


def smooth_displacements(
    displacements: np.ndarray,
    stroke_ids: np.ndarray,
    sigma: float,
) -> np.ndarray:
    result = displacements.copy()
    if sigma <= 0:
        return result
    for stroke_id in np.unique(stroke_ids):
        indices = np.flatnonzero(stroke_ids == stroke_id)
        if len(indices) > 1:
            result[indices] = gaussian_filter1d(
                result[indices], sigma=sigma, axis=0, mode="nearest"
            )
    return result


def point_to_skeleton_distance(xy: np.ndarray, skeleton: np.ndarray) -> np.ndarray:
    _, distance = nearest_skeleton_displacements(
        xy, skeleton, max_snap_px=float("inf")
    )
    return distance


def main(args: argparse.Namespace) -> None:
    if not 0 < args.blend <= 1:
        raise ValueError("--blend must be in (0,1]")
    if args.max_snap_px <= 0 or args.smooth_sigma < 0:
        raise ValueError("snap radius must be positive and sigma non-negative")
    samples = load_trajectory_csv(args.trajectory_csv)
    sample = pick_sample(
        samples,
        sample_id=args.sample_id,
        character=args.character,
        index=args.index,
    )
    posture, xy_source, gamma = load_initial_pose_csv(args.pose_csv, sample)
    xy_canvas = source_xy_to_canvas(
        sample, xy_source, args.image_size, args.padding
    )
    target = load_target_image(args.target_image, args.image_size)
    skeleton = thin_binary(target >= args.threshold)
    stroke_ids = np.asarray(
        [point.stroke_id for point in sample.all_points()], dtype=np.int64
    )
    raw_displacement, before_distance = nearest_skeleton_displacements(
        xy_canvas, skeleton, args.max_snap_px
    )
    displacement = smooth_displacements(
        raw_displacement, stroke_ids, args.smooth_sigma
    )
    norms = np.linalg.norm(displacement, axis=1)
    clip = np.minimum(1.0, args.max_snap_px / np.maximum(norms, 1e-6))
    displacement *= clip[:, None]
    snapped_canvas = xy_canvas + args.blend * displacement
    after_distance = point_to_skeleton_distance(snapped_canvas, skeleton)
    snapped_source = canvas_xy_to_source(
        sample, snapped_canvas, args.image_size, args.padding
    )

    with open(args.pose_csv, "r", encoding="utf-8-sig", newline="") as file:
        first = next(csv.DictReader(file))
    basis = first.get("regression_angle_basis") or "paper_declared_radian"
    fixed = {
        "source": "initial_pose_csv",
        "confidence": "low",
        "reason": "preserved_during_target_skeleton_initialization",
    }
    planar = {
        "source": "target_skeleton_initialized",
        "confidence": "low_simulation",
        "reason": "bounded_local_target_skeleton_snap",
    }
    decisions = {
        "H": dict(fixed),
        "alpha": dict(fixed),
        "beta": dict(fixed),
        "gamma": dict(fixed),
        "x": dict(planar),
        "y": dict(planar),
    }
    output = Path(args.output_csv)
    save_pose_csv(
        sample,
        posture,
        output,
        basis,
        decisions,
        xy_source=snapped_source,
        gamma=gamma,
        prototype="paper_target_skeleton_initializer_v1",
    )

    panel = Image.new("RGB", (args.image_size * 3, args.image_size), "black")
    target_image = Image.fromarray(
        np.rint(np.clip(target, 0, 1) * 255).astype(np.uint8), mode="L"
    ).convert("RGB")
    skeleton_image = Image.fromarray(
        skeleton.astype(np.uint8) * 255, mode="L"
    ).convert("RGB")
    overlay = target_image.copy()
    draw = ImageDraw.Draw(overlay)
    for stroke_id in np.unique(stroke_ids):
        indices = np.flatnonzero(stroke_ids == stroke_id)
        if len(indices) > 1:
            draw.line(
                [tuple(map(float, point)) for point in snapped_canvas[indices]],
                fill=(255, 0, 0),
                width=1,
            )
    panel.paste(target_image, (0, 0))
    panel.paste(skeleton_image, (args.image_size, 0))
    panel.paste(overlay, (2 * args.image_size, 0))
    panel.save(output.with_suffix(".png"))

    report = {
        "format": "paper_target_skeleton_initializer_v1",
        "target_image": args.target_image,
        "input_pose_csv": args.pose_csv,
        "output_pose_csv": str(output),
        "point_count": int(len(xy_canvas)),
        "target_skeleton_pixels": int(skeleton.sum()),
        "max_snap_px": float(args.max_snap_px),
        "blend": float(args.blend),
        "smooth_sigma": float(args.smooth_sigma),
        "distance_to_target_skeleton_px": {
            "before_mean": float(before_distance.mean()),
            "before_max": float(before_distance.max()),
            "after_mean": float(after_distance.mean()),
            "after_max": float(after_distance.max()),
        },
        "applied_displacement_px": {
            "mean": float(np.linalg.norm(
                snapped_canvas - xy_canvas, axis=1
            ).mean()),
            "max": float(np.linalg.norm(
                snapped_canvas - xy_canvas, axis=1
            ).max()),
        },
        "simulation_only": True,
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory_csv", required=True)
    parser.add_argument("--pose_csv", required=True)
    parser.add_argument("--target_image", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--character", default=None)
    parser.add_argument("--sample_id", default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--padding", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_snap_px", type=float, default=10.0)
    parser.add_argument("--blend", type=float, default=0.75)
    parser.add_argument("--smooth_sigma", type=float, default=0.75)
    main(parser.parse_args())
