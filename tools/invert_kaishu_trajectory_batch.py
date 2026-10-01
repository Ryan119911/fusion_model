"""Batch-export one refined pose trajectory for every database 楷书 character.

This command is intentionally a coordinator around ``invert_paper_trajectory``
rather than a second implementation of PSOC/LM.  It joins two independent
sources by character:

* the trajectory CSV supplies one complete input trajectory per character;
* the LabelMe/data.csv index supplies a 楷书 target crop for that character.

Each character is optimized in an isolated child process.  Isolation keeps a
large batch from accumulating CUDA graphs and makes failed characters
restartable.  A target crop, ``pose_refined.csv`` and the original inversion
report are retained for every completed character.  Characters present in the
楷书 database but absent from the trajectory CSV are reported explicitly;
they are never silently fabricated.

The generated pose files are paper-frame simulation candidates.  They are not
robot/TCP-calibrated trajectories.
"""
from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
try:  # Keep --help and manifest planning usable in minimal environments.
    from scipy.ndimage import distance_transform_edt
except ImportError:  # pragma: no cover - production dependencies include SciPy
    distance_transform_edt = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.geometry import normalize_trajectory_xy
from utils.image_preprocessing import letterbox_character_image


FORMAT = "kaishu_pose_trajectory_batch_v1"


def character_stem(character: str) -> str:
    """Return a collision-free, portable directory name for a Unicode char."""
    text = str(character)
    codepoints = "_".join(f"u{ord(value):04x}" for value in text)
    return f"char_{codepoints or 'empty'}"


def _sample_point_count(sample: Any) -> int:
    return len(sample.all_points())


def select_trajectory_samples(
    samples: Iterable[Any], selection: str = "longest"
) -> tuple[dict[str, Any], dict[str, int]]:
    """Select one deterministic trajectory for each character.

    ``trajectories.csv`` can contain multiple samples for a character.  The
    longest sample is the default because it usually preserves the most stroke
    detail; ties are resolved by sample id.  The returned duplicate counts are
    written to the manifest so this choice is auditable.
    """
    if selection not in {"longest", "first"}:
        raise ValueError("selection must be 'longest' or 'first'")
    grouped: dict[str, list[Any]] = defaultdict(list)
    for sample in samples:
        character = str(sample.character or "").strip()
        if character:
            grouped[character].append(sample)
    selected: dict[str, Any] = {}
    duplicates: dict[str, int] = {}
    for character, candidates in grouped.items():
        candidates = sorted(
            candidates,
            key=lambda item: str(item.meta.get("sample_id", "")),
        )
        if selection == "longest":
            candidates = sorted(
                candidates,
                key=lambda item: (
                    -_sample_point_count(item),
                    str(item.meta.get("sample_id", "")),
                ),
            )
        selected[character] = candidates[0]
        duplicates[character] = len(candidates)
    return selected, duplicates


def build_kaishu_dataset(
    image_dir: str,
    json_dir: str,
    data_csv: str,
    image_ext: str,
    chirography: str,
) -> CalligraphyImageDataset:
    """Build a strictly style-filtered LabelMe index.

    The dataset class deliberately warns and disables filtering when its
    metadata is missing.  That fallback is unsafe for this batch command, so
    fail loudly instead of mixing non-楷书 images.
    """
    # Import lazily so planning/help and unit tests do not require PyTorch.
    from datasets.calligraphy_image_dataset import CalligraphyImageDataset

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
    if dataset.allowed_image_rels is None:
        raise RuntimeError(
            "楷书 filter is not active; check --data_csv and its "
            "img_path/chirography columns before running the batch"
        )
    return dataset


def group_target_items(dataset: CalligraphyImageDataset) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in dataset.index:
        character = str(item.get("character", "")).strip()
        if character:
            grouped[character].append(item)
    for values in grouped.values():
        values.sort(
            key=lambda item: (
                str(item.get("image_path", "")),
                int(item.get("shape_index", 0)),
            )
        )
    return dict(grouped)


def load_target_exclusions(path: str | None) -> list[dict[str, Any]]:
    """Load exact image/bbox exclusions used by the dataset audit pipeline."""
    if not path:
        return []
    exclusion_path = Path(path).expanduser().resolve()
    if not exclusion_path.exists():
        raise FileNotFoundError(f"Target exclusion file not found: {path}")
    values = json.loads(exclusion_path.read_text(encoding="utf-8"))
    if not isinstance(values, list):
        raise ValueError("Target exclusion JSON must contain a list")
    for index, value in enumerate(values):
        if not isinstance(value, dict) or not value.get("image_path"):
            raise ValueError(
                f"Target exclusion #{index} requires an image_path field"
            )
    return [dict(value) for value in values]


def _normalise_candidate_path(value: Any) -> str:
    return str(value or "").replace("\\", "/").removeprefix("./")


def _candidate_excluded(
    item: dict[str, Any], character: str, exclusions: list[dict[str, Any]]
) -> dict[str, Any] | None:
    candidate_path = _normalise_candidate_path(item.get("image_path"))
    candidate_bbox = item.get("bbox")
    for exclusion in exclusions:
        excluded_character = exclusion.get("character")
        if excluded_character and str(excluded_character) != str(character):
            continue
        excluded_path = _normalise_candidate_path(exclusion.get("image_path"))
        if excluded_path != candidate_path and not candidate_path.endswith("/" + excluded_path):
            continue
        excluded_bbox = exclusion.get("bbox")
        if excluded_bbox is not None:
            if candidate_bbox is None or len(candidate_bbox) != len(excluded_bbox):
                continue
            if any(abs(float(actual) - float(expected)) > 1e-3 for actual, expected in zip(candidate_bbox, excluded_bbox)):
                continue
        return exclusion
    return None


def filter_target_items(
    grouped: dict[str, list[dict[str, Any]]], exclusions: list[dict[str, Any]]
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Remove audited bad candidates while retaining empty character keys."""
    removed: list[dict[str, Any]] = []
    filtered: dict[str, list[dict[str, Any]]] = {}
    for character, items in grouped.items():
        kept = []
        for item in items:
            exclusion = _candidate_excluded(item, character, exclusions)
            if exclusion is None:
                kept.append(item)
            else:
                removed.append({
                    "character": character,
                    "image_path": str(item.get("image_path")),
                    "bbox": item.get("bbox"),
                    "reason": exclusion.get("reason"),
                })
        filtered[character] = kept
    return filtered, removed


def trajectory_canvas(sample: Any, image_size: int, padding: int) -> np.ndarray:
    """Flatten a trajectory into the same Y-down canvas used by the renderer."""
    normalized = normalize_trajectory_xy(
        sample, canvas_size=image_size, padding=padding
    )
    points: list[tuple[float, float]] = []
    for stroke in normalized:
        points.extend(stroke)
    if not points:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def _sample_image_target(
    dataset: CalligraphyImageDataset,
    item: dict[str, Any],
    image_size: int,
    padding: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    sample = dataset._build_sample(item)
    canvas, transform = letterbox_character_image(
        sample["image"],
        canvas_size=image_size,
        padding=padding,
        crop_foreground=True,
    )
    return canvas.astype(np.float32), {
        "source_image": str(item["image_path"]),
        "source_json": str(item["json_path"]),
        "bbox": [float(value) for value in item["bbox"]],
        "shape_index": int(item["shape_index"]),
        "group_id": item.get("group_id"),
        "transform": transform,
    }


def target_match_score(
    target: np.ndarray,
    trajectory_xy: np.ndarray,
    tolerance_px: int = 5,
    threshold: float = 0.35,
    model_support: np.ndarray | None = None,
) -> dict[str, float]:
    """Score a target crop against a trajectory without using the B-BSMG model."""
    if trajectory_xy.size == 0:
        return {"coverage": 0.0, "mean_distance_px": float("inf"), "score": -float("inf")}
    mask = np.asarray(target >= threshold, dtype=bool)
    if not np.any(mask):
        return {"coverage": 0.0, "mean_distance_px": float("inf"), "score": -float("inf")}
    radius = max(1, int(tolerance_px))
    # Pillow's MaxFilter performs the square dilation in optimized native code;
    # a Python loop over every offset becomes prohibitive for 20k candidates.
    support = np.asarray(
        Image.fromarray((mask.astype(np.uint8) * 255), mode="L").filter(
            ImageFilter.MaxFilter(2 * radius + 1)
        )
    ) > 0
    x = np.clip(np.rint(trajectory_xy[:, 0]).astype(int), 0, mask.shape[1] - 1)
    y = np.clip(np.rint(trajectory_xy[:, 1]).astype(int), 0, mask.shape[0] - 1)
    coverage = float(support[y, x].mean())
    if distance_transform_edt is None:
        # Candidate ranking remains valid through the dominant coverage term;
        # the optional distance tie-break is unavailable without SciPy.
        mean_distance = 0.0
    else:
        distance = distance_transform_edt(~mask)
        mean_distance = float(distance[y, x].mean())
    result = {
        "coverage": coverage,
        "mean_distance_px": mean_distance,
        # Legacy score remains stable for best_trajectory_coverage.
        "score": float(coverage - 0.02 * min(mean_distance, 50.0)),
    }
    if model_support is not None:
        support = np.asarray(model_support, dtype=bool)
        intersection = float(np.logical_and(mask, support).sum())
        target_area = float(mask.sum())
        support_area = float(support.sum())
        dice = 2.0 * intersection / max(target_area + support_area, 1.0)
        union = float(np.logical_or(mask, support).sum())
        iou = intersection / max(union, 1.0)
        ink_ratio = target_area / max(support_area, 1.0)
        ink_balance = min(ink_ratio, 1.0 / max(ink_ratio, 1e-6))
        # A centerline-only criterion accepts arbitrarily thick targets.  The
        # renderer-support term selects a database variant that the bounded
        # paper model can actually express, while coverage still prevents a
        # visually similar but misplaced crop from winning.
        result.update(
            {
                "model_support_dice": float(dice),
                "model_support_iou": float(iou),
                "target_to_model_support_ink_ratio": float(ink_ratio),
                "model_support_ink_balance": float(ink_balance),
                "model_compatibility_score": float(
                    0.55 * dice
                    + 0.25 * coverage
                    + 0.20 * ink_balance
                    - 0.005 * min(mean_distance, 20.0)
                ),
            }
        )
    return result


def trajectory_model_support(
    sample: Any,
    image_size: int,
    padding: int,
    width_px: float,
) -> np.ndarray:
    """Rasterize per-stroke segments at the renderer's maximum useful width."""
    if width_px <= 0:
        raise ValueError("model support width must be positive")
    normalized = normalize_trajectory_xy(
        sample, canvas_size=image_size, padding=padding
    )
    image = Image.new("L", (image_size, image_size), 0)
    draw = ImageDraw.Draw(image)
    width = max(1, int(round(width_px)))
    radius = max(1, width // 2)
    for stroke in normalized:
        points = [(float(x), float(y)) for x, y in stroke]
        if not points:
            continue
        if len(points) >= 2:
            draw.line(points, fill=255, width=width, joint="curve")
        for x, y in (points[0], points[-1]):
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=255,
            )
    return np.asarray(image, dtype=np.uint8) > 0


def choose_target(
    dataset: CalligraphyImageDataset,
    items: list[dict[str, Any]],
    sample: Any,
    image_size: int,
    padding: int,
    selection: str,
    tolerance_px: int,
    threshold: float,
    model_support_width_px: float = 7.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not items:
        raise ValueError("No target candidates for character")
    trajectory_xy = trajectory_canvas(sample, image_size, padding)
    model_support = trajectory_model_support(
        sample, image_size, padding, model_support_width_px
    )
    candidates: list[tuple[np.ndarray, dict[str, Any], dict[str, float]]] = []
    for item in items:
        canvas, metadata = _sample_image_target(dataset, item, image_size, padding)
        score = target_match_score(
            canvas,
            trajectory_xy,
            tolerance_px,
            threshold,
            model_support=model_support,
        )
        candidates.append((canvas, metadata, score))
    if selection == "first":
        selected = candidates[0]
    elif selection == "best_trajectory_coverage":
        selected = max(
            candidates,
            key=lambda value: (
                value[2]["score"],
                value[2]["coverage"],
                -value[2]["mean_distance_px"],
                str(value[1]["source_image"]),
                int(value[1]["shape_index"]),
            ),
        )
    elif selection == "best_model_support":
        selected = max(
            candidates,
            key=lambda value: (
                value[2]["model_compatibility_score"],
                value[2]["model_support_dice"],
                value[2]["coverage"],
                -value[2]["mean_distance_px"],
                str(value[1]["source_image"]),
                int(value[1]["shape_index"]),
            ),
        )
    else:
        raise ValueError(
            "target selection must be 'first', 'best_trajectory_coverage', "
            "or 'best_model_support'"
        )
    canvas, metadata, score = selected
    metadata = dict(metadata)
    metadata["selection"] = selection
    metadata["candidate_count"] = len(candidates)
    metadata["model_support_width_px"] = float(model_support_width_px)
    metadata["trajectory_match"] = score
    return canvas, metadata


def save_target_image(ink_positive: np.ndarray, path: Path) -> None:
    """Save visually conventional black-ink/white-background PNG."""
    visual = np.rint(np.clip(1.0 - ink_positive, 0.0, 1.0) * 255.0).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(visual, mode="L").save(path)


def target_compatibility_failures(
    metadata: dict[str, Any], args: argparse.Namespace
) -> list[str]:
    """Reject mislabeled/corrupt targets before an expensive pose inversion."""
    if metadata.get("selection") != "best_model_support":
        return []
    match = metadata.get("trajectory_match", {})
    failures: list[str] = []
    if float(match.get("coverage", 0.0)) < args.min_target_coverage:
        failures.append("trajectory_coverage_below_threshold")
    if (
        float(match.get("model_support_dice", 0.0))
        < args.min_model_support_dice
    ):
        failures.append("model_support_dice_below_threshold")
    ink_ratio = float(match.get("target_to_model_support_ink_ratio", 0.0))
    if ink_ratio < args.min_target_support_ink_ratio:
        failures.append("target_support_ink_ratio_too_small")
    if ink_ratio > args.max_target_support_ink_ratio:
        failures.append("target_support_ink_ratio_too_large")
    return failures


def build_skeleton_snap_initial_pose(
    sample: Any,
    target: np.ndarray,
    output_csv: Path,
    image_size: int,
    padding: int,
    threshold: float,
    max_snap_px: float,
    blend: float,
    smooth_sigma: float,
    initial_h_mm: float,
    initial_alpha_deg: float,
    initial_beta_deg: float,
    initial_gamma_deg: float,
) -> dict[str, Any]:
    """Create a bounded skeleton-snapped pose used by the fast H-only stage.

    The original trajectory remains the coordinate reference.  Only a smooth,
    bounded x/y initialization is written to the staged pose CSV; H and the
    angles use the command-line defaults until the differentiable renderer
    refines H and derives alpha/beta/gamma.
    """
    from tools.invert_paper_trajectory import canvas_xy_to_source, save_pose_csv
    from tools.snap_trajectory_to_target import (
        nearest_skeleton_displacements,
        point_to_skeleton_distance,
        smooth_displacements,
        thin_binary,
    )

    if not 0.0 < blend <= 1.0:
        raise ValueError("--snap_blend must be in (0,1]")
    if max_snap_px <= 0.0 or smooth_sigma < 0.0:
        raise ValueError("snap radius must be positive and sigma non-negative")
    xy_canvas = trajectory_canvas(sample, image_size, padding)
    points = sample.all_points()
    if len(xy_canvas) != len(points):
        raise ValueError("Trajectory canvas point count mismatch")
    skeleton = thin_binary(np.asarray(target >= threshold, dtype=bool))
    stroke_ids = np.asarray([point.stroke_id for point in points], dtype=np.int64)
    raw_displacement, before_distance = nearest_skeleton_displacements(
        xy_canvas, skeleton, max_snap_px
    )
    displacement = smooth_displacements(
        raw_displacement, stroke_ids, smooth_sigma
    )
    norms = np.linalg.norm(displacement, axis=1)
    clip = np.minimum(1.0, max_snap_px / np.maximum(norms, 1e-6))
    displacement *= clip[:, None]
    snapped_canvas = xy_canvas + blend * displacement
    after_distance = point_to_skeleton_distance(snapped_canvas, skeleton)
    snapped_source = canvas_xy_to_source(
        sample, snapped_canvas, image_size, padding
    )
    posture = np.tile(
        np.asarray(
            [
                initial_h_mm,
                np.deg2rad(initial_alpha_deg),
                np.deg2rad(initial_beta_deg),
            ],
            dtype=np.float32,
        ),
        (len(points), 1),
    )
    gamma = np.full(
        len(points), np.deg2rad(initial_gamma_deg), dtype=np.float32
    )
    fixed = {
        "source": "command_line_defaults",
        "confidence": "low_simulation",
        "reason": "initial_value_before_fast_h_only_inversion",
    }
    planar = {
        "source": "target_skeleton_initialized",
        "confidence": "low_simulation",
        "reason": "bounded_smooth_local_target_skeleton_snap",
    }
    save_pose_csv(
        sample,
        posture,
        output_csv,
        "paper_declared_radian",
        {
            "H": dict(fixed),
            "alpha": dict(fixed),
            "beta": dict(fixed),
            "gamma": dict(fixed),
            "x": dict(planar),
            "y": dict(planar),
        },
        xy_source=snapped_source,
        gamma=gamma,
        prototype="paper_target_skeleton_initializer_v1",
    )
    applied = np.linalg.norm(snapped_canvas - xy_canvas, axis=1)
    return {
        "format": "paper_target_skeleton_initializer_v1",
        "point_count": int(len(points)),
        "target_skeleton_pixels": int(skeleton.sum()),
        "max_snap_px": float(max_snap_px),
        "blend": float(blend),
        "smooth_sigma": float(smooth_sigma),
        "distance_to_target_skeleton_px": {
            "before_mean": float(before_distance.mean()),
            "before_max": float(before_distance.max()),
            "after_mean": float(after_distance.mean()),
            "after_max": float(after_distance.max()),
        },
        "applied_displacement_px": {
            "mean": float(applied.mean()),
            "max": float(applied.max()),
        },
        "simulation_only": True,
    }


def parse_target_overrides(entries: Iterable[str]) -> dict[str, str]:
    """Parse repeated ``CHAR=IMAGE`` overrides used for canonical targets."""
    overrides: dict[str, str] = {}
    for entry in entries:
        text = str(entry)
        if "=" not in text:
            raise ValueError(
                f"Invalid --target_override {entry!r}; expected CHARACTER=IMAGE_PATH"
            )
        character, path = text.split("=", 1)
        character = character.strip()
        path = path.strip()
        if not character or not path:
            raise ValueError(
                f"Invalid --target_override {entry!r}; character and path are required"
            )
        if character in overrides:
            raise ValueError(f"Duplicate --target_override for {character!r}")
        overrides[character] = path
    return overrides


def load_override_target(path: str, image_size: int, padding: int) -> tuple[np.ndarray, dict[str, Any]]:
    from utils.image_preprocessing import load_character_image

    canvas, transform = load_character_image(path, canvas_size=image_size, padding=padding)
    return canvas.astype(np.float32), {
        "selection": "override",
        "candidate_count": 1,
        "source_image": str(Path(path).expanduser().resolve()),
        "source_json": None,
        "bbox": None,
        "shape_index": None,
        "group_id": None,
        "transform": transform,
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _child_command(
    args: argparse.Namespace,
    character: str,
    sample_id: str,
    target_path: Path,
    output_dir: Path,
    trajectory_csv: str | Path | None = None,
    initial_pose_csv: str | Path | None = None,
) -> list[str]:
    trajectory_csv = trajectory_csv or args.trajectory_csv
    command = [
        args.python,
        "-u",
        str(ROOT / "tools" / "invert_paper_trajectory.py"),
        "--trajectory_csv",
        str(Path(trajectory_csv)),
        "--target_image",
        str(target_path),
        "--bbsmg_ckpt",
        str(Path(args.bbsmg_ckpt)),
        "--character",
        character,
        "--sample_id",
        sample_id,
        "--output_dir",
        str(output_dir),
        "--output_stem",
        "inversion",
        "--device",
        args.device,
        "--image_size",
        str(args.image_size),
        "--padding",
        str(args.padding),
        "--order",
        str(args.order),
        "--max_steps",
        str(args.max_steps),
        "--damping",
        str(args.damping),
        "--optimization_size",
        str(args.optimization_size),
        "--point_batch_size",
        str(args.point_batch_size),
        "--pixel_weight",
        str(args.pixel_weight),
        "--h_smoothness_weight",
        str(args.h_smoothness_weight),
        "--h_point_velocity_weight",
        str(args.h_point_velocity_weight),
        "--h_point_acceleration_weight",
        str(args.h_point_acceleration_weight),
        "--initial_h_mm",
        str(args.initial_h_mm),
        "--initial_alpha_deg",
        str(args.initial_alpha_deg),
        "--initial_beta_deg",
        str(args.initial_beta_deg),
        "--initial_gamma_deg",
        str(args.initial_gamma_deg),
        "--xy_max_offset_px",
        str(args.xy_max_offset_px),
        "--xy_smoothness_weight",
        str(args.xy_smoothness_weight),
        "--xy_prior_weight",
        str(args.xy_prior_weight),
        "--dynamic_profile",
        args.dynamic_profile,
        "--pixels_per_model_unit",
        str(args.pixels_per_model_unit),
        "--patch_floor",
        str(args.patch_floor),
        "--footprint_longitudinal_scale",
        str(args.footprint_longitudinal_scale),
        "--footprint_transverse_scale",
        str(args.footprint_transverse_scale),
        "--render_max_step_px",
        str(args.render_max_step_px),
    ]
    if initial_pose_csv is not None:
        command.extend(
            [
                "--initial_pose_csv",
                str(Path(initial_pose_csv)),
                "--initial_pose_xy_source",
                "csv",
            ]
        )
    if args.optimize_xy:
        command.append("--optimize_xy")
    if args.fused_pose_from_height:
        command.extend(["--fused_pose_from_height", "--field_mode", "h_only"])
    elif args.field_mode:
        command.extend(["--field_mode", args.field_mode])
    if args.optimize_gamma:
        command.extend(["--optimize_gamma", "--gamma_max_abs_deg", str(args.gamma_max_abs_deg)])
    if args.search_orders:
        command.extend(["--search_orders", "--order_min", str(args.order_min), "--order_max", str(args.order_max)])
    if args.cap_order_to_points:
        command.append("--cap_order_to_points")
    return command


def _load_report(output_dir: Path) -> dict[str, Any] | None:
    path = output_dir / "inversion_report.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def run_batch(args: argparse.Namespace) -> dict[str, Any]:
    from datasets.trajectory_dataset import load_trajectory_csv
    from utils.trajectory_processing import write_trajectory_csv

    if args.fused_pose_from_height and args.optimize_gamma:
        raise ValueError("--fused_pose_from_height and --optimize_gamma are mutually exclusive")
    if args.xy_initializer != "none" and args.optimize_xy:
        raise ValueError(
            "--xy_initializer and --optimize_xy are mutually exclusive; "
            "the skeleton initializer is followed by H-only inversion"
        )
    if args.max_characters < 0:
        raise ValueError("--max_characters must be non-negative")
    if args.shard_count < 1:
        raise ValueError("--shard_count must be at least 1")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("--shard_index must be in [0, shard_count)")

    # Child processes run with ``ROOT`` as cwd.  Resolve all user paths once so
    # launching the command from another directory cannot make a character
    # fail merely because a relative path was interpreted twice.
    for name in (
        "trajectory_csv",
        "bbsmg_ckpt",
        "image_dir",
        "json_dir",
        "data_csv",
        "output_dir",
    ):
        setattr(args, name, str(Path(getattr(args, name)).expanduser().resolve()))

    dataset = build_kaishu_dataset(
        args.image_dir,
        args.json_dir,
        args.data_csv,
        args.image_ext,
        args.chirography,
    )
    target_items = group_target_items(dataset)
    exclusions = load_target_exclusions(args.exclude_targets_json)
    target_items, removed_targets = filter_target_items(target_items, exclusions)
    if not target_items:
        raise RuntimeError(
            "The filtered database contains no target characters; verify "
            "the 楷书 data.csv entries and image/json roots"
        )
    target_overrides = {
        character: str(Path(path).expanduser().resolve())
        for character, path in parse_target_overrides(args.target_override).items()
    }
    samples = load_trajectory_csv(args.trajectory_csv)
    selected, duplicate_counts = select_trajectory_samples(
        samples, selection=args.trajectory_selection
    )
    characters = sorted(target_items)
    if args.include_characters:
        allowed = set(args.include_characters)
        characters = [char for char in characters if char in allowed]
    if args.shard_count > 1:
        characters = [
            character
            for position, character in enumerate(characters)
            if position % args.shard_count == args.shard_index
        ]
    # When a bounded smoke run is requested, choose from the actual
    # database/trajectory intersection.  Taking the first N database keys
    # can otherwise select rare characters with no trajectory and produce a
    # misleading all-missing smoke test.  Full runs still retain every
    # database character so missing trajectories remain auditable.
    if args.max_characters > 0:
        characters = [char for char in characters if char in selected]
        characters = characters[: args.max_characters]
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    shard_suffix = (
        f"_shard_{args.shard_index}_of_{args.shard_count}"
        if args.shard_count > 1
        else ""
    )
    manifest_path = output_root / f"manifest{shard_suffix}.jsonl"
    summary: dict[str, Any] = {
        "format": FORMAT,
        "simulation_only": True,
        "trajectory_csv": str(Path(args.trajectory_csv)),
        "bbsmg_checkpoint": str(Path(args.bbsmg_ckpt)),
        "style_filter": {
            "chirography": args.chirography,
            "data_csv": args.data_csv,
            "image_dir": args.image_dir,
            "json_dir": args.json_dir,
            "indexed_target_boxes": len(dataset.index),
            "database_character_count": len(target_items),
            "target_exclusion_file": args.exclude_targets_json,
            "excluded_target_candidates": len(removed_targets),
            "excluded_targets": removed_targets,
        },
        "selection": {
            "trajectory_selection": args.trajectory_selection,
            "target_selection": args.target_selection,
            "target_match_tolerance_px": args.target_match_tolerance_px,
            "target_match_threshold": args.target_match_threshold,
            "model_support_width_px": args.model_support_width_px,
            "min_target_coverage": args.min_target_coverage,
            "min_model_support_dice": args.min_model_support_dice,
            "target_support_ink_ratio_range": [
                args.min_target_support_ink_ratio,
                args.max_target_support_ink_ratio,
            ],
            "target_overrides": target_overrides,
        },
        "parameters": {
            key: getattr(args, key)
            for key in (
                "image_size", "padding", "order", "max_steps", "damping",
                "optimization_size", "point_batch_size", "optimize_xy",
                "fused_pose_from_height", "field_mode", "optimize_gamma",
                "pixel_weight", "h_smoothness_weight",
                "h_point_velocity_weight", "h_point_acceleration_weight",
                "xy_max_offset_px", "xy_initializer", "snap_threshold",
                "snap_max_px", "snap_blend", "snap_smooth_sigma", "device",
            )
        },
        "database_characters": len(target_items),
        "trajectory_characters": len(selected),
        "shard": {
            "index": args.shard_index,
            "count": args.shard_count,
            "manifest": str(manifest_path),
        },
        "requested_characters": len(characters),
        "processed": 0,
        "completed": 0,
        "accepted": 0,
        "low_quality": 0,
        "failed": 0,
        "target_compatible": 0,
        "target_incompatible": 0,
        "skipped_existing": 0,
        "missing_trajectory": [],
        "trajectory_without_kaishu_target": [],
        "duplicate_trajectory_counts": duplicate_counts,
        "characters": [],
    }
    missing_trajectory = sorted(set(target_items) - set(selected))
    summary["missing_trajectory"] = missing_trajectory
    extra_trajectory_characters = sorted(set(selected) - set(target_items))
    summary["trajectory_without_kaishu_target"] = extra_trajectory_characters
    # The requested universe is the filtered database.  Extra trajectory
    # characters (for example 隶书 or 行楷 samples in the same CSV) are
    # expected and are intentionally ignored, so strict mode only rejects a
    # database 楷书 character with no matching trajectory.
    if args.require_complete and missing_trajectory:
        raise RuntimeError(
            "Some database 楷书 characters have no trajectory; "
            f"missing_trajectory={len(missing_trajectory)}"
        )
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index, character in enumerate(characters):
            summary["processed"] += 1
            char_dir = output_root / character_stem(character)
            char_dir.mkdir(parents=True, exist_ok=True)
            sample = selected.get(character)
            record: dict[str, Any] = {
                "index": index,
                "character": character,
                "output_dir": str(char_dir),
                "status": "failed",
                "trajectory_sample_id": None if sample is None else sample.meta.get("sample_id"),
                "trajectory_point_count": 0 if sample is None else _sample_point_count(sample),
            }
            if sample is None:
                record.update({"status": "missing_trajectory", "error": "no trajectory sample for this database character"})
                summary["failed"] += 1
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                summary["characters"].append(record)
                continue
            output_csv = char_dir / "pose_refined.csv"
            inversion_report = _load_report(char_dir)
            if args.resume and output_csv.exists() and inversion_report is not None:
                record.update({
                    "status": "skipped_existing",
                    "pose_csv": str(output_csv),
                    "inversion_report": str(char_dir / "inversion_report.json"),
                })
                summary["skipped_existing"] += 1
                summary["completed"] += 1
                metrics = inversion_report.get("metrics", {})
                if float(metrics.get("iou_at_0.5", 0.0)) >= args.accept_iou:
                    summary["accepted"] += 1
                else:
                    summary["low_quality"] += 1
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                summary["characters"].append(record)
                print(
                    f"[BATCH] {index + 1}/{len(characters)} {character} "
                    "status=skipped_existing",
                    flush=True,
                )
                continue
            try:
                if character in target_overrides:
                    target_canvas, target_meta = load_override_target(
                        target_overrides[character], args.image_size, args.padding
                    )
                else:
                    target_canvas, target_meta = choose_target(
                        dataset,
                        target_items[character],
                        sample,
                        args.image_size,
                        args.padding,
                        args.target_selection,
                        args.target_match_tolerance_px,
                        args.target_match_threshold,
                        args.model_support_width_px,
                    )
                target_path = char_dir / "target.png"
                save_target_image(target_canvas, target_path)
                _write_json(char_dir / "target_source.json", target_meta)
                record.update({"target_image": str(target_path), "target_source": target_meta})
                compatibility_failures = target_compatibility_failures(
                    target_meta, args
                )
                record["target_compatibility"] = {
                    "accepted": not compatibility_failures,
                    "failures": compatibility_failures,
                }
                if compatibility_failures:
                    record.update(
                        {
                            "status": "target_incompatible",
                            "error": ",".join(compatibility_failures),
                        }
                    )
                    summary["target_incompatible"] += 1
                    summary["failed"] += 1
                    manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                    summary["characters"].append(record)
                    print(
                        f"[BATCH] {index + 1}/{len(characters)} {character} "
                        "status=target_incompatible",
                        flush=True,
                    )
                    continue
                summary["target_compatible"] += 1
                if args.target_preflight_only:
                    record["status"] = "target_compatible"
                    manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                    summary["characters"].append(record)
                    print(
                        f"[BATCH] {index + 1}/{len(characters)} {character} "
                        "status=target_compatible",
                        flush=True,
                    )
                    continue
            except Exception as exc:  # per-character failure must not lose the batch
                record.update({"status": "target_failed", "error": f"{type(exc).__name__}: {exc}"})
                summary["failed"] += 1
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                summary["characters"].append(record)
                continue
            input_trajectory_path = char_dir / "input_trajectory.csv"
            if not input_trajectory_path.exists():
                write_trajectory_csv(sample, input_trajectory_path)
            record["trajectory_input"] = str(input_trajectory_path)
            initial_pose_path = None
            if args.xy_initializer == "skeleton_snap":
                initial_pose_path = char_dir / "skeleton_snap_initial_pose.csv"
                snap_report = build_skeleton_snap_initial_pose(
                    sample,
                    target_canvas,
                    initial_pose_path,
                    args.image_size,
                    args.padding,
                    args.snap_threshold,
                    args.snap_max_px,
                    args.snap_blend,
                    args.snap_smooth_sigma,
                    args.initial_h_mm,
                    args.initial_alpha_deg,
                    args.initial_beta_deg,
                    args.initial_gamma_deg,
                )
                _write_json(char_dir / "skeleton_snap_report.json", snap_report)
                record["xy_initializer"] = {
                    "pose_csv": str(initial_pose_path),
                    "report": snap_report,
                }
            command = _child_command(
                args,
                character,
                str(sample.meta.get("sample_id", "")),
                target_path,
                char_dir,
                trajectory_csv=input_trajectory_path,
                initial_pose_csv=initial_pose_path,
            )
            log_path = char_dir / "inversion.log"
            record["command"] = shlex.join(command)
            try:
                with log_path.open("w", encoding="utf-8") as log:
                    completed = subprocess.run(
                        command,
                        cwd=str(ROOT),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                        timeout=args.timeout_seconds if args.timeout_seconds > 0 else None,
                    )
                inversion_report = _load_report(char_dir)
                generated = char_dir / "inversion_trajectory.csv"
                if completed.returncode != 0 or inversion_report is None or not generated.exists():
                    raise RuntimeError(
                        f"inversion failed returncode={completed.returncode}; "
                        f"see {log_path}"
                    )
                generated.replace(output_csv)
                # The child uses a generic stem, while the batch contract uses
                # one stable filename for every character.  Keep the report
                # path in sync after this atomic rename.
                inversion_report["pose_output"] = str(output_csv)
                inversion_report["batch_pose_csv"] = str(output_csv)
                _write_json(char_dir / "inversion_report.json", inversion_report)
                record.update({
                    "status": "completed",
                    "pose_csv": str(output_csv),
                    "inversion_report": str(char_dir / "inversion_report.json"),
                    "returncode": int(completed.returncode),
                    "metrics": inversion_report.get("metrics", {}),
                })
                summary["completed"] += 1
                iou = float(inversion_report.get("metrics", {}).get("iou_at_0.5", 0.0))
                if iou >= args.accept_iou:
                    record["quality"] = "accepted"
                    summary["accepted"] += 1
                else:
                    record["quality"] = "low_quality"
                    summary["low_quality"] += 1
            except Exception as exc:
                record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}", "log": str(log_path)})
                summary["failed"] += 1
                if args.fail_fast:
                    manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                    summary["characters"].append(record)
                    raise
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            summary["characters"].append(record)
            print(
                f"[BATCH] {index + 1}/{len(characters)} {character} "
                f"status={record['status']}",
                flush=True,
            )
    summary["completed_fraction"] = (
        summary["completed"] / summary["requested_characters"]
        if summary["requested_characters"]
        else 0.0
    )
    _write_json(output_root / f"batch_summary{shard_suffix}.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Invert one refined pose CSV for every strictly 楷书 database character"
    )
    parser.add_argument("--trajectory_csv", required=True)
    parser.add_argument("--bbsmg_ckpt", required=True)
    parser.add_argument("--output_dir", default="outputs/kaishu_pose_trajectory_batch_v1")
    parser.add_argument("--image_dir", default="data/raw/images")
    parser.add_argument("--json_dir", default="data/raw/json_files")
    parser.add_argument("--data_csv", default="data/raw/data.csv")
    parser.add_argument("--image_ext", default=".jpg")
    parser.add_argument("--chirography", default="楷")
    parser.add_argument(
        "--exclude_targets_json",
        default=None,
        help="optional exact image/bbox exclusion list produced by target audits",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--padding", type=int, default=16)
    parser.add_argument("--trajectory_selection", choices=["longest", "first"], default="longest")
    parser.add_argument(
        "--target_selection",
        choices=["first", "best_trajectory_coverage", "best_model_support"],
        default="best_model_support",
    )
    parser.add_argument(
        "--target_override",
        action="append",
        default=[],
        metavar="CHAR=IMAGE",
        help="use an explicit canonical target image for one character; repeatable",
    )
    parser.add_argument("--target_match_tolerance_px", type=int, default=5)
    parser.add_argument("--target_match_threshold", type=float, default=0.35)
    parser.add_argument(
        "--model_support_width_px",
        type=float,
        default=7.0,
        help=(
            "maximum useful full brush width used only to rank same-character "
            "database targets for renderer compatibility"
        ),
    )
    parser.add_argument("--min_target_coverage", type=float, default=0.75)
    parser.add_argument("--min_model_support_dice", type=float, default=0.30)
    parser.add_argument(
        "--min_target_support_ink_ratio", type=float, default=0.45
    )
    parser.add_argument(
        "--max_target_support_ink_ratio", type=float, default=1.65
    )
    parser.add_argument(
        "--target_preflight_only",
        action="store_true",
        help="select and audit targets without launching pose inversion",
    )
    parser.add_argument("--accept_iou", type=float, default=0.70)
    parser.add_argument("--max_characters", type=int, default=0, help="0 means all matched 楷书 characters")
    parser.add_argument("--include_characters", nargs="*", default=None)
    parser.add_argument(
        "--shard_count",
        type=int,
        default=1,
        help="number of disjoint deterministic character shards",
    )
    parser.add_argument(
        "--shard_index",
        type=int,
        default=0,
        help="zero-based shard processed by this coordinator",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--require_complete", action="store_true")
    parser.add_argument("--fail_fast", action="store_true")
    parser.add_argument("--timeout_seconds", type=int, default=0)
    parser.add_argument("--order", type=int, default=5)
    parser.add_argument("--search_orders", action="store_true")
    parser.add_argument("--order_min", type=int, default=3)
    parser.add_argument("--order_max", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=15)
    parser.add_argument("--damping", type=float, default=0.05)
    parser.add_argument("--optimization_size", type=int, default=64)
    parser.add_argument("--point_batch_size", type=int, default=64)
    parser.add_argument("--pixel_weight", type=float, default=12.0)
    parser.add_argument("--h_smoothness_weight", type=float, default=0.2)
    parser.add_argument("--h_point_velocity_weight", type=float, default=5.0)
    parser.add_argument("--h_point_acceleration_weight", type=float, default=10.0)
    parser.add_argument("--initial_h_mm", type=float, default=15.5)
    parser.add_argument("--initial_alpha_deg", type=float, default=0.0)
    parser.add_argument("--initial_beta_deg", type=float, default=0.0)
    parser.add_argument("--initial_gamma_deg", type=float, default=0.0)
    parser.add_argument("--optimize_xy", action="store_true")
    parser.add_argument(
        "--xy_initializer",
        choices=["none", "skeleton_snap"],
        default="none",
        help=(
            "optional bounded smooth target-skeleton x/y initializer; "
            "use without --optimize_xy for the fast H-only stage"
        ),
    )
    parser.add_argument("--snap_threshold", type=float, default=0.5)
    parser.add_argument("--snap_max_px", type=float, default=4.0)
    parser.add_argument("--snap_blend", type=float, default=0.75)
    parser.add_argument("--snap_smooth_sigma", type=float, default=0.75)
    parser.add_argument("--xy_max_offset_px", type=float, default=4.0)
    parser.add_argument("--xy_smoothness_weight", type=float, default=1.0)
    parser.add_argument("--xy_prior_weight", type=float, default=0.15)
    parser.add_argument("--fused_pose_from_height", action="store_true")
    parser.add_argument("--field_mode", choices=["auto", "all", "h_only", "xy_only"], default=None)
    parser.add_argument("--optimize_gamma", action="store_true")
    parser.add_argument("--gamma_max_abs_deg", type=float, default=180.0)
    parser.add_argument("--cap_order_to_points", action="store_true")
    parser.add_argument("--dynamic_profile", default="wang2020_figure4_digitized_v1")
    parser.add_argument("--pixels_per_model_unit", type=float, default=20.0)
    parser.add_argument("--patch_floor", type=float, default=0.05)
    parser.add_argument("--footprint_longitudinal_scale", type=float, default=0.22)
    parser.add_argument("--footprint_transverse_scale", type=float, default=0.262)
    parser.add_argument("--render_max_step_px", type=float, default=2.0)
    return parser


if __name__ == "__main__":
    run_batch(build_parser().parse_args())
