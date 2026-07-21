from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image


def _weighted_moments(weights: np.ndarray) -> Tuple[float, float, float, float]:
    weights = np.asarray(weights, dtype=np.float64)
    total = float(weights.sum())
    height, width = weights.shape
    if total <= 1e-8:
        return (width - 1) / 2.0, (height - 1) / 2.0, 1.0, 1.0
    ys, xs = np.indices(weights.shape, dtype=np.float64)
    center_x = float((weights * xs).sum() / total)
    center_y = float((weights * ys).sum() / total)
    std_x = float(np.sqrt((weights * (xs - center_x) ** 2).sum() / total + 1e-8))
    std_y = float(np.sqrt((weights * (ys - center_y) ** 2).sum() / total + 1e-8))
    return center_x, center_y, std_x, std_y


def transform_target(
    target: np.ndarray,
    scale: float,
    shift_x: int,
    shift_y: int,
) -> np.ndarray:
    """Scale around the canvas center, then translate without wrapping."""
    target = np.asarray(target, dtype=np.float32)
    height, width = target.shape
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = Image.fromarray(
        np.clip(target * 255.0, 0, 255).astype(np.uint8), mode="L"
    ).resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    resized_array = np.asarray(resized, dtype=np.float32) / 255.0

    left = (width - resized_width) // 2 + int(shift_x)
    top = (height - resized_height) // 2 + int(shift_y)
    dst_left = max(left, 0)
    dst_top = max(top, 0)
    dst_right = min(left + resized_width, width)
    dst_bottom = min(top + resized_height, height)
    canvas = np.zeros((height, width), dtype=np.float32)
    if dst_right <= dst_left or dst_bottom <= dst_top:
        return canvas
    src_left = dst_left - left
    src_top = dst_top - top
    canvas[dst_top:dst_bottom, dst_left:dst_right] = resized_array[
        src_top:src_top + (dst_bottom - dst_top),
        src_left:src_left + (dst_right - dst_left),
    ]
    return canvas


def alignment_metrics(
    target: np.ndarray,
    centerline: np.ndarray,
    proximity: np.ndarray,
    threshold: float = 0.5,
    support_threshold: float = 0.25,
) -> Dict[str, float]:
    target_binary = np.asarray(target) >= threshold
    centerline_binary = np.asarray(centerline) >= threshold
    support_binary = np.asarray(proximity) >= support_threshold
    centerline_count = int(centerline_binary.sum())
    target_count = int(target_binary.sum())
    support_count = int(support_binary.sum())
    coverage = float(
        np.logical_and(target_binary, centerline_binary).sum() / max(centerline_count, 1)
    )
    intersection = int(np.logical_and(target_binary, support_binary).sum())
    support_dice = float(
        (2.0 * intersection + 1e-6) / (target_count + support_count + 1e-6)
    )
    score = 0.7 * coverage + 0.3 * support_dice
    return {
        "coverage": coverage,
        "support_dice": support_dice,
        "score": score,
        "target_ink_fraction": float(target_binary.mean()),
        "background_median": float(np.median(np.asarray(target)[~support_binary]))
        if np.any(~support_binary)
        else 0.0,
    }


def align_target_to_trajectory(
    target: np.ndarray,
    centerline: np.ndarray,
    proximity: np.ndarray,
    min_scale: float = 0.60,
    max_scale: float = 1.25,
    local_shift: int = 4,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Register a cleaned target to a fixed trajectory using scale/translation."""
    target = np.asarray(target, dtype=np.float32)
    before = alignment_metrics(target, centerline, proximity)
    target_x, target_y, target_std_x, target_std_y = _weighted_moments(target)
    reference_weights = np.asarray(proximity, dtype=np.float32) ** 2
    ref_x, ref_y, ref_std_x, ref_std_y = _weighted_moments(reference_weights)
    scale_x = ref_std_x / max(target_std_x, 1e-6)
    scale_y = ref_std_y / max(target_std_y, 1e-6)
    base_scale = float(np.clip(np.sqrt(scale_x * scale_y), min_scale, max_scale))
    canvas_center_x = (target.shape[1] - 1) / 2.0
    canvas_center_y = (target.shape[0] - 1) / 2.0

    best_target = target
    best_metrics = before
    best_transform = {"scale": 1.0, "shift_x": 0, "shift_y": 0}
    for scale_factor in (0.90, 1.0, 1.10):
        scale = float(np.clip(base_scale * scale_factor, min_scale, max_scale))
        scaled_center_x = canvas_center_x + scale * (target_x - canvas_center_x)
        scaled_center_y = canvas_center_y + scale * (target_y - canvas_center_y)
        base_shift_x = int(round(ref_x - scaled_center_x))
        base_shift_y = int(round(ref_y - scaled_center_y))
        for delta_y in (-local_shift, 0, local_shift):
            for delta_x in (-local_shift, 0, local_shift):
                shift_x = base_shift_x + delta_x
                shift_y = base_shift_y + delta_y
                candidate = transform_target(target, scale, shift_x, shift_y)
                metrics = alignment_metrics(candidate, centerline, proximity)
                if metrics["score"] > best_metrics["score"]:
                    best_target = candidate
                    best_metrics = metrics
                    best_transform = {
                        "scale": scale,
                        "shift_x": shift_x,
                        "shift_y": shift_y,
                    }

    report: Dict[str, Any] = {
        "version": 1,
        "before": before,
        "after": best_metrics,
        **best_transform,
    }
    return np.clip(best_target, 0.0, 1.0), report
