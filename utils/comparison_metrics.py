import csv
from pathlib import Path
from typing import Dict, List

import numpy as np

from models.geometry import resample_stroke
from utils.types import CharacterTrajectory


def _filter2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    padding_y = kernel.shape[0] // 2
    padding_x = kernel.shape[1] // 2
    padded = np.pad(
        image,
        ((padding_y, padding_y), (padding_x, padding_x)),
        mode="constant",
    )
    windows = np.lib.stride_tricks.sliding_window_view(padded, kernel.shape)
    return np.einsum("ijkl,kl->ij", windows, kernel, optimize=True)


def _ssim_score(
    prediction: np.ndarray,
    target: np.ndarray,
    window_size: int = 11,
    sigma: float = 1.5,
    eps: float = 1e-6,
) -> float:
    coords = np.arange(window_size, dtype=np.float64) - window_size // 2
    gaussian = np.exp(-(coords ** 2) / (2 * sigma ** 2))
    gaussian /= gaussian.sum()
    window = np.outer(gaussian, gaussian)
    mu_x = _filter2d(prediction, window)
    mu_y = _filter2d(target, window)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x2 = _filter2d(prediction * prediction, window) - mu_x2
    sigma_y2 = _filter2d(target * target, window) - mu_y2
    sigma_xy = _filter2d(prediction * target, window) - mu_xy
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    score = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + eps
    )
    return float(np.clip(score.mean(), 0.0, 1.0))


def image_metrics(prediction: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    prediction = np.clip(np.asarray(prediction, dtype=np.float64), 0.0, 1.0)
    target = np.clip(np.asarray(target, dtype=np.float64), 0.0, 1.0)
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction and target shapes differ: {prediction.shape} != {target.shape}"
        )
    difference = prediction - target
    foreground = target > 0.1
    prediction_binary = prediction >= 0.5
    target_binary = target >= 0.5
    intersection = float(np.logical_and(prediction_binary, target_binary).sum())
    prediction_sum = float(prediction_binary.sum())
    target_sum = float(target_binary.sum())
    union = float(np.logical_or(prediction_binary, target_binary).sum())

    prediction_mean = float(prediction.mean())
    target_mean = float(target.mean())
    prediction_variance = float(prediction.var())
    target_variance = float(target.var())
    covariance = float(
        ((prediction - prediction_mean) * (target - target_mean)).mean()
    )
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    global_ssim = (
        (2 * prediction_mean * target_mean + c1) * (2 * covariance + c2)
    ) / (
        (prediction_mean ** 2 + target_mean ** 2 + c1)
        * (prediction_variance + target_variance + c2)
    )
    soft_intersection = float((prediction * target).sum())
    soft_union = float(prediction.sum() + target.sum())
    return {
        "mse": float(np.mean(difference ** 2)),
        "mae": float(np.mean(np.abs(difference))),
        "foreground_mae": (
            float(np.mean(np.abs(difference[foreground])))
            if np.any(foreground)
            else float(np.mean(np.abs(difference)))
        ),
        "ssim_score": _ssim_score(prediction, target),
        "global_ssim": float(np.clip(global_ssim, -1.0, 1.0)),
        "dice_score": (2.0 * soft_intersection + 1e-6) / (soft_union + 1e-6),
        "dice_at_0.5": (2.0 * intersection + 1e-6) / (
            prediction_sum + target_sum + 1e-6
        ),
        "iou_at_0.5": (intersection + 1e-6) / (union + 1e-6),
        "ink_mean": prediction_mean,
        "target_ink_mean": target_mean,
        "ink_delta": prediction_mean - target_mean,
    }


def signed_difference_image(
    generated: np.ndarray, target: np.ndarray
) -> np.ndarray:
    generated = np.asarray(generated, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    false_positive = np.clip(generated - target, 0.0, 1.0)
    false_negative = np.clip(target - generated, 0.0, 1.0)
    agreement = np.minimum(generated, target)
    return np.stack([false_positive, agreement, false_negative], axis=-1)


def trajectory_metrics(
    initial: CharacterTrajectory,
    generated: CharacterTrajectory,
    points_per_stroke: int,
) -> Dict[str, float]:
    initial_by_id = {stroke.stroke_id: stroke for stroke in initial.sorted_strokes()}
    generated_by_id = {
        stroke.stroke_id: stroke for stroke in generated.sorted_strokes()
    }
    common_ids = sorted(set(initial_by_id) & set(generated_by_id))
    xyz_differences: List[np.ndarray] = []
    angle_differences: List[np.ndarray] = []
    for stroke_id in common_ids:
        left = resample_stroke(initial_by_id[stroke_id], points_per_stroke)
        right = resample_stroke(generated_by_id[stroke_id], points_per_stroke)
        left_values = np.asarray(
            [point.as_tuple() for point in left.sorted_points()], dtype=np.float64
        )
        right_values = np.asarray(
            [point.as_tuple() for point in right.sorted_points()], dtype=np.float64
        )
        xyz_differences.append(right_values[:, :3] - left_values[:, :3])
        angle_differences.append(
            np.arctan2(
                np.sin(right_values[:, 3:] - left_values[:, 3:]),
                np.cos(right_values[:, 3:] - left_values[:, 3:]),
            )
        )
    if not xyz_differences:
        raise ValueError("Initial and generated trajectories share no stroke IDs")
    xyz = np.concatenate(xyz_differences)
    angles = np.concatenate(angle_differences)
    xyz_norm = np.linalg.norm(xyz, axis=1)
    angle_norm = np.linalg.norm(angles, axis=1)
    return {
        "common_strokes": len(common_ids),
        "initial_strokes": len(initial.sorted_strokes()),
        "generated_strokes": len(generated.sorted_strokes()),
        "points_compared": int(len(xyz)),
        "xyz_rmse": float(np.sqrt(np.mean(xyz ** 2))),
        "xyz_mean_distance": float(xyz_norm.mean()),
        "xyz_max_distance": float(xyz_norm.max()),
        "x_mae": float(np.abs(xyz[:, 0]).mean()),
        "y_mae": float(np.abs(xyz[:, 1]).mean()),
        "z_mae": float(np.abs(xyz[:, 2]).mean()),
        "angle_rmse_radians": float(np.sqrt(np.mean(angles ** 2))),
        "angle_mean_distance_radians": float(angle_norm.mean()),
        "angle_max_distance_radians": float(angle_norm.max()),
    }


def read_comparison_manifest(path: str) -> List[Dict[str, str]]:
    manifest_path = Path(path).resolve()
    with open(manifest_path, "r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    required = {"sample_id", "target_image"}
    missing = required - set(rows[0] if rows else {})
    if missing:
        raise ValueError(f"Comparison manifest is missing columns: {sorted(missing)}")
    for row in rows:
        if not str(row.get("sample_id", "")).strip():
            raise ValueError("Comparison manifest contains an empty sample_id")
        image_path = Path(row["target_image"])
        if not image_path.is_absolute():
            cwd_candidate = (Path.cwd() / image_path).resolve()
            manifest_candidate = (manifest_path.parent / image_path).resolve()
            image_path = (
                cwd_candidate if cwd_candidate.exists() else manifest_candidate
            )
        if not image_path.is_file():
            raise FileNotFoundError(f"Target image not found: {image_path}")
        row["target_image"] = str(image_path.resolve())
    return rows
