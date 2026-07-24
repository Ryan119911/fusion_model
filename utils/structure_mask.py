from collections import deque
from typing import Any, Dict, Tuple

import numpy as np


STRUCTURE_TARGET_MODE = "binary_structure_mask"


def binary_dilate(binary: np.ndarray, radius: int = 1) -> np.ndarray:
    binary = np.asarray(binary, dtype=bool)
    if radius <= 0:
        return binary.copy()
    height, width = binary.shape
    padded = np.pad(binary, radius, mode="constant", constant_values=False)
    return np.logical_or.reduce([
        padded[
            offset_y:offset_y + height,
            offset_x:offset_x + width,
        ]
        for offset_y in range(2 * radius + 1)
        for offset_x in range(2 * radius + 1)
    ])


def binary_erode(binary: np.ndarray, radius: int = 1) -> np.ndarray:
    binary = np.asarray(binary, dtype=bool)
    if radius <= 0:
        return binary.copy()
    height, width = binary.shape
    padded = np.pad(binary, radius, mode="constant", constant_values=False)
    return np.logical_and.reduce([
        padded[
            offset_y:offset_y + height,
            offset_x:offset_x + width,
        ]
        for offset_y in range(2 * radius + 1)
        for offset_x in range(2 * radius + 1)
    ])


def binary_opening(binary: np.ndarray, iterations: int = 1) -> np.ndarray:
    result = np.asarray(binary, dtype=bool).copy()
    for _ in range(max(int(iterations), 0)):
        result = binary_dilate(binary_erode(result, radius=1), radius=1)
    return result


def skeletonize_binary(binary: np.ndarray, max_iterations: int = 128) -> np.ndarray:
    """Vectorized Zhang-Suen thinning for target/trajectory geometry checks."""
    image = np.pad(np.asarray(binary, dtype=np.uint8), 1)
    for _ in range(max_iterations):
        changed = False
        for step in (0, 1):
            p2 = image[:-2, 1:-1]
            p3 = image[:-2, 2:]
            p4 = image[1:-1, 2:]
            p5 = image[2:, 2:]
            p6 = image[2:, 1:-1]
            p7 = image[2:, :-2]
            p8 = image[1:-1, :-2]
            p9 = image[:-2, :-2]
            center = image[1:-1, 1:-1]
            neighbor_count = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            transitions = (
                ((p2 == 0) & (p3 == 1)).astype(np.uint8)
                + ((p3 == 0) & (p4 == 1)).astype(np.uint8)
                + ((p4 == 0) & (p5 == 1)).astype(np.uint8)
                + ((p5 == 0) & (p6 == 1)).astype(np.uint8)
                + ((p6 == 0) & (p7 == 1)).astype(np.uint8)
                + ((p7 == 0) & (p8 == 1)).astype(np.uint8)
                + ((p8 == 0) & (p9 == 1)).astype(np.uint8)
                + ((p9 == 0) & (p2 == 1)).astype(np.uint8)
            )
            removable = (
                (center == 1)
                & (neighbor_count >= 2)
                & (neighbor_count <= 6)
                & (transitions == 1)
            )
            if step == 0:
                removable &= (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                removable &= (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            if np.any(removable):
                center[removable] = 0
                changed = True
        if not changed:
            break
    return image[1:-1, 1:-1].astype(bool)


def symmetric_structure_metrics(
    target: Any,
    centerline: Any,
    proximity: Any,
    support_threshold: float = 0.25,
    skeleton_tolerance: int = 5,
) -> Dict[str, float]:
    """Measure both target→trajectory and trajectory→target skeleton agreement."""
    target_binary = np.asarray(target) >= 0.5
    centerline_binary = np.asarray(centerline) >= 0.5
    support_binary = np.asarray(proximity) >= float(support_threshold)
    target_skeleton = skeletonize_binary(target_binary)
    skeleton_neighborhood = binary_dilate(
        target_skeleton,
        radius=int(skeleton_tolerance),
    )
    target_skeleton_count = int(target_skeleton.sum())
    centerline_count = int(centerline_binary.sum())
    target_to_trajectory = float(
        np.logical_and(target_skeleton, support_binary).sum()
        / max(target_skeleton_count, 1)
    )
    trajectory_to_target = float(
        np.logical_and(centerline_binary, skeleton_neighborhood).sum()
        / max(centerline_count, 1)
    )
    symmetric_score = float(
        2.0 * target_to_trajectory * trajectory_to_target
        / max(target_to_trajectory + trajectory_to_target, 1e-6)
    )
    return {
        "target_skeleton_in_support_fraction": target_to_trajectory,
        "trajectory_near_target_skeleton_fraction": trajectory_to_target,
        "symmetric_skeleton_score": symmetric_score,
        "target_skeleton_pixels": target_skeleton_count,
        "skeleton_tolerance": int(skeleton_tolerance),
    }


def _remove_small_components(
    binary: np.ndarray,
    min_component_pixels: int,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Remove isolated 8-connected ink components smaller than the requested area."""
    binary = np.asarray(binary, dtype=bool)
    if min_component_pixels <= 1:
        return binary.copy(), {
            "components_total": 0,
            "components_removed": 0,
            "pixels_removed": 0,
        }

    height, width = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    cleaned = binary.copy()
    components_total = 0
    components_removed = 0
    pixels_removed = 0
    for start_y, start_x in zip(*np.nonzero(binary)):
        if visited[start_y, start_x]:
            continue
        components_total += 1
        queue = deque([(int(start_y), int(start_x))])
        visited[start_y, start_x] = True
        component = []
        while queue:
            y, x = queue.popleft()
            component.append((y, x))
            for neighbor_y in range(max(0, y - 1), min(height, y + 2)):
                for neighbor_x in range(max(0, x - 1), min(width, x + 2)):
                    if (
                        binary[neighbor_y, neighbor_x]
                        and not visited[neighbor_y, neighbor_x]
                    ):
                        visited[neighbor_y, neighbor_x] = True
                        queue.append((neighbor_y, neighbor_x))
        if len(component) < min_component_pixels:
            components_removed += 1
            pixels_removed += len(component)
            for y, x in component:
                cleaned[y, x] = False

    return cleaned, {
        "components_total": components_total,
        "components_removed": components_removed,
        "pixels_removed": pixels_removed,
    }


def build_structure_mask(
    target: Any,
    threshold: float = 0.35,
    min_component_pixels: int = 8,
    opening_iterations: int = 1,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Convert a normalized ink-positive target into a clean binary glyph mask."""
    if not 0.0 < threshold < 1.0:
        raise ValueError("structure threshold must satisfy 0 < value < 1")
    if min_component_pixels < 0:
        raise ValueError("min_component_pixels must be non-negative")
    if opening_iterations < 0:
        raise ValueError("opening_iterations must be non-negative")
    array = np.asarray(target, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"structure target must be 2-D, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("structure target contains NaN or Inf")

    array = np.clip(array, 0.0, 1.0)
    initial = array >= float(threshold)
    opened = binary_opening(initial, iterations=opening_iterations)
    cleaned, component_info = _remove_small_components(
        opened,
        min_component_pixels=int(min_component_pixels),
    )
    if not np.any(cleaned):
        raise ValueError("structure cleanup removed all foreground pixels")

    mask = cleaned.astype(np.float32)
    info: Dict[str, Any] = {
        "mode": STRUCTURE_TARGET_MODE,
        "threshold": float(threshold),
        "min_component_pixels": int(min_component_pixels),
        "opening_iterations": int(opening_iterations),
        "foreground_pixels_before": int(initial.sum()),
        "foreground_pixels_after_opening": int(opened.sum()),
        "foreground_pixels_after": int(cleaned.sum()),
        "foreground_fraction": float(cleaned.mean()),
        "gray_transition_fraction": float(
            np.logical_and(array > 0.1, array < 0.9).mean()
        ),
        **component_info,
    }
    return mask, info
