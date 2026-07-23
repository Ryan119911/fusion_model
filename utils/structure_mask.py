from collections import deque
from typing import Any, Dict, Tuple

import numpy as np


STRUCTURE_TARGET_MODE = "binary_structure_mask"


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
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Convert a normalized ink-positive target into a clean binary glyph mask."""
    if not 0.0 < threshold < 1.0:
        raise ValueError("structure threshold must satisfy 0 < value < 1")
    if min_component_pixels < 0:
        raise ValueError("min_component_pixels must be non-negative")
    array = np.asarray(target, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"structure target must be 2-D, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("structure target contains NaN or Inf")

    array = np.clip(array, 0.0, 1.0)
    initial = array >= float(threshold)
    cleaned, component_info = _remove_small_components(
        initial,
        min_component_pixels=int(min_component_pixels),
    )
    if not np.any(cleaned):
        raise ValueError("structure cleanup removed all foreground pixels")

    mask = cleaned.astype(np.float32)
    info: Dict[str, Any] = {
        "mode": STRUCTURE_TARGET_MODE,
        "threshold": float(threshold),
        "min_component_pixels": int(min_component_pixels),
        "foreground_pixels_before": int(initial.sum()),
        "foreground_pixels_after": int(cleaned.sum()),
        "foreground_fraction": float(cleaned.mean()),
        "gray_transition_fraction": float(
            np.logical_and(array > 0.1, array < 0.9).mean()
        ),
        **component_info,
    }
    return mask, info
