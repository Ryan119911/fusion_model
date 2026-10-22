"""Attribute original-target forward-image residuals to trajectory strokes.

This is diagnostic only.  Overlapping strokes cannot be uniquely separated in
the target image, so target pixels are assigned to the nearest sampled path.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _nearest_stroke_regions(
    shape: tuple[int, int], xy_m: np.ndarray, stroke_ids: np.ndarray,
    origin_m: np.ndarray, pixel_size_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(pixel_size_m) or pixel_size_m <= 0:
        raise ValueError("invalid neural ink pixel size")
    height, width = shape
    yy, xx = np.indices(shape, dtype=np.float64)
    pixels = np.stack((
        origin_m[0] + xx.ravel() * pixel_size_m,
        origin_m[1] - yy.ravel() * pixel_size_m,
    ), axis=1)
    nearest_squared = np.full(len(pixels), np.inf, dtype=np.float64)
    nearest_stroke = np.full(len(pixels), -1, dtype=np.int64)
    for start in range(0, len(xy_m), 128):
        points = xy_m[start:start + 128]
        squared = np.sum((pixels[:, None, :] - points[None, :, :]) ** 2, axis=2)
        index = np.argmin(squared, axis=1)
        distance = squared[np.arange(len(pixels)), index]
        improved = distance < nearest_squared
        nearest_squared[improved] = distance[improved]
        nearest_stroke[improved] = stroke_ids[start + index[improved]]
    return nearest_stroke.reshape(shape), np.sqrt(nearest_squared).reshape(shape) / pixel_size_m


def _centroid(mask: np.ndarray) -> np.ndarray | None:
    yy, xx = np.nonzero(mask)
    if not len(xx):
        return None
    return np.asarray((float(xx.mean()), float(yy.mean())))


def attribute_stroke_residuals(
    frames: np.ndarray, xy_m: np.ndarray, stroke_ids: np.ndarray,
    origin_m: np.ndarray, pixel_size_m: float, target_ink: np.ndarray,
    *, threshold: float = 0.5, support_radius_px: float = 7.0,
) -> dict[str, Any]:
    frames = np.asarray(frames, dtype=np.float32)
    xy_m = np.asarray(xy_m, dtype=np.float64)
    stroke_ids = np.asarray(stroke_ids, dtype=np.int64)
    origin_m = np.asarray(origin_m, dtype=np.float64)
    target_ink = np.asarray(target_ink, dtype=np.float32)
    if (frames.ndim != 3 or frames.shape[0] == 0
            or target_ink.shape != frames.shape[1:]
            or xy_m.shape != (len(frames), 2)
            or stroke_ids.shape != (len(frames),)
            or origin_m.shape != (2,)
            or not np.isfinite(frames).all()
            or not np.isfinite(xy_m).all()
            or not np.isfinite(target_ink).all()
            or not 0.0 < threshold < 1.0):
        raise ValueError("invalid neural ink stream or target geometry")
    region, distance = _nearest_stroke_regions(
        target_ink.shape, xy_m, stroke_ids, origin_m, pixel_size_m
    )
    predicted = frames[-1]
    target_mask = target_ink >= threshold
    predicted_mask = predicted >= threshold
    support = distance <= support_radius_px
    target_within = float(target_ink[support].sum())
    result: dict[str, Any] = {
        "method": "nearest_dense_path_voronoi_v2",
        "threshold": threshold,
        "support_radius_px": support_radius_px,
        "diagnostic_thresholds": {
            "placement_centroid_px": 2.0,
            "ink_area_ratio_band": [0.8, 1.2],
            "mean_radius_delta_px": 0.25,
        },
        "target_ink_outside_support_over_inside": (
            float(target_ink[~support].sum()) / max(target_within, 1.0e-9)
        ),
        "global_iou": float(np.count_nonzero(target_mask & predicted_mask)
                            / max(np.count_nonzero(target_mask | predicted_mask), 1)),
        "strokes": [],
        "limitation": "Target overlap is assigned to its nearest sampled trajectory; "
                      "width/placement hints are diagnostics, not ground-truth stroke segmentation.",
    }
    previous = np.concatenate((np.zeros_like(frames[:1]), frames[:-1]), axis=0)
    increments = np.maximum(frames - previous, 0.0)
    for stroke in np.unique(stroke_ids):
        owned = region == stroke
        target_owned = target_mask & owned
        predicted_owned = predicted_mask & owned
        extra = predicted_owned & ~target_mask
        missing = target_owned & ~predicted_mask
        intersection = target_owned & predicted_owned
        union = target_owned | predicted_owned
        target_area = int(np.count_nonzero(target_owned))
        predicted_area = int(np.count_nonzero(predicted_owned))
        tc, pc = _centroid(target_owned), _centroid(predicted_owned)
        centroid_offset = float(np.linalg.norm(pc - tc)) if tc is not None and pc is not None else None
        target_radius = float(distance[target_owned].mean()) if target_area else None
        predicted_radius = float(distance[predicted_owned].mean()) if predicted_area else None
        area_ratio = predicted_area / target_area if target_area else None
        radius_delta = (predicted_radius - target_radius
                        if target_radius is not None and predicted_radius is not None else None)
        placement_evidence = centroid_offset is not None and centroid_offset > 2.0
        area_direction = (1 if area_ratio is not None and area_ratio > 1.2 else
                          -1 if area_ratio is not None and area_ratio < .8 else 0)
        # Area excess alone is not proof of a wide brush: wrong position,
        # path length and overlap can all increase occupied pixels.
        width_evidence = (not placement_evidence and radius_delta is not None
                          and abs(radius_delta) >= .25
                          and area_direction * radius_delta > 0)
        if not target_area or not predicted_area:
            hint = "missing_or_extra_stroke"
        elif placement_evidence:
            hint = "placement_and_ink_volume_candidate" if area_direction else "placement_candidate"
        elif width_evidence:
            hint = "overwide_candidate" if area_direction > 0 else "underwide_candidate"
        elif area_direction:
            hint = "ink_volume_or_path_length_candidate"
        else:
            hint = "mixed_or_uncertain"
        result["strokes"].append({
            "stroke_id": int(stroke),
            "target_area_px": target_area,
            "predicted_area_px": predicted_area,
            "area_ratio_predicted_over_target": area_ratio,
            "extra_ink_px": int(np.count_nonzero(extra)),
            "missing_ink_px": int(np.count_nonzero(missing)),
            "intersection_px": int(np.count_nonzero(intersection)),
            "iou": float(np.count_nonzero(intersection) / max(np.count_nonzero(union), 1)),
            "centroid_offset_px": centroid_offset,
            "target_mean_distance_to_path_px": target_radius,
            "predicted_mean_distance_to_path_px": predicted_radius,
            "predicted_minus_target_mean_radius_px": radius_delta,
            "placement_evidence": placement_evidence,
            "width_evidence": width_evidence,
            "attributed_forward_ink_mass": float(increments[stroke_ids == stroke].sum()),
            "residual_hint": hint,
        })
    return result


def audit_candidate(candidate: Path, output: Path) -> dict[str, Any]:
    candidate = Path(candidate).expanduser().resolve()
    with np.load(candidate / "neural_ink.npz") as stream:
        arrays = {key: stream[key] for key in
                  ("frames", "xy_m", "stroke_ids", "origin_m", "pixel_size_m")}
    with Image.open(candidate / "scaled_target.png") as opened:
        target = 1.0 - np.asarray(opened.convert("L"), dtype=np.float32) / 255.0
    result = attribute_stroke_residuals(
        arrays["frames"], arrays["xy_m"], arrays["stroke_ids"],
        arrays["origin_m"], float(arrays["pixel_size_m"]), target,
    )
    metadata = json.loads((candidate / "offline_inversion.json").read_text(encoding="utf-8"))
    result["character"] = metadata["character"]
    result["sample_id"] = metadata["sample_id"]
    result["font_size_m"] = metadata["font_size_m"]
    result["target_sha256"] = metadata["target_scaling"]["target_sha256"]
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    return result
