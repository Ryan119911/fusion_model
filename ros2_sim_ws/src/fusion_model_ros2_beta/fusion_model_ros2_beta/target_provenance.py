"""Fail-closed provenance checks for the original calligraphy target images.

The V16 batch stores a conventional black-on-white ``target.png`` while the
model consumes an ink-positive float canvas.  This module reproduces the exact
batch preprocessing without importing PyTorch or the dataset class, and proves
that a manifest's original image/LabelMe crop really produced that target.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from .offline_fontsize_inversion import _sha256


PROVENANCE_CONTRACT = "v16_original_target_provenance_v2"
_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR


def _require_sha256(value: Any, label: str) -> str:
    digest = str(value or "").lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"invalid {label} SHA256")
    return digest


def _resolved(value: Any, label: str) -> Path:
    if not value:
        raise ValueError(f"missing {label}")
    path = Path(str(value)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _verify_hash(path: Path, expected: Any, label: str) -> str:
    expected_digest = _require_sha256(expected, label)
    actual = _sha256(path).lower()
    if actual != expected_digest:
        raise ValueError(f"{label} hash mismatch: {actual} != {expected_digest}")
    return actual


def _bbox_from_shape(shape: Mapping[str, Any]) -> list[float]:
    points = shape.get("points")
    if not isinstance(points, list) or len(points) < 2:
        raise ValueError("target annotation shape has no rectangle points")
    x1, y1 = points[0]
    x2, y2 = points[1]
    return [float(min(x1, x2)), float(min(y1, y2)),
            float(max(x1, x2)), float(max(y1, y2))]


def _same_bbox(left: Any, right: Any) -> bool:
    try:
        return len(left) == len(right) == 4 and all(
            math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1.0e-6)
            for a, b in zip(left, right)
        )
    except (TypeError, ValueError):
        return False


def _dataset_crop(image: Image.Image, bbox: Any) -> np.ndarray:
    if not _same_bbox(bbox, bbox):
        raise ValueError("invalid target bbox")
    width, height = image.size
    x1, y1, x2, y2 = (float(value) for value in bbox)
    # Exact CalligraphyImageDataset behaviour for padding=0.0.
    left = max(0, min(int(round(x1)), width - 1))
    top = max(0, min(int(round(y1)), height - 1))
    right = max(left + 1, min(int(round(x2)), width))
    bottom = max(top + 1, min(int(round(y2)), height))
    array = np.asarray(image.crop((left, top, right, bottom)).convert("L"),
                       dtype=np.float32) / 255.0
    # Exact polarity pre-pass in CalligraphyImageDataset._crop_to_tensor.
    if float(array.mean()) > 0.5:
        array = 1.0 - array
    return array


def _as_grayscale_float(image: Any) -> np.ndarray:
    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("L"))
    else:
        array = np.asarray(image)
    if array.ndim == 3:
        if array.shape[0] in (1, 3, 4):
            array = array[0] if array.shape[0] == 1 else array[:3].mean(axis=0)
        else:
            array = array[..., 0] if array.shape[-1] == 1 else array[..., :3].mean(axis=-1)
    if array.ndim != 2:
        raise ValueError(f"original target must be 2-D after grayscale conversion, got {array.shape}")
    array = array.astype(np.float32, copy=False)
    if float(array.max(initial=0.0)) > 1.0:
        array = array / 255.0
    return np.clip(array, 0.0, 1.0)


def _border_pixels(array: np.ndarray) -> np.ndarray:
    if min(array.shape) < 2:
        return array.reshape(-1)
    return np.concatenate((array[0], array[-1], array[1:-1, 0], array[1:-1, -1]))


def _letterbox(
    image: Any, *, image_size: int, padding: int, glyph_scale: float = 1.0,
    normalization_override: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if image_size < 1 or padding < 0 or 2 * padding >= image_size:
        raise ValueError("invalid original-target canvas geometry")
    if not math.isfinite(glyph_scale) or glyph_scale <= 0.0:
        raise ValueError("invalid original-target glyph scale")
    array = _as_grayscale_float(image)
    border_level = float(np.median(_border_pixels(array)))
    dark_level = float(np.quantile(array, 0.02))
    bright_level = float(np.quantile(array, 0.98))
    if normalization_override is not None:
        # NumPy 1.x and 2.x interpolate float32 quantiles differently. The
        # hash-pinned batch transform is the authoritative replay recipe; the
        # independently measured transform must first pass _compare_transform.
        border_level = float(normalization_override["border_background_level"])
        dark_level = float(normalization_override["dark_level_p02"])
        bright_level = float(normalization_override["bright_level_p98"])
        if (not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in
                    (border_level, dark_level, bright_level))
                or float(normalization_override["contrast_floor"]) != 0.08):
            raise ValueError("invalid pinned target normalization")
    dark_contrast = max(border_level - dark_level, 0.0)
    bright_contrast = max(bright_level - border_level, 0.0)
    if dark_contrast >= bright_contrast:
        polarity = "dark_ink_on_light_background"
        contrast = max(dark_contrast, 1.0e-6)
        ink = (border_level - array) / contrast
    else:
        polarity = "light_ink_on_dark_background"
        contrast = max(bright_contrast, 1.0e-6)
        ink = (array - border_level) / contrast
    if (normalization_override is not None
            and normalization_override["polarity"] != polarity):
        raise ValueError("pinned target normalization polarity mismatch")
    ink = np.clip(ink, 0.0, 1.0)
    contrast_floor = 0.08
    ink = np.clip((ink - contrast_floor) / (1.0 - contrast_floor), 0.0, 1.0)
    foreground = ink > 0.05
    if np.any(foreground):
        ys, xs = np.nonzero(foreground)
        crop_box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        cropped = ink[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
    else:
        crop_box = (0, 0, int(ink.shape[1]), int(ink.shape[0]))
        cropped = ink
    available = image_size - 2 * padding
    height, width = cropped.shape
    scale = glyph_scale * min(available / max(width, 1), available / max(height, 1))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    if resized_width > image_size or resized_height > image_size:
        raise ValueError("scaled original target would clip the model canvas")
    resized = Image.fromarray(
        np.clip(cropped * 255.0, 0, 255).astype(np.uint8), mode="L"
    ).resize((resized_width, resized_height), _BILINEAR)
    offset_x = (image_size - resized_width) // 2
    offset_y = (image_size - resized_height) // 2
    canvas = np.zeros((image_size, image_size), dtype=np.float32)
    canvas[offset_y:offset_y + resized_height,
           offset_x:offset_x + resized_width] = np.asarray(resized, dtype=np.float32) / 255.0
    transform = {
        "version": 2,
        "canvas_size": image_size,
        "padding": padding,
        "crop_foreground": True,
        "foreground_threshold": 0.05,
        "original_size": [int(array.shape[1]), int(array.shape[0])],
        "crop_box": list(crop_box),
        "resized_size": [resized_width, resized_height],
        "offset": [offset_x, offset_y],
        "scale": scale,
        "normalization": {
            "polarity": polarity,
            "border_background_level": border_level,
            "dark_level_p02": dark_level,
            "bright_level_p98": bright_level,
            "contrast": contrast,
            "contrast_floor": contrast_floor,
            "normalized_background_median": float(np.median(_border_pixels(ink))),
        },
    }
    return np.clip(canvas, 0.0, 1.0), transform


def render_scaled_original_target(
    provenance: Mapping[str, Any], destination: Path, ratio: float,
    *, image_size: int = 128, padding: int = 16,
) -> dict[str, Any]:
    """Make a size target with one resample from the verified source crop."""
    if not math.isfinite(ratio) or ratio <= 0.0:
        raise ValueError("invalid original-target size ratio")
    original = _resolved(provenance.get("original_target_image"), "original target image")
    _verify_hash(original, provenance.get("original_target_sha256"), "original target")
    source_meta = _resolved(provenance.get("target_source_json"), "target source JSON")
    _verify_hash(source_meta, provenance.get("target_source_sha256"), "target source JSON")
    metadata = json.loads(source_meta.read_text(encoding="utf-8"))
    if (provenance.get("contract") != PROVENANCE_CONTRACT
            or provenance.get("pixel_reproduction_exact") is not True
            or provenance.get("preprocessing_transform") != metadata.get("transform")):
        raise ValueError("scaled target requires validated pinned preprocessing")
    with Image.open(original) as opened:
        source = (_dataset_crop(opened, metadata["bbox"])
                  if metadata.get("source_json") else opened.convert("L").copy())
    canvas, transform = _letterbox(
        source, image_size=image_size, padding=padding, glyph_scale=ratio,
        normalization_override=metadata["transform"]["normalization"],
    )
    pixels = np.rint((1.0 - canvas) * 255.0).astype(np.uint8)
    if math.isclose(ratio, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        normalized = _resolved(provenance.get("normalized_target_image"), "normalized target image")
        _verify_hash(normalized, provenance.get("normalized_target_sha256"), "normalized target")
        with Image.open(normalized) as opened:
            expected = np.asarray(opened.convert("L"), dtype=np.uint8)
        if not np.array_equal(pixels, expected):
            raise ValueError("direct original-target scaling differs from canonical target at unit size")
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, mode="L").save(destination)
    return {
        "method": "single_resample_from_original_source_crop_v1",
        "normalization_replay": "hash_pinned_batch_transform_v1",
        "ratio": ratio,
        "resized_size": transform["resized_size"],
        "offset": transform["offset"],
        "target_sha256": _sha256(destination),
        "source_sha256": provenance["original_target_sha256"],
    }


def _compare_transform(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> None:
    for key in ("version", "canvas_size", "padding", "crop_foreground",
                "foreground_threshold", "original_size", "crop_box",
                "resized_size", "offset"):
        if expected.get(key) != actual.get(key):
            raise ValueError(f"target transform mismatch: {key}")
    if not math.isclose(float(expected.get("scale", -1.0)), float(actual["scale"]),
                        rel_tol=1.0e-9, abs_tol=1.0e-9):
        raise ValueError("target transform mismatch: scale")
    expected_norm = expected.get("normalization", {})
    actual_norm = actual["normalization"]
    if expected_norm.get("polarity") != actual_norm["polarity"]:
        raise ValueError("target normalization polarity mismatch")
    for key in ("border_background_level", "dark_level_p02", "bright_level_p98",
                "contrast", "contrast_floor", "normalized_background_median"):
        # A 2e-5 absolute bound covers the measured NumPy 1.21/2.2 float32
        # quantile drift (max 1.04e-5 in the audited corpus). It is never a
        # substitute for the exact canonical-pixel comparison below.
        tolerance = 2.0e-5 if key in ("dark_level_p02", "bright_level_p98", "contrast") else 1.0e-7
        if not math.isclose(float(expected_norm.get(key, float("nan"))), float(actual_norm[key]),
                            rel_tol=0.0, abs_tol=tolerance):
            raise ValueError(f"target normalization mismatch: {key}")


def validate_original_target_record(
    record: Mapping[str, Any], *, image_size: int = 128, padding: int = 16,
    allow_screened: bool = False,
) -> dict[str, Any]:
    """Validate all source identities and reproduce the canonical target pixels."""
    character = str(record.get("character", ""))
    sample_id = str(record.get("sample_id", ""))
    if len(character) != 1 or not sample_id:
        raise ValueError("original-target record requires one character and a sample id")
    glyph_identity = record.get("glyph_identity")
    status = glyph_identity.get("status") if isinstance(glyph_identity, dict) else None
    if (not isinstance(glyph_identity, dict)
            or (status != "verified" and not (allow_screened and status == "screened"))
            or glyph_identity.get("character") != character
            or not str(glyph_identity.get("method", "")).strip()
            or glyph_identity.get("method") == "annotation_label_only"):
        raise ValueError(
            "original target glyph identity is not independently verified; "
            "an annotation label alone cannot exclude simplified/traditional mismatch"
        )
    original = _resolved(record.get("original_target_image"), "original target image")
    normalized = _resolved(record.get("normalized_target_image"), "normalized target image")
    source_meta_path = _resolved(record.get("target_source_json"), "target source JSON")
    source_trajectory = _resolved(record.get("source_trajectory"), "source trajectory")
    digests = {
        "original_target_sha256": _verify_hash(original, record.get("original_target_sha256"), "original target"),
        "normalized_target_sha256": _verify_hash(normalized, record.get("normalized_target_sha256"), "normalized target"),
        "target_source_sha256": _verify_hash(source_meta_path, record.get("target_source_sha256"), "target source JSON"),
        "source_sha256": _verify_hash(source_trajectory, record.get("source_sha256"), "source trajectory"),
    }
    if status == "screened":
        from .glyph_screen import SCREEN_METHOD
        if (glyph_identity.get("method") != SCREEN_METHOD
                or glyph_identity.get("semantic_glyph_identity_proven") is not False
                or glyph_identity.get("script_invariant") is not True
                or glyph_identity.get("reasons") != []
                or not math.isfinite(float(glyph_identity.get("target_to_model_support_ink_ratio", float("nan"))))
                or not math.isfinite(float(glyph_identity.get("ratio_max", float("nan"))))
                or float(glyph_identity["ratio_max"]) <= 0.0
                or float(glyph_identity["target_to_model_support_ink_ratio"]) > float(glyph_identity["ratio_max"])
                or any(glyph_identity.get(key) != digests[key] for key in
                       ("original_target_sha256", "normalized_target_sha256", "target_source_sha256"))):
            raise ValueError("automated glyph screen evidence is invalid or stale")
    metadata = json.loads(source_meta_path.read_text(encoding="utf-8"))
    if Path(str(metadata.get("source_image", ""))).expanduser().resolve() != original:
        raise ValueError("target source metadata points to a different original image")
    bbox = metadata.get("bbox")
    source_annotation = metadata.get("source_json")
    if source_annotation is None:
        if record.get("source_annotation_json") or record.get("source_annotation_sha256"):
            raise ValueError("override target must not declare a source annotation")
        if metadata.get("selection") != "override" or bbox is not None:
            raise ValueError("unannotated target must be an explicit override")
        with Image.open(original) as opened:
            model_input: Any = opened.convert("L").copy()
        annotation_path = None
        annotation_digest = None
    else:
        annotation_path = _resolved(record.get("source_annotation_json"), "source annotation JSON")
        if Path(str(source_annotation)).expanduser().resolve() != annotation_path:
            raise ValueError("target source metadata points to a different annotation JSON")
        annotation_digest = _verify_hash(annotation_path, record.get("source_annotation_sha256"),
                                         "source annotation JSON")
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        shape_index = int(metadata.get("shape_index", -1))
        shapes = annotation.get("shapes", [])
        if not 0 <= shape_index < len(shapes):
            raise ValueError("target shape_index is outside the source annotation")
        shape = shapes[shape_index]
        if (str(shape.get("label", "")) != character
                or shape.get("shape_type") != "rectangle"
                or not _same_bbox(_bbox_from_shape(shape), bbox)
                or shape.get("group_id") != metadata.get("group_id")):
            raise ValueError("target annotation identity does not match character/bbox/group")
        with Image.open(original) as opened:
            model_input = _dataset_crop(opened, bbox)
    _, transform = _letterbox(model_input, image_size=image_size, padding=padding)
    expected_transform = metadata.get("transform")
    if not isinstance(expected_transform, dict):
        raise ValueError("target source metadata is missing the preprocessing transform")
    _compare_transform(expected_transform, transform)
    canvas, replay_transform = _letterbox(
        model_input, image_size=image_size, padding=padding,
        normalization_override=expected_transform["normalization"],
    )
    _compare_transform(expected_transform, replay_transform)
    with Image.open(normalized) as opened:
        canonical = np.asarray(opened.convert("L"), dtype=np.uint8)
    if canonical.shape != (image_size, image_size):
        raise ValueError("normalized target has the wrong model canvas size")
    reproduced = np.rint(np.clip(1.0 - canvas, 0.0, 1.0) * 255.0).astype(np.uint8)
    if not np.array_equal(reproduced, canonical):
        difference = np.abs(reproduced.astype(np.int16) - canonical.astype(np.int16))
        raise ValueError(
            "normalized target is not reproducible from its original source "
            f"(different_pixels={int(np.count_nonzero(difference))}, max_abs={int(difference.max())})"
        )
    return {
        "contract": PROVENANCE_CONTRACT,
        "character": character,
        "sample_id": sample_id,
        "source_trajectory": str(source_trajectory),
        "original_target_image": str(original),
        "normalized_target_image": str(normalized),
        "target_source_json": str(source_meta_path),
        "source_annotation_json": str(annotation_path) if annotation_path else None,
        "source_annotation_sha256": annotation_digest,
        "glyph_identity": glyph_identity,
        **digests,
        "preprocessing_transform": expected_transform,
        "normalization_replay": "hash_pinned_batch_transform_v1",
        "pixel_reproduction_exact": True,
    }
