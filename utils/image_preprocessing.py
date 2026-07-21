from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image


DEFAULT_CANVAS_PADDING = 4
DEFAULT_CONTRAST_FLOOR = 0.08


def _as_grayscale_float(image: Any) -> np.ndarray:
    """Convert supported image inputs to a two-dimensional [0, 1] array."""
    if hasattr(image, "detach"):
        array = image.detach().cpu().numpy()
    elif isinstance(image, Image.Image):
        array = np.asarray(image.convert("L"))
    else:
        array = np.asarray(image)

    if array.ndim == 3:
        if array.shape[0] in (1, 3, 4):
            array = array[0] if array.shape[0] == 1 else array[:3].mean(axis=0)
        else:
            array = array[..., 0] if array.shape[-1] == 1 else array[..., :3].mean(axis=-1)
    if array.ndim != 2:
        raise ValueError(f"Character image must be 2-D after grayscale conversion, got {array.shape}")

    array = array.astype(np.float32, copy=False)
    if float(array.max(initial=0.0)) > 1.0:
        array = array / 255.0
    return np.clip(array, 0.0, 1.0)


def _border_pixels(array: np.ndarray) -> np.ndarray:
    if min(array.shape) < 2:
        return array.reshape(-1)
    return np.concatenate((array[0], array[-1], array[1:-1, 0], array[1:-1, -1]))


def normalize_image_polarity_with_info(
    image: Any,
    contrast_floor: float = DEFAULT_CONTRAST_FLOOR,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Remove paper tone and return an ink-positive image plus diagnostics."""
    if not 0.0 <= contrast_floor < 1.0:
        raise ValueError("contrast_floor must satisfy 0 <= value < 1")
    array = _as_grayscale_float(image)
    border_level = float(np.median(_border_pixels(array)))
    dark_level = float(np.quantile(array, 0.02))
    bright_level = float(np.quantile(array, 0.98))
    dark_contrast = max(border_level - dark_level, 0.0)
    bright_contrast = max(bright_level - border_level, 0.0)

    if dark_contrast >= bright_contrast:
        polarity = "dark_ink_on_light_background"
        contrast = max(dark_contrast, 1e-6)
        ink = (border_level - array) / contrast
    else:
        polarity = "light_ink_on_dark_background"
        contrast = max(bright_contrast, 1e-6)
        ink = (array - border_level) / contrast

    ink = np.clip(ink, 0.0, 1.0)
    ink = np.clip((ink - contrast_floor) / max(1.0 - contrast_floor, 1e-6), 0.0, 1.0)
    info = {
        "polarity": polarity,
        "border_background_level": border_level,
        "dark_level_p02": dark_level,
        "bright_level_p98": bright_level,
        "contrast": contrast,
        "contrast_floor": float(contrast_floor),
        "normalized_background_median": float(np.median(_border_pixels(ink))),
    }
    return ink.astype(np.float32), info


def normalize_image_polarity(
    image: Any,
    contrast_floor: float = DEFAULT_CONTRAST_FLOOR,
) -> np.ndarray:
    """Convert an image to a 2-D float mask with background=0 and ink=1."""
    normalized, _ = normalize_image_polarity_with_info(image, contrast_floor=contrast_floor)
    return normalized


def _foreground_crop(
    image: np.ndarray,
    threshold: float,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    foreground = image > threshold
    if not np.any(foreground):
        height, width = image.shape
        return image, (0, 0, width, height)
    ys, xs = np.nonzero(foreground)
    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    return image[top:bottom, left:right], (left, top, right, bottom)


def letterbox_character_image(
    image: Any,
    canvas_size: int = 128,
    padding: int = DEFAULT_CANVAS_PADDING,
    crop_foreground: bool = True,
    foreground_threshold: float = 0.05,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Trim blank margins and place a character on the training canvas."""
    if canvas_size < 1:
        raise ValueError("canvas_size must be positive")
    if padding < 0 or padding * 2 >= canvas_size:
        raise ValueError("padding must satisfy 0 <= 2 * padding < canvas_size")

    normalized, normalization_info = normalize_image_polarity_with_info(image)
    original_height, original_width = normalized.shape
    crop_box = (0, 0, original_width, original_height)
    cropped = normalized
    if crop_foreground:
        cropped, crop_box = _foreground_crop(normalized, foreground_threshold)

    height, width = cropped.shape
    available = canvas_size - 2 * padding
    scale = min(available / max(width, 1), available / max(height, 1))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = Image.fromarray(
        np.clip(cropped * 255.0, 0, 255).astype(np.uint8), mode="L"
    ).resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    offset_x = padding + (available - resized_width) // 2
    offset_y = padding + (available - resized_height) // 2
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    canvas[
        offset_y:offset_y + resized_height,
        offset_x:offset_x + resized_width,
    ] = np.asarray(resized, dtype=np.float32) / 255.0

    transform = {
        "version": 2,
        "canvas_size": canvas_size,
        "padding": padding,
        "crop_foreground": crop_foreground,
        "foreground_threshold": foreground_threshold,
        "original_size": [original_width, original_height],
        "crop_box": list(crop_box),
        "resized_size": [resized_width, resized_height],
        "offset": [offset_x, offset_y],
        "scale": scale,
        "normalization": normalization_info,
    }
    return np.clip(canvas, 0.0, 1.0), transform


def load_character_image(
    path: str,
    canvas_size: int = 128,
    padding: int = DEFAULT_CANVAS_PADDING,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Character image not found: {path}")
    with Image.open(image_path) as image:
        return letterbox_character_image(
            image.convert("L"),
            canvas_size=canvas_size,
            padding=padding,
            crop_foreground=True,
        )
