from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image


DEFAULT_CANVAS_PADDING = 4


def normalize_image_polarity(image: Any) -> np.ndarray:
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
        raise ValueError(f"Character image must be 2D after grayscale conversion, got {array.shape}")
    array = array.astype(np.float32, copy=False)
    if array.max(initial=0.0) > 1.0:
        array = array / 255.0
    array = np.clip(array, 0.0, 1.0)
    if float(array.mean()) > 0.5:
        array = 1.0 - array
    return array


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
    if canvas_size < 1:
        raise ValueError("canvas_size must be positive")
    if padding < 0 or padding * 2 >= canvas_size:
        raise ValueError("padding must satisfy 0 <= 2 * padding < canvas_size")
    normalized = normalize_image_polarity(image)
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
        "version": 1,
        "canvas_size": canvas_size,
        "padding": padding,
        "crop_foreground": crop_foreground,
        "foreground_threshold": foreground_threshold,
        "original_size": [original_width, original_height],
        "crop_box": list(crop_box),
        "resized_size": [resized_width, resized_height],
        "offset": [offset_x, offset_y],
        "scale": scale,
    }
    return np.clip(canvas, 0.0, 1.0), transform


def load_character_image(
    path: str,
    canvas_size: int = 128,
    padding: int = DEFAULT_CANVAS_PADDING,
) -> np.ndarray:
    with Image.open(path) as image:
        canvas, _ = letterbox_character_image(
            image.convert("L"),
            canvas_size=canvas_size,
            padding=padding,
            crop_foreground=True,
        )
    return canvas
