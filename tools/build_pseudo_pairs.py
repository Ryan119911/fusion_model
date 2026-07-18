import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw

from config import ensure_dirs, load_config
from datasets.calligraphy_image_dataset import CalligraphyImageDataset
from datasets.makehanzi_dataset import MakeHanziDataset
from datasets.trajectory_dataset import load_trajectory_csv
from models.dynamic_brush import build_dynamic_brush
from models.geometry import compute_heading, normalize_trajectory_xy
from utils.character_groups import (
    validate_character_target_mapping,
    validate_group_consistency,
)
from utils.feature_schema import build_stroke_features, get_feature_schema
from utils.image_preprocessing import (
    DEFAULT_CANVAS_PADDING,
    letterbox_character_image,
    normalize_image_polarity,
)


NORM_PADDING = DEFAULT_CANVAS_PADDING


def rasterize_polyline(
    points: Sequence[Tuple[float, float]],
    canvas_size: int = 128,
    width: int = 5,
) -> np.ndarray:
    image = Image.new("L", (canvas_size, canvas_size), 0)
    draw = ImageDraw.Draw(image)
    if len(points) >= 2:
        draw.line(list(points), fill=255, width=width)
    elif len(points) == 1:
        x, y = points[0]
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=255)
    return np.asarray(image, dtype=np.float32) / 255.0


def rasterize_stroke_mask(
    points: Sequence[Tuple[float, float]],
    canvas_size: int = 128,
    width: int = 8,
) -> np.ndarray:
    return rasterize_polyline(points, canvas_size, width)


def normalize_target_polarity(target: np.ndarray) -> np.ndarray:
    return normalize_image_polarity(target)


def letterbox_char_to_canvas(
    image_tensor: Any,
    canvas_size: int = 128,
    padding: int = NORM_PADDING,
) -> np.ndarray:
    canvas, _ = letterbox_character_image(
        image_tensor,
        canvas_size=canvas_size,
        padding=padding,
        crop_foreground=True,
    )
    return canvas


def stroke_target_from_canvas(
    canvas: np.ndarray,
    points: Sequence[Tuple[float, float]],
    canvas_size: int,
) -> np.ndarray:
    return np.clip(
        canvas * rasterize_stroke_mask(points, canvas_size, width=8),
        0.0,
        1.0,
    ).astype(np.float32)


def build_image_index(
    dataset: CalligraphyImageDataset,
) -> Dict[str, List[Dict[str, Any]]]:
    result: Dict[str, List[Dict[str, Any]]] = {}
    for item in dataset.index:
        character = item.get("character")
        if character:
            result.setdefault(character, []).append(item)
    return result


def _source_metadata(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in item.items()
        if key in {
            "image_path", "json_path", "bbox", "group_id",
            "shape_index", "folder", "file_stem",
        }
    }


def main(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    if args.chirography is not None:
        cfg.data.chirography_filter = args.chirography
    ensure_dirs(cfg)
    schema = get_feature_schema(cfg.bbsmg.feature_schema)
    if schema.input_dim != cfg.bbsmg.input_dim:
        raise ValueError("Configured feature schema and input_dim disagree")

    trajectories = load_trajectory_csv(
        cfg.data.trajectory_csv,
        timestamp_column=cfg.data.timestamp_column,
        validate=cfg.data.validate_trajectories,
        points_per_stroke=cfg.data.points_per_stroke,
    )
    brush = build_dynamic_brush(cfg.dynamic_brush)

    makehanzi_index: Dict[str, Dict[str, Any]] = {}
    if Path(cfg.data.dictionary_txt).exists() and Path(cfg.data.graphics_txt).exists():
        makehanzi = MakeHanziDataset(
            cfg.data.dictionary_txt, cfg.data.graphics_txt
        )
        makehanzi_index = {
            sample["character"]: sample for sample in makehanzi.samples
        }

    image_dataset = None
    image_index: Dict[str, List[Dict[str, Any]]] = {}
    if Path(cfg.data.image_dir).exists() and Path(cfg.data.json_dir).exists():
        image_dataset = CalligraphyImageDataset(
            image_dir=cfg.data.image_dir,
            json_dir=cfg.data.json_dir,
            image_ext=cfg.data.image_ext,
            image_size=None,
            grayscale=True,
            padding=0.0,
            data_csv=cfg.data.data_csv,
            chirography_filter=cfg.data.chirography_filter,
        )
        image_index = build_image_index(image_dataset)

    inputs: List[List[float]] = []
    targets: List[np.ndarray] = []
    character_targets: List[np.ndarray] = []
    character_target_indices: List[int] = []
    metadata: List[Dict[str, Any]] = []
    image_counters: Dict[str, int] = {}
    canvas_size = cfg.data.canvas_size

    for sample_index, sample in enumerate(trajectories):
        if not sample.character:
            continue
        strokes = sample.sorted_strokes()
        normalized_strokes = normalize_trajectory_xy(
            sample, canvas_size=canvas_size, padding=NORM_PADDING
        )
        medians = []
        makehanzi_sample = makehanzi_index.get(sample.character)
        if makehanzi_sample and makehanzi_sample.get("graphics"):
            medians = makehanzi_sample["graphics"].medians or []

        used_real_image = False
        source: Dict[str, Any] = {}
        canvas = None
        canvas_transform = None
        candidates = image_index.get(sample.character, [])
        if image_dataset is not None and candidates:
            candidate_index = image_counters.get(sample.character, 0) % len(candidates)
            image_counters[sample.character] = candidate_index + 1
            item = candidates[candidate_index]
            image_sample = image_dataset._build_sample(item)
            canvas, canvas_transform = letterbox_character_image(
                image_sample["image"],
                canvas_size=canvas_size,
                padding=NORM_PADDING,
                crop_foreground=True,
            )
            used_real_image = True
            source = _source_metadata(item)

        if canvas is None:
            character_canvas = np.zeros((canvas_size, canvas_size), dtype=np.float32)
            for normalized in normalized_strokes:
                character_canvas = np.maximum(
                    character_canvas,
                    rasterize_polyline(normalized, canvas_size, width=5),
                )
        else:
            character_canvas = canvas
        character_target_index = len(character_targets)
        character_targets.append(normalize_target_polarity(character_canvas))
        samples_before = len(inputs)

        for stroke_order, (stroke, normalized) in enumerate(
            zip(strokes, normalized_strokes)
        ):
            points = stroke.sorted_points()
            if not points or not normalized:
                continue
            headings = compute_heading([(point.x, point.y) for point in points])
            states = brush.simulate_stroke(stroke, theta0=headings[0])
            feature = build_stroke_features(
                schema.name, points[0], states[0], normalized
            )

            if canvas is not None:
                target = stroke_target_from_canvas(canvas, normalized, canvas_size)
            else:
                target = rasterize_polyline(normalized, canvas_size, width=5)

            inputs.append(feature)
            targets.append(normalize_target_polarity(target))
            character_target_indices.append(character_target_index)
            metadata.append(
                {
                    "character": sample.character,
                    "sample_id": sample.meta.get("sample_id"),
                    "trajectory_sample_index": sample_index,
                    "stroke_id": stroke.stroke_id,
                    "stroke_order": stroke_order,
                    "num_strokes_in_traj": len(strokes),
                    "num_medians_in_makehanzi": len(medians),
                    "used_real_image": used_real_image,
                    "character_target_index": character_target_index,
                    "canvas_transform": canvas_transform,
                    "feature_schema": schema.name,
                    "dynamic_brush_mode": cfg.dynamic_brush.mode,
                    **source,
                }
            )
        if len(inputs) == samples_before:
            character_targets.pop()

    if not inputs:
        raise RuntimeError("No pseudo pairs were generated")
    consistency_errors = validate_group_consistency(metadata)
    if consistency_errors:
        preview = "\n".join(consistency_errors[:20])
        raise RuntimeError(
            f"Generated inconsistent character groups ({len(consistency_errors)}):\n{preview}"
        )
    mapping_errors = validate_character_target_mapping(
        metadata,
        character_target_indices,
        len(character_targets),
    )
    if mapping_errors:
        preview = "\n".join(mapping_errors[:20])
        raise RuntimeError(
            f"Generated invalid character target mapping ({len(mapping_errors)}):\n{preview}"
        )
    input_array = np.asarray(inputs, dtype=np.float32)
    target_array = np.asarray(targets, dtype=np.float32)
    character_target_array = np.asarray(character_targets, dtype=np.float32)
    character_target_index_array = np.asarray(character_target_indices, dtype=np.int64)
    metadata_array = np.asarray(metadata, dtype=object)
    output = Path(args.output_npz)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        inputs=input_array,
        targets=target_array,
        character_targets=character_target_array,
        character_target_indices=character_target_index_array,
        meta=metadata_array,
        feature_schema=np.asarray(schema.name),
        feature_fields=np.asarray(schema.fields),
        dynamic_brush_mode=np.asarray(cfg.dynamic_brush.mode),
    )
    real_count = sum(bool(item["used_real_image"]) for item in metadata)
    print(f"[DONE] output={output}")
    print(f"[DONE] inputs={input_array.shape}, targets={target_array.shape}")
    print(f"[DONE] character_targets={character_target_array.shape}")
    print(f"[DONE] real={real_count}, synthetic={len(metadata) - real_count}")
    print(f"[DONE] schema={schema.name}, dynamic_brush={cfg.dynamic_brush.mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output_npz", default="data/processed/bbsmg_train.npz")
    parser.add_argument("--chirography")
    main(parser.parse_args())
