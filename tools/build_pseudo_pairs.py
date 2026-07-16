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
from utils.feature_schema import build_stroke_features, get_feature_schema


NORM_PADDING = 4


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
    target = np.asarray(target, dtype=np.float32)
    if target.max(initial=0.0) > 1.0:
        target = target / 255.0
    target = np.clip(target, 0.0, 1.0)
    if float(target.mean()) > 0.5:
        target = 1.0 - target
    return target


def letterbox_char_to_canvas(
    image_tensor: Any,
    canvas_size: int = 128,
    padding: int = NORM_PADDING,
) -> np.ndarray:
    array = image_tensor.detach().cpu().numpy()
    if array.ndim == 3:
        array = array[0] if array.shape[0] == 1 else array.mean(axis=0)
    height, width = array.shape
    available = max(canvas_size - 2 * padding, 1)
    scale = min(available / max(width, 1), available / max(height, 1))
    new_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    resized = Image.fromarray(
        np.clip(array * 255.0, 0, 255).astype(np.uint8)
    ).resize(new_size, Image.Resampling.BILINEAR)
    offset_x = padding + (available - new_size[0]) // 2
    offset_y = padding + (available - new_size[1]) // 2
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    canvas[
        offset_y:offset_y + new_size[1],
        offset_x:offset_x + new_size[0],
    ] = np.asarray(resized, dtype=np.float32) / 255.0
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

            used_real_image = False
            source: Dict[str, Any] = {}
            candidates = image_index.get(sample.character, [])
            if image_dataset is not None and candidates:
                candidate_index = image_counters.get(sample.character, 0) % len(candidates)
                image_counters[sample.character] = candidate_index + 1
                item = candidates[candidate_index]
                image_sample = image_dataset._build_sample(item)
                canvas = letterbox_char_to_canvas(
                    image_sample["image"], canvas_size, NORM_PADDING
                )
                target = stroke_target_from_canvas(canvas, normalized, canvas_size)
                used_real_image = True
                source = _source_metadata(item)
            else:
                target = rasterize_polyline(normalized, canvas_size, width=5)

            inputs.append(feature)
            targets.append(normalize_target_polarity(target))
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
                    "feature_schema": schema.name,
                    "dynamic_brush_mode": cfg.dynamic_brush.mode,
                    **source,
                }
            )

    if not inputs:
        raise RuntimeError("No pseudo pairs were generated")
    input_array = np.asarray(inputs, dtype=np.float32)
    target_array = np.asarray(targets, dtype=np.float32)
    metadata_array = np.asarray(metadata, dtype=object)
    output = Path(args.output_npz)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        inputs=input_array,
        targets=target_array,
        meta=metadata_array,
        feature_schema=np.asarray(schema.name),
        feature_fields=np.asarray(schema.fields),
        dynamic_brush_mode=np.asarray(cfg.dynamic_brush.mode),
    )
    real_count = sum(bool(item["used_real_image"]) for item in metadata)
    print(f"[DONE] output={output}")
    print(f"[DONE] inputs={input_array.shape}, targets={target_array.shape}")
    print(f"[DONE] real={real_count}, synthetic={len(metadata) - real_count}")
    print(f"[DONE] schema={schema.name}, dynamic_brush={cfg.dynamic_brush.mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output_npz", default="data/processed/bbsmg_train.npz")
    parser.add_argument("--chirography")
    main(parser.parse_args())
