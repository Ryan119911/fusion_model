import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt

from utils.comparison_metrics import read_comparison_manifest
from utils.types import (
    CharacterTrajectory,
    PointState,
    StrokeTrajectory,
    TrajectoryPoint,
)


TRANSFORM_NAMES = (
    "identity",
    "flip_x",
    "flip_y",
    "rotate_90",
    "rotate_180",
    "rotate_270",
    "transpose",
    "anti_transpose",
)


def _safe_stem(value: Any) -> str:
    text = str(value).strip() or "sample"
    return re.sub(r"[^\w.-]+", "_", text).strip("_") or "sample"


def sample_index(
    samples: Sequence[CharacterTrajectory],
) -> Dict[str, CharacterTrajectory]:
    result: Dict[str, CharacterTrajectory] = {}
    for sample in samples:
        sample_id = str(sample.meta.get("sample_id"))
        if sample_id in result:
            raise ValueError(f"Duplicate trajectory sample_id: {sample_id}")
        result[sample_id] = sample
    return result


def _sample_key(row: Mapping[str, Any]) -> str:
    for key in ("sample_id", "character", "char_id", "file_stem"):
        if row.get(key) not in (None, ""):
            return str(row[key])
    return "default_sample"


def _point_state(value: Any) -> PointState:
    if value in (None, ""):
        return PointState.MOVE
    if isinstance(value, str):
        names = {
            "down": PointState.DOWN,
            "move": PointState.MOVE,
            "up": PointState.UP,
            "transition": PointState.TRANSITION,
        }
        name = value.strip().lower()
        if name in names:
            return names[name]
    return PointState.from_value(int(float(value)))


def load_alignment_trajectory_csv(
    csv_path: str, timestamp_column: Optional[str] = None
) -> List[CharacterTrajectory]:
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Trajectory CSV not found: {csv_path}")
    grouped: Dict[str, List[Dict[str, str]]] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"stroke_id", "point_id", "x", "y"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Trajectory CSV is missing columns: {sorted(missing)}")
        for row in reader:
            grouped.setdefault(_sample_key(row), []).append(row)

    samples: List[CharacterTrajectory] = []
    for sample_id, rows in grouped.items():
        strokes: Dict[int, List[TrajectoryPoint]] = {}
        for row in rows:
            stroke_id = int(float(row["stroke_id"]))
            timestamp = None
            if timestamp_column and row.get(timestamp_column) not in (None, ""):
                timestamp = float(row[timestamp_column])
            point = TrajectoryPoint(
                stroke_id=stroke_id,
                point_id=int(float(row["point_id"])),
                x=float(row["x"]),
                y=float(row["y"]),
                z=float(row.get("z") or 0.0),
                alpha=float(row.get("alpha") or 0.0),
                beta=float(row.get("beta") or 0.0),
                gamma=float(row.get("gamma") or 0.0),
                state=_point_state(row.get("state")),
                timestamp=timestamp,
            )
            if not math.isfinite(point.x) or not math.isfinite(point.y):
                raise ValueError(
                    f"Non-finite x/y in sample={sample_id}, "
                    f"stroke={stroke_id}, point={point.point_id}"
                )
            strokes.setdefault(stroke_id, []).append(point)
        first = rows[0]
        samples.append(
            CharacterTrajectory(
                character=first.get("character") or None,
                strokes=[
                    StrokeTrajectory(stroke_id=stroke_id, points=points)
                    for stroke_id, points in sorted(strokes.items())
                ],
                meta={"sample_id": sample_id},
            )
        )
    return samples


def load_ink_image(path: str, image_size: int) -> np.ndarray:
    with Image.open(path) as image:
        gray = image.convert("L").resize(
            (image_size, image_size), Image.Resampling.BILINEAR
        )
    array = np.asarray(gray, dtype=np.float32) / 255.0
    if float(array.mean()) > 0.5:
        array = 1.0 - array
    return np.clip(array, 0.0, 1.0)


def trajectory_to_unit_strokes(
    sample: CharacterTrajectory,
) -> List[np.ndarray]:
    points = sample.all_points()
    if not points:
        raise ValueError("Trajectory sample is empty")
    x_values = np.asarray([point.x for point in points], dtype=np.float64)
    y_values = np.asarray([point.y for point in points], dtype=np.float64)
    center_x = 0.5 * (float(x_values.min()) + float(x_values.max()))
    center_y = 0.5 * (float(y_values.min()) + float(y_values.max()))
    extent = max(
        float(x_values.max() - x_values.min()),
        float(y_values.max() - y_values.min()),
        1e-9,
    )
    strokes: List[np.ndarray] = []
    for stroke in sample.sorted_strokes():
        rows = [
            [
                (point.x - center_x) / extent + 0.5,
                (center_y - point.y) / extent + 0.5,
            ]
            for point in stroke.sorted_points()
        ]
        if rows:
            strokes.append(np.asarray(rows, dtype=np.float64))
    return strokes


def apply_coordinate_transform(points: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("points must have shape [N, 2]")
    x_values, y_values = values[:, 0], values[:, 1]
    transforms = {
        "identity": (x_values, y_values),
        "flip_x": (1.0 - x_values, y_values),
        "flip_y": (x_values, 1.0 - y_values),
        "rotate_90": (1.0 - y_values, x_values),
        "rotate_180": (1.0 - x_values, 1.0 - y_values),
        "rotate_270": (y_values, 1.0 - x_values),
        "transpose": (y_values, x_values),
        "anti_transpose": (1.0 - y_values, 1.0 - x_values),
    }
    if name not in transforms:
        raise ValueError(f"Unknown coordinate transform: {name}")
    transformed = np.column_stack(transforms[name])
    return np.clip(transformed, 0.0, 1.0)


def transform_strokes(
    strokes: Sequence[np.ndarray], name: str
) -> List[np.ndarray]:
    return [apply_coordinate_transform(stroke, name) for stroke in strokes]


def place_strokes(
    strokes: Sequence[np.ndarray],
    image_size: int,
    scale: float,
    offset_x: float,
    offset_y: float,
) -> List[np.ndarray]:
    center = 0.5 * (image_size - 1)
    return [
        (np.asarray(stroke) - 0.5) * (image_size - 1) * scale
        + np.asarray([center + offset_x, center + offset_y])
        for stroke in strokes
    ]


def _polyline_length(strokes: Sequence[np.ndarray]) -> float:
    return float(
        sum(
            np.linalg.norm(np.diff(stroke, axis=0), axis=1).sum()
            for stroke in strokes
            if len(stroke) >= 2
        )
    )


def estimate_line_width(
    strokes: Sequence[np.ndarray], target_area: int, image_size: int
) -> int:
    length = _polyline_length(strokes)
    if length <= 1e-9:
        return 1
    return int(np.clip(round(target_area / length), 1, max(2, image_size // 8)))


def rasterize_strokes(
    strokes: Sequence[np.ndarray], image_size: int, line_width: int
) -> np.ndarray:
    image = Image.new("1", (image_size, image_size), 0)
    draw = ImageDraw.Draw(image)
    radius = max(0.5, line_width / 2.0)
    for stroke in strokes:
        points = [(float(row[0]), float(row[1])) for row in stroke]
        if len(points) >= 2:
            draw.line(points, fill=1, width=line_width, joint="curve")
        elif points:
            x_value, y_value = points[0]
            draw.ellipse(
                (
                    x_value - radius,
                    y_value - radius,
                    x_value + radius,
                    y_value + radius,
                ),
                fill=1,
            )
    return np.asarray(image, dtype=bool)


def alignment_metrics(
    candidate_mask: np.ndarray,
    target_mask: np.ndarray,
    target_distance: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    if candidate_mask.shape != target_mask.shape:
        raise ValueError("candidate and target mask shapes differ")
    if not np.any(candidate_mask):
        return {
            "symmetric_chamfer_px": float("inf"),
            "trajectory_to_target_px": float("inf"),
            "target_to_trajectory_px": float("inf"),
            "dice": 0.0,
            "iou": 0.0,
        }
    if target_distance is None:
        target_distance = distance_transform_edt(~target_mask)
    candidate_distance = distance_transform_edt(~candidate_mask)
    trajectory_to_target = float(target_distance[candidate_mask].mean())
    target_to_trajectory = float(candidate_distance[target_mask].mean())
    intersection = float(np.logical_and(candidate_mask, target_mask).sum())
    union = float(np.logical_or(candidate_mask, target_mask).sum())
    candidate_area = float(candidate_mask.sum())
    target_area = float(target_mask.sum())
    return {
        "symmetric_chamfer_px": 0.5
        * (trajectory_to_target + target_to_trajectory),
        "trajectory_to_target_px": trajectory_to_target,
        "target_to_trajectory_px": target_to_trajectory,
        "dice": (2.0 * intersection + 1e-6)
        / (candidate_area + target_area + 1e-6),
        "iou": (intersection + 1e-6) / (union + 1e-6),
    }


def _evaluate_candidate(
    strokes: Sequence[np.ndarray],
    target_mask: np.ndarray,
    target_distance: np.ndarray,
    scale: float,
    offset_x: float,
    offset_y: float,
) -> Dict[str, Any]:
    image_size = target_mask.shape[0]
    placed = place_strokes(strokes, image_size, scale, offset_x, offset_y)
    line_width = estimate_line_width(placed, int(target_mask.sum()), image_size)
    candidate_mask = rasterize_strokes(placed, image_size, line_width)
    values: Dict[str, Any] = {
        "scale": float(scale),
        "offset_x_px": float(offset_x),
        "offset_y_px": float(offset_y),
        "line_width_px": line_width,
        **alignment_metrics(candidate_mask, target_mask, target_distance),
    }
    return values


def _search_grid(
    strokes: Sequence[np.ndarray],
    target_mask: np.ndarray,
    target_distance: np.ndarray,
    scales: Sequence[float],
    offsets_x: Sequence[float],
    offsets_y: Sequence[float],
) -> Dict[str, Any]:
    best: Optional[Dict[str, Any]] = None
    for scale in scales:
        for offset_x in offsets_x:
            for offset_y in offsets_y:
                candidate = _evaluate_candidate(
                    strokes,
                    target_mask,
                    target_distance,
                    float(scale),
                    float(offset_x),
                    float(offset_y),
                )
                if best is None or candidate["symmetric_chamfer_px"] < best[
                    "symmetric_chamfer_px"
                ]:
                    best = candidate
    if best is None:
        raise RuntimeError("Alignment search produced no candidates")
    return best


def search_alignment(
    strokes: Sequence[np.ndarray],
    target_mask: np.ndarray,
    min_scale: float = 0.65,
    max_scale: float = 1.15,
    max_offset_ratio: float = 0.25,
    coarse_steps: int = 7,
    offset_steps: int = 9,
    fine_steps: int = 5,
) -> Dict[str, Any]:
    if target_mask.ndim != 2 or target_mask.shape[0] != target_mask.shape[1]:
        raise ValueError("target_mask must be a square 2D array")
    if not np.any(target_mask):
        raise ValueError("Target mask contains no foreground pixels")
    if min_scale <= 0.0 or max_scale < min_scale:
        raise ValueError("Invalid scale range")
    if min(coarse_steps, offset_steps, fine_steps) < 2:
        raise ValueError("Search step counts must be >= 2")

    image_size = target_mask.shape[0]
    max_offset = max_offset_ratio * image_size
    target_distance = distance_transform_edt(~target_mask)
    scale_grid = np.linspace(min_scale, max_scale, coarse_steps)
    offset_grid = np.linspace(-max_offset, max_offset, offset_steps)
    coarse = _search_grid(
        strokes,
        target_mask,
        target_distance,
        scale_grid,
        offset_grid,
        offset_grid,
    )

    scale_step = (max_scale - min_scale) / max(coarse_steps - 1, 1)
    offset_step = 2.0 * max_offset / max(offset_steps - 1, 1)
    fine_scales = np.linspace(
        max(min_scale, coarse["scale"] - scale_step),
        min(max_scale, coarse["scale"] + scale_step),
        fine_steps,
    )
    fine_offsets_x = np.linspace(
        max(-max_offset, coarse["offset_x_px"] - offset_step),
        min(max_offset, coarse["offset_x_px"] + offset_step),
        fine_steps,
    )
    fine_offsets_y = np.linspace(
        max(-max_offset, coarse["offset_y_px"] - offset_step),
        min(max_offset, coarse["offset_y_px"] + offset_step),
        fine_steps,
    )
    return _search_grid(
        strokes,
        target_mask,
        target_distance,
        fine_scales,
        fine_offsets_x,
        fine_offsets_y,
    )


def diagnose_transforms(
    strokes: Sequence[np.ndarray],
    target_mask: np.ndarray,
    **search_options: Any,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name in TRANSFORM_NAMES:
        values = search_alignment(
            transform_strokes(strokes, name), target_mask, **search_options
        )
        rows.append({"transform": name, **values})
    rows.sort(key=lambda row: row["symmetric_chamfer_px"])
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def _overlay_image(candidate_mask: np.ndarray, target_mask: np.ndarray) -> Image.Image:
    output = np.full((*target_mask.shape, 3), 255, dtype=np.uint8)
    target_only = np.logical_and(target_mask, ~candidate_mask)
    candidate_only = np.logical_and(candidate_mask, ~target_mask)
    overlap = np.logical_and(target_mask, candidate_mask)
    output[target_only] = (60, 105, 220)
    output[candidate_only] = (220, 55, 45)
    output[overlap] = (35, 155, 85)
    return Image.fromarray(output, mode="RGB")


def _candidate_mask(
    strokes: Sequence[np.ndarray], image_size: int, row: Mapping[str, Any]
) -> Tuple[np.ndarray, List[np.ndarray]]:
    transformed = transform_strokes(strokes, str(row["transform"]))
    placed = place_strokes(
        transformed,
        image_size,
        float(row["scale"]),
        float(row["offset_x_px"]),
        float(row["offset_y_px"]),
    )
    mask = rasterize_strokes(placed, image_size, int(row["line_width_px"]))
    return mask, placed


def save_diagnostic_figure(
    output_path: Path,
    strokes: Sequence[np.ndarray],
    target_mask: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    panel_size = 256
    label_height = 52
    columns = 4
    canvas = Image.new(
        "RGB",
        (columns * panel_size, 2 * (panel_size + label_height)),
        (244, 244, 244),
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    by_name = {str(row["transform"]): row for row in rows}
    for index, name in enumerate(TRANSFORM_NAMES):
        row = by_name[name]
        mask, _ = _candidate_mask(strokes, target_mask.shape[0], row)
        panel = _overlay_image(mask, target_mask).resize(
            (panel_size, panel_size), Image.Resampling.NEAREST
        )
        column, panel_row = index % columns, index // columns
        x_value = column * panel_size
        y_value = panel_row * (panel_size + label_height)
        canvas.paste(panel, (x_value, y_value))
        color = (15, 120, 55) if int(row["rank"]) == 1 else (20, 20, 20)
        if int(row["rank"]) == 1:
            draw.rectangle(
                (x_value, y_value, x_value + panel_size - 1, y_value + panel_size - 1),
                outline=color,
                width=5,
            )
        draw.text(
            (x_value + 7, y_value + panel_size + 5),
            f"#{row['rank']} {name}  chamfer={row['symmetric_chamfer_px']:.2f}px",
            fill=color,
            font=font,
        )
        draw.text(
            (x_value + 7, y_value + panel_size + 24),
            (
                f"s={row['scale']:.3f}  dx={row['offset_x_px']:+.1f}  "
                f"dy={row['offset_y_px']:+.1f}  IoU={row['iou']:.3f}"
            ),
            fill=color,
            font=font,
        )
    canvas.save(output_path)


def write_aligned_points(
    path: Path,
    sample: CharacterTrajectory,
    placed_strokes: Sequence[np.ndarray],
) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["sample_id", "stroke_id", "point_id", "image_x", "image_y"],
        )
        writer.writeheader()
        for stroke, placed in zip(sample.sorted_strokes(), placed_strokes):
            for point, pixel in zip(stroke.sorted_points(), placed):
                writer.writerow(
                    {
                        "sample_id": sample.meta.get("sample_id"),
                        "stroke_id": point.stroke_id,
                        "point_id": point.point_id,
                        "image_x": float(pixel[0]),
                        "image_y": float(pixel[1]),
                    }
                )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _selected_manifest(
    manifest: Sequence[Dict[str, str]], sample_id: Optional[str]
) -> List[Dict[str, str]]:
    if sample_id is None:
        return list(manifest)
    selected = [row for row in manifest if str(row["sample_id"]) == sample_id]
    if not selected:
        raise ValueError(f"sample_id not found in comparison manifest: {sample_id}")
    return selected


def main(args: argparse.Namespace) -> None:
    from config import load_config

    cfg = load_config(args.config)
    samples = load_alignment_trajectory_csv(
        args.trajectory_csv or cfg.data.trajectory_csv,
        timestamp_column=cfg.data.timestamp_column,
    )
    samples_by_id = sample_index(samples)
    manifest = _selected_manifest(
        read_comparison_manifest(args.manifest), args.sample_id
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict[str, Any]] = []
    sample_reports: List[Dict[str, Any]] = []

    for index, item in enumerate(manifest):
        sample_id = str(item["sample_id"])
        if sample_id not in samples_by_id:
            raise ValueError(f"sample_id not found in trajectory CSV: {sample_id}")
        sample = samples_by_id[sample_id]
        target = load_ink_image(item["target_image"], cfg.bbsmg.image_size)
        target_mask = target >= args.target_threshold
        strokes = trajectory_to_unit_strokes(sample)
        rows = diagnose_transforms(
            strokes,
            target_mask,
            min_scale=args.min_scale,
            max_scale=args.max_scale,
            max_offset_ratio=args.max_offset_ratio,
            coarse_steps=args.coarse_steps,
            offset_steps=args.offset_steps,
            fine_steps=args.fine_steps,
        )
        for row in rows:
            row.update(
                {
                    "sample_id": sample_id,
                    "character": item.get("character") or sample.character or "",
                    "target_image": item["target_image"],
                }
            )
        all_rows.extend(rows)
        best = rows[0]
        sample_dir = output_dir / f"{index:03d}_{_safe_stem(sample_id)}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        save_diagnostic_figure(
            sample_dir / "coordinate_diagnostics.png", strokes, target_mask, rows
        )
        best_mask, best_points = _candidate_mask(
            strokes, target_mask.shape[0], best
        )
        _overlay_image(best_mask, target_mask).save(
            sample_dir / "best_alignment.png"
        )
        write_aligned_points(
            sample_dir / "aligned_trajectory_pixels.csv", sample, best_points
        )
        report = {
            "sample_id": sample_id,
            "character": item.get("character") or sample.character or "",
            "target_image": item["target_image"],
            "target_threshold": args.target_threshold,
            "best_transform": best["transform"],
            "current_mapping_rank": next(
                row["rank"] for row in rows if row["transform"] == "identity"
            ),
            "results": rows,
            "note": (
                "This diagnostic checks global 2D coordinate conventions only; "
                "it does not measure robot trajectory accuracy or model domain fit."
            ),
        }
        with open(sample_dir / "alignment_metrics.json", "w", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)
        sample_reports.append(report)
        print(
            f"[SAMPLE {index + 1}/{len(manifest)}] id={sample_id} "
            f"best={best['transform']} chamfer={best['symmetric_chamfer_px']:.3f}px "
            f"IoU={best['iou']:.4f} identity_rank={report['current_mapping_rank']}"
        )

    write_csv(output_dir / "coordinate_diagnostics.csv", all_rows)
    root_report = {
        "config": args.config,
        "trajectory_csv": args.trajectory_csv or cfg.data.trajectory_csv,
        "manifest": args.manifest,
        "samples": sample_reports,
    }
    with open(
        output_dir / "coordinate_diagnostics.json", "w", encoding="utf-8"
    ) as file:
        json.dump(root_report, file, ensure_ascii=False, indent=2)
    print(f"[DONE] Saved coordinate diagnostics to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--trajectory_csv")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sample_id")
    parser.add_argument("--target_threshold", type=float, default=0.5)
    parser.add_argument("--min_scale", type=float, default=0.65)
    parser.add_argument("--max_scale", type=float, default=1.15)
    parser.add_argument("--max_offset_ratio", type=float, default=0.25)
    parser.add_argument("--coarse_steps", type=int, default=7)
    parser.add_argument("--offset_steps", type=int, default=9)
    parser.add_argument("--fine_steps", type=int, default=5)
    parser.add_argument("--output_dir", default="outputs/trajectory_alignment")
    main(parser.parse_args())
