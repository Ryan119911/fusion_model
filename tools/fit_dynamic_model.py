import argparse
import csv
from pathlib import Path
import sys
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from config import load_config
from models.dynamic_brush import (
    DynamicBrushModel,
    DynamicBrushParams,
    fit_poly_from_pairs,
)


def load_calibration_csv(path: str) -> Dict[str, List[float]]:
    values = {"z": [], "width": [], "drag": [], "offset": []}
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            for key in values:
                values[key].append(float(row[key]))
    if not values["z"]:
        raise ValueError("Calibration CSV is empty")
    return values


def aggregate_by_z(values: Dict[str, List[float]], method: str) -> Dict[str, List[float]]:
    grouped: Dict[float, Dict[str, List[float]]] = {}
    for z, width, drag, offset in zip(
        values["z"], values["width"], values["drag"], values["offset"]
    ):
        bucket = grouped.setdefault(z, {"width": [], "drag": [], "offset": []})
        bucket["width"].append(width)
        bucket["drag"].append(drag)
        bucket["offset"].append(offset)
    reducer = np.mean if method == "mean" else np.median
    output = {"z": [], "width": [], "drag": [], "offset": []}
    for z in sorted(grouped):
        output["z"].append(z)
        for key in ("width", "drag", "offset"):
            output[key].append(float(reducer(grouped[z][key])))
    return output


def validation_mse(coeffs: List[float], xs: List[float], ys: List[float]) -> float:
    predictions = np.polynomial.polynomial.polyval(xs, coeffs)
    return float(np.mean((predictions - np.asarray(ys)) ** 2))


def choose_polynomial(
    xs: List[float],
    ys: List[float],
    minimum: int,
    maximum: int,
    val_ratio: float,
):
    count = len(xs)
    val_count = max(1, int(round(count * val_ratio))) if count > 2 else count
    train_count = max(1, count - val_count)
    x_train, y_train = xs[:train_count], ys[:train_count]
    x_val, y_val = xs[train_count:], ys[train_count:]
    if not x_val:
        x_val, y_val = xs, ys
    candidates = []
    for degree in range(minimum, maximum + 1):
        fit = fit_poly_from_pairs(x_train, y_train, degree)
        candidates.append((validation_mse(fit.coeffs, x_val, y_val), degree, fit))
    _, degree, _ = min(candidates, key=lambda item: item[0])
    final_fit = fit_poly_from_pairs(xs, ys, degree)
    return final_fit, degree, validation_mse(final_fit.coeffs, xs, ys)


def main(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    raw = load_calibration_csv(args.calibration_csv)
    curves = aggregate_by_z(raw, args.aggregate)
    fits = {}
    diagnostics = {}
    for key in ("width", "drag", "offset"):
        fit, degree, mse = choose_polynomial(
            curves["z"], curves[key], args.min_degree, args.max_degree, args.val_ratio
        )
        fits[key] = fit
        diagnostics[f"{key}_degree"] = degree
        diagnostics[f"{key}_mse"] = mse
    model = DynamicBrushModel(
        DynamicBrushParams(
            mode="calibrated",
            kw=cfg.dynamic_brush.kw,
            kd=cfg.dynamic_brush.kd,
            dt=cfg.dynamic_brush.dt,
            snap_clip_min=cfg.dynamic_brush.snap_clip_min,
            width_fn=fits["width"],
            drag_fn=fits["drag"],
            offset_fn=fits["offset"],
        )
    )
    model.save_json(args.output_json)
    output_csv = Path(args.output_fit_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    z_grid = np.linspace(min(curves["z"]), max(curves["z"]), args.num_fit_points)
    with open(output_csv, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=["z", "width_fit", "drag_fit", "offset_fit"]
        )
        writer.writeheader()
        for z in z_grid:
            writer.writerow({
                "z": float(z),
                "width_fit": model.width(float(z)),
                "drag_fit": model.drag(float(z)),
                "offset_fit": model.offset(float(z)),
            })
    print(f"[DONE] model={args.output_json}")
    print(f"[DONE] samples={args.output_fit_csv}")
    print(f"[FIT] {diagnostics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--calibration_csv", required=True)
    parser.add_argument("--output_json", default="data/processed/dynamic_brush_coeffs.json")
    parser.add_argument("--output_fit_csv", default="data/processed/dynamic_brush_fit.csv")
    parser.add_argument("--aggregate", choices=["mean", "median"], default="mean")
    parser.add_argument("--min_degree", type=int, default=1)
    parser.add_argument("--max_degree", type=int, default=4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--num_fit_points", type=int, default=100)
    main(parser.parse_args())
