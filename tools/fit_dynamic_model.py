# 中文注释：本文件从标定 CSV 中拟合动态笔刷的宽度、阻尼和偏移多项式。
import argparse
import csv
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from config import load_config, ensure_dirs
from models.dynamic_brush import fit_poly_from_pairs


# 中文注释：读取动态笔刷标定数据。
def load_calibration_csv(csv_path: str) -> Dict[str, List[float]]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Calibration csv not found: {csv_path}")
    cols = {"z": [], "width": [], "drag": [], "offset": []}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cols["z"].append(float(row["z"]))
            cols["width"].append(float(row["width"]))
            cols["drag"].append(float(row["drag"]))
            cols["offset"].append(float(row["offset"]))
    return cols


# 中文注释：按 z 深度聚合标定样本，降低噪声。
def aggregate_by_z(curves: Dict[str, List[float]], method: str = "mean") -> Dict[str, List[float]]:
    bucket: Dict[float, Dict[str, List[float]]] = {}
    for z, w, d, o in zip(curves["z"], curves["width"], curves["drag"], curves["offset"]):
        bucket.setdefault(float(z), {"width": [], "drag": [], "offset": []})
        bucket[float(z)]["width"].append(float(w))
        bucket[float(z)]["drag"].append(float(d))
        bucket[float(z)]["offset"].append(float(o))
    zs = sorted(bucket.keys())
    out = {"z": [], "width": [], "drag": [], "offset": []}
    reduce_fn = np.mean if method == "mean" else np.median
    for z in zs:
        out["z"].append(z)
        out["width"].append(float(reduce_fn(bucket[z]["width"])))
        out["drag"].append(float(reduce_fn(bucket[z]["drag"])))
        out["offset"].append(float(reduce_fn(bucket[z]["offset"])))
    return out


# 中文注释：按比例划分训练集和验证集。
def split_train_val(xs: List[float], ys: List[float], val_ratio: float = 0.2):
    idx = np.arange(len(xs))
    if len(idx) <= 2:
        return (xs, ys), (xs, ys)
    val_len = max(1, int(len(idx) * val_ratio))
    train_idx = idx[:-val_len]
    val_idx = idx[-val_len:]
    xs_np = np.asarray(xs, dtype=float)
    ys_np = np.asarray(ys, dtype=float)
    return (xs_np[train_idx].tolist(), ys_np[train_idx].tolist()), (xs_np[val_idx].tolist(), ys_np[val_idx].tolist())


# 中文注释：计算多项式系数在样本上的均方误差。
def mse_for_coeffs(coeffs: List[float], xs: List[float], ys: List[float]) -> float:
    pred = []
    for x in xs:
        pred.append(sum(c * (float(x) ** i) for i, c in enumerate(coeffs)))
    pred_np = np.asarray(pred, dtype=float)
    ys_np = np.asarray(ys, dtype=float)
    return float(np.mean((pred_np - ys_np) ** 2))


# 中文注释：在候选阶数中选择验证误差最低的多项式拟合。
def auto_fit_curve(xs: List[float], ys: List[float], min_degree: int, max_degree: int, val_ratio: float = 0.2) -> Tuple[List[float], int, float]:
    (xtr, ytr), (xva, yva) = split_train_val(xs, ys, val_ratio=val_ratio)
    best_coeffs = None
    best_degree = min_degree
    best_mse = float("inf")
    for degree in range(min_degree, max_degree + 1):
        coeffs = fit_poly_from_pairs(xtr, ytr, degree=degree).coeffs
        mse = mse_for_coeffs(coeffs, xva, yva)
        if mse < best_mse:
            best_mse = mse
            best_degree = degree
            best_coeffs = coeffs
    return best_coeffs, best_degree, best_mse


# 中文注释：分别拟合宽度、阻尼和偏移曲线。
def fit_all_auto(curves: Dict[str, List[float]], min_degree: int, max_degree: int, val_ratio: float = 0.2) -> Dict[str, object]:
    z = curves["z"]
    width_coeffs, width_deg, width_mse = auto_fit_curve(z, curves["width"], min_degree, max_degree, val_ratio)
    drag_coeffs, drag_deg, drag_mse = auto_fit_curve(z, curves["drag"], min_degree, max_degree, val_ratio)
    offset_coeffs, offset_deg, offset_mse = auto_fit_curve(z, curves["offset"], min_degree, max_degree, val_ratio)
    return {
        "width_coeffs": width_coeffs,
        "drag_coeffs": drag_coeffs,
        "offset_coeffs": offset_coeffs,
        "width_degree": width_deg,
        "drag_degree": drag_deg,
        "offset_degree": offset_deg,
        "width_val_mse": width_mse,
        "drag_val_mse": drag_mse,
        "offset_val_mse": offset_mse,
    }


# 中文注释：把拟合得到的多项式系数保存为 JSON。
def save_coeffs_json(coeffs: Dict[str, object], out_json: str) -> None:
    path = Path(out_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(coeffs, f, ensure_ascii=False, indent=2)


# 中文注释：保存拟合曲线采样点，便于检查。
def save_fit_samples(coeffs: Dict[str, object], z_values: List[float], out_csv: str) -> None:
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 中文注释：按多项式系数计算给定 x 的拟合值。
    def poly(c, x):
        return sum(ci * (x ** i) for i, ci in enumerate(c))
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["z", "width_fit", "drag_fit", "offset_fit"])
        writer.writeheader()
        for z in z_values:
            writer.writerow({
                "z": z,
                "width_fit": poly(coeffs["width_coeffs"], z),
                "drag_fit": poly(coeffs["drag_coeffs"], z),
                "offset_fit": poly(coeffs["offset_coeffs"], z),
            })

# 中文注释：绘制标定散点与拟合曲线。
def plot_fit_curves(raw_curves, agg_curves, coeffs, out_dir: str) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # 中文注释：按多项式系数计算给定 x 的拟合值。
    def poly(c, x):
        return sum(ci * (x ** i) for i, ci in enumerate(c))
    z_min, z_max = min(agg_curves["z"]), max(agg_curves["z"] )
    z_grid = np.linspace(z_min, z_max, 200)
    specs = [("width", "width_coeffs", "width_fit.png"), ("drag", "drag_coeffs", "drag_fit.png"), ("offset", "offset_coeffs", "offset_fit.png")]
    for name, key, file_name in specs:
        plt.figure(figsize=(6, 4))
        plt.scatter(raw_curves["z"], raw_curves[name], s=18, alpha=0.35, label="raw")
        plt.scatter(agg_curves["z"], agg_curves[name], s=30, alpha=0.9, label="aggregated")
        y_fit = [poly(coeffs[key], float(z)) for z in z_grid]
        plt.plot(z_grid, y_fit, linewidth=2.0, label="fitted curve")
        plt.xlabel("z")
        plt.ylabel(name)
        plt.title(f"{name} vs z")
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(out_dir) / file_name, dpi=160)
        plt.close()


# 中文注释：解析命令行参数，准备日志文件并分派到对应子命令。
def main(args):
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    raw_curves = load_calibration_csv(args.calibration_csv)
    curves = aggregate_by_z(raw_curves, method=args.aggregate)
    coeffs = fit_all_auto(curves, min_degree=args.min_degree, max_degree=args.max_degree, val_ratio=args.val_ratio)
    save_coeffs_json(coeffs, args.output_json)
    z_min, z_max = min(curves["z"]), max(curves["z"] )
    z_grid = np.linspace(z_min, z_max, args.num_fit_points).tolist()
    save_fit_samples(coeffs, z_grid, args.output_fit_csv)
    print(f"Saved coeffs to: {args.output_json}")
    print(f"Saved fit samples to: {args.output_fit_csv}")
    print(f"Selected degrees => width: {coeffs['width_degree']}, drag: {coeffs['drag_degree']}, offset: {coeffs['offset_degree']}")
    plot_fit_curves(raw_curves, curves, coeffs, args.output_plot_dir)


# 中文注释：作为脚本直接运行时，从这里进入命令行流程或示例测试。
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to yaml config")
    parser.add_argument("--calibration_csv", type=str, required=True, help="CSV with columns: z,width,drag,offset")
    parser.add_argument("--output_json", type=str, default="data/processed/dynamic_brush_coeffs.json", help="Output coeffs json")
    parser.add_argument("--output_fit_csv", type=str, default="data/processed/dynamic_brush_fit.csv", help="Output fitted curve csv")
    parser.add_argument("--output_plot_dir", type=str, default="data/processed/fit_plots", help="Output directory for fit plots")
    parser.add_argument("--num_fit_points", type=int, default=100, help="Number of sampled fit points")
    parser.add_argument("--aggregate", type=str, default="mean", choices=["mean", "median"], help="Aggregate repeated measurements at same z")
    parser.add_argument("--min_degree", type=int, default=1, help="Minimum polynomial degree")
    parser.add_argument("--max_degree", type=int, default=4, help="Maximum polynomial degree")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation split ratio for auto degree selection")
    args = parser.parse_args()
    main(args)
# 使用说明：该脚本用于把动态笔触模型所需的标定数据拟合成多项式系数。
# 输入应为一个包含表头的 CSV，至少包含四列：z、width、drag、offset，分别表示下压深度与对应测得的笔触宽度、拖拽长度和偏移量。
# 脚本会根据 config.py 中设置的多项式阶数，分别拟合 Width(z)、Drag(z) 和 Offset(z)，并输出两个文件：一个 JSON 系数文件（可供后续替换 DynamicBrushModel 中的占位函数），以及一个采样后的拟合曲线 CSV（便于可视化检查拟合质量）。
# 典型运行方式为：python tools/fit_dynamic_model.py --config configs/default.yaml --calibration_csv data/raw/dynamic_calibration.csv --output_json data/processed/dynamic_brush_coeffs.json --output_fit_csv data/processed/dynamic_brush_fit.csv。

