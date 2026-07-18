import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import load_config
from datasets.trajectory_dataset import load_trajectory_csv
from models.dynamic_brush import build_dynamic_brush
from models.fusion_renderer import FusionRenderer
from models.geometry import normalize_trajectory_xy_with_bounds, trajectory_bounds
from optim.trajectory_optimizer import TrajectoryOptimizer, load_target_image
from tools.optimize_trajectory import save_character_trajectory_csv, save_image
from utils.comparison_metrics import (
    image_metrics,
    read_comparison_manifest,
    signed_difference_image,
    trajectory_metrics,
)
from utils.types import CharacterTrajectory


IMAGE_METRIC_FIELDS = (
    "mse",
    "mae",
    "foreground_mae",
    "ssim_score",
    "global_ssim",
    "dice_score",
    "dice_at_0.5",
    "iou_at_0.5",
    "ink_mean",
    "ink_delta",
)


def _finite_float(value: Any, default: float = 0.0) -> float:
    result = float(value)
    return result if math.isfinite(result) else default


def _safe_stem(value: Any) -> str:
    text = str(value).strip() or "sample"
    return re.sub(r"[^0-9A-Za-z._-]+", "_", text).strip("_") or "sample"


def _gray_panel(array: np.ndarray, size: int) -> Image.Image:
    image = Image.fromarray(
        np.clip(array * 255.0, 0, 255).astype(np.uint8), mode="L"
    )
    return image.resize((size, size), Image.Resampling.NEAREST).convert("RGB")


def trajectory_overlay(
    initial: CharacterTrajectory,
    generated: CharacterTrajectory,
    size: int,
) -> Image.Image:
    bounds_a = trajectory_bounds(initial)
    bounds_b = trajectory_bounds(generated)
    bounds = (
        min(bounds_a[0], bounds_b[0]),
        max(bounds_a[1], bounds_b[1]),
        min(bounds_a[2], bounds_b[2]),
        max(bounds_a[3], bounds_b[3]),
    )
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    initial_points = normalize_trajectory_xy_with_bounds(
        initial, bounds, canvas_size=size, padding=8
    )
    generated_points = normalize_trajectory_xy_with_bounds(
        generated, bounds, canvas_size=size, padding=8
    )
    for points in initial_points:
        if len(points) >= 2:
            draw.line(points, fill=(35, 110, 220), width=3)
    for points in generated_points:
        if len(points) >= 2:
            draw.line(points, fill=(220, 55, 45), width=2)
    return canvas


def save_rgb(array: np.ndarray, output_path: Path) -> None:
    image = Image.fromarray(
        np.clip(array * 255.0, 0, 255).astype(np.uint8), mode="RGB"
    )
    image.save(output_path)


def save_comparison_figure(
    output_path: Path,
    target: np.ndarray,
    initial: np.ndarray,
    generated: np.ndarray,
    overlay: Image.Image,
    initial_metrics: Dict[str, float],
    generated_metrics: Dict[str, float],
) -> None:
    panel_size = 256
    label_height = 32
    columns = [
        ("Target", _gray_panel(target, panel_size)),
        ("Initial render", _gray_panel(initial, panel_size)),
        ("Generated render", _gray_panel(generated, panel_size)),
        (
            "Absolute difference",
            _gray_panel(np.abs(generated - target), panel_size),
        ),
        (
            "Signed difference",
            Image.fromarray(
                np.clip(
                    signed_difference_image(generated, target) * 255.0,
                    0,
                    255,
                ).astype(np.uint8),
                mode="RGB",
            ).resize((panel_size, panel_size), Image.Resampling.NEAREST),
        ),
        ("Trajectory overlay", overlay.resize((panel_size, panel_size))),
    ]
    canvas = Image.new(
        "RGB",
        (panel_size * 3, (panel_size + label_height) * 2 + 72),
        (245, 245, 245),
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, (label, image) in enumerate(columns):
        column = index % 3
        row = index // 3
        x = column * panel_size
        y = row * (panel_size + label_height)
        canvas.paste(image, (x, y))
        draw.text((x + 8, y + panel_size + 8), label, fill=(20, 20, 20), font=font)
    metric_y = 2 * (panel_size + label_height) + 8
    draw.text(
        (8, metric_y),
        (
            "Initial: "
            f"MSE={initial_metrics['mse']:.6f}  "
            f"Dice={initial_metrics['dice_score']:.4f}  "
            f"IoU={initial_metrics['iou_at_0.5']:.4f}"
        ),
        fill=(35, 110, 220),
        font=font,
    )
    draw.text(
        (8, metric_y + 24),
        (
            "Generated: "
            f"MSE={generated_metrics['mse']:.6f}  "
            f"Dice={generated_metrics['dice_score']:.4f}  "
            f"IoU={generated_metrics['iou_at_0.5']:.4f}"
        ),
        fill=(200, 45, 35),
        font=font,
    )
    canvas.save(output_path)


def sample_index(samples: Sequence[CharacterTrajectory]) -> Dict[str, CharacterTrajectory]:
    result: Dict[str, CharacterTrajectory] = {}
    for sample in samples:
        sample_id = str(sample.meta.get("sample_id"))
        if sample_id in result:
            raise ValueError(f"Duplicate trajectory sample_id: {sample_id}")
        result[sample_id] = sample
    return result


def prefix_metrics(prefix: str, values: Dict[str, Any]) -> Dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    numeric_fields = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and (
                key.startswith("initial_")
                or key.startswith("generated_")
                or key.startswith("delta_")
                or key.startswith("trajectory_")
                or key.startswith("optimizer_")
            )
        }
    )
    means = {
        key: float(np.mean([_finite_float(row[key]) for row in rows if key in row]))
        for key in numeric_fields
    }
    accepted = sum(bool(row["accepted_update"]) for row in rows)
    return {
        "samples": len(rows),
        "accepted_updates": accepted,
        "acceptance_rate": accepted / max(len(rows), 1),
        "mean_metrics": means,
    }


def grouped_summaries(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("target_kind") or "unspecified"), []).append(row)
    return {name: summarize(values) for name, values in sorted(groups.items())}


def _percentage_change(initial: float, generated: float) -> str:
    if abs(initial) < 1e-12:
        return "n/a"
    return f"{100.0 * (generated - initial) / initial:+.2f}%"


def write_analysis(path: Path, rows: Sequence[Dict[str, Any]], report: Dict[str, Any]) -> None:
    means = report["mean_metrics"]
    initial_mse = means["initial_mse"]
    generated_mse = means["generated_mse"]
    initial_iou = means["initial_iou_at_0.5"]
    generated_iou = means["generated_iou_at_0.5"]
    mean_ink_delta = means["generated_ink_delta"]
    lines = [
        "# 轨迹与真实样本对比分析",
        "",
        "## 汇总",
        "",
        f"- 样本数：{report['samples']}",
        f"- LM 更新被接受：{report['accepted_updates']}（{report['acceptance_rate']:.1%}）",
        f"- 平均 MSE：{initial_mse:.6f} -> {generated_mse:.6f}（{_percentage_change(initial_mse, generated_mse)}）",
        f"- 平均 IoU@0.5：{initial_iou:.4f} -> {generated_iou:.4f}（{generated_iou - initial_iou:+.4f}）",
        f"- 生成图平均墨量差：{mean_ink_delta:+.6f}（正值为多墨，负值为缺墨）",
        "",
        "## 逐样本记录",
        "",
        "| sample_id | 目标类型 | 接受更新 | 优化前 MSE | 优化后 MSE | 优化前 IoU | 优化后 IoU | 墨量差 | XYZ RMSE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['sample_id']} | {row['target_kind']} | "
            f"{'是' if row['accepted_update'] else '否'} | "
            f"{row['initial_mse']:.6f} | {row['generated_mse']:.6f} | "
            f"{row['initial_iou_at_0.5']:.4f} | {row['generated_iou_at_0.5']:.4f} | "
            f"{row['generated_ink_delta']:+.6f} | {row['trajectory_xyz_rmse']:.6f} |"
        )
    lines.extend([
        "",
        "## 原因诊断",
        "",
    ])
    if generated_mse < initial_mse:
        lines.append("- 优化后平均像素误差下降；仍需查看有符号差异图，确认收益不是只来自笔画粗细变化。")
    else:
        lines.append("- 平均像素误差未下降。LM 使用前景加权目标，因此某些已接受更新可能以全局 MSE 换取前景覆盖。")
    if generated_iou > initial_iou:
        lines.append("- 二值重合度提高，说明在 0.5 阈值下笔画位置或覆盖范围更接近目标。")
    else:
        lines.append("- 二值重合度未提高，常见原因是笔画偏移、粗细不匹配，或灰度恰好跨过 0.5 阈值。")
    if mean_ink_delta > 0.005:
        lines.append("- 生成图整体多墨；重点检查红色区域是否来自笔画过宽、下压过深或多笔叠加。")
    elif mean_ink_delta < -0.005:
        lines.append("- 生成图整体缺墨；重点检查蓝色区域是否来自下压不足、笔画过窄或轨迹覆盖不完整。")
    else:
        lines.append("- 总墨量基本平衡，剩余误差更可能来自空间对齐或局部粗细，而不是整体墨量。")
    kinds = report.get("by_target_kind", {})
    if "real" in kinds and "synthetic" in kinds:
        real_mse = kinds["real"]["mean_metrics"].get("generated_mse")
        synthetic_mse = kinds["synthetic"]["mean_metrics"].get("generated_mse")
        if real_mse is not None and synthetic_mse is not None and real_mse > synthetic_mse:
            lines.append("- 真实目标的 MSE 高于合成目标，符合纹理、抗锯齿、笔画宽度或配准造成的合成到真实域差异。")
    lines.extend([
        "- XYZ 和角度差表示优化轨迹相对输入参考轨迹的改变量，不是相对独立机器人真值的误差。",
        "- 这些规则用于定位问题，不构成因果证明；应结合红/蓝/绿差异图和轨迹叠加图逐样本确认。",
        "",
        "## 颜色图例",
        "",
        "- 有符号差异图红色：生成图多出的墨迹。",
        "- 有符号差异图蓝色：真实目标中存在、生成图缺失的墨迹。",
        "- 有符号差异图绿色：生成图与真实目标重合的墨迹。",
        "- 轨迹叠加图蓝色：输入参考轨迹；红色：优化后轨迹。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    samples = load_trajectory_csv(
        args.trajectory_csv or cfg.data.trajectory_csv,
        timestamp_column=cfg.data.timestamp_column,
        validate=cfg.data.validate_trajectories,
    )
    samples_by_id = sample_index(samples)
    manifest = read_comparison_manifest(args.manifest)
    renderer = FusionRenderer(
        image_size=cfg.bbsmg.image_size,
        device=cfg.train.device,
        input_dim=cfg.bbsmg.input_dim,
        feature_schema=cfg.bbsmg.feature_schema,
        latent_dim=cfg.bbsmg.latent_dim,
        base_channels=cfg.bbsmg.base_channels,
        out_channels=cfg.bbsmg.out_channels,
        use_tanh=cfg.bbsmg.use_tanh,
        brush=build_dynamic_brush(cfg.dynamic_brush),
    )
    renderer.load_weights(args.bbsmg_ckpt, args.normalization_npz)
    optimizer = TrajectoryOptimizer(
        renderer=renderer,
        render_samples=cfg.optim.render_samples_per_stroke,
        jacobian_epsilon=cfg.optim.jacobian_epsilon,
        xy_margin_ratio=cfg.optim.xy_margin_ratio,
        z_margin=cfg.optim.z_margin,
        angle_margin_radians=cfg.optim.angle_margin_radians,
        xyz_reg_weight=cfg.optim.xyz_reg_weight,
        z_reg_weight=cfg.optim.z_reg_weight,
        angle_reg_weight=cfg.optim.angle_reg_weight,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    order = args.order if args.order is not None else cfg.optim.cheb_order_max
    if not cfg.optim.cheb_order_min <= order <= cfg.optim.cheb_order_max:
        raise ValueError(
            f"order must be in [{cfg.optim.cheb_order_min}, {cfg.optim.cheb_order_max}]"
        )
    rows: List[Dict[str, Any]] = []
    for item_index, item in enumerate(manifest):
        sample_id = str(item["sample_id"])
        if sample_id not in samples_by_id:
            raise ValueError(f"sample_id not found in trajectory CSV: {sample_id}")
        sample = samples_by_id[sample_id]
        target = load_target_image(item["target_image"], cfg.bbsmg.image_size)
        result = optimizer.optimize(
            sample,
            target,
            order=order,
            damping=cfg.optim.lm_damping,
            max_steps=cfg.optim.lm_max_steps,
            optimize_angles=bool(args.use_6d or cfg.optim.optimize_angles),
        )
        initial_values = image_metrics(result.initial_image, result.target_image)
        generated_values = image_metrics(result.rendered_image, result.target_image)
        trajectory_values = trajectory_metrics(
            sample,
            result.optimized_sample,
            args.trajectory_points,
        )
        row: Dict[str, Any] = {
            "sample_id": sample_id,
            "character": item.get("character") or sample.character or "",
            "target_image": item["target_image"],
            "target_kind": item.get("target_kind", "real"),
            "order": order,
            "lm_success": result.lm_result.success,
            "lm_steps": result.lm_result.num_steps,
            "lm_message": result.lm_result.message,
            "accepted_update": result.final_score < result.initial_score,
            "optimizer_initial_score": result.initial_score,
            "optimizer_final_score": result.final_score,
            **prefix_metrics("initial", initial_values),
            **prefix_metrics("generated", generated_values),
            **prefix_metrics("trajectory", trajectory_values),
        }
        for metric in IMAGE_METRIC_FIELDS:
            row[f"delta_{metric}"] = generated_values[metric] - initial_values[metric]
        rows.append(row)

        sample_dir = output_dir / f"{item_index:03d}_{_safe_stem(sample_id)}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        save_image(result.target_image, sample_dir / "target.png")
        save_image(result.initial_image, sample_dir / "initial.png")
        save_image(result.rendered_image, sample_dir / "generated.png")
        save_image(
            np.abs(result.rendered_image - result.target_image),
            sample_dir / "abs_diff.png",
        )
        save_rgb(
            signed_difference_image(result.rendered_image, result.target_image),
            sample_dir / "signed_diff.png",
        )
        overlay = trajectory_overlay(
            sample, result.optimized_sample, cfg.bbsmg.image_size * 2
        )
        overlay.save(sample_dir / "trajectory_overlay.png")
        save_comparison_figure(
            sample_dir / "comparison.png",
            result.target_image,
            result.initial_image,
            result.rendered_image,
            overlay,
            initial_values,
            generated_values,
        )
        save_character_trajectory_csv(
            result.optimized_sample, sample_dir / "generated_trajectory.csv"
        )
        with open(sample_dir / "metrics.json", "w", encoding="utf-8") as file:
            json.dump(row, file, ensure_ascii=False, indent=2)
        print(
            f"[SAMPLE {item_index + 1}/{len(manifest)}] id={sample_id} "
            f"accepted={row['accepted_update']} "
            f"MSE={initial_values['mse']:.6f}->{generated_values['mse']:.6f} "
            f"IoU={initial_values['iou_at_0.5']:.4f}->{generated_values['iou_at_0.5']:.4f}"
        )

    if not rows:
        raise RuntimeError("Comparison manifest contains no samples")
    write_csv(output_dir / "sample_metrics.csv", rows)
    report = {
        "config": args.config,
        "trajectory_csv": args.trajectory_csv or cfg.data.trajectory_csv,
        "bbsmg_checkpoint": args.bbsmg_ckpt,
        "feature_schema": cfg.bbsmg.feature_schema,
        "dynamic_brush_mode": cfg.dynamic_brush.mode,
        "optimize_angles": bool(args.use_6d or cfg.optim.optimize_angles),
        "order": order,
        **summarize(rows),
        "by_target_kind": grouped_summaries(rows),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    write_analysis(output_dir / "analysis.md", rows, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[DONE] Saved trajectory comparison to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--trajectory_csv")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--bbsmg_ckpt", required=True)
    parser.add_argument("--normalization_npz")
    parser.add_argument("--order", type=int)
    parser.add_argument("--use_6d", action="store_true")
    parser.add_argument("--trajectory_points", type=int, default=128)
    parser.add_argument("--output_dir", default="outputs/trajectory_comparison")
    main(parser.parse_args())
