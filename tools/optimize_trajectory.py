# 中文注释：本文件命令行工具：加载目标图像和初始样本，执行轨迹优化并保存结果。
import argparse
from pathlib import Path
import csv

import numpy as np
from PIL import Image

from config import load_config, ensure_dirs
from datasets.trajectory_dataset import load_trajectory_csv
from models.fusion_renderer import FusionRenderer
from optim.trajectory_optimizer import TrajectoryOptimizer, load_target_image


# 中文注释：把优化后的字符轨迹保存为 CSV。
def save_character_trajectory_csv(sample, out_csv: str) -> None:
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["character", "sample_id", "stroke_id", "point_id", "x", "y", "z", "alpha", "beta", "gamma", "state"]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in sample.all_points():
            writer.writerow({
                "character": sample.character,
                "sample_id": sample.meta.get("sample_id"),
                "stroke_id": p.stroke_id,
                "point_id": p.point_id,
                "x": p.x,
                "y": p.y,
                "z": p.z,
                "alpha": p.alpha,
                "beta": p.beta,
                "gamma": p.gamma,
                "state": int(p.state),
            })


# 中文注释：把张量或数组形式的灰度图保存为图片。
def save_image(arr: np.ndarray, out_path: str) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), mode="L")
    img.save(path)


# 中文注释：从轨迹数据集中按编号或字符选择初始样本。
def pick_sample(samples, sample_id=None, character=None, index=0):
    if sample_id is not None:
        for s in samples:
            if str(s.meta.get("sample_id")) == str(sample_id):
                return s
    if character is not None:
        cands = [s for s in samples if s.character == character]
        if len(cands) > 0:
            return cands[min(index, len(cands) - 1)]
    if len(samples) == 0:
        raise RuntimeError("No trajectory samples found")
    return samples[min(index, len(samples) - 1)]


# 中文注释：解析命令行参数，准备日志文件并分派到对应子命令。
def main(args):
    cfg = load_config(args.config)
    ensure_dirs(cfg)

    samples = load_trajectory_csv(args.trajectory_csv or cfg.data.trajectory_csv)
    sample = pick_sample(samples, sample_id=args.sample_id, character=args.character, index=args.index)

    target = load_target_image(args.target_image, image_size=cfg.bbsmg.image_size)

    renderer = FusionRenderer(
        image_size=cfg.bbsmg.image_size,
        device=cfg.train.device,
        input_dim=cfg.bbsmg.input_dim,
        latent_dim=cfg.bbsmg.latent_dim,
        base_channels=cfg.bbsmg.base_channels,
    )

    if args.bbsmg_ckpt is not None:
        renderer.load_weights(
            args.bbsmg_ckpt,
            normalization_npz=args.normalization_npz,
        )

    # 若 trajectory_optimizer.py 已升级为 6D 版，则 use_6d 仅用于日志提示；否则可在此处切换不同优化器实现。
    optimizer = TrajectoryOptimizer(
        renderer=renderer,
        z_reg_weight=cfg.optim.z_reg_weight,
        angle_reg_weight=cfg.optim.angle_reg_weight,
        render_samples=cfg.optim.render_samples_per_stroke,
    )
    print(f"Use 6D optimization: {bool(args.use_6d)}")

    print(f"[CHECK] cfg.optim.lm_max_steps={cfg.optim.lm_max_steps}", flush=True)

    result = optimizer.optimize(
        template=sample,
        target_image=target,
        order=args.order,
        damping=cfg.optim.lm_damping,
        max_steps=cfg.optim.lm_max_steps,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{sample.character or 'sample'}_{sample.meta.get('sample_id', '0')}"

    save_character_trajectory_csv(result.optimized_sample, str(out_dir / f"{stem}_optimized.csv"))
    save_image(result.target_image, str(out_dir / f"{stem}_target.png"))
    save_image(result.rendered_image, str(out_dir / f"{stem}_rendered.png"))

    diff = np.abs(result.rendered_image - result.target_image)
    save_image(diff, str(out_dir / f"{stem}_diff.png"))

    print(f"Optimization success: {result.lm_result.success}")
    print(f"Final cost: {result.lm_result.final_cost:.6f}")
    print(f"Saved outputs to: {out_dir}")


# 中文注释：作为脚本直接运行时，从这里进入命令行流程或示例测试。
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to yaml config")
    parser.add_argument("--trajectory_csv", type=str, default=None, help="Override trajectory csv path")
    parser.add_argument("--target_image", type=str, required=True, help="Target character image path")
    parser.add_argument("--bbsmg_ckpt", type=str, default=None, help="Path to trained B-BSMG checkpoint")
    parser.add_argument("--normalization_npz", type=str, default=None, help="Training NPZ for legacy checkpoints without saved input normalization")
    parser.add_argument("--sample_id", type=str, default=None, help="Select trajectory sample by sample_id")
    parser.add_argument("--character", type=str, default=None, help="Select first trajectory sample for given character")
    parser.add_argument("--index", type=int, default=0, help="Index inside character subset if character is used")
    parser.add_argument("--order", type=int, default=5, help="Chebyshev order")
    parser.add_argument("--use_6d", action="store_true", help="Enable 6D optimization path if supported by trajectory optimizer")
    parser.add_argument("--output_dir", type=str, default="outputs/optimize", help="Output directory")
    args = parser.parse_args()
    main(args)
# 使用说明：该脚本用于把整条闭环真正跑起来。
# 它会读取轨迹 CSV、目标字符图像以及可选的 B-BSMG 权重文件，
# 然后选取一个初始整字轨迹样本，构造 FusionRenderer 与 TrajectoryOptimizer，并执行基于 Chebyshev + LM 的反向优化。
# 新增参数 --use_6d 用于与增强版 main.py 对齐：
# 如果你的 optim/trajectory_optimizer.py 已替换为 6D 版本，则该脚本会按当前优化器实现直接执行 6D 优化；
# 如果仍是 3D 版本，则该参数只作为兼容占位和日志提示，不会破坏现有流程。
# 优化完成后，脚本会导出优化后的轨迹 CSV，以及 target、rendered 和 diff 三张 PNG，便于直接检查误差和收敛效果。
# 典型运行方式为：python tools/optimize_trajectory.py --config configs/default.yaml --target_image data/raw/targets/yong.png --bbsmg_ckpt outputs/bbsmg_best.pt --character 永 --order 5 --use_6d --output_dir outputs/optimize。
