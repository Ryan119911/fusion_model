# Fusion Model

该项目将轨迹条件、B-BSMG 单笔生成器和 Chebyshev/LM 轨迹优化组合为书法轨迹实验流水线。

## 数据语义

- 现有 `bbsmg_train_10d.npz` 使用 `stroke10_v1`：
  `h, heading, heading_copy, x0, y0, x1, y1, dx, dy, length`。
- 无 schema 元数据的旧 10D NPZ/检查点会按 `stroke10_v1` 读取，不会静默改成姿态语义。
- 新姿态实验使用 `stroke10_pose_v2`：
  `z, alpha, beta, x0, y0, x1, y1, dx, dy, length`。
- `z` 越大表示下压越深；角度按弧度记录。当前数据的姿态角全为零，因此 6D 优化只能在新数据训练的 pose-v2 模型上使用。
- 未提供真实 `z -> width/drag/offset` 标定时，动态笔刷默认 `disabled`。

## Ubuntu 环境

现有 `ddpm` 环境可补齐依赖：

```bash
conda activate ddpm
pip install -r requirements.txt
```

首次训练会把 7 GB NPZ 中的 `targets.npy` 提取到
`data/cache/npz_arrays`，随后使用磁盘映射，避免全部目标图常驻内存。需要至少约 8 GB 额外磁盘空间。

## 继续训练

`--epochs` 是目标总 epoch，不是额外 epoch 数：

```bash
python -u tools/train_bbsmg.py \
  --config configs/default.yaml \
  --npz_path data/processed/bbsmg_train_10d.npz \
  --resume outputs/bbsmg_10d_full/bbsmg_last.pt \
  --epochs 30 \
  --val_ratio 0.1 \
  --lr_factor 0.5 \
  --lr_patience 3 \
  --min_lr 0.000001
```

每个 epoch 更新 `bbsmg_last.pt`，验证最优模型写入 `bbsmg_best.pt`，
每 10 epoch 保留一个快照。默认启用 FP16 AMP，适配 GTX 1660 6GB。

## 审计与评估

```bash
python tools/audit_npz.py \
  --npz_path data/processed/bbsmg_train_10d.npz \
  --output outputs/data_audit.json

python tools/evaluate_bbsmg.py \
  --config configs/default.yaml \
  --npz_path data/processed/bbsmg_train_10d.npz \
  --checkpoint outputs/bbsmg_10d_full/bbsmg_best.pt \
  --output_dir outputs/evaluate_bbsmg
```

评估报告分别包含全部、真实图像目标和合成目标三个子集。

## 消融实验

默认实验比较随机划分、分组划分、完整损失和去除拓扑/结构损失。分组实验共用同一 manifest：

```bash
python -u tools/run_ablation.py \
  --npz_path data/processed/bbsmg_train_10d.npz \
  --epochs 30 \
  --resume

python tools/summarize_experiments.py
```

同步回 `outputs/ablations` 和训练日志即可继续分析。汇总脚本会生成 `summary.csv`、
`subgroup_summary.csv` 和中文 `analysis.md`。

四组实验的变量表、当前结果分析和下一轮实验建议见
`docs/ablation_and_trajectory_analysis.md`。

批量生成优化轨迹与真实目标的差异图：

```bash
python -u tools/compare_trajectories.py \
  --config configs/ablations/stroke10_v1_c_grouped_full.yaml \
  --trajectory_csv data/raw/trajectories.csv \
  --manifest configs/trajectory_comparison_manifest.csv \
  --bbsmg_ckpt outputs/ablations/stroke10_v1_c_grouped_full/bbsmg_best.pt \
  --normalization_npz data/processed/bbsmg_train_10d.npz \
  --output_dir outputs/trajectory_comparison_c
```

manifest 格式参考 `configs/trajectory_comparison_manifest.example.csv`。输出包含逐样本对比图、
彩色差异图、优化轨迹 CSV、指标 CSV/JSON 和自动生成的中文分析报告。

## 姿态模型与轨迹优化

先用 `configs/pose_v2.yaml` 重建 NPZ 并训练，才能启用 `--use_6d`。
旧 `stroke10_v1` 模型只消费高度、方向和二维几何，代码会拒绝对它进行无效的角度优化。

## 测试

```bash
python -m unittest discover -s tests -v
python -m compileall -q .
```

## Character-level inference comparison

This command renders the complete `武_fake_sim` trajectory directly with B-BSMG and compares it with the target image on one 128x128 canvas. It does not read or stitch per-stroke images from `outputs/eval_10d_full`.

```bash
python -u main.py render-character \
  --config configs/default.yaml \
  --trajectory_csv data/raw/trajectories.csv \
  --character 武 \
  --sample_id 武_fake_sim \
  --target_image data/raw/targets/wu_kaishu_target.png \
  --checkpoint outputs/bbsmg_10d_full/bbsmg_best.pt \
  --normalization_npz data/processed/bbsmg_train_10d.npz \
  --output_dir outputs/character_comparisons
```

The output directory contains target, prediction, absolute difference, and `target | prediction | absolute difference` images, plus a JSON report with MAE, MSE, Dice, and IoU. Omit `--normalization_npz` when the checkpoint already stores input normalization.
