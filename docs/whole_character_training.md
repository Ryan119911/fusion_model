# 整字一致性训练与评估

本流程解决旧版伪标签中“同一个字的不同笔画来自不同整字图片”的问题，并在单笔损失之外增加整字融合损失。

## 重要说明

- 必须重新生成 NPZ；旧版 `bbsmg_train_10d.npz` 不包含完整整字标签。
- 必须从头训练；不要通过 `--resume` 继续旧的 epoch 30 checkpoint。
- 默认训练目标为：`单笔复合损失 + 0.25 × 整字复合损失`。
- 每个 batch 会保留完整 `sample_id`，不会把同一个字的笔画拆到不同 batch。
- 训练、评估和外部目标图统一使用：黑底白墨、前景裁剪、等比例缩放、4 像素留白。

## Ubuntu 22.04 命令

进入仓库并更新代码：

```bash
cd ~/coppeliasim/machine_learning/model
git switch main
git pull origin main
conda activate ddpm
```

重新构建一致性数据。输出使用新文件名，保留旧 NPZ 方便回溯：

```bash
python -u tools/build_pseudo_pairs.py \
  --config configs/default.yaml \
  --output_npz data/processed/bbsmg_train_10d_consistent.npz \
  --chirography 楷
```

如果原始书法数据没有书体字段，去掉 `--chirography 楷`。

审计数据。命令返回 0，且报告中的 `consistent_character_groups` 和
`valid_character_target_mapping` 都为 `true` 才能继续：

```bash
python -u tools/audit_npz.py \
  --npz_path data/processed/bbsmg_train_10d_consistent.npz \
  --output outputs/bbsmg_10d_character/data_audit.json
```

从头训练 30 epoch：

```bash
python -u tools/train_bbsmg.py \
  --config configs/default.yaml \
  --npz_path data/processed/bbsmg_train_10d_consistent.npz \
  --output_dir outputs/bbsmg_10d_character \
  --epochs 30 \
  --val_ratio 0.1 \
  --character_loss_weight 0.25 \
  --lr_factor 0.5 \
  --lr_patience 3 \
  --min_lr 0.000001
```

训练完成后同时评估单笔和整字。整字对比图命名中包含汉字与 `sample_id`：

```bash
python -u tools/evaluate_bbsmg.py \
  --config configs/default.yaml \
  --npz_path data/processed/bbsmg_train_10d_consistent.npz \
  --checkpoint outputs/bbsmg_10d_character/bbsmg_best.pt \
  --split_manifest outputs/bbsmg_10d_character/split_manifest.json \
  --output_dir outputs/eval_10d_character \
  --num_images 20 \
  --num_character_images 20
```

报告位置：

```text
outputs/eval_10d_character/metrics.json
outputs/eval_10d_character/metrics.csv
outputs/eval_10d_character/character_*_comparison.png
```

重点观察 `character_metrics.all` 中的：

```text
composite_loss
dice_at_0.5
iou_at_0.5
ssim_score
ink_delta
```

## 生成“武”字外部目标对比图

将目标图放到 `assets/targets/wu_kaishu_target.png`，然后执行：

```bash
python -u main.py render-character \
  --config configs/default.yaml \
  --trajectory_csv data/raw/trajectories.csv \
  --character 武 \
  --sample_id 武_fake_sim \
  --target_image assets/targets/wu_kaishu_target.png \
  --checkpoint outputs/bbsmg_10d_character/bbsmg_best.pt \
  --output_dir outputs/wu_character_comparison
```

新版 checkpoint 自带输入归一化，因此不需要 `--normalization_npz`。输出布局为：

```text
目标图 | 生成图 | 绝对差异图
```

## 继续训练

只有新版整字监督 checkpoint 才能继续训练。例如从 30 训练到总计 50 epoch：

```bash
python -u tools/train_bbsmg.py \
  --config configs/default.yaml \
  --npz_path data/processed/bbsmg_train_10d_consistent.npz \
  --resume outputs/bbsmg_10d_character/bbsmg_last.pt \
  --epochs 50 \
  --character_loss_weight 0.25 \
  --lr_factor 0.5 \
  --lr_patience 3 \
  --min_lr 0.000001
```

这里 `--epochs 50` 表示训练到总计 50 epoch，不是额外增加 50 epoch。

## 显存不足时

整字分组不能拆开，所以单个笔画很多的字可能使实际 batch 略大于 `batch_size`。如果 GTX 1660 6GB 出现 CUDA OOM，先将 `configs/default.yaml` 中的 `batch_size` 从 16 改为 8；不要关闭整字分组。
