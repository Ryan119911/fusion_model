# 纯 U-Net 整字模型（以“武”为例）

## 模型变化

旧整字模型将每笔压缩成 10D token，随后使用 Transformer 和全局池化。这会丢失弯曲、转折、斜钩和精确空间位置，容易生成平均灰图。

当前版本改为：

```text
完整轨迹
  → 6×128×128 空间条件图
  → 纯 U-Net（GroupNorm + 多尺度跳跃连接）
  → 1×128×128 完整字图
```

六个输入通道依次为：

```text
centerline      完整字中心线
proximity       中心线的连续邻域（抑制平均灰雾）
pressure        归一化压力
stroke_order    笔顺
direction_cos   局部轨迹方向余弦
direction_sin   局部轨迹方向正弦
```

U-Net 直接处理完整空间结构，不会逐笔生成后叠加，也不包含 Transformer。
输出端以 `proximity` 形成轨迹先验，最终卷积头从零初始化；整字损失额外约束加权 BCE、背景灰雾、总墨量和轨迹覆盖，避免重新退化成大面积灰色平均图。

## 重要兼容说明

必须重新构建整字 NPZ。旧 U-Net 权重可以迁移，但不能原样续训：

```text
旧 character_sequence_v1 NPZ：不兼容
旧 character_generator_v1 checkpoint：不兼容
旧 character_spatial_v2 NPZ：不兼容，必须重建
旧 character_spatial_v3 NPZ：不兼容，缺少目标净化与配准
旧 character_spatial_v4 NPZ：不兼容，缺少繁简匹配与双向质量过滤
旧 character_unet_v2 checkpoint：只能通过 --init_character_checkpoint 迁移
单笔 bbsmg_best.pt：不兼容
新 NPZ：character_spatial_v5
新 checkpoint：character_unet_v3
```

程序会主动拒绝旧文件，避免静默加载错误模型。

## 1. Ubuntu 更新代码

```bash
cd ~/coppeliasim/machine_learning/model
git switch main
git pull origin main
conda activate ddpm
```

目标图已经位于：

```text
assets/targets/wu_kaishu_target.png
```

## 2. 重新构建 U-Net 整字数据

先安装单字标签繁简转换依赖：

```bash
python -m pip install -r requirements-character.txt
```

古代书法图像中的字形通常是繁体，但原始标注可能是简体。v5 默认使用
`--target_script traditional`，把图像标注映射到对应繁体轨迹。例如“丑”图像会配给
“醜”轨迹，“儿”图像会配给“兒”轨迹。`trajectories.csv` 必须包含这些繁体字符。

不要覆盖旧 NPZ，使用新的文件名：

```bash
python -u tools/build_character_pairs.py \
  --config configs/default.yaml \
  --trajectory_csv data/raw/trajectories.csv \
  --output_npz data/processed/character_unet_clean_v5.npz \
  --chirography 楷 \
  --target_character 武 \
  --target_image assets/targets/wu_kaishu_target.png \
  --require_real_target \
  --trajectory_padding 16 \
  --trajectory_width 3 \
  --target_script traditional \
  --min_alignment_coverage 0.55 \
  --min_support_dice 0.45 \
  --max_outside_support_fraction 0.35 \
  --min_target_support_area_ratio 0.30 \
  --max_target_support_area_ratio 1.70 \
  --max_target_ink_fraction 0.45 \
  --max_foreground_bbox_fill_fraction 0.55 \
  --max_border_ink_fraction 0.02 \
  --max_target_candidates 64 \
  --max_registered_candidates 8 \
  --audit_limit_per_status 40
```

正确输出必须类似：

```text
inputs shape: (字符数, 6, 128, 128)
targets shape: (字符数, 1, 128, 128)
channels: centerline, proximity, pressure, stroke_order, direction_cos, direction_sin
trajectory/target coverage: mean=...
target quality filter: rejected=...
quality failures: {...}
audit panels: {'accepted': ..., 'rejected': ...}
```

如果 `--require_real_target` 导致字符样本过少，可以去掉它，让没有真实楷书目标的字符使用完整轨迹栅格目标。
程序会先根据图像边界估计纸张背景并归零，再对同字候选进行排序和目标配准。配准只改变监督目标，轨迹输入保持不动。v5 同时检查覆盖率、support Dice、支撑区域外墨迹、目标/支撑面积比、总墨量、前景框填充率和画布边界污染。完整拒绝详情写入同名 `.rejected.json`，聚合统计写入 `.summary.json`，审计图保存在同名 `_audit/accepted` 和 `_audit/rejected` 目录。审计图从左到右为 proximity、配准前目标、配准后目标、红色目标/绿色中心线重叠。

构建全部可用楷书字符时，不要传 `--character`、`--target_character` 或
`--target_image`：

```bash
python -u tools/build_character_pairs.py \
  --config configs/default.yaml \
  --trajectory_csv data/raw/trajectories.csv \
  --output_npz data/processed/character_all_kaishu_clean_v5.npz \
  --chirography 楷 \
  --require_real_target \
  --trajectory_padding 16 \
  --trajectory_width 3 \
  --target_script traditional \
  --min_alignment_coverage 0.55 \
  --min_support_dice 0.45 \
  --max_outside_support_fraction 0.35 \
  --min_target_support_area_ratio 0.30 \
  --max_target_support_area_ratio 1.70 \
  --max_target_ink_fraction 0.45 \
  --max_foreground_bbox_fill_fraction 0.55 \
  --max_border_ink_fraction 0.02 \
  --max_target_candidates 64 \
  --max_registered_candidates 8 \
  --audit_limit_per_status 40
```

训练前必须先人工查看 `_audit/accepted` 和 `_audit/rejected`。如果 accepted 中仍出现
错误字形，不应通过放宽阈值扩大数据集，而应检查对应的 `annotation_character`、
`mapped_target_character` 和源图路径。

## 3. 从头训练 v5 通用 U-Net

GTX 1660 6GB 建议先使用 `batch_size=4`：

```bash
python -u tools/train_character.py \
  --config configs/default.yaml \
  --npz_path data/processed/character_all_kaishu_clean_v5.npz \
  --output_dir outputs/character_general_v5 \
  --epochs 30 \
  --batch_size 4 \
  --val_ratio 0.1 \
  --split_mode character \
  --lr_factor 0.5 \
  --lr_patience 3 \
  --min_lr 0.000001
```

如果显存足够可将 batch size 增加到 8。不要加载或 resume v4 通用模型，因为其权重已经学习了繁简错配和异常目标；v5 首次实验必须从头训练。模型结构仍是 `character_unet_v3`，这里的 v5 指数据格式和监督质量流程。

## 4. 验证整字效果

```bash
python -u tools/evaluate_character.py \
  --config configs/default.yaml \
  --npz_path data/processed/character_all_kaishu_clean_v5.npz \
  --checkpoint outputs/character_general_v5/character_best.pt \
  --output_dir outputs/eval_character_general_v5 \
  --split val \
  --num_images 20
```

重点查看：

```text
dice_score
iou_at_0.5
ssim_score
plain_mse
prediction_ink / target_ink
background_mean
trajectory_prediction_coverage / trajectory_target_coverage
```

每张 `character_*_comparison.png` 从左到右为轨迹、目标、预测、绝对差异。
评估还会生成 `per_character_metrics.csv`，并在 `metrics.json` 中写入宏平均和逐字符指标。泛化实验必须使用按字符互斥的训练/验证划分。

## 5. 生成“武”字对比图

```bash
python -u tools/predict_character.py \
  --config configs/default.yaml \
  --trajectory_csv data/raw/trajectories.csv \
  --checkpoint outputs/character_general_v5/character_best.pt \
  --character 武 \
  --target_image assets/targets/wu_kaishu_target.png \
  --output_dir outputs/wu_unet \
  --output_stem wu_kaishu \
  --trajectory_padding 16 \
  --trajectory_width 3
```

结果：

```text
outputs/wu_unet/wu_kaishu_target.png
outputs/wu_unet/wu_kaishu_trajectory.png
outputs/wu_unet/wu_kaishu_proximity.png
outputs/wu_unet/wu_kaishu_prediction.png
outputs/wu_unet/wu_kaishu_diff.png
outputs/wu_unet/wu_kaishu_comparison.png
outputs/wu_unet/wu_kaishu_metrics.json
```

## 6. 只针对“武”微调

先建立单字空间数据：

```bash
python -u tools/build_character_pairs.py \
  --config configs/default.yaml \
  --trajectory_csv data/raw/trajectories.csv \
  --output_npz data/processed/wu_unet_finetune_v5.npz \
  --character 武 \
  --target_character 武 \
  --target_image assets/targets/wu_kaishu_target.png \
  --trajectory_padding 16 \
  --trajectory_width 3 \
  --min_alignment_coverage 0.55
```

从通用 U-Net 开始微调：

```bash
python -u tools/train_character.py \
  --config configs/default.yaml \
  --npz_path data/processed/wu_unet_finetune_v5.npz \
  --init_character_checkpoint outputs/character_general_v5/character_best.pt \
  --output_dir outputs/wu_unet_finetune \
  --epochs 200 \
  --batch_size 1 \
  --val_ratio 0 \
  --lr_factor 0.5 \
  --lr_patience 10 \
  --min_lr 0.000001
```

微调后使用：

```text
outputs/wu_unet_finetune/character_best.pt
```

## 7. 继续训练

只有新 U-Net checkpoint 才能继续：

```bash
python -u tools/train_character.py \
  --config configs/default.yaml \
  --npz_path data/processed/character_all_kaishu_clean_v5.npz \
  --resume outputs/character_general_v5/character_last.pt \
  --epochs 80 \
  --batch_size 4 \
  --lr_factor 0.5 \
  --lr_patience 3 \
  --min_lr 0.000001
```

`--epochs 80` 表示训练到总计 80 epoch。

## 8. 未见字符泛化测试

不能用单字符训练集自身评价泛化。以“永”为完全未见字符时，只构建测试 NPZ，不参与训练：

```bash
python -u tools/build_character_pairs.py \
  --config configs/default.yaml \
  --trajectory_csv data/raw/trajectories.csv \
  --output_npz data/processed/yong_unseen_clean_v5.npz \
  --character 永 \
  --chirography 楷 \
  --require_real_target \
  --trajectory_padding 16 \
  --trajectory_width 3 \
  --min_alignment_coverage 0.55 \
  --max_target_candidates 64 \
  --max_registered_candidates 8
```

构建成功意味着至少一个目标通过 `0.55` 覆盖阈值；若全部被拒绝，应更换字符或检查 `.rejected.json`，不能降低阈值后强行报告指标。

使用冻结的 checkpoint 直接测试：

```bash
python -u tools/evaluate_character.py \
  --config configs/default.yaml \
  --npz_path data/processed/yong_unseen_clean_v5.npz \
  --checkpoint outputs/character_general_v5/character_best.pt \
  --output_dir outputs/yong_unseen_clean_v5 \
  --split all \
  --character 永 \
  --batch_size 1 \
  --num_images 20
```

正式泛化结论应至少包含 20 个训练中完全未出现的字符，并同时报告：

```text
metrics.json -> macro_metrics.dice_at_0.5
metrics.json -> macro_metrics.iou_at_0.5
metrics.json -> macro_metrics.ink_ratio
per_character_metrics.csv
```
