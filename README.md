# Fusion Model：v6 结构优先整字生成

本仓库包含两条彼此独立的流程：

- 单笔 B-BSMG：保留原有单笔训练与推理代码。
- 整字 U-Net：输入完整汉字轨迹，一次生成完整 `128×128` 汉字结构。

当前整字版本是 **v6 结构优先模型**。它只学习清晰的笔画结构 Mask，暂不学习书法
纹理、纸张纹理、拓片噪声或风格迁移。

```text
trajectories.csv 中的一条完整汉字轨迹
  ↓
6×128×128 空间条件图
  centerline / proximity / pressure / stroke_order / direction_cos / direction_sin
  ↓
纯 U-Net（GroupNorm、多尺度跳跃连接，不使用 Transformer）
  ↓
完整汉字概率图
  ↓ 指定阈值
完整二值结构 Mask
```

v6 的目标是先回答一个明确问题：

> 模型能否根据未见汉字的完整轨迹，生成位置正确、边界清楚、不过度铺墨的完整字形？

风格迁移不属于本版本范围。仅有轨迹而没有风格参考时，模型也无法唯一确定某位
书法家的笔锋和墨色。

## 1. v6 相对 v5 的变化

v5 直接回归灰度书法图。训练目标混有扫描灰度、拓片颗粒和不同笔画粗细，模型容易
输出数据集平均的模糊光带。v6 改为：

1. 对真实目标进行背景归一化、裁剪、繁简映射和轨迹配准。
2. 将配准目标按阈值转换成二值结构 Mask。
3. 删除小型孤立连通域，减少拓片噪点。
4. 使用结构 Mask 重新执行双向质量检查。
5. 使用结构损失训练，不再使用灰度 MSE/SSIM 作为整字主目标。
6. 将固定的平滑 proximity 偏置改为较窄、可学习衰减的 logit 先验。
7. 同时输出概率图和阈值化 Mask，并扫描验证集最佳阈值。

当前格式：

```text
NPZ format       character_spatial_v6
target_mode      binary_structure_mask
checkpoint       character_unet_v4
preprocessing    clean_register_script_structure_v3
```

以下旧文件不兼容：

```text
character_sequence_v1
character_spatial_v2 / v3 / v4 / v5
character_generator_v1
character_unet_v2 / v3
单笔 bbsmg_best.pt
```

v6 必须从原始轨迹和书法图重新构建数据，并从头训练。不能使用 v5 checkpoint
`--resume` 或 `--init_character_checkpoint`。

## 2. Ubuntu 环境准备

```bash
cd ~/coppeliasim/machine_learning/model
git switch main
git pull origin main
conda activate ddpm
python -m pip install -r requirements-character.txt
```

确认以下原始资源存在：

```text
data/raw/trajectories.csv
data/raw/images/
data/raw/json_files/
data/raw/data.csv
assets/targets/wu_kaishu_target.png
```

`trajectories.csv` 至少要包含：

```text
character, stroke_id, point_id, x, y, z, ...
```

项目默认筛选 `data.csv` 中书体为“楷”的图像。OpenCC 用于把图像简体标注映射为
繁体轨迹身份，例如“丑”映射为“醜”、“儿”映射为“兒”。

## 3. 异常目标黑名单

几何检查不能识别所有拓片内容。已确认的异常候选记录在：

```text
configs/character_target_exclusions.json
```

当前文件精确排除“乘”的一个噪声候选：

```json
{
  "character": "乘",
  "image_path": "data/raw/images/152/32.jpg",
  "bbox": [0.0, 2023.0, 391.0, 2408.0]
}
```

它只排除这个图像中的这个边界框，不排除“乘”字的其他候选。后续发现异常图时，
按照审计清单中的 `character`、`image_path` 和 `bbox` 添加新项，不要直接删除审计
PNG，也不要删除整个汉字。

## 4. 构建 v6 全字符楷书数据

不要覆盖 v5 文件：

```bash
python -u tools/build_character_pairs.py \
  --config configs/default.yaml \
  --trajectory_csv data/raw/trajectories.csv \
  --output_npz data/processed/character_all_kaishu_structure_v6.npz \
  --chirography 楷 \
  --require_real_target \
  --trajectory_padding 16 \
  --trajectory_width 3 \
  --target_script traditional \
  --structure_threshold 0.35 \
  --min_component_pixels 8 \
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
  --exclude_candidates_json configs/character_target_exclusions.json \
  --audit_limit_per_status 2000
```

`--audit_limit_per_status 2000` 的目的是尽可能保存全部通过和质量拒绝样本，避免只看
前 40 张产生误判。磁盘空间不足时可以改为 100。

正确输出应包含：

```text
inputs shape: (N, 6, 128, 128)
targets shape: (N, 1, 128, 128)
target mode: binary_structure_mask
target sources: {'real': N}
target quality filter: rejected=...
manually excluded target candidates: 1
trajectory/target coverage: mean=...
```

生成文件：

```text
data/processed/character_all_kaishu_structure_v6.npz
data/processed/character_all_kaishu_structure_v6.summary.json
data/processed/character_all_kaishu_structure_v6.rejected.json
data/processed/character_all_kaishu_structure_v6_audit/
  accepted/
  rejected/
  audit_manifest.json
```

审计图从左到右依次为：

```text
proximity
配准前灰度目标
配准后灰度目标
v6 二值结构目标
红色结构目标 / 绿色轨迹重叠
```

训练前必须检查：

- accepted 中是否仍有错字、大片墨块、错误裁剪或残留拓片内容；
- 二值结构是否保留所有有效笔画；
- rejected 中是否有大量明显正确样本；
- `.summary.json` 中接受数量、拒绝原因和人工排除数量是否合理。

不要为了增加数量直接降低 `0.55` 覆盖阈值。先检查字形映射、源图和候选选择。

## 5. 从头训练 v6

GTX 1660 6GB 建议：

```bash
python -u tools/train_character.py \
  --config configs/default.yaml \
  --npz_path data/processed/character_all_kaishu_structure_v6.npz \
  --output_dir outputs/character_general_structure_v6 \
  --epochs 40 \
  --batch_size 4 \
  --val_ratio 0.1 \
  --split_mode character \
  --lr_factor 0.5 \
  --lr_patience 3 \
  --min_lr 0.000001
```

显存足够时可以使用 `--batch_size 8`。`--split_mode character` 会保证训练字符与验证
字符身份互斥，因此验证集可以评价未见汉字泛化。

主要损失分项：

```text
weighted_bce       二值像素分类
dice_loss          整体重叠
tversky_loss       更强惩罚假阳性和过度铺墨
boundary_loss      笔画边界
background_loss    背景漏墨
confidence_loss    灰色不确定区域
ink_loss           总含墨量
trajectory_loss    仅在目标确有笔画的中心线上施加弱约束
```

输出：

```text
outputs/character_general_structure_v6/
  character_best.pt
  character_last.pt
  character_epoch_*.pt
  training_metrics.csv
  split_manifest.json
```

正式评估始终使用 `character_best.pt`。增加 epoch 不等于效果更好；如果最佳 checkpoint
很早出现，应优先检查验证指标和图像。

## 6. 验证集评估与阈值选择

```bash
python -u tools/evaluate_character.py \
  --config configs/default.yaml \
  --npz_path data/processed/character_all_kaishu_structure_v6.npz \
  --checkpoint outputs/character_general_structure_v6/character_best.pt \
  --output_dir outputs/eval_character_general_structure_v6 \
  --split val \
  --batch_size 4 \
  --num_images 100 \
  --threshold 0.50 \
  --thresholds 0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70
```

重点指标：

```text
macro_metrics.dice_at_threshold
macro_metrics.iou_at_threshold
macro_metrics.boundary_f1
macro_metrics.uncertain_fraction
macro_metrics.ink_ratio
background_mean
trajectory_prediction_coverage
trajectory_target_coverage
best_threshold_by_macro_iou
```

理想方向：

```text
Dice / IoU / Boundary F1       越高越好
uncertain_fraction             越低越好
background_mean                越低越好
ink_ratio                      接近 1
prediction/target coverage     差距缩小
```

输出的 `character_*_comparison.png` 从左到右为：

```text
轨迹
目标结构 Mask
预测概率图
预测二值 Mask
二值绝对差异
```

完整结果位于：

```text
metrics.json
metrics.csv
per_character_metrics.csv
```

从 `metrics.json -> best_threshold_by_macro_iou.threshold` 读取最佳阈值。最终部署和单字
预测应使用同一个阈值，而不是默认永远使用 `0.5`。

## 7. 用冻结模型生成“武”字

假设验证集最佳阈值为 `0.45`：

```bash
python -u tools/predict_character.py \
  --config configs/default.yaml \
  --trajectory_csv data/raw/trajectories.csv \
  --checkpoint outputs/character_general_structure_v6/character_best.pt \
  --character 武 \
  --target_image assets/targets/wu_kaishu_target.png \
  --output_dir outputs/wu_structure_v6 \
  --output_stem wu_structure \
  --trajectory_padding 16 \
  --trajectory_width 3 \
  --threshold 0.45
```

结果：

```text
wu_structure_trajectory.png
wu_structure_proximity.png
wu_structure_target_gray.png
wu_structure_target.png
wu_structure_prediction.png
wu_structure_prediction_mask.png
wu_structure_diff.png
wu_structure_mask_diff.png
wu_structure_comparison.png
wu_structure_metrics.json
```

`prediction.png` 是网络概率图；用于机器人轨迹后续处理或结构评价的正式结果是
`prediction_mask.png`。目标图只用于比较，不会在推理时输入模型。

## 8. 泛化能力的正确验证方式

上面的 `--split val --split_mode character` 已经是多字符泛化测试。还可以建立一个完全
独立、从未参与训练的字符集合。不要使用单个“永”字代表总体泛化能力，建议至少
20 个字符。

以单个“永”做流程检查：

```bash
python -u tools/build_character_pairs.py \
  --config configs/default.yaml \
  --trajectory_csv data/raw/trajectories.csv \
  --output_npz data/processed/yong_unseen_structure_v6.npz \
  --character 永 \
  --chirography 楷 \
  --require_real_target \
  --trajectory_padding 16 \
  --trajectory_width 3 \
  --target_script traditional \
  --structure_threshold 0.35 \
  --min_component_pixels 8 \
  --min_alignment_coverage 0.55 \
  --exclude_candidates_json configs/character_target_exclusions.json
```

冻结 checkpoint 直接评价，不能微调：

```bash
python -u tools/evaluate_character.py \
  --config configs/default.yaml \
  --npz_path data/processed/yong_unseen_structure_v6.npz \
  --checkpoint outputs/character_general_structure_v6/character_best.pt \
  --output_dir outputs/yong_unseen_structure_v6 \
  --split all \
  --character 永 \
  --batch_size 1 \
  --num_images 1 \
  --threshold 0.45
```

必须先确认 `split_manifest.json` 的 `train_characters` 中没有被测字符。

## 9. 继续训练与微调

在完全相同的 v6 NPZ 上继续训练到总计 80 epoch：

```bash
python -u tools/train_character.py \
  --config configs/default.yaml \
  --npz_path data/processed/character_all_kaishu_structure_v6.npz \
  --resume outputs/character_general_structure_v6/character_last.pt \
  --epochs 80 \
  --batch_size 4 \
  --val_ratio 0.1 \
  --split_mode character \
  --lr_factor 0.5 \
  --lr_patience 3 \
  --min_lr 0.000001
```

`--epochs 80` 表示训练到总计 80 epoch，不是额外再训练 80 epoch。

只针对“武”微调可以用于检查模型容量或制作单字演示，但它属于拟合实验，不能再用于
证明泛化能力。微调也只能加载 v6 `character_unet_v4` checkpoint。

## 10. 常见问题

### 提示 NPZ 格式不兼容

说明使用了 v5 或更早数据。重新运行第 4 节，不能仅修改文件名或
`format_version`。

### 提示 checkpoint 不兼容

v6 不能加载 `character_unet_v3` 或 `bbsmg_best.pt`。从头训练 v6。

### 输出概率图仍有灰色边缘

概率图允许存在不确定值。先查看 `prediction_mask.png`，并使用验证集扫描得到的最佳
阈值。若二值 Mask 仍明显偏粗，再根据 `ink_ratio`、`background_mean` 和 Boundary F1
调整模型，而不是仅延长训练。

### 目标 Mask 丢失细笔画

先降低 `--structure_threshold`，例如从 `0.35` 改为 `0.30`；如果只丢失孤立的小笔画，
再把 `--min_component_pixels` 从 `8` 改为 `4`。修改后必须重新检查 accepted 审计图。

### accepted 中出现噪声图

把审计清单中的精确 `character`、`image_path` 和 `bbox` 加入
`configs/character_target_exclusions.json`，然后重新构建 NPZ。

### CUDA 显存不足

依次尝试：

```text
batch_size 8 → 4 → 2
```

不要修改输入尺寸或通道数，否则 checkpoint 与数据格式会不兼容。

## 11. 不属于 v6 的内容

v6 不做：

- 根据参考图迁移书法家风格；
- 生成纸张、拓片和墨色纹理；
- 用单字微调结果证明泛化；
- 逐笔预测后再叠加整字；
- Transformer 或全局 token pooling。

完成结构泛化后，风格迁移应作为独立的第二阶段：

```text
v6 结构 Mask + 风格参考图 → 风格化书法图
```

该阶段不会与当前 v6 结构模型混在一起训练。
