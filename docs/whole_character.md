# 纯 U-Net 整字模型（以“武”为例）

## 模型变化

旧整字模型将每笔压缩成 10D token，随后使用 Transformer 和全局池化。这会丢失弯曲、转折、斜钩和精确空间位置，容易生成平均灰图。

当前版本改为：

```text
完整轨迹
  → 5×128×128 空间条件图
  → 纯 U-Net（GroupNorm + 多尺度跳跃连接）
  → 1×128×128 完整字图
```

五个输入通道依次为：

```text
centerline      完整字中心线
pressure        归一化压力
stroke_order    笔顺
direction_cos   局部轨迹方向余弦
direction_sin   局部轨迹方向正弦
```

U-Net 直接处理完整空间结构，不会逐笔生成后叠加，也不包含 Transformer。

## 重要兼容说明

必须重新构建整字 NPZ，并从头训练：

```text
旧 character_sequence_v1 NPZ：不兼容
旧 character_generator_v1 checkpoint：不兼容
单笔 bbsmg_best.pt：不兼容
新 NPZ：character_spatial_v2
新 checkpoint：character_unet_v2
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

不要覆盖旧 NPZ，使用新的文件名：

```bash
python -u tools/build_character_pairs.py \
  --config configs/default.yaml \
  --trajectory_csv data/raw/trajectories.csv \
  --output_npz data/processed/character_unet_5ch.npz \
  --chirography 楷 \
  --target_character 武 \
  --target_image assets/targets/wu_kaishu_target.png \
  --require_real_target \
  --trajectory_width 3
```

正确输出必须类似：

```text
inputs shape: (字符数, 5, 128, 128)
targets shape: (字符数, 1, 128, 128)
channels: centerline, pressure, stroke_order, direction_cos, direction_sin
```

如果 `--require_real_target` 导致字符样本过少，可以去掉它，让没有真实楷书目标的字符使用完整轨迹栅格目标。

## 3. 从头训练 U-Net

GTX 1660 6GB 建议先使用 `batch_size=4`：

```bash
python -u tools/train_character.py \
  --config configs/default.yaml \
  --npz_path data/processed/character_unet_5ch.npz \
  --output_dir outputs/character_unet \
  --epochs 50 \
  --batch_size 4 \
  --val_ratio 0.1 \
  --lr_factor 0.5 \
  --lr_patience 3 \
  --min_lr 0.000001
```

如果显存足够可将 batch size 增加到 8。不要对旧 Transformer checkpoint 使用 `--resume` 或 `--init_character_checkpoint`。

## 4. 验证整字效果

```bash
python -u tools/evaluate_character.py \
  --config configs/default.yaml \
  --npz_path data/processed/character_unet_5ch.npz \
  --checkpoint outputs/character_unet/character_best.pt \
  --output_dir outputs/eval_character_unet \
  --split val \
  --num_images 20
```

重点查看：

```text
dice_score
iou_at_0.5
ssim_score
plain_mse
```

每张 `character_*_comparison.png` 从左到右为目标、预测、绝对差异。

## 5. 生成“武”字对比图

```bash
python -u tools/predict_character.py \
  --config configs/default.yaml \
  --trajectory_csv data/raw/trajectories.csv \
  --checkpoint outputs/character_unet/character_best.pt \
  --character 武 \
  --target_image assets/targets/wu_kaishu_target.png \
  --output_dir outputs/wu_unet \
  --output_stem wu_kaishu \
  --trajectory_width 3
```

结果：

```text
outputs/wu_unet/wu_kaishu_target.png
outputs/wu_unet/wu_kaishu_trajectory.png
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
  --output_npz data/processed/wu_unet_finetune.npz \
  --character 武 \
  --target_character 武 \
  --target_image assets/targets/wu_kaishu_target.png \
  --trajectory_width 3
```

从通用 U-Net 开始微调：

```bash
python -u tools/train_character.py \
  --config configs/default.yaml \
  --npz_path data/processed/wu_unet_finetune.npz \
  --init_character_checkpoint outputs/character_unet/character_best.pt \
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
  --npz_path data/processed/character_unet_5ch.npz \
  --resume outputs/character_unet/character_last.pt \
  --epochs 80 \
  --batch_size 4 \
  --lr_factor 0.5 \
  --lr_patience 3 \
  --min_lr 0.000001
```

`--epochs 80` 表示训练到总计 80 epoch。
