# 整字模型（以“武”为例）

## 1. Ubuntu 22.04 更新仓库

远端 `main` 的历史已按要求恢复到 `014c0f1` 后重新开发。如果 Ubuntu 上已有旧的 `main`，先确认 `git status` 没有未提交内容，再同步：

```bash
cd ~/coppeliasim/machine_learning/model
git status
git fetch origin
git switch main
git reset --hard origin/main
conda activate ddpm
```

原来的实验代码完整保存在远端 `style` 分支。

## 2. 放置“武”字目标图

把目标图复制为：

```text
assets/targets/wu_kaishu_target.png
```

图像可以是白底黑字或黑底白字。程序会自动统一为黑底白墨、裁掉空白边、等比缩放并留 4 像素边距。

## 3. 构建整字数据

推荐使用全部楷书整字训练，并用指定图片覆盖“武”的监督目标：

```bash
python -u tools/build_character_pairs.py \
  --config configs/default.yaml \
  --trajectory_csv data/raw/trajectories.csv \
  --output_npz data/processed/character_train_10d.npz \
  --chirography 楷 \
  --target_character 武 \
  --target_image assets/targets/wu_kaishu_target.png \
  --require_real_target
```

成功时输入应为：

```text
[字符样本数, 64, 10]
```

而不是旧数据的 `[笔画样本数, 10]`。`64` 是补齐后的最大笔画数，`stroke_masks` 会标记每个字实际有几笔。

如果 `--require_real_target` 后样本数太少，可去掉该参数，缺少真实楷书图的字符会使用完整轨迹栅格图作为整字监督。

## 4. 从头训练整字模型

推荐先从头训练 50 epoch：

```bash
python -u tools/train_character.py \
  --config configs/default.yaml \
  --npz_path data/processed/character_train_10d.npz \
  --output_dir outputs/character_10d \
  --epochs 50 \
  --batch_size 16 \
  --val_ratio 0.1 \
  --lr_factor 0.5 \
  --lr_patience 3 \
  --min_lr 0.000001
```

也可以用已经训练好的单笔模型做参数初始化，但不能使用 `--resume`：

```bash
python -u tools/train_character.py \
  --config configs/default.yaml \
  --npz_path data/processed/character_train_10d.npz \
  --output_dir outputs/character_10d_from_strokes \
  --init_stroke_checkpoint outputs/bbsmg_10d_full/bbsmg_best.pt \
  --epochs 50 \
  --batch_size 16 \
  --val_ratio 0.1 \
  --lr_factor 0.5 \
  --lr_patience 3 \
  --min_lr 0.000001
```

`--init_stroke_checkpoint` 只迁移兼容的编码器/解码器参数；新 Transformer 仍从头学习。旧的 `bbsmg_last.pt` 不能作为整字训练的 `--resume`。

## 5. 评估整字验证集

```bash
python -u tools/evaluate_character.py \
  --config configs/default.yaml \
  --npz_path data/processed/character_train_10d.npz \
  --checkpoint outputs/character_10d/character_best.pt \
  --output_dir outputs/eval_character_10d \
  --split val \
  --num_images 20
```

关注 `dice_score`、`iou_at_0.5`、`ssim_score`、`plain_mse`，并查看：

```text
outputs/eval_character_10d/character_*_comparison.png
```

每张对比图从左到右固定为：目标图、整字预测图、绝对差异图。

## 6. 生成“武”字对比图

```bash
python -u tools/predict_character.py \
  --config configs/default.yaml \
  --trajectory_csv data/raw/trajectories.csv \
  --checkpoint outputs/character_10d/character_best.pt \
  --character 武 \
  --target_image assets/targets/wu_kaishu_target.png \
  --output_dir outputs/wu_whole_character \
  --output_stem wu_kaishu
```

结果：

```text
outputs/wu_whole_character/wu_kaishu_target.png
outputs/wu_whole_character/wu_kaishu_prediction.png
outputs/wu_whole_character/wu_kaishu_diff.png
outputs/wu_whole_character/wu_kaishu_comparison.png
outputs/wu_whole_character/wu_kaishu_metrics.json
```

如果轨迹 CSV 中有多个“武”，可加 `--sample_id 武_fake_sim` 精确选择，或使用 `--index 0`、`--index 1`。

如果通用整字模型已经能生成完整结构，但“武”与指定目标仍有明显字形差异，可以再做只针对“武”的微调。先构建单字 NPZ：

```bash
python -u tools/build_character_pairs.py \
  --config configs/default.yaml \
  --trajectory_csv data/raw/trajectories.csv \
  --output_npz data/processed/wu_character_finetune.npz \
  --character 武 \
  --target_character 武 \
  --target_image assets/targets/wu_kaishu_target.png
```

再从通用整字权重开始微调。这里必须使用 `--init_character_checkpoint`，不能用 `--resume`，因为训练数据和划分已经改变：

```bash
python -u tools/train_character.py \
  --config configs/default.yaml \
  --npz_path data/processed/wu_character_finetune.npz \
  --init_character_checkpoint outputs/character_10d/character_best.pt \
  --output_dir outputs/character_wu_finetune \
  --epochs 200 \
  --batch_size 1 \
  --val_ratio 0 \
  --lr_factor 0.5 \
  --lr_patience 10 \
  --min_lr 0.000001
```

微调后，把预测命令中的 checkpoint 改为：

```text
outputs/character_wu_finetune/character_best.pt
```

## 7. 继续增加 epoch

下面表示从已有 checkpoint 继续训练到总计 80 epoch：

```bash
python -u tools/train_character.py \
  --config configs/default.yaml \
  --npz_path data/processed/character_train_10d.npz \
  --resume outputs/character_10d/character_last.pt \
  --epochs 80 \
  --batch_size 16 \
  --lr_factor 0.5 \
  --lr_patience 3 \
  --min_lr 0.000001
```

这里的 `80` 是总 epoch 数，不是额外训练 80 个 epoch。
