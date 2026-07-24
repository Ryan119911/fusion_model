# Fusion Model：v8 轨迹忠实整字生成

本仓库包含两条独立流程：

- 单笔 B-BSMG：保留原有单笔训练与推理代码。
- 整字 U-Net：输入一条完整汉字轨迹，一次生成完整 `128×128` 字形。

当前整字主流程是 **v8 轨迹几何优先模型**。它解决 v7 中最关键的数据冲突：

```text
v7：一条轨迹 + 另一位书写者的同字图片
     → 笔画长度、位置、粗细和转折并非真正配对

v8：一条轨迹 + 由这条轨迹自身生成的压力感知目标
     → 长度、位置、起止点和交叉关系严格同源
```

v8 暂不做书法家风格迁移，也不使用 Transformer。

```text
完整 trajectories.csv 样本
  ↓
6×128×128 空间条件图
  centerline / proximity / pressure / stroke_order / direction_cos / direction_sin
  ↓
压力感知的同源轨迹目标 Mask
  ↓
纯 U-Net + 轨迹邻域几何门控
  ↓
完整字形概率图与二值 Mask
```

## 1. v8 设计原则

### 轨迹是几何真值

模型输出必须忠实于输入轨迹。目标图不再由随机楷书图片决定，因此不会出现同一笔画
长度、位置和走向互相冲突的问题。

### 压力决定局部笔宽

构建目标时，将每个轨迹点的 `z` 归一化，并映射到：

```text
render_min_width ～ render_max_width
```

如果一条轨迹的所有 `z` 相同，使用中间宽度，不会全部渲染成最粗笔画。

### 输出受轨迹邻域约束

U-Net 输出乘以由 `proximity` 生成的几何门控。轨迹邻域之外的概率被压为零，防止模型
生成与输入轨迹无关的额外墨迹。

### 真实楷书图不再是像素真值

`data/raw/images/` 可以保留给以后独立的风格阶段，但 v8 结构训练不读取这些图像。
指定的外部武字图片在推理时也只用于额外视觉比较，不会输入模型。

## 2. 格式与兼容性

```text
NPZ format       character_spatial_v8
target_mode      trajectory_faithful_mask
checkpoint       character_unet_v6
preprocessing    trajectory_pressure_render_v1
```

v8 必须重新构建数据并从头训练，不能从 v7 checkpoint 恢复或初始化。

旧 v7 数据和 checkpoint 仍可用于只读评估，但不能进入 v8 训练。

## 3. Ubuntu 环境

```bash
cd ~/coppeliasim/machine_learning/model
git switch main
git pull origin main
conda activate ddpm
python -m pip install -r requirements-character.txt
```

确认轨迹文件存在：

```text
data/raw/trajectories.csv
```

至少包含：

```text
character, sample_id, stroke_id, point_id, x, y, z, ...
```

## 4. 构建 v8 同源轨迹数据

```bash
python -u tools/build_trajectory_character_pairs.py \
  --config configs/default.yaml \
  --trajectory_csv data/raw/trajectories.csv \
  --output_npz data/processed/character_trajectory_faithful_v8.npz \
  --trajectory_padding 16 \
  --trajectory_width 3 \
  --render_min_width 4 \
  --render_max_width 8 \
  --pressure_gamma 1.0 \
  --min_trajectory_coverage 0.999 \
  --skeleton_tolerance 3 \
  --audit_limit 500
```

默认解释为 `z` 越大，笔画越宽。如果数据的物理含义相反，增加：

```bash
--pressure_invert
```

不要在未检查压力数据前随意使用这个选项。

正确输出的关键检查应接近：

```text
target mode: trajectory_faithful_mask
trajectory/target coverage: mean=1.000000, min≈1.000000
symmetric skeleton score: 接近 1
```

生成文件：

```text
data/processed/character_trajectory_faithful_v8.npz
data/processed/character_trajectory_faithful_v8.summary.json
data/processed/character_trajectory_faithful_v8.rejected.json
data/processed/character_trajectory_faithful_v8_audit/
```

审计图从左到右：

```text
轨迹中心线
proximity
pressure
同源目标 Mask
红色目标 / 绿色中心线重叠
```

训练前检查：

- 目标必须包含完整中心线；
- 笔画长度和位置必须与第一幅中心线一致；
- 压力变化是否产生合理的局部宽度；
- 如果整体过粗，先调整 `4～8`，不要用阈值补救数据目标；
- 如果整体过细，可尝试 `5～9`。

## 5. 从头训练 v8

GTX 1660 6GB 建议：

```bash
python -u tools/train_character.py \
  --config configs/default.yaml \
  --npz_path data/processed/character_trajectory_faithful_v8.npz \
  --output_dir outputs/character_trajectory_faithful_v8 \
  --epochs 30 \
  --batch_size 4 \
  --val_ratio 0.1 \
  --split_mode character \
  --lr_factor 0.5 \
  --lr_patience 3 \
  --min_lr 0.000001
```

显存不足时把 `batch_size` 改为 `2`。

v8 主要损失：

```text
weighted_bce       二值像素分类
dice_loss          整体重叠
tversky_loss       过度铺墨
cldice_loss         骨架拓扑
boundary_loss      边界
background_loss    背景漏墨
confidence_loss    灰色不确定区域
ink_loss           全字总墨量
local_ink_loss     局部笔宽与局部墨量
trajectory_loss    输入中心线连续覆盖
```

`character_best.pt` 仍按结构综合评分选择。正式评估不要使用
`character_last.pt`。

## 6. 验证未见字符

```bash
python -u tools/evaluate_character.py \
  --config configs/default.yaml \
  --npz_path data/processed/character_trajectory_faithful_v8.npz \
  --checkpoint outputs/character_trajectory_faithful_v8/character_best.pt \
  --output_dir outputs/eval_character_trajectory_faithful_v8 \
  --split val \
  --batch_size 4 \
  --num_images 200 \
  --threshold 0.50 \
  --thresholds 0.35,0.40,0.45,0.50,0.55,0.60,0.65
```

重点指标：

```text
macro_metrics.iou_at_threshold
macro_metrics.dice_at_threshold
macro_metrics.boundary_f1
trajectory_prediction_coverage
mask_ink_ratio
uncertain_fraction
best_threshold_by_balanced_score
```

在 v8 中，验证字符身份与训练字符互斥，但验证目标仍由各自轨迹生成，因此评价的是：

> 模型能否把从未见过的轨迹结构可靠地渲染成完整字形。

## 7. 生成“武”字

先使用验证集平衡阈值。假设最佳阈值为 `0.50`：

```bash
python -u tools/predict_character.py \
  --config configs/default.yaml \
  --trajectory_csv data/raw/trajectories.csv \
  --checkpoint outputs/character_trajectory_faithful_v8/character_best.pt \
  --character 武 \
  --output_dir outputs/wu_trajectory_faithful_v8 \
  --output_stem wu_v8 \
  --trajectory_padding 16 \
  --trajectory_width 3 \
  --threshold 0.50
```

主要对比图：

```text
wu_v8_comparison.png
```

面板顺序：

```text
输入轨迹
同源压力渲染目标
U-Net 概率预测
二值预测
二值差异
```

如需额外显示原来的楷书武字参考：

```bash
python -u tools/predict_character.py \
  --config configs/default.yaml \
  --trajectory_csv data/raw/trajectories.csv \
  --checkpoint outputs/character_trajectory_faithful_v8/character_best.pt \
  --character 武 \
  --target_image assets/targets/wu_kaishu_target.png \
  --output_dir outputs/wu_trajectory_faithful_v8 \
  --output_stem wu_v8 \
  --trajectory_padding 16 \
  --trajectory_width 3 \
  --threshold 0.50
```

这会额外生成：

```text
wu_v8_external_reference_comparison.png
```

外部楷书图只用于说明两套写法的差异，不参与同源轨迹指标。

## 8. 继续训练

只能在完全相同的 v8 NPZ 上恢复：

```bash
python -u tools/train_character.py \
  --config configs/default.yaml \
  --npz_path data/processed/character_trajectory_faithful_v8.npz \
  --resume outputs/character_trajectory_faithful_v8/character_last.pt \
  --epochs 50 \
  --batch_size 4 \
  --val_ratio 0.1 \
  --split_mode character \
  --lr_factor 0.5 \
  --lr_patience 3 \
  --min_lr 0.000001
```

`--epochs 50` 表示训练到总计 50 epoch。

## 9. 如何解释 v8 结果

### 同源目标效果好，外部楷书参考差

这是正常现象：两者笔画几何不同。说明模型忠实于轨迹，但没有学习外部图片的写法。

### 同源目标仍有长度或位置偏移

优先检查：

```text
trajectory_prediction_coverage
geometry_gate_threshold
comparison 图第一、二、四幅
```

v8 的几何门控应使轨迹之外的远距离墨迹为零。

### 整体粗细不合适

重新构建数据并调整：

```text
render_min_width
render_max_width
```

不要只调整推理阈值，因为目标笔宽本身应先符合项目需求。

## 10. v8 不解决的内容

v8 不做：

- 复现任意外部楷书图片的具体笔画布局；
- 书法家风格迁移；
- 纸张、拓片和墨色纹理；
- 用训练集中的武字证明字符泛化；
- Transformer 或逐笔图像叠加。

后续风格阶段应保持几何分离：

```text
v8 轨迹忠实结构 Mask + 风格参考 → 风格化书法图
```
