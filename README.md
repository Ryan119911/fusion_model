# Fusion Model：v8 轨迹忠实整字生成

本仓库包含两条独立流程：

- 单笔 B-BSMG：保留原有单笔训练与推理代码。
- 整字 U-Net：输入一条完整汉字轨迹，一次生成完整 `128×128` 字形。

本项目后续统一使用楷书目标
`data/raw/targets/wu_kaishu_target.png`。旧的 `武.png` 和
`wu_target_xingkai.png` 属于已经废弃的行楷参考，不得用于训练、反演、渲染或
指标比较。整字目标只进入反演/评价流程；B-BSMG 仍使用姿态—足迹配对数据训练，
不能把单张“武”字当作 B-BSMG 训练样本。

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

### 楷书目标的骨架驱动 x/y 初始化

当外部楷书目标与输入轨迹的笔画位置不一致时，不能先用 H/姿态角补偿二维结构
误差。先将现有轨迹在局部范围内吸附到目标骨架：

```bash
python -u tools/snap_trajectory_to_target.py \
  --trajectory_csv data/raw/trajectories.csv \
  --pose_csv outputs/wu_kaishu_target_v25_xy_refine/wu_trajectory.csv \
  --target_image data/raw/targets/wu_kaishu_target.png \
  --output_csv outputs/wu_kaishu_target_v26_skeleton/wu_initialized.csv \
  --character 武 \
  --max_snap_px 10 \
  --blend 0.8 \
  --smooth_sigma 0.75
```

工具输出新的 x/y、骨架覆盖图和 JSON 距离报告，保留原有 z/alpha/beta/gamma。
随后再用 `invert_paper_trajectory.py` 分阶段优化姿态。该初始化仅是图像坐标配准，
输出仍是仿真候选，不是经过机器人坐标系标定的安全轨迹。

数据库中同一个字存在多个书写样本时，先审计候选，不要直接对灰度图求平均：

```bash
python -u tools/audit_character_target_variants.py \
  --trajectory_csv data/raw/trajectories.csv \
  --target_image data/raw/targets/wu_kaishu_target.png \
  --character 武 \
  --chirography 楷 \
  --output_dir outputs/wu_kaishu_variant_audit_v1
```

排名同时考虑轨迹—候选的对称骨架一致性、候选—规范目标 Dice 和墨量平衡。
输出的前若干候选可用于几何初始化和风格统计，但它们没有 H/alpha/beta/gamma
真值，不能直接作为姿态 B-BSMG 的监督样本。

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
  --target_image data/raw/targets/wu_kaishu_target.png \
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

## 11. 论文融合仿真原型：B-BSMG + Dynamic Brush + PSOC/LM

这是一条与 v8 并行的新链路，不替换整字 U-Net。它用于从固定的二维轨迹和目标图像反演论文姿态参数：

```text
固定 x/y 轨迹
  → CGL 节点表示 H/α/β
  → 动态宽度、拖曳和笔尖偏移
  → (Lt,Lh,Lr) 反解虚拟姿态
  → 可微 B-BSMG 逐点渲染并集
  → 图像残差
  → LM 更新 H/α/β 节点
```

原型范围和单位固定为：

```text
H       11–20 mm
alpha   0–10° = 0–0.174532925 rad
beta    0–5°  = 0–0.087266463 rad
gamma   0 rad（固定，不参与优化）
```

这里的 `alpha` 是倾角，`beta` 是纸面内旋转角，`gamma` 是当前轴对称笔刷模型不可观测的第三轴向角。CSV 中所有角度均输出弧度。`z` 暂存论文参数 `H`，单位 mm；它还不是机器人基坐标系中的 TCP z。

代码使用 B-BSMG 论文给出的暂定回归式：

```text
Lt = 0.0672 H + 0.0263 alpha + 0.0191 beta + 0.0267
Lh = 0.0196 H + 0.0039 alpha + 0.0073 beta + 0.0372
Lr = 0.0239 H + 0.0061 alpha + 0.0096 beta + 0.1137
```

论文正文明确写明 `alpha/beta represent the radian`，因此当前 v1 数据和检查点按
弧度代入上式；但同一论文的实验采样和图 9 又以 `0°/5°/10°` 展示角度，按系数
量级观察也存在“回归时实际使用角度数值”的可能。这是论文内部的单位歧义，不应
在已有检查点上静默切换。当前版本会用独立正则和 Jacobian 敏感度报告暴露该问题；
后续如测试 degree-fitted 假设，必须另建数据版本、从头训练检查点并与 radian
版本做 A/B 对照。无论内部标定采用何种基底，对外 CSV 始终输出弧度。

正向渲染中按动态笔刷论文对完整宽度 `w=2Lr` 和拖曳长度
`d=Lt+Lh` 做一阶惯性更新，再用带参考姿态正则的回归逆解得到 B-BSMG
的虚拟 `(H,alpha,beta)`。v7 的笔尖偏移和方向使用保存笔根的摩擦状态更新；
轨迹切向角与动态笔刷方向分别记录，二者都不冒充第三姿态角。

### 11.1 构建新的论文 B-BSMG 数据

旧 `bbsmg_train_10d.npz` 的姿态列没有覆盖上述范围，不能用于这条反演链路。重新生成：

```bash
python -u tools/build_paper_bbsmg_dataset.py \
  --output_npz data/processed/paper_bbsmg_v1.npz \
  --count 50000 \
  --image_size 128 \
  --pixels_per_model_unit 20 \
  --supersample 4 \
  --anchor_margin 4 \
  --seed 42
```

`pixels_per_model_unit=20` 用于生成单笔训练图。整字正向融合默认再使用
`footprint_scale=0.22`，即当前归一化整字画布中的有效比例为约
`4.4 pixel/model-unit`。这是针对当前 128×128 目标与轨迹尺度的仿真桥接参数，
后续必须由真实相机与毛笔标定替换。

输出：

```text
data/processed/paper_bbsmg_v1.npz
data/processed/paper_bbsmg_v1.summary.json
```

NPZ 输入严格为：

```text
[H_mm, alpha_rad, beta_rad, x0_px, y0_px]
```

目标是论文对称三次 Bézier B-BSM 的抗锯齿 `128×128` 笔触图。NPZ 内保存了特征名、单位、上下限和归一化尺度；训练和推理必须读取同一份尺度。

### 11.2 训练论文参数化 B-BSMG

```bash
python -u tools/train_bbsmg.py \
  --config configs/paper_bbsmg.yaml \
  --npz_path data/processed/paper_bbsmg_v1.npz \
  --output_dir outputs/paper_bbsmg_v1 \
  --epochs 50 \
  --val_ratio 0.1 \
  --lr_factor 0.5 \
  --lr_patience 3 \
  --min_lr 0.000001
```

必须使用：

```text
outputs/paper_bbsmg_v1/bbsmg_best.pt
```

新 checkpoint 标记为 `paper_bbsmg_v1`、`input_dim=5`，并保存训练归一化。反演器会拒绝 10D checkpoint 或特征语义不匹配的 checkpoint。

先评估单笔参数模型，再进行整字反演：

```bash
python -u tools/evaluate_bbsmg.py \
  --config configs/paper_bbsmg.yaml \
  --npz_path data/processed/paper_bbsmg_v1.npz \
  --checkpoint outputs/paper_bbsmg_v1/bbsmg_best.pt \
  --output_dir outputs/eval_paper_bbsmg_v1 \
  --val_ratio 0.1 \
  --num_images 40
```

#### degree-fitted 独立 A/B 版本

不要用 degree-fitted 数据续训 `paper_bbsmg_v1`。它具有相同的弧度输入和输出
接口，但内部将弧度转换成论文实验中的角度数值后再代入回归系数，必须使用独立
数据、独立输出目录并从头训练：

```bash
python -u tools/build_paper_bbsmg_dataset.py \
  --output_npz data/processed/paper_bbsmg_degree_fitted_v2.npz \
  --regression_angle_basis degree_fitted \
  --count 50000 \
  --image_size 128 \
  --pixels_per_model_unit 20 \
  --supersample 4 \
  --anchor_margin 4 \
  --seed 42

python -u tools/train_bbsmg.py \
  --config configs/paper_bbsmg.yaml \
  --npz_path data/processed/paper_bbsmg_degree_fitted_v2.npz \
  --output_dir outputs/paper_bbsmg_degree_fitted_v2 \
  --epochs 50 \
  --val_ratio 0.1 \
  --lr_factor 0.5 \
  --lr_patience 3 \
  --min_lr 0.000001

python -u tools/evaluate_bbsmg.py \
  --config configs/paper_bbsmg.yaml \
  --npz_path data/processed/paper_bbsmg_degree_fitted_v2.npz \
  --checkpoint outputs/paper_bbsmg_degree_fitted_v2/bbsmg_best.pt \
  --output_dir outputs/eval_paper_bbsmg_degree_fitted_v2 \
  --val_ratio 0.1 \
  --num_images 40
```

checkpoint 会标记为 `paper_bbsmg_degree_fitted_v2`。渲染器会自动读取其角度基底，
并拒绝格式与基底不一致的 checkpoint。外部输入、反演 CSV 和最终机器人接口中的
alpha/beta 仍然全部使用弧度。degree-fitted 仿真数据的平均单笔墨迹面积约比
radian v1 大 24%，因此先对整字比例做独立扫描：

```bash
for SCALE in 0.18 0.20 0.22; do
  python -u tools/render_paper_trajectory.py \
    --trajectory_csv data/raw/trajectories.csv \
    --target_image data/raw/targets/wu_kaishu_target.png \
    --bbsmg_ckpt outputs/paper_bbsmg_degree_fitted_v2/bbsmg_best.pt \
    --character 武 \
    --h_mm 15.5 \
    --alpha_deg 0 \
    --beta_deg 0 \
    --footprint_scale "$SCALE" \
    --render_max_step_px 2.0 \
    --output_image "outputs/wu_paper_degree_scale/scale_${SCALE}.png"
done
```

每张图旁边的 JSON 都会记录 `target_metrics`。以 IoU、Dice、墨迹比例和肉眼结构
共同选择尺度；`0.20` 只是根据面积比给出的扫描起点。当前“武”字实测中
`0.22` 的 Dice、IoU 和墨迹比例最好，因此下面的正式反演使用 `0.22`。更换
目标、轨迹、毛笔模型或相机尺度后必须重新扫描。

### 11.3 先检查正向融合渲染

用默认姿态运行 Dynamic Brush + B-BSMG：

```bash
python -u tools/render_paper_trajectory.py \
  --trajectory_csv data/raw/trajectories.csv \
  --bbsmg_ckpt outputs/paper_bbsmg_v1/bbsmg_best.pt \
  --character 武 \
  --h_mm 15.5 \
  --alpha_deg 0 \
  --beta_deg 0 \
  --footprint_scale 0.22 \
  --render_max_step_px 2.0 \
  --output_image outputs/wu_paper_forward/default_pose.png
```

反演完成后，也可以重新正向验证导出的弧度姿态：

```bash
python -u tools/render_paper_trajectory.py \
  --trajectory_csv data/raw/trajectories.csv \
  --bbsmg_ckpt outputs/paper_bbsmg_v1/bbsmg_best.pt \
  --pose_csv outputs/wu_paper_inverse_v4/wu_trajectory.csv \
  --character 武 \
  --footprint_scale 0.22 \
  --render_max_step_px 2.0 \
  --output_image outputs/wu_paper_forward/inverted_pose.png
```

每次正向运行还会生成同名 `.states.csv`，逐点记录虚拟姿态、`Lt/Lh/Lr`、动态偏移、x/y 切向角和实际接触画布坐标，用于后续标定审计。

### 11.4 对“武”字执行固定 x/y 的 PSOC/LM 反演

```bash
python -u tools/invert_paper_trajectory.py \
  --trajectory_csv data/raw/trajectories.csv \
  --target_image data/raw/targets/wu_kaishu_target.png \
  --bbsmg_ckpt outputs/paper_bbsmg_v1/bbsmg_best.pt \
  --character 武 \
  --output_dir outputs/wu_paper_inverse_v5_radian \
  --output_stem wu \
  --device cuda \
  --padding 16 \
  --order 3 \
  --optimization_size 16 \
  --max_steps 15 \
  --damping 0.05 \
  --jacobian_mode finite_difference \
  --finite_difference_eps 0.01 \
  --h_prior_weight 0.001 \
  --alpha_prior_weight 0.05 \
  --beta_prior_weight 0.05 \
  --h_smoothness_weight 0.02 \
  --alpha_smoothness_weight 0.10 \
  --beta_smoothness_weight 0.10 \
  --initial_h_mm 15.5 \
  --initial_alpha_deg 0 \
  --initial_beta_deg 0 \
  --footprint_scale 0.22 \
  --render_max_step_px 2.0
```

显存或速度不足时可以先做烟雾测试：

```bash
python -u tools/invert_paper_trajectory.py \
  --trajectory_csv data/raw/trajectories.csv \
  --target_image data/raw/targets/wu_kaishu_target.png \
  --bbsmg_ckpt outputs/paper_bbsmg_v1/bbsmg_best.pt \
  --character 武 \
  --output_dir outputs/wu_paper_inverse_smoke \
  --order 2 \
  --optimization_size 8 \
  --render_stride 2 \
  --max_steps 2
```

正式结果包含：

```text
wu_trajectory.csv   原始 x/y + 反演 z(H)/alpha/beta + gamma=0
wu_target.png
wu_initial.png
wu_rendered.png
wu_diff.png
wu_comparison.png
wu_report.json
```

目标图必须与输入 x/y 轨迹在位置、长度、笔画走向上基本对齐。该工具故意不允许 LM 移动 x/y，所以它只能用 H/α/β 修正局部笔触宽度、拖曳、方向细节和接触形态；不能把一套骨架变成另一套字形。若目标与轨迹骨架不一致，应先完成二维配准或更换匹配轨迹。

LM 每一步都要构造图像残差对 CGL 姿态节点的 Jacobian，运行时间明显长于普通神经网络推理。6GB 显存默认使用 `finite_difference`：逐个扰动 CGL 变量并在 `torch.no_grad()` 下计算数值 Jacobian，不保留整字反向图。该方式与 Wang 论文的数值 Jacobian 路径一致，显存接近普通推理，但速度较慢。`autograd` 只建议在更大显存的 GPU 上使用。`order`、`optimization_size`、`render_stride` 和加密采样间距共同决定速度与精度。

检查 `wu_trajectory.csv` 时必须满足：

- `x/y` 与输入轨迹逐点完全一致；
- `z` 位于 `[11,20]` mm；
- `alpha` 位于 `[0,0.174532925]` rad；
- `beta` 位于 `[0,0.087266463]` rad；
- `gamma` 每行严格为 `0`；
- `pose_frame=paper_model`、`prototype=paper_psoc_lm_v6_observability_gated`；
- `regression_angle_basis` 必须与所用 checkpoint 一致。

如果旧版 `paper_psoc_lm_v1` CSV 报姿态越界，可先显式裁剪并查看图像：

```bash
python -u tools/render_paper_trajectory.py \
  --trajectory_csv data/raw/trajectories.csv \
  --bbsmg_ckpt outputs/paper_bbsmg_v1/bbsmg_best.pt \
  --pose_csv outputs/wu_paper_inverse_v1/wu_trajectory.csv \
  --character 武 \
  --clip_pose_limits \
  --footprint_scale 0.35 \
  --render_max_step_px 2.0 \
  --output_image outputs/wu_paper_forward/inverted_pose_legacy_clipped.png
```

裁剪结果只用于诊断。正式 CSV 必须用 v6 反演器重新生成。v6 保留 v5 的
逐点有界 sigmoid，并沿相邻固定 x/y 线段按不超过 2 px 的间距插入可微
渲染样本，解决稀疏轨迹被渲染成离散印章点的问题。插值样本只进入正向渲染，
导出的原始 x/y 点数与坐标不变。v5 为 H、alpha、beta 分别设置先验和
平滑权重：H 保持弱约束，图像辨识能力较弱的 alpha/beta 使用更强约束，避免
角度大量贴到物理上下限。`wu_report.json` 的
`lm.diagnostics.image_jacobian_sensitivity` 会报告三类变量对图像的相对敏感度，
`bound_fraction_within_1pct` 会报告接近上下限的点比例。当前 alpha/beta 应视为
带先验的仿真估计值，不能当作真实机器人姿态真值。

degree-fitted checkpoint 训练完成后，用完全相同的优化配置执行 A/B 反演：

```bash
python -u tools/invert_paper_trajectory.py \
  --trajectory_csv data/raw/trajectories.csv \
  --target_image data/raw/targets/wu_kaishu_target.png \
  --bbsmg_ckpt outputs/paper_bbsmg_degree_fitted_v2/bbsmg_best.pt \
  --character 武 \
  --output_dir outputs/wu_paper_inverse_v5_degree \
  --output_stem wu \
  --device cuda \
  --padding 16 \
  --order 3 \
  --optimization_size 16 \
  --max_steps 20 \
  --damping 0.05 \
  --jacobian_mode finite_difference \
  --finite_difference_eps 0.01 \
  --h_prior_weight 0.001 \
  --alpha_prior_weight 0.05 \
  --beta_prior_weight 0.05 \
  --h_smoothness_weight 0.02 \
  --alpha_smoothness_weight 0.10 \
  --beta_smoothness_weight 0.10 \
  --initial_h_mm 15.5 \
  --initial_alpha_deg 0 \
  --initial_beta_deg 0 \
  --footprint_scale 0.22 \
  --render_max_step_px 2.0
```

A/B 判断不能只看最终 loss。优先比较 `dice_at_0.5`、`iou_at_0.5`、
`relative_median`、alpha/beta 的 `bound_fraction_within_1pct`，以及实际对比图。
degree-fitted 只有在图像指标不下降、角度贴边显著减少且中位敏感度提高时才保留。

当前“武”字 A/B 中，degree-fitted 的 Dice/IoU 均低于 radian v1，beta 贴边比例
也更高，因此默认仍保留论文正文声明的 radian v1；degree-fitted 只作为论文单位
歧义实验留档。

### 11.5 v6 可观测性门控反演

单幅二值字形无法可靠恢复所有三维姿态。v6 默认先计算一次完整 Jacobian，以各
字段“归一化物理范围内的中位图像敏感度”做门控。H 始终参与优化；alpha/beta
只有达到相对阈值才进入后续 LM，未通过的字段严格保持命令行默认值：

```bash
python -u tools/invert_paper_trajectory.py \
  --trajectory_csv data/raw/trajectories.csv \
  --target_image data/raw/targets/wu_kaishu_target.png \
  --bbsmg_ckpt outputs/paper_bbsmg_v1/bbsmg_best.pt \
  --character 武 \
  --output_dir outputs/wu_paper_inverse_v6_gated \
  --output_stem wu \
  --device cuda \
  --padding 16 \
  --order 3 \
  --optimization_size 16 \
  --max_steps 20 \
  --damping 0.05 \
  --jacobian_mode finite_difference \
  --finite_difference_eps 0.01 \
  --field_mode auto \
  --min_relative_median_sensitivity 0.45 \
  --h_prior_weight 0.001 \
  --alpha_prior_weight 0.05 \
  --beta_prior_weight 0.05 \
  --h_smoothness_weight 0.02 \
  --alpha_smoothness_weight 0.10 \
  --beta_smoothness_weight 0.10 \
  --initial_h_mm 15.5 \
  --initial_alpha_deg 0 \
  --initial_beta_deg 0 \
  --footprint_scale 0.22 \
  --render_max_step_px 2.0
```

`auto` 会先审计全部字段，之后通常只需计算被保留字段的 Jacobian。也可用
`--field_mode all` 重现 v5 全变量 A/B，或用 `--field_mode h_only` 跳过审计并
强制只优化 H。正式结果的 JSON 会记录 `observability_gate` 和
`field_decisions`；CSV 每行增加 `z/alpha/beta/gamma_source` 与对应
`*_confidence`。`medium_simulation` 只表示仿真内部可辨识，仍不等于机器人真值。

当前“武”字在阈值 `0.35` 下曾放行 alpha，但其最终贴边比例达到 69%，因此默认
阈值提高为 `0.45`。已经完成一次审计并确认 alpha/beta 不可靠时，可在后续正式
导出中直接使用 `--field_mode h_only`，避免重复计算完整审计 Jacobian。

### 11.6 当前原型不能直接下发机器人

论文回归系数和 `Kw=Kd=0.02` 来自论文，Figure 4 曲线来自数字化近似；
`20 pixel/model-unit`、`footprint_scale=0.22` 与跨论文尺度桥梁仍是仿真参数。
真实执行前必须依次替换：

1. 用真实毛笔采集 `(H,alpha,beta) → (Lt,Lh,Lr)` 标定数据并重拟合回归；
2. 用连续书写数据拟合宽度、拖曳、偏移和惯性参数；
3. 完成相机像素、纸面坐标、机器人基坐标和 TCP 的外参标定；
4. 明确定义 `paper_model` 姿态到机器人控制器 Euler/四元数的旋转顺序；
5. 加入关节限位、速度、加速度、碰撞和纸面接触力约束；
6. 低速、离纸、单笔验证后才允许接触纸面。

因此当前导出的六维序列是“论文纸面坐标系中的仿真反演结果”，不是可直接执行的机器人轨迹真值。

### 11.7 v7：使用论文数据的动态笔刷与阶数搜索

v7 把 v6 中人为设定的 `offset_fraction=0.25` 默认路径替换为 Wang 等（2020）
的动态笔刷数据：

- `Kw=Kd=0.02` 是论文正文给出的精确实验值；
- `Width(z)`、`Drag(z)`、`Offset(z)` 的作者多项式系数没有在正文或表格中公布；
- 本仓库从论文 Figure 4 的橙色拟合曲线数字化得到一组近似系数，并以
  `wang2020_figure4_digitized_v1` 单独版本化；
- B-BSMG 的 `H=11–20 mm` 线性映射到 Figure 4 的 `z=0–1.5 cm` 只是一条
  归一化范围桥梁，不是机器人 z 标定；
- 两篇论文使用的毛笔不同，因此把 Wang 的 `Width/Drag` 曲线进度重映射到
  B-BSMG 的端点尺寸，并以无量纲 `Offset(z)/Drag(z)` 传递偏移；绝不把
  Figure 4 的绝对 cm 静默当作 B-BSM 模型单位。

正向状态更新现按 Wang Eq. (1)、(6)–(9) 保存笔根。短距离运动时笔根受摩擦保持，
超过自由偏移后再弹回；`.states.csv` 会同时输出自由偏移、保持偏移、实际偏移、
轨迹切向角和动态笔刷角。论文数据、数字化误差和论文未报告的参数都会写入输出
JSON 的 `paper_calibration`，对应代码在
`models/paper_calibration.py`。

先在 Ubuntu 上运行新的正向基线：

```bash
python -u tools/render_paper_trajectory.py \
  --trajectory_csv data/raw/trajectories.csv \
  --target_image data/raw/targets/wu_kaishu_target.png \
  --bbsmg_ckpt outputs/paper_bbsmg_v1/bbsmg_best.pt \
  --character 武 \
  --h_mm 15.5 \
  --alpha_deg 0 \
  --beta_deg 0 \
  --dynamic_profile wang2020_figure4_digitized_v1 \
  --offset_transfer_scale 1.0 \
  --footprint_scale 0.22 \
  --render_max_step_px 2.0 \
  --output_image outputs/wu_paper_forward_v7/default_pose.png
```

先查看 `default_pose.png`、同名 JSON 和 `.states.csv`。新偏移模型会改变接触根位置，
所以不要把 v6 与 v7 的 LM loss 直接比较；应比较同一目标下的 MSE、Dice、IoU、
接触位置和最终对比图。

本次“武”字实测表明直接传递 `Offset/Drag`（尺度 `1.0`）会带来约 `6.35 px`
的平均接触根位移。由于两篇论文的毛笔和 B-BSMG 锚点定义并不相同，必须先扫描
这条跨论文传递尺度，不能直接把 `1.0` 当作真实标定：

```bash
for OFFSET_SCALE in 0.0 0.25 0.5 0.75 1.0; do
  python -u tools/render_paper_trajectory.py \
    --trajectory_csv data/raw/trajectories.csv \
    --target_image data/raw/targets/wu_kaishu_target.png \
    --bbsmg_ckpt outputs/paper_bbsmg_v1/bbsmg_best.pt \
    --character 武 \
    --h_mm 15.5 \
    --alpha_deg 0 \
    --beta_deg 0 \
    --dynamic_profile wang2020_figure4_digitized_v1 \
    --offset_transfer_scale "${OFFSET_SCALE}" \
    --footprint_scale 0.22 \
    --output_image \
      "outputs/wu_paper_offset_scan_v7/offset_${OFFSET_SCALE}.png"
done
```

用 JSON 中的 MSE、Dice、IoU，加上对比图的笔画位置共同选择
`BEST_OFFSET_SCALE`。完成尺度扫描后，再按论文报告的 CGL 阶数 `3–8`
做完整字符级搜索：

```bash
read -p "BEST_OFFSET_SCALE: " BEST_OFFSET_SCALE

python -u tools/invert_paper_trajectory.py \
  --trajectory_csv data/raw/trajectories.csv \
  --target_image data/raw/targets/wu_kaishu_target.png \
  --bbsmg_ckpt outputs/paper_bbsmg_v1/bbsmg_best.pt \
  --character 武 \
  --output_dir outputs/wu_paper_inverse_v7_wang \
  --output_stem wu \
  --device cuda \
  --padding 16 \
  --search_orders \
  --order_min 3 \
  --order_max 8 \
  --optimization_size 16 \
  --max_steps 20 \
  --damping 0.05 \
  --jacobian_mode finite_difference \
  --finite_difference_eps 0.01 \
  --field_mode h_only \
  --dynamic_profile wang2020_figure4_digitized_v1 \
  --offset_transfer_scale "${BEST_OFFSET_SCALE}" \
  --h_prior_weight 0.001 \
  --alpha_prior_weight 0.05 \
  --beta_prior_weight 0.05 \
  --h_smoothness_weight 0.02 \
  --alpha_smoothness_weight 0.10 \
  --beta_smoothness_weight 0.10 \
  --initial_h_mm 15.5 \
  --initial_alpha_deg 0 \
  --initial_beta_deg 0 \
  --footprint_scale 0.22 \
  --render_max_step_px 2.0
```

Wang 论文是“每个分解笔画分别测试 3–8 阶”；当前目标图只有整字而没有逐笔目标，
所以 v7 采用“整字共享一个阶数，依次比较 3–8 阶”的近似，并把每个候选的 cost、
MSE、Dice 和 IoU 写入 `psoc_order_search.candidates`。跨阶数时正则残差数量会随
CGL 节点数变化，因此不能按总 cost 选择；程序按与论文图像误差对应的全分辨率
plain MSE 选择最终阶数。每个候选的图像和姿态 CSV 另存于
`order_candidates/`。完整搜索约等于执行 6 次反演。
先做功能检查时去掉 `--search_orders`，使用 `--order 3 --max_steps 2`。

论文 Eq. (19) 的末端抬笔权重 `beta_k` 没有公开。代码提供
`--terminal_lift_weight` 和 `--terminal_lift_nodes`，但默认权重严格为 `0`，
不会把人为数值伪装成论文数据。只有完成真实毛笔标定或明确进行仿真消融时才应设置。

### 11.8 v8：按全分辨率图像保存最佳 LM 迭代

v7 的 LM 接受条件使用 `optimization_size×optimization_size` 的加权残差。实测中，
`16×16` cost 在第 20–30 步继续下降时，`128×128` MSE 可能反而上升，导致更好的
中间姿态被最后一步覆盖。v8 保留论文的低分辨率数值 Jacobian 和 LM 更新，但在
初始点及每个被 LM 接受的迭代上额外计算一次完整 `128×128` plain MSE，并最终
返回其中最好的 checkpoint。

报告格式更新为 `paper_psoc_lm_v8_fullres_checkpoint`，并新增：

```text
lm.history.full_resolution_mse
lm.diagnostics.checkpoint_selection.initial_mse
lm.diagnostics.checkpoint_selection.best_mse
lm.diagnostics.checkpoint_selection.best_step
lm.diagnostics.checkpoint_selection.terminal_mse
lm.diagnostics.checkpoint_selection.returned_best_checkpoint
```

`lm.final_cost` 现在对应被返回 checkpoint 的正则化 cost；优化循环最后一步的 cost
保存在 `terminal_regularized_cost`。因此可以安全增加 `max_steps`，较差的后续迭代
不会再覆盖先前更好的全分辨率结果。

### 11.9 各向异性笔触尺度：只加粗，不加长

原 `footprint_scale` 同时缩放笔触局部坐标的纵向长度和横向宽度。“武”字实测中，
从 `0.22` 增大到 `0.24` 后，预测墨迹率由 `0.08954` 提高到 `0.10425`，接近目标
`0.10547`；但笔画端点也被拉长，Dice 从 `0.49578` 降至 `0.48196`。因此新增：

```text
--footprint_longitudinal_scale   沿轨迹方向的长度尺度
--footprint_transverse_scale     垂直轨迹方向的宽度尺度
```

两者未指定时都回退到 `--footprint_scale`，旧命令和旧结果保持兼容。动态 Offset
仍沿笔画方向计算，因此像素换算使用纵向尺度。当前仿真建议先固定纵向 `0.22`，
单独扫描横向尺度；这只是 B-BSMG 图像桥梁标定，不是机器人毛笔物理尺寸。

### 11.10 v9：受限 x/y 轨迹修正

v8 只优化 H/alpha/beta，严格锁定输入 x/y。“武”字在固定 x/y 下搜索 CGL
3–8 阶后，order 8 最好，但 Dice 仍约为 `0.5052`；差异图显示剩余误差主要来自
笔画位置和长度，而不是全局粗细。v9 因此新增可选的逐笔 CGL 平面偏移：

```text
--optimize_xy                 开启 x/y 联合反演，默认关闭
--xy_max_offset_px            每个画布坐标分量的最大修正，默认 6 px
--xy_smoothness_weight        相邻 x/y CGL 节点平滑权重，默认 0.10
--xy_prior_weight             回到原轨迹的零偏移先验，默认 0.05
```

偏移在 CGL 插值后经过 `tanh`，因此每一个轨迹点都严格位于
`[-xy_max_offset_px,+xy_max_offset_px]`。`xy_max_offset_px` 不允许超过
`padding`，防止修正后的轨迹离开画布。未传 `--optimize_xy` 时，行为与 v8
完全兼容。

当前“武”字推荐运行：

```bash
python -u tools/invert_paper_trajectory.py \
  --trajectory_csv data/raw/trajectories.csv \
  --target_image data/raw/targets/wu_kaishu_target.png \
  --bbsmg_ckpt outputs/paper_bbsmg_v1/bbsmg_best.pt \
  --character 武 \
  --output_dir outputs/wu_paper_inverse_v9_xy6 \
  --output_stem wu \
  --device cuda \
  --padding 16 \
  --order 8 \
  --optimization_size 32 \
  --max_steps 30 \
  --damping 0.05 \
  --jacobian_mode finite_difference \
  --finite_difference_eps 0.01 \
  --field_mode h_only \
  --optimize_xy \
  --xy_max_offset_px 6 \
  --xy_smoothness_weight 0.10 \
  --xy_prior_weight 0.05 \
  --dynamic_profile wang2020_figure4_digitized_v1 \
  --offset_transfer_scale 1.0 \
  --pixel_weight 5 \
  --h_prior_weight 0.001 \
  --alpha_prior_weight 0.05 \
  --beta_prior_weight 0.05 \
  --h_smoothness_weight 0.02 \
  --alpha_smoothness_weight 0.10 \
  --beta_smoothness_weight 0.10 \
  --initial_h_mm 15.5 \
  --initial_alpha_deg 0 \
  --initial_beta_deg 0 \
  --footprint_longitudinal_scale 0.22 \
  --footprint_transverse_scale 0.245 \
  --render_max_step_px 2.0
```

v9 报告格式为 `paper_psoc_lm_v9_bounded_xy`，新增
`xy_optimization`、修正前/后的 `trajectory_target_coverage_at_5px`，并直接
报告二值及软墨量比。导出的 `wu_trajectory.csv` 中 x/y 已反变换回输入
`trajectories.csv` 的坐标系；新增 `x_source/x_confidence` 和
`y_source/y_confidence` 字段说明其来源。正向渲染工具读取 v9 CSV 时也会使用
其中的新 x/y。

这些 x/y 仍是输入轨迹坐标系中的仿真配准结果，不是机器人基坐标或 TCP
坐标。送入真实机器人之前，必须再经过纸面坐标、相机、TCP 和机器人基座标定。

### 11.11 H 可辨识性消融

如果联合反演结果中大量 H 位于 11/20 mm 边界，不能仅凭图像指标把它解释为
真实 z。使用 `--field_mode xy_only --optimize_xy` 可以固定
`H=initial_h_mm`、alpha 和 beta，只优化完全相同的受限 x/y 变量：

```bash
python -u tools/invert_paper_trajectory.py \
  --trajectory_csv data/raw/trajectories.csv \
  --target_image data/raw/targets/wu_kaishu_target.png \
  --bbsmg_ckpt outputs/paper_bbsmg_v1/bbsmg_best.pt \
  --character 武 \
  --output_dir outputs/wu_paper_inverse_v9_xy_only \
  --output_stem wu \
  --device cuda \
  --padding 16 \
  --order 8 \
  --optimization_size 32 \
  --max_steps 30 \
  --damping 0.05 \
  --jacobian_mode finite_difference \
  --finite_difference_eps 0.01 \
  --field_mode xy_only \
  --optimize_xy \
  --xy_max_offset_px 6 \
  --xy_smoothness_weight 0.10 \
  --xy_prior_weight 0.05 \
  --dynamic_profile wang2020_figure4_digitized_v1 \
  --offset_transfer_scale 1.0 \
  --pixel_weight 5 \
  --initial_h_mm 15.5 \
  --initial_alpha_deg 0 \
  --initial_beta_deg 0 \
  --footprint_longitudinal_scale 0.22 \
  --footprint_transverse_scale 0.245 \
  --render_max_step_px 2.0
```

将它与 `wu_paper_inverse_v9_xy6` 比较。若固定 H 后 MSE、Dice、IoU 几乎不变，
说明单张目标图不能为 z 提供足够独立信息，联合反演的 H 不应进入真实机器人
数据集。只有联合 H 在留出图像上稳定改善指标、不过度贴边，并通过真实毛笔
标定验证后，才能把它升级为可用 z 标签。

### 11.12 v10：逐点 H 连续性与短笔画阶数保护

`wu_paper_inverse_v9_xy6_hreg_medium` 将节点级 H 平滑权重提高后，H 贴边率由
`40.48%` 降到 `19.05%`，但仍出现同一笔画内 `20→11→18 mm` 一类跳变。
原因是旧正则只约束 CGL 节点，没有直接约束插值后实际导出的轨迹点；同时“武”
字部分笔画只有 3–7 个输入点，却统一分配 order 8（9 个节点），形成不可辨识的
冗余变量。

v10 新增三个显式开关，默认值保持 v9 行为：

```text
--cap_order_to_points
    每个笔画的有效阶数限制为 min(requested_order, point_count-1)。
    单点笔画退化为常数；未使用的预分配节点不进入 Jacobian 或正则项。

--h_point_velocity_weight
    对同一笔画内、相邻解码轨迹点的归一化 H 一阶差分加权。

--h_point_acceleration_weight
    对同一笔画内、相邻解码轨迹点的归一化 H 二阶差分加权。
```

差分不会跨越笔画边界。由于当前 CSV 没有时间戳，这两个量只表示按输入点序号的
连续性，不能解释为真实 `mm/s` 或 `mm/s²`。正式机器人速度和加速度约束必须在
完成时间参数化、TCP 及纸面标定后重新定义。

“武”字 v10 首轮建议保持 v9 的节点级弱先验，只单独验证短笔画保护和逐点约束：

```bash
python -u tools/invert_paper_trajectory.py \
  --trajectory_csv data/raw/trajectories.csv \
  --target_image data/raw/targets/wu_kaishu_target.png \
  --bbsmg_ckpt outputs/paper_bbsmg_v1/bbsmg_best.pt \
  --character 武 \
  --output_dir outputs/wu_paper_inverse_v10_point_continuity \
  --output_stem wu \
  --device cuda \
  --padding 16 \
  --order 8 \
  --cap_order_to_points \
  --optimization_size 32 \
  --max_steps 30 \
  --damping 0.05 \
  --jacobian_mode finite_difference \
  --finite_difference_eps 0.01 \
  --field_mode h_only \
  --optimize_xy \
  --xy_max_offset_px 6 \
  --xy_smoothness_weight 0.10 \
  --xy_prior_weight 0.05 \
  --h_point_velocity_weight 2.0 \
  --h_point_acceleration_weight 4.0 \
  --dynamic_profile wang2020_figure4_digitized_v1 \
  --offset_transfer_scale 1.0 \
  --pixel_weight 5 \
  --h_prior_weight 0.001 \
  --h_smoothness_weight 0.02 \
  --alpha_prior_weight 0.05 \
  --beta_prior_weight 0.05 \
  --alpha_smoothness_weight 0.10 \
  --beta_smoothness_weight 0.10 \
  --initial_h_mm 15.5 \
  --initial_alpha_deg 0 \
  --initial_beta_deg 0 \
  --footprint_longitudinal_scale 0.22 \
  --footprint_transverse_scale 0.245 \
  --render_max_step_px 2.0
```

启用任一 v10 开关后，报告和 CSV 的 `format/prototype` 为
`paper_psoc_lm_v10_point_continuity`。报告新增：

```text
lm.diagnostics.cgl_layout
lm.diagnostics.trajectory_continuity.first_difference
lm.diagnostics.trajectory_continuity.second_difference
lm.diagnostics.trajectory_continuity.per_stroke
```

验收时同时比较 v9 `xy_only`、v9 联合反演和 v10：图像指标不能掩盖 H 跳变。
如果 v10 仍需要频繁触及 11/20 mm，或连续性改善完全来自 x/y 贴边，则 H 继续
标记为仿真低置信估计，不进入真实机器人监督数据。

### 11.13 v11：从已有六维 CSV 分阶段继续反演

v10 以前每次运行都会从原始 x/y 和统一的 H/alpha/beta 默认值重新开始，不能在
已获得的连续 x/y/H 上单独审计角度。v11 新增：

```text
--initial_pose_csv PATH
```

该 CSV 必须按 `stroke_id + point_id` 与 `--trajectory_csv` 选中的样本严格匹配，
x/y 使用输入轨迹坐标系，z 使用 mm，alpha/beta/gamma 使用 rad。当前轴对称原型
仍要求 gamma 为 0。工具会用原始轨迹的同一个画布变换映射 CSV x/y，并把逐点
H/alpha/beta 最小二乘拟合回每笔有效 CGL 节点，然后从该状态继续 LM。

使用当前最佳 v10 轨迹，仅执行一次 H/alpha/beta 可观测性审计而不更新参数：

```bash
python -u tools/invert_paper_trajectory.py \
  --trajectory_csv data/raw/trajectories.csv \
  --initial_pose_csv \
    outputs/wu_paper_inverse_v10_velocity8_w0258/wu_trajectory.csv \
  --target_image data/raw/targets/wu_kaishu_target.png \
  --bbsmg_ckpt outputs/paper_bbsmg_v1/bbsmg_best.pt \
  --character 武 \
  --output_dir outputs/wu_paper_inverse_v11_pose_audit \
  --output_stem wu \
  --device cuda \
  --padding 16 \
  --order 8 \
  --cap_order_to_points \
  --optimization_size 32 \
  --max_steps 0 \
  --jacobian_mode finite_difference \
  --finite_difference_eps 0.01 \
  --field_mode auto \
  --min_relative_median_sensitivity 0.45 \
  --dynamic_profile wang2020_figure4_digitized_v1 \
  --offset_transfer_scale 1.0 \
  --pixel_weight 5 \
  --h_point_velocity_weight 8.0 \
  --h_point_acceleration_weight 4.0 \
  --h_prior_weight 0.001 \
  --h_smoothness_weight 0.02 \
  --alpha_prior_weight 0.05 \
  --beta_prior_weight 0.05 \
  --alpha_smoothness_weight 0.10 \
  --beta_smoothness_weight 0.10 \
  --footprint_longitudinal_scale 0.22 \
  --footprint_transverse_scale 0.258 \
  --render_max_step_px 2.0
```

报告格式为 `paper_psoc_lm_v11_staged_pose`，并在 `initialization` 和
`lm.diagnostics.regularization.initial_posture_source` 中记录初始化来源。只有
alpha/beta 的 `relative_median` 达到门槛，且后续留出目标验证不出现边界饱和时，
才能进入角度优化；gamma 在加入非轴对称笔刷观测或真实姿态标定前继续固定为 0。

### 11.14 v14：逐 CGL 节点的噪声可观测性门控

v13 的 6D B-BSMG 让 gamma 真正进入神经渲染器；v14 进一步把整字段相对灵敏度
门控替换为逐节点 SNR 门控。默认噪声取自 checkpoint 验证集 plain MSE 的平方根：

```text
节点 SNR = 该节点跨完整物理范围的像素 RMS 响应 / 验证噪声 RMSE
```

启用参数：

```text
--observability_gate_mode node_snr
--min_observability_snr 1.0
```

只有超过噪声的 H/alpha/beta/gamma CGL 节点进入 LM；其余节点严格保留初值。
正向渲染工具现在也读取姿态 CSV 中的非零 gamma。提交真实字帖反演前，应先用
`tools/build_paper_roundtrip_probe.py` 和
`tools/evaluate_paper_pose_recovery.py` 做已知真值的局部闭环测试。完整 v13 训练、
v14 审计、闭环命令和真实机器人标定边界见
[`docs/robot_brush_calibration.md`](docs/robot_brush_calibration.md)。

### 11.16 v16：联合剪枝与显式 CUDA 失败

v16 在节点 SNR 门控后重新审计联合 Jacobian。如果字段间典型相关性超过门槛，
`--joint_gate_action prune` 会依次固定较弱字段，直到保留的姿态变量通过有效秩、
条件数和字段相关性检查。显式传入 `--device cuda` 时，如果 PyTorch 无法初始化
CUDA，反演和渲染工具会直接报错，不再静默回退 CPU。

### 11.17 v17：多初值姿态稳定性验证

单次反演得到很高的图像 IoU，并不能证明 z/alpha/beta/gamma 是唯一解。v17 对同一
组合成真值施加正负、多幅度初始扰动，顺序执行 v16 反演，并同时检查：

```text
每次反演相对已知真值的 normalized RMSE
不同初值解之间的 normalized cross-start RMS standard deviation
每次反演的 IoU、联合 Jacobian 可辨识性和物理边界饱和
每个姿态字段是否允许进入仿真共识输出
```

“武”字完整 GPU 验证命令：

```bash
python -u tools/run_paper_multistart_validation.py \
  --trajectory_csv data/raw/trajectories.csv \
  --truth_pose_csv outputs/wu_paper_roundtrip_v14/wu_truth.csv \
  --target_image outputs/wu_paper_roundtrip_v14/wu_truth_render.png \
  --bbsmg_ckpt outputs/paper_bbsmg_gamma_v13/bbsmg_best.pt \
  --character 武 \
  --output_dir outputs/wu_paper_multistart_v17 \
  --device cuda \
  --perturbation_scales -2 -1 -0.5 0.5 1 2 \
  --order 1 \
  --optimization_size 64 \
  --max_steps 5 \
  --resume_completed
```

### 11.21 v21：共享盆地中心先验

v20 局部复验中 z/alpha/beta/gamma 的真值精度、IoU、Jacobian 和边界均通过，
但 alpha/beta 的跨初值离散度略高于 `0.02`。v21 将“每次不同的优化初值”和
“所有运行共享的物理先验中心”分离。`--posture_prior_pose_csv` 只定义正则化
中心，不覆盖每次扰动初值：

```bash
python -u tools/run_paper_multistart_validation.py \
  --trajectory_csv data/raw/trajectories.csv \
  --truth_pose_csv outputs/wu_paper_roundtrip_v14/wu_truth.csv \
  --initial_base_pose_csv \
    outputs/wu_paper_single_field_multistart_v19/p0p5/wu_trajectory.csv \
  --posture_prior_pose_csv \
    outputs/wu_paper_single_field_multistart_v19/p0p5/wu_trajectory.csv \
  --target_image outputs/wu_paper_roundtrip_v14/wu_truth_render.png \
  --bbsmg_ckpt outputs/paper_bbsmg_gamma_v13/bbsmg_best.pt \
  --character 武 \
  --output_dir outputs/wu_paper_shared_prior_v21 \
  --device cuda \
  --perturbation_scales -0.5 -0.25 0.25 0.5 \
  --order 1 \
  --optimization_size 64 \
  --max_steps 8 \
  --pose_prior_weight 0.005 \
  --resume_completed
```

共享先验还定义节点 SNR 门控后的回退值：不可观测、被联合剪枝或未选中的姿态节点
回到 `--posture_prior_pose_csv`，而不是保留每次不同的扰动初值。这样局部稳定性统计
只衡量可观测节点的优化差异，不会被明确不可观测的自由变量人为放大。

v22 使用这一回退语义完成“武”字四初值合成闭环验收：

```text
字段    worst normalized RMSE    cross-start normalized RMS std
z       0.006760                 0.000749
alpha   0.043674                 0.018525
beta    0.033467                 0.017420
gamma   0.013930                 0.005507

minimum IoU              0.994798
joint Jacobian           passed
maximum boundary fraction 0
overall_passed           true
```

完整机器可读报告：

```text
outputs/wu_paper_shared_prior_nodes_v22/multistart_summary.json
```

这只证明同一仿真正向模型上的局部反演已经达到既定精度和稳定性，不证明输出是现实
机器人的安全姿态。真实部署前仍必须采集毛笔高度/姿态/足迹及机器人坐标系标定数据。

各初值的日志、CSV、图像和单次恢复指标保存在对应的 `m2/m1/m0p5/p0p5/p1/p2`
子目录。最终汇总为：

```text
outputs/wu_paper_multistart_v17/multistart_summary.json
```

默认验收线为 z normalized RMSE 不超过 `0.01`，alpha/beta 不超过 `0.06`，
gamma 不超过 `0.04`，且每个字段的跨初值 normalized RMS 标准差不超过 `0.02`。
未通过字段会标记为 `withhold_unstable_or_inaccurate`。该结论只适用于仿真闭环，
不能替代真实毛笔和机器人的姿态标定。

### 11.18 v18：分阶段信赖域与零初值锚定偏差验证

v17 的六初值实验发现，默认 alpha/beta/gamma 姿态先验会把解锚定到不同初值。
v18 在合成唯一性测试中将姿态初值先验设为 0，并用
`--allowed_pose_fields` 依次执行三个块坐标阶段：

```text
H
alpha + beta
gamma
```

每阶段只更新白名单字段，其余字段严格继承上一阶段 CSV。每个阶段仍执行节点 SNR
和联合 Jacobian 检查，避免通过关闭可观测性验证来获得表面更好的结果。

```bash
python -u tools/run_paper_staged_multistart_v18.py \
  --trajectory_csv data/raw/trajectories.csv \
  --truth_pose_csv outputs/wu_paper_roundtrip_v14/wu_truth.csv \
  --target_image outputs/wu_paper_roundtrip_v14/wu_truth_render.png \
  --bbsmg_ckpt outputs/paper_bbsmg_gamma_v13/bbsmg_best.pt \
  --character 武 \
  --output_dir outputs/wu_paper_staged_multistart_v18 \
  --device cuda \
  --perturbation_scales -2 -1 -0.5 0.5 1 2 \
  --cycles 1 \
  --stage_steps 5 \
  --order 1 \
  --optimization_size 64 \
  --pose_prior_weight 0 \
  --resume_completed
```

### 11.20 v20：全局候选选择与局部盆地复验

宽范围初值用于发现多个优化盆地，不要求所有局部盆地给出同一姿态。v20 先从 v19
结果中筛除 IoU、不满足联合 Jacobian 或存在边界饱和的候选，再按全分辨率 MSE
选择最佳候选，同时保留近最优候选的姿态离散度作为全局歧义证据：

```bash
python -u tools/select_paper_multistart_candidate.py \
  --summary_json \
    outputs/wu_paper_single_field_multistart_v19/multistart_summary.json \
  --near_optimal_factor 2 \
  --output_json \
    outputs/wu_paper_selected_basin_v20/candidate_selection.json
```

随后围绕所选候选做更小的局部扰动；真值 CSV 仍独立用于合成验收，不能用候选自身
替代真值：

```bash
python -u tools/run_paper_multistart_validation.py \
  --trajectory_csv data/raw/trajectories.csv \
  --truth_pose_csv outputs/wu_paper_roundtrip_v14/wu_truth.csv \
  --initial_base_pose_csv \
    outputs/wu_paper_single_field_multistart_v19/p0p5/wu_trajectory.csv \
  --target_image outputs/wu_paper_roundtrip_v14/wu_truth_render.png \
  --bbsmg_ckpt outputs/paper_bbsmg_gamma_v13/bbsmg_best.pt \
  --character 武 \
  --output_dir outputs/wu_paper_selected_basin_v20 \
  --device cuda \
  --perturbation_scales -0.5 -0.25 0.25 0.5 \
  --order 1 \
  --optimization_size 64 \
  --max_steps 8 \
  --pose_prior_weight 0 \
  --resume_completed
```

该零先验只用于已知真值的仿真可辨识性测试。真实字帖反演仍必须使用经过物理标定
的姿态先验和不确定度，不能把仿真零先验直接解释为安全机器人命令。

### 11.19 v19：α/β 单字段坐标下降

v18 三循环表明，把 alpha 与 beta 放在同一阶段仍允许二者互相补偿。v19 使用同一
工具的 `--stage_scheme separate`，把每轮拆成 H→alpha→beta→gamma 四个单字段
信赖域阶段；每阶段的联合 Jacobian 退化为单字段审计，后续循环再处理字段间耦合。

```bash
python -u tools/run_paper_staged_multistart_v18.py \
  --trajectory_csv data/raw/trajectories.csv \
  --truth_pose_csv outputs/wu_paper_roundtrip_v14/wu_truth.csv \
  --target_image outputs/wu_paper_roundtrip_v14/wu_truth_render.png \
  --bbsmg_ckpt outputs/paper_bbsmg_gamma_v13/bbsmg_best.pt \
  --character 武 \
  --output_dir outputs/wu_paper_single_field_multistart_v19 \
  --device cuda \
  --perturbation_scales -2 -1 -0.5 0.5 1 2 \
  --stage_scheme separate \
  --cycles 3 \
  --stage_steps 5 \
  --order 1 \
  --optimization_size 64 \
  --pose_prior_weight 0 \
  --resume_completed
```

### 11.15 v15：联合 Jacobian 可辨识性与姿态降阶

单节点 SNR 高只表示该节点能够改变图像，不表示多个姿态字段能够被分别恢复。
v15 对通过 SNR 的完整像素 Jacobian 继续计算：

```text
有效秩及秩比例
条件数
H/alpha/beta/gamma 字段子空间的两两典型相关系数
```

保守验收条件为有效秩比例不低于 `0.90`、条件数不超过 `100`、最大字段相关性
不超过 `0.95`。未通过时，所有参与联合优化的姿态字段自动标记
`confidence=low`、`reason=optimized_but_jointly_nonidentifiable`。

“武”字合成闭环中，order 3 的 126 个节点只有 116 个有效秩，条件数约
`8.38e4`；降到每笔 order 1 后变为 64/64、条件数约 `32.77`。这说明姿态反演
应先降低自由度，再考虑增加 LM 迭代。即使 order 1 图像 MSE 达到 `0.000063`，
alpha 与 gamma 的字段相关性仍约 `0.95947`，超过保守阈值，因此角度仍不能作为
机器人真值。完整数据和解释见
[`docs/robot_brush_calibration.md`](docs/robot_brush_calibration.md)。

### 11.27 v27：多参考楷书外观细化（姿态与风格解耦）

规范目标固定为：

```text
data/raw/targets/wu_kaishu_target.png
```

不得再使用已废弃的行楷 `data/raw/targets/武.png` 或
`wu_target_xingkai.png`。v22 已通过的姿态 B-BSMG 保持冻结；新增的
`StyleRefinerUNet` 只学习从几何掩膜、骨架、内部距离场和软边界到真实毛笔灰度外观
的映射，不能被解释为 `H/alpha/beta/gamma` 的真实标签。

先构建所有楷书单字的风格数据。数据文件保留来源图片和字符字段，训练程序按来源作品
分组切分，并从通用训练/验证中完全排除“武”：

```bash
python -u tools/build_kaishu_style_dataset.py \
  --output data/processed/kaishu_style_v27.npz \
  --heldout_character 武
```

再进行通用训练。排名前 5 的同字候选只用于低学习率适配，其余候选保持为独立测试；
排名来自 `audit_character_target_variants.py`，规范目标始终是最终验收对象：

```bash
python -u tools/train_kaishu_style_refiner.py \
  --npz data/processed/kaishu_style_v27.npz \
  --output_dir outputs/kaishu_style_refiner_v27 \
  --epochs 30 \
  --batch_size 12 \
  --device cuda \
  --variant_audit_json \
    outputs/wu_kaishu_variant_audit_v27_baseline/variants.json \
  --adapt_top_k 5 \
  --adapt_epochs 20 \
  --adapt_lr 0.00003
```

输出包括通用/适配检查点、机器可读 `training_metrics.json`，以及
`generic_wu_panels/` 和 `adapted_wu_test_panels/` 中的
geometry/refined/target/abs-diff 图。通用验证、少样本适配和留出测试按来源严格隔离。
最终反演仍需分别报告细化前几何误差和细化后外观误差；若几何 IoU、轨迹覆盖率、
姿态连续性、边界饱和或联合 Jacobian 不合格，不能用外观细化后的低 MSE 覆盖失败。

训练完成后，将冻结的 v26 安全姿态渲染与适配后的外观模型组合评估：

```bash
python -u tools/evaluate_style_refined_render.py \
  --render_image outputs/wu_kaishu_target_v26_gamma_safe/wu_rendered.png \
  --target_image data/raw/targets/wu_kaishu_target.png \
  --style_ckpt outputs/kaishu_style_refiner_v27b/style_refiner_adapted.pt \
  --pose_report outputs/wu_kaishu_target_v26_gamma_safe/wu_report.json \
  --trajectory_csv outputs/wu_kaishu_target_v26_gamma_safe/wu_trajectory.csv \
  --output_dir outputs/wu_kaishu_style_refined_v27 \
  --device cuda
```

该工具分别记录 `geometry_before_refinement` 与
`appearance_after_refinement`，并从原姿态报告继承轨迹覆盖率、各字段边界比例和联合
Jacobian 审计，同时重新计算 z/alpha/beta/gamma 的逐笔连续性。它不会用外观指标
替代姿态验收。规范目标在此处与反演器保持一致：保留原始画布空白并直接缩放到
128×128，不再次裁边或 letterbox。外观输出只允许在原几何硬支持内做全局墨量
标定，默认增益限制为 0.8–1.25；只有 MSE 改善、IoU 基本不退化且墨量平衡不下降
时才标记为 `appearance_accepted`。少样本适配同样必须同时改善 20 个留出候选的
loss 与 MSE，否则 `style_refiner_selected.pt` 自动回退到通用检查点。
v36 的全局墨量增益使用有界二分法直接求解
`mean(clip(prediction * gain, 0, 1)) == target_mean`；不能再用简单均值比代替，
因为高灰度像素截断会系统性低估所需增益。若 0.8–1.25 内不可达，仍保留边界值并
让后续墨量平衡门槛决定是否拒绝。

v33 起，风格细化损失额外约束全局墨量、16×16 局部墨量和支持域内欠墨/过墨。
其中欠墨惩罚高于过墨惩罚，因为硬几何门已经禁止网络在轨迹支持域外补结构。
训练报告新增 `ink_loss`、`local_ink_loss`、`under_ink`、`over_ink`、
`ink_ratio` 和 `ink_balance_score`；少样本适配除 loss/MSE 外还要求留出集墨量
平衡下降不超过 0.01。推荐从通用无“武”训练重新开始：

```bash
python -u tools/train_kaishu_style_refiner.py \
  --npz data/processed/kaishu_style_v27.npz \
  --output_dir outputs/kaishu_style_refiner_v33_ink \
  --epochs 30 \
  --batch_size 12 \
  --device cuda \
  --ink_weight 0.75 \
  --local_ink_weight 0.75 \
  --tone_balance_weight 0.25 \
  --variant_audit_json \
    outputs/wu_kaishu_variant_audit_v27_baseline/variants.json \
  --adapt_top_k 5 \
  --adapt_epochs 20 \
  --adapt_lr 0.00003
```

v34 修正训练/推理支持域不一致：旧版 `mask_only` 用二值结构掩膜直接裁掉
B-BSMG 的抗锯齿软边缘，造成规范目标欠墨。新训练默认
`--support_mode mask_or_soft`，输出门取结构掩膜与软几何支持域的逐像素最大值；
它仍只来自冻结的正向几何渲染，不引入目标图像或姿态标签。旧检查点没有该字段，
加载时继续使用 `mask_only`，因此结果可复现。v34 训练命令为：

```bash
python -u tools/train_kaishu_style_refiner.py \
  --npz data/processed/kaishu_style_v27.npz \
  --output_dir outputs/kaishu_style_refiner_v34_soft_support \
  --epochs 30 \
  --batch_size 12 \
  --device cuda \
  --support_mode mask_or_soft \
  --ink_weight 0.75 \
  --local_ink_weight 0.75 \
  --tone_balance_weight 0.25 \
  --variant_audit_json \
    outputs/wu_kaishu_variant_audit_v27_baseline/variants.json \
  --adapt_top_k 5 \
  --adapt_epochs 20 \
  --adapt_lr 0.00003
```

训练后不得只按规范目标放大软支持域。v35 使用全部未参与适配的“武”候选作为
泛化约束，在留出 MSE 回退不超过 0.0001、墨量平衡不下降且规范目标 IoU 不下降
的候选中，选择规范目标 MSE 最低的尺度，并输出新检查点、JSON 与四联对比图：

```bash
python -u tools/calibrate_style_support.py \
  --npz data/processed/kaishu_style_v27.npz \
  --style_ckpt \
    outputs/kaishu_style_refiner_v34_soft_support/style_refiner_selected.pt \
  --render_image \
    outputs/wu_kaishu_target_v32_width_scan/render_t0.262.png \
  --target_image data/raw/targets/wu_kaishu_target.png \
  --output_dir outputs/wu_kaishu_target_v35_support_calibration \
  --scales 1.0 1.1 1.2 1.3 1.5 1.7 2.0 2.5 \
  --device cuda
```

该步骤只校准由冻结几何渲染导出的软支持域，不修改也不识别
`x/y/z/alpha/beta/gamma`。随后应把
`style_refiner_support_calibrated.pt` 交给
`evaluate_style_refined_render.py`，继续执行姿态安全与外观接受门槛。

若当前阶段只优化 x/y，额外传入
`--posture_report outputs/wu_kaishu_target_v26_gamma_safe/wu_report.json`，
以继承被冻结姿态的联合 Jacobian 审计，同时仍从当前报告读取 x/y 位移和边界比例。

不同阻尼或差分步长的 x/y 独立复验完成后，量化跨运行离散度：

```bash
python -u tools/compare_xy_refinements.py \
  --trajectory_csvs \
    outputs/wu_kaishu_target_v28b_xy_stability/wu_trajectory.csv \
    outputs/wu_kaishu_target_v28c_xy_restart/wu_trajectory.csv \
  --report_jsons \
    outputs/wu_kaishu_target_v28b_xy_stability/wu_report.json \
    outputs/wu_kaishu_target_v28c_xy_restart/wu_report.json \
  --output_json outputs/wu_kaishu_target_v28_xy_stability.json \
  --max_normalized_rms_std 0.02
```

报告同时检查 x/y 的 normalized RMS 标准差、画布像素差，以及
z/alpha/beta/gamma 是否逐点完全不变。默认只在边界比例不超过 5%、平均位移不
超过 2 px、目标覆盖率不低于 0.99 的候选中按 IoU 选择结果；不合格候选不会因
像素误差较低而被选中。
# Restart-stability audit

For simulation-only pose restarts without real pose truth, use
`tools/compare_pose_refinements.py`. The complete command, thresholds, and
physical-calibration limitation are documented in
`docs/pose_restart_stability.md`.

## v41: prevent false strokes caused by x/y foldbacks

The v31 trajectory has valid DOWN/UP boundaries, but its third stroke contains
an invalid geometric deformation: a short SVG entry segment was stretched and
folded back into a long visible connector. This is not an inter-stroke
pen-lift failure. CGL-node smoothness and a bounded point offset do not prevent
this failure by themselves.

v41 adds decoded point-space constraints on every segment inside each stroke:

- `--xy_segment_length_weight` preserves the original segment-length ratio.
- `--xy_segment_direction_weight` prevents segment reversal while still
  allowing a whole stroke to translate.

Always restart this correction from an undistorted trajectory, not from v31:

```bash
python -u tools/invert_paper_trajectory.py \
  --trajectory_csv data/raw/trajectories.csv \
  --initial_pose_csv \
    outputs/wu_kaishu_target_v26_gamma_safe/wu_trajectory.csv \
  --posture_prior_pose_csv \
    outputs/wu_kaishu_target_v26_gamma_safe/wu_trajectory.csv \
  --target_image data/raw/targets/wu_kaishu_target.png \
  --bbsmg_ckpt outputs/paper_bbsmg_gamma_v13/bbsmg_best.pt \
  --character 武 \
  --output_dir outputs/wu_kaishu_target_v41_shape_safe_a \
  --order 5 \
  --max_steps 15 \
  --damping 0.1 \
  --field_mode xy_only \
  --optimize_xy \
  --xy_max_offset_px 4.0 \
  --xy_smoothness_weight 0.8 \
  --xy_prior_weight 0.2 \
  --xy_segment_length_weight 2.0 \
  --xy_segment_direction_weight 1.0 \
  --cap_order_to_points \
  --footprint_longitudinal_scale 0.22 \
  --footprint_transverse_scale 0.262
```

Inspect `diagnostics.xy_optimization.segment_length_ratio` and
`segment_direction_cosine` in `wu_report.json` together with the rendered
image. A candidate should normally keep maximum length ratio below 1.5,
minimum direction cosine above 0.7, mean point displacement below 2 px,
component-bound fraction below 5%, and target coverage at 1.0. Pixel MSE or
IoU alone must not select a trajectory that creates a false stroke.

## v42: fused z/alpha/beta/gamma derivation

v42 implements the project-specific combination of the two papers:

1. PSOC/LM optimizes `H` (prototype CSV `z`, in millimetres) while x/y
   remains fixed at the accepted v41 trajectory.
2. The Wang dynamic model converts H into stateful width/drag geometry.
3. With H fixed, the B-BSMG regression is inverted as a regularized 3-by-2
   least-squares problem to derive alpha and beta.
4. Gamma is `atan2(y[i+1]-y[i], x[i+1]-x[i])` within each stroke. The final
   point inherits the preceding direction; no direction crosses a pen-up.

```bash
python -u tools/invert_paper_trajectory.py \
  --trajectory_csv data/raw/trajectories.csv \
  --initial_pose_csv \
    outputs/wu_kaishu_target_v41_shape_safe_a/武_paper_inverse_trajectory.csv \
  --posture_prior_pose_csv \
    outputs/wu_kaishu_target_v41_shape_safe_a/武_paper_inverse_trajectory.csv \
  --target_image data/raw/targets/wu_kaishu_target.png \
  --bbsmg_ckpt outputs/paper_bbsmg_gamma_v13/bbsmg_best.pt \
  --character 武 \
  --output_dir outputs/wu_kaishu_target_v42b_fused_pose_smooth \
  --order 5 \
  --max_steps 15 \
  --damping 0.1 \
  --field_mode h_only \
  --fused_pose_from_height \
  --pose_inverse_regularization 0.00001 \
  --cap_order_to_points \
  --h_smoothness_weight 0.05 \
  --h_prior_weight 0.01 \
  --h_point_velocity_weight 0.5 \
  --h_point_acceleration_weight 1.0 \
  --footprint_longitudinal_scale 0.22 \
  --footprint_transverse_scale 0.262 \
  --device cuda
```

Forward verification must use the same fusion mode. It recomputes alpha,
beta and gamma from z/x/y and verifies that the CSV agrees. Gamma is not
applied a second time as an axial footprint rotation.

```bash
python -u tools/render_paper_trajectory.py \
  --trajectory_csv data/raw/trajectories.csv \
  --bbsmg_ckpt outputs/paper_bbsmg_gamma_v13/bbsmg_best.pt \
  --pose_csv \
    outputs/wu_kaishu_target_v42b_fused_pose_smooth/武_paper_inverse_trajectory.csv \
  --character 武 \
  --target_image data/raw/targets/wu_kaishu_target.png \
  --output_image outputs/wu_kaishu_target_v42b_forward/render.png \
  --fused_pose_from_height \
  --pose_inverse_regularization 0.00001 \
  --footprint_longitudinal_scale 0.22 \
  --footprint_transverse_scale 0.262 \
  --device cuda
```

Inspect `fused_pose_derivation` for geometry reconstruction error and
alpha/beta bound fractions, and `fused_pose_validation` for CSV recomputation
errors. High bound saturation means the two paper models need real cross-brush
calibration; it must be reported rather than hidden by invented non-zero
angles. Camera, paper, TCP and real-brush calibration remain mandatory before
robot execution.

## v43: discard x/y inherited from an obsolete target

`--initial_pose_csv` historically initialized both pose and x/y. Therefore an
old xingkai run could silently replace the current raw kaishu trajectory, and
the v41 segment constraints would preserve that already-corrupted path. Use
`--initial_pose_xy_source trajectory` to inherit only H/angles while taking
x/y from `--trajectory_csv`:

```bash
python -u tools/invert_paper_trajectory.py \
  --trajectory_csv data/raw/trajectories.csv \
  --initial_pose_csv outputs/wu_kaishu_target_v42b_fused_pose_smooth/武_paper_inverse_trajectory.csv \
  --initial_pose_xy_source trajectory \
  --posture_prior_pose_csv outputs/wu_kaishu_target_v42b_fused_pose_smooth/武_paper_inverse_trajectory.csv \
  --target_image data/raw/targets/wu_kaishu_target.png \
  --bbsmg_ckpt outputs/paper_bbsmg_gamma_v13/bbsmg_best.pt \
  --character 武 \
  --output_dir outputs/wu_kaishu_target_v43_kaishu_raw_xy \
  --order 5 --max_steps 15 --damping 0.1 \
  --field_mode xy_only --optimize_xy --fused_pose_from_height \
  --pose_inverse_regularization 0.00001 \
  --xy_max_offset_px 4 --xy_smoothness_weight 0.8 --xy_prior_weight 0.2 \
  --xy_segment_length_weight 2 --xy_segment_direction_weight 1 \
  --cap_order_to_points \
  --footprint_longitudinal_scale 0.22 \
  --footprint_transverse_scale 0.262 --device cuda
```

The report field `initialization.initial_pose_xy_source` must be `trajectory`
for this kaishu reset.

## v44: target-local footprint calibration

After v43 accepts the geometry, measure target width on the path normal and
local length on its tangent. The reported aspect ratio and confidence feed a
fixed-H 2-by-2 solve `[Lt+Lh, Lr] -> [alpha, beta]`. x/y and H are preserved;
gamma is the within-stroke forward `atan2(dy,dx)` direction.

```bash
python -u tools/calibrate_target_local_footprints.py \
  --trajectory_csv data/raw/trajectories.csv \
  --pose_csv outputs/wu_kaishu_target_v43_kaishu_raw_xy/武_paper_inverse_trajectory.csv \
  --target_image data/raw/targets/wu_kaishu_target.png \
  --bbsmg_ckpt outputs/paper_bbsmg_gamma_v13/bbsmg_best.pt \
  --character 武 \
  --output_dir outputs/wu_kaishu_target_v44_target_footprint \
  --footprint_longitudinal_scale 0.22 \
  --footprint_transverse_scale 0.262 \
  --radius_px 12 --ink_threshold 0.35 --target_blend 0.5 \
  --axis_scale_blend 0.35 \
  --minimum_confidence 0.1 --angle_regularization 0.01 \
  --device cuda
```

Inspect `footprint_overlay.png`, `local_footprints.csv`, `diff.png`, and
`report.json`. Clipped or off-centre cross sections are blended back toward
Wang geometry. `render_scale_only.png` and `axis_scale_calibration.accepted`
separate global axis-scale evidence from pointwise alpha/beta evidence.
`acceptance.accepted` is false unless IoU is non-decreasing,
at least 20 cross sections are valid, and every angle boundary fraction is at
most 0.25; a rejected candidate has no `recommended_pose_csv`. This is an
image-derived simulation candidate, not real brush/TCP/robot calibration.

## v45: structure-preserving target-skeleton x/y refinement

When local footprint confidence is low, first move the safe trajectory toward
the current target skeleton. `--xy_target_skeleton_weight` samples a clipped
target-skeleton distance field at every decoded point. The sampling is
differentiable with respect to x/y, never crosses stroke boundaries, and is
used together with segment length/direction constraints. When enabled,
checkpoint selection uses the complete regularized objective rather than
silently restoring the lowest-MSE checkpoint and discarding skeleton gains.

```bash
python -u tools/invert_paper_trajectory.py \
  --trajectory_csv data/raw/trajectories.csv \
  --initial_pose_csv outputs/wu_kaishu_target_v45a_xy_stage2/武_paper_inverse_trajectory.csv \
  --initial_pose_xy_source csv \
  --posture_prior_pose_csv outputs/wu_kaishu_target_v43_kaishu_raw_xy/武_paper_inverse_trajectory.csv \
  --target_image data/raw/targets/wu_kaishu_target.png \
  --bbsmg_ckpt outputs/paper_bbsmg_gamma_v13/bbsmg_best.pt \
  --character 武 \
  --output_dir outputs/wu_kaishu_target_v45b_xy_skeleton \
  --order 5 --max_steps 10 --damping 0.1 \
  --field_mode xy_only --optimize_xy --fused_pose_from_height \
  --pose_inverse_regularization 0.00001 \
  --xy_max_offset_px 4 --xy_smoothness_weight 1 \
  --xy_prior_weight 0.15 --xy_segment_length_weight 3 \
  --xy_segment_direction_weight 2 \
  --xy_target_skeleton_weight 2 \
  --xy_target_skeleton_max_distance_px 12 \
  --xy_target_skeleton_threshold 0.35 \
  --cap_order_to_points \
  --footprint_longitudinal_scale 0.22 \
  --footprint_transverse_scale 0.262 --device cuda
```

Accept only if target coverage and local-footprint confidence improve while
segment direction, segment length ratio, displacement bounds, and the false
stroke visual audit remain safe. Then rerun v44 calibration using the accepted
v45 CSV; do not calibrate alpha/beta against a rejected x/y candidate.
## Gamma 语义修复与整字端到端精修（仿真）

`paper_bbsmg_gamma_v14` 的训练 gamma 是局部笔刷旋转，训练范围约为 ±30°。
轨迹 CSV 中的 gamma 则是每个笔画的绝对 `atan2(dy, dx)` 前进方向。当前
`PaperDynamicConfig.gamma_mode=relative_to_heading` 会在进入 6D B-BSMG 前自动
计算 `wrap(gamma_csv - forward_xy_heading)`，再由渲染器负责整笔旋转；旧的绝对
语义可用 `--gamma_mode absolute_heading` 仅作兼容对照。

冻结 B-BSMG、直接对最终整字图像优化有限 x/y 和 H/alpha/beta：

```bash
/home/robot/miniconda3/envs/ddpm/bin/python -u tools/optimize_character_end_to_end.py \
  --trajectory_csv data/raw/trajectories.csv \
  --pose_csv outputs/wu_joint_xy_pose_footprint_candidates_v1/candidate_00_xy_pose_base/pose.csv \
  --bbsmg_ckpt outputs/paper_bbsmg_gamma_v14/bbsmg_best.pt \
  --target_image data/raw/targets/wu_kaishu_target.png \
  --output_dir outputs/wu_character_e2e_v1 \
  --character 武 --sample_id 武_fake_sim --device cuda \
  --iterations 60 --xy_max_delta_px 4 --h_max_delta_mm 2 \
  --alpha_max_delta_deg 2 --beta_max_delta_deg 1.5
```

输出包括 `comparison.png`、`diff.png`、`report.json` 和可重放的
`pose_refined.csv`。必须再用 `tools/render_paper_trajectory.py --fail_on_unsafe`
重放；报告中的 `gamma_csv_semantics` 与 `gamma_model_semantics` 用于区分导出
字段和网络输入语义。该流程仍是仿真候选，不能替代真实毛笔/TCP 标定。

## v15：楷书限定的通用 B-BSMG（按字符分组验证）

局部 patch 的随机切分会把几乎相同的解析笔触同时放进训练和验证，不能
代表整字泛化。v15 使用 `data/raw/data.csv` 的 `chirography=楷` 索引，只保留
数据库中出现的楷书字符；`武` 完全留出，不参与训练。每个楷书字符只采样少量
姿态作为一个独立 group，验证集按字符 group 切分。B-BSMG 内部仍使用
`ink=1/background=0` 计算损失，gamma 是相对当前轨迹方向的局部角度。

```bash
/home/robot/miniconda3/envs/ddpm/bin/python -u tools/build_general_paper_bbsmg_dataset.py \
  --trajectory_csv data/raw/trajectories.csv \
  --style_data_csv data/raw/data.csv \
  --style_image_dir data/raw/images \
  --style_json_dir data/raw/json_files \
  --chirography 楷 \
  --holdout_character 武 \
  --samples_per_character 5 \
  --output_npz data/processed/paper_bbsmg_general_v15_kaishu.npz

PYTHONPATH=. /home/robot/miniconda3/envs/ddpm/bin/python -u tools/train_bbsmg.py \
  --config configs/paper_bbsmg_gamma_v13.yaml \
  --npz_path data/processed/paper_bbsmg_general_v15_kaishu.npz \
  --output_dir outputs/paper_bbsmg_general_v15_kaishu \
  --epochs 50 --val_ratio 0.1 --group_split \
  --lr_factor 0.5 --lr_patience 4 --min_lr 0.000001
```

`paper_bbsmg_general_v15_kaishu.summary.json` 记录楷书过滤、留出字符和
分组信息。该版本是跨楷书字符的解析几何通用基线，不等同于真实毛笔墨迹
标定；真实风格监督仍需单独的局部足迹数据。PNG 导出统一为黑墨白底，
但指标仍在内部 ink=1/background=0 空间计算。

## v16：楷书风格细化（局部足迹 + 速度/压力代理）

v16 冻结已验证的 v15 通用 B-BSMG，仅训练楷书数据库的可微风格细化器。
每个样本输入 12 个通道：目标真实墨迹的结构掩膜、骨架、距离/宽度和软几何，
以及匹配轨迹的中心线、邻近度、笔画顺序、方向、z 高度压力代理和逐采样位移
速度代理。当前 `trajectories.csv` 没有力传感器和时间戳，因此压力、速度是可复现
的轨迹代理量，不能解释为真实物理标定值。`武` 只作为留出字符评估，不参与通用
训练，避免整字泄漏。

优先使用已有楷书 NPZ 增广，避免重复解析 LabelMe 数据：

```bash
PYTHONPATH=. /home/robot/miniconda3/envs/ddpm/bin/python -u \
  tools/augment_kaishu_style_dataset_v16.py \
  --base_npz data/processed/kaishu_style_v27.npz \
  --trajectory_csv data/raw/trajectories.csv \
  --output_npz data/processed/kaishu_style_v16.npz \
  --heldout_character 武 --min_trajectory_coverage 0.30

PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/robot/miniconda3/envs/ddpm/bin/python -u \
  tools/train_kaishu_style_refiner.py \
  --npz data/processed/kaishu_style_v16.npz \
  --output_dir outputs/kaishu_style_v16 \
  --heldout_character 武 --epochs 50 --batch_size 16 \
  --base_channels 24 --workers 2 --lr 0.0003 --val_ratio 0.15 \
  --device cuda --support_mode mask_or_soft
```

将冻结的 v15 整字渲染送入细化器，并同时报告几何图与外观图，只有墨量、端点
和局部足迹改善且几何 IoU 不下降时才接受细化结果：

```bash
PYTHONPATH=. /home/robot/miniconda3/envs/ddpm/bin/python -u \
  tools/evaluate_style_refined_render.py \
  --render_image outputs/wu_character_e2e_v15_kaishu/render_black_white_replay.png \
  --target_image data/raw/targets/wu_kaishu_target.png \
  --style_ckpt outputs/kaishu_style_v16/style_refiner_selected.pt \
  --pose_report outputs/wu_character_e2e_v15_kaishu/render_black_white_replay.json \
  --trajectory_csv data/raw/trajectories.csv --character 武 \
  --output_dir outputs/wu_style_v16_eval --device cuda \
  --trajectory_padding 4 --trajectory_width 3 \
  --footprint_width_scale_px 16 --structure_threshold 0.35 \
  --metric_threshold 0.35 --min_ink_gain 0.8 --max_ink_gain 1.25
```

输出 `metrics.json`、`comparison.png`、`diff.png` 和黑墨白底结果图。几何指标仍由
冻结 v15 B-BSMG 负责；风格细化器不得掩盖姿态、抬笔、笔画顺序或轨迹安全问题。
真实毛笔足迹/压力和时间标定接入前，v16 仍属于仿真风格适配结果。

## robot 分支：CoppeliaSim 虚拟机械臂轨迹回放

`robot` 分支提供一个不依赖真实机械臂和毛笔的 CoppeliaSim 可视化原型。
它读取完整的 `x/y/z/alpha/beta/gamma/state` 轨迹 CSV，在纸面坐标系中创建
六个关节标记、末端姿态标记和分笔轨迹绘图对象。状态 `2=UP`、`3=TRANSITION`
只执行抬笔移动，不绘制轨迹，因此不会把相邻笔画连接起来。当前原型使用直接
末端位姿回放、圆柱形末端工具标记和可视化关节链，不包含真实机器人 IK、碰撞、
动力学或毛笔接触模型。

### 1. 启动 CoppeliaSim

在 Ubuntu 远程主机上启动带 ZMQ Remote API 的实例。若已有可用的 X0 显示会话，
查看已经保存的回放场景可直接执行：

```bash
cd /home/robot/CoppeliaSim_Edu_V4_7_0_rev4_Ubuntu22_04
DISPLAY=:0 ./coppeliaSim.sh -h \
  -f /home/robot/coppeliasim/machine_learning/model/outputs/coppeliasim_wu_v1/virtual_arm_wu.ttt \
  > /tmp/coppeliasim_virtual_arm.log 2>&1 &
```

如果要观察 GUI，把 `-h` 去掉。确认 CoppeliaSim 的 `ZMQ remote API server`
插件已启用，并保持默认端口 `23000`。

第一次没有 `.ttt` 场景时，可先启动 CoppeliaSim 后执行第 3 步回放，并用
`--scene_output` 生成它；
不要使用不带参数的 `-s`，CoppeliaSim 4.7 要求 `-s` 后提供自动启动毫秒数。

若需要从空场景开始，可先建立一个最小场景并保持服务运行：

```bash
DISPLAY=:0 ./coppeliaSim.sh -h -c \
  'sim.createDummy(0.1); sim.saveScene("/home/robot/coppeliasim/machine_learning/model/outputs/coppeliasim_empty.ttt")' \
  > /tmp/coppeliasim_virtual_arm.log 2>&1 &
```

### 2. 先做离线安全检查

这一步不连接模拟器，会生成坐标映射后的预览图、状态统计和安全报告：

```bash
cd /home/robot/coppeliasim/machine_learning/model
PYTHONPATH=. /home/robot/miniconda3/envs/ddpm/bin/python -u \
  tools/coppeliasim_virtual_arm.py \
  --trajectory_csv outputs/wu_character_e2e_v15_kaishu/pose_refined.csv \
  --require_trajectory_prototype paper_target_local_footprint_v44 \
  --require_trajectory_sha256 52e15c5f1b4cdf454a78a5345cd6516896740aebc72b2b56b436016ba0df3251 \
  --character 武 --sample_id 武_fake_sim \
  --output_dir outputs/coppeliasim_wu_v1 --offline
```

应看到 `cross_stroke_segments=0`，并在
`outputs/coppeliasim_wu_v1/trajectory_preview.png` 查看轨迹方向。

### 3. 连接模拟器并回放

```bash
PYTHONPATH=. /home/robot/miniconda3/envs/ddpm/bin/python -u \
  tools/coppeliasim_virtual_arm.py \
  --trajectory_csv outputs/wu_character_e2e_v15_kaishu/pose_refined.csv \
  --require_trajectory_prototype paper_target_local_footprint_v44 \
  --require_trajectory_sha256 52e15c5f1b4cdf454a78a5345cd6516896740aebc72b2b56b436016ba0df3251 \
  --character 武 --sample_id 武_fake_sim \
  --output_dir outputs/coppeliasim_wu_latest_csv_pose_v1 \
  --orientation_mode csv_pose --strict_ik \
  --interval 0.015 --max_step_m 0.002 --client_port 23000 \
  --keep_scene
```

默认把图像坐标 y 轴翻转为纸面世界坐标；若需要保持原始方向，添加
`--no_flip_y`。`trajectory_report.json` 会记录输入 CSV 的绝对路径、SHA256、
prototype、姿态单位/范围、状态计数、抬笔规则和仿真安全声明；来源或哈希不匹配
时会在加载 UR5 前终止。`--orientation_mode csv_pose` 将 CSV 的
`alpha/beta/gamma` 作为相对于纸面平行基准姿态的局部 XYZ 弧度偏移，
`--strict_ik` 禁止将未达容差的部分 IK 解计为成功。该原型的验收目标是轨迹显示、
笔画边界和姿态字段回放正确，不能
据此宣称真实机器人可执行性；接入 UR5 等真实机械臂前还需重新做 TCP、纸面和
关节限位标定。
