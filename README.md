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

正向渲染中先按动态笔刷论文对宽度 `w=Lr` 和拖曳长度 `d=Lt+Lh` 做一阶惯性更新，再用带参考姿态正则的回归逆解得到 B-BSMG 的虚拟 `(H,alpha,beta)`。笔尖偏移采用受自由偏移与相邻点位移共同限制的 `min` 原型。轨迹切向角由固定 x/y 计算，不冒充第三姿态角。

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
`footprint_scale=0.35`，即当前归一化整字画布中的有效比例为约
`7 pixel/model-unit`。这是针对当前 128×128 目标与轨迹尺度的仿真桥接参数，
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
  --footprint_scale 0.35 \
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
  --target_image assets/targets/wu_kaishu_target.png \
  --bbsmg_ckpt outputs/paper_bbsmg_v1/bbsmg_best.pt \
  --character 武 \
  --output_dir outputs/wu_paper_inverse_v4 \
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
  --target_image assets/targets/wu_kaishu_target.png \
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
- `pose_frame=paper_model`、`prototype=paper_psoc_lm_v4`。

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

裁剪结果只用于诊断。正式 CSV 必须用 v4 反演器重新生成。v4 保留 v3 的
逐点有界 sigmoid，并沿相邻固定 x/y 线段按不超过 2 px 的间距插入可微
渲染样本，解决稀疏轨迹被渲染成离散印章点的问题。插值样本只进入正向渲染，
导出的原始 x/y 点数与坐标不变。v4 进一步为 H、alpha、beta 分别设置先验和
平滑权重：H 保持弱约束，图像辨识能力较弱的 alpha/beta 使用更强约束，避免
角度大量贴到物理上下限。`wu_report.json` 的
`lm.diagnostics.image_jacobian_sensitivity` 会报告三类变量对图像的相对敏感度，
`bound_fraction_within_1pct` 会报告接近上下限的点比例。当前 alpha/beta 应视为
带先验的仿真估计值，不能当作真实机器人姿态真值。

### 11.5 当前原型不能直接下发机器人

论文回归系数、`20 pixel/model-unit`、`footprint_scale=0.35`、惯性系数和偏移比例目前都是仿真参数。真实执行前必须依次替换：

1. 用真实毛笔采集 `(H,alpha,beta) → (Lt,Lh,Lr)` 标定数据并重拟合回归；
2. 用连续书写数据拟合宽度、拖曳、偏移和惯性参数；
3. 完成相机像素、纸面坐标、机器人基坐标和 TCP 的外参标定；
4. 明确定义 `paper_model` 姿态到机器人控制器 Euler/四元数的旋转顺序；
5. 加入关节限位、速度、加速度、碰撞和纸面接触力约束；
6. 低速、离纸、单笔验证后才允许接触纸面。

因此当前导出的六维序列是“论文纸面坐标系中的仿真反演结果”，不是可直接执行的机器人轨迹真值。
