# 真实毛笔与机器人标定接口

单张整字图像不能唯一确定六自由度轨迹。真实阶段应先采集同步的机器人 TCP
姿态和笔触观测，再替换论文曲线与当前仿真尺度。仓库定义
`robot_brush_calibration_v1` CSV，模板为
`examples/robot_brush_calibration_v1.csv`。

字段单位直接写在列名中：

```text
trial_id, point_id, timestamp_s
x_mm, y_mm, z_mm
alpha_rad, beta_rad, gamma_rad
contact
footprint_width_mm, footprint_length_mm, footprint_angle_rad, image_path
```

每组标定应只改变一个主要变量，并覆盖多个重复试验。`timestamp_s` 在每个
trial 内必须严格递增，角度统一使用弧度。先运行：

```bash
python -u tools/validate_robot_brush_calibration.py \
  --calibration_csv data/raw/robot_brush_calibration.csv \
  --output_json data/processed/robot_brush_calibration.summary.json
```

校验器检查键、单位、数值范围、接触样本、测量字段和各姿态变量的激励跨度。
`excitation_ready=true` 只说明实验改变了该变量，不代表图像能够辨识它。随后仍须
用独立留出 trial 做 Jacobian/灵敏度审计。只有输入激励和输出可辨识性同时通过，
才开放对应变量：

```text
z/H      需要压力或高度扫描以及宽度/长度观测
alpha    需要独立倾角扫描
beta     需要独立纸面倾角扫描
gamma    需要非轴对称笔锋、footprint_angle 或外部姿态观测
```

在完成上述标定前，v11 导出的 H/alpha/beta/gamma 是仿真候选值，不是可以直接
下发真实机器人的物理真值。

## v12 非轴对称 gamma 通道

v12 把 gamma 定义为笔触局部足迹相对于轨迹切向的附加轴向角。默认不开启，因此
旧 checkpoint、旧命令和旧 CSV 的行为不变。开启时必须满足：

- `footprint_longitudinal_scale != footprint_transverse_scale`；
- `gamma_max_abs_deg` 给出对称仿真边界；
- `field_mode=auto` 的 gamma 相对中位灵敏度达到门槛。

只做审计而不更新参数：

```bash
python -u tools/invert_paper_trajectory.py \
  --trajectory_csv data/raw/trajectories.csv \
  --initial_pose_csv outputs/wu_paper_inverse_v10_velocity8_w0258/wu_trajectory.csv \
  --target_image assets/targets/wu_kaishu_target.png \
  --bbsmg_ckpt outputs/paper_bbsmg_v1/bbsmg_best.pt \
  --character 武 \
  --output_dir outputs/wu_paper_inverse_v12_gamma_audit \
  --output_stem wu \
  --device cuda \
  --order 8 \
  --cap_order_to_points \
  --optimization_size 32 \
  --max_steps 0 \
  --field_mode auto \
  --min_relative_median_sensitivity 0.45 \
  --optimize_gamma \
  --gamma_max_abs_deg 30 \
  --gamma_smoothness_weight 0.10 \
  --gamma_prior_weight 0.05 \
  --footprint_longitudinal_scale 0.22 \
  --footprint_transverse_scale 0.258
```

当前“武”字审计的相对中位灵敏度为：

```text
H       1.0000
alpha   0.1542
beta    0.0479
gamma   0.1078
```

因此阈值 `0.45` 下仍只开放 H，gamma 保持初始 0。不能为了得到非零角度而降低
门槛；应先通过真实非轴对称笔锋的 `footprint_angle_rad` 或同步外部姿态观测完成
标定。

## v13 姿态敏感 B-BSMG

v12 的 5D checkpoint 不把 gamma 输入神经网络，只能用几何旋转近似。v13 新增
6D checkpoint：

```text
H_mm, alpha_rad, beta_rad, gamma_rad, x0_px, y0_px
```

合成数据使用拉丁超立方分层采样，使各姿态变量独立覆盖其范围：

```bash
python -u tools/build_paper_bbsmg_dataset.py \
  --output_npz data/processed/paper_bbsmg_gamma_v13.npz \
  --count 50000 \
  --image_size 128 \
  --pixels_per_model_unit 20 \
  --supersample 4 \
  --include_gamma \
  --gamma_max_abs_deg 30 \
  --sampling_mode latin_hypercube \
  --seed 42
```

训练：

```bash
python -u tools/train_bbsmg.py \
  --config configs/paper_bbsmg_gamma_v13.yaml \
  --npz_path data/processed/paper_bbsmg_gamma_v13.npz \
  --val_ratio 0.1 \
  --epochs 50 \
  --output_dir outputs/paper_bbsmg_gamma_v13
```

常规验证：

```bash
PYTHONPATH=. python -u tools/evaluate_bbsmg.py \
  --config configs/paper_bbsmg_gamma_v13.yaml \
  --npz_path data/processed/paper_bbsmg_gamma_v13.npz \
  --checkpoint outputs/paper_bbsmg_gamma_v13/bbsmg_best.pt \
  --output_dir outputs/eval_paper_bbsmg_gamma_v13
```

姿态独立扫描：

```bash
python -u tools/evaluate_paper_pose_sensitivity.py \
  --checkpoint outputs/paper_bbsmg_gamma_v13/bbsmg_best.pt \
  --output_json outputs/eval_paper_bbsmg_gamma_v13/pose_sensitivity.json \
  --device cuda \
  --samples_per_field 9
```

`response_ratio` 接近 1 表示网络复现了对应姿态端点之间的解析笔触变化。该指标只
验证仿真网络是否使用了该输入，不证明真实毛笔姿态可以从单张图像唯一反演。

## v14 节点级噪声可观测性门控

旧门控按整类变量的相对中位 Jacobian 灵敏度决定是否开放 H/alpha/beta/gamma，
会掩盖“同一字段只有少数 CGL 节点可辨识”的情况。v14 改为逐节点计算：

```text
SNR = 单个节点跨完整物理范围引起的像素 RMS / 图像噪声 RMSE
```

默认噪声来自 v13 checkpoint 的验证集 `plain_mse` 平方根。只有
`SNR >= --min_observability_snr` 的节点进入 LM，其他节点保持初值。报告格式
为 `paper_psoc_lm_v14_node_snr_gate`，并在
`lm.diagnostics.observability_gate.selected_node_columns` 中保存每个字段的
候选数、入选数、SNR 分布和决策变量列号。

只审计、不更新姿态：

```bash
python -u tools/invert_paper_trajectory.py \
  --trajectory_csv data/raw/trajectories.csv \
  --initial_pose_csv outputs/wu_paper_inverse_v10_velocity8_w0258/wu_trajectory.csv \
  --target_image data/raw/targets/wu_kaishu_target.png \
  --bbsmg_ckpt outputs/paper_bbsmg_gamma_v13/bbsmg_best.pt \
  --character 武 \
  --output_dir outputs/wu_paper_inverse_v14_node_audit \
  --output_stem wu \
  --device cuda \
  --order 8 \
  --cap_order_to_points \
  --optimization_size 32 \
  --max_steps 0 \
  --field_mode auto \
  --observability_gate_mode node_snr \
  --min_observability_snr 1.0 \
  --optimize_gamma \
  --gamma_max_abs_deg 30 \
  --footprint_longitudinal_scale 0.22 \
  --footprint_transverse_scale 0.258
```

对于真实目标，不得通过人为降低 checkpoint 噪声来放行角度节点。若一次 LM
更新降低正则化代价却增加全分辨率 MSE，优化器会返回历史上全分辨率 MSE 最好的
姿态，而不是最后一步。

### 合成闭环恢复测试

在真实字帖反演前，先验证“正向生成已知姿态，再从扰动初值恢复”的局部闭环。
这不会证明真实姿态唯一，但能发现 gamma 未进入正向渲染、边界参数化错误或 LM
实现错误。

```bash
python -u tools/build_paper_roundtrip_probe.py \
  --input_pose_csv outputs/wu_paper_inverse_v10_velocity8_w0258/wu_trajectory.csv \
  --output_pose_csv outputs/wu_paper_roundtrip_v14/wu_truth.csv

python -u tools/render_paper_trajectory.py \
  --trajectory_csv data/raw/trajectories.csv \
  --bbsmg_ckpt outputs/paper_bbsmg_gamma_v13/bbsmg_best.pt \
  --pose_csv outputs/wu_paper_roundtrip_v14/wu_truth.csv \
  --character 武 \
  --output_image outputs/wu_paper_roundtrip_v14/wu_truth_render.png \
  --device cuda \
  --footprint_longitudinal_scale 0.22 \
  --footprint_transverse_scale 0.258

python -u tools/build_paper_roundtrip_probe.py \
  --input_pose_csv outputs/wu_paper_roundtrip_v14/wu_truth.csv \
  --output_pose_csv outputs/wu_paper_roundtrip_v14/wu_perturbed_initial.csv \
  --profile perturbed_initial
```

随后以 `wu_perturbed_initial.csv` 作为 `--initial_pose_csv` 运行 v14 反演，并
量化姿态恢复误差：

```bash
python -u tools/evaluate_paper_pose_recovery.py \
  --reference_csv outputs/wu_paper_roundtrip_v14/wu_truth.csv \
  --estimate_csv outputs/wu_paper_roundtrip_v14/local_inverse/wu_trajectory.csv \
  --output_json outputs/wu_paper_roundtrip_v14/local_pose_error.json
```

从统一默认姿态恢复完整轨迹属于全局逆问题；LM 只保证局部更新，不能把局部闭环
通过解释为“能够从单张二维字帖无条件生成唯一六维机器人轨迹”。

当前“武”字合成闭环实测还揭示了分辨率和联合可辨识性两个限制：

```text
32×32 LM：更新降低低分辨率代价，却增加 128×128 MSE，最终回退第 0 步
64×64 LM：128×128 MSE 0.001300 -> 0.000188，Dice=0.993039

姿态 RMSE（扰动初值 -> 64×64 LM）
H       0.400000 -> 0.285197 mm
alpha   0.008727 -> 0.032125 rad
beta    0.004363 -> 0.016127 rad
gamma   0.034907 -> 0.080182 rad
```

因此需要姿态反演时建议 `--optimization_size 64`；但高图像相似度仍不能作为姿态
正确性的证据。alpha/beta/gamma 会与 H 和彼此补偿，即使各自节点 SNR 高于噪声，
联合姿态仍可能非唯一。当前导出必须保留 `simulation_only=true` 和字段置信度。
要生成可监督真实机器人或扩散模型的数据，下一阶段必须加入至少一种独立观测：

- 同步机器人编码器/TCP 姿态；
- 接触力或压深；
- 笔触宽度、长度和主轴角；
- 同一轨迹在独立姿态激励下的多幅标定图像。

## v15 联合 Jacobian 审计与姿态降阶

v14 的节点 SNR 是必要条件，不是充分条件。v15 对所有入选姿态节点按完整物理
范围缩放 Jacobian，并继续计算有效秩、条件数和字段子空间的典型相关系数。默认
保守阈值为：

```text
effective_rank / selected_columns >= 0.90
condition_number <= 100
max_field_canonical_correlation <= 0.95
```

任一条件失败时，报告仍允许保留仿真消融结果，但所有参与联合优化的姿态字段都
写为：

```text
confidence = low
reason = optimized_but_jointly_nonidentifiable
jointly_identifiable = false
```

同一个“武”字合成闭环的阶数对照：

```text
order 3:
  selected/effective rank = 126/116
  condition number = 83792.86
  max correlation = 0.994769 (alpha/beta)

order 1:
  selected/effective rank = 64/64
  condition number = 32.77
  max correlation = 0.959469 (alpha/gamma)
```

order 1 的五步 LM 将图像 MSE 从 `0.001328` 降到 `0.000063`，但姿态真值恢复
并不一致：

```text
姿态 RMSE（扰动初值 -> order 1 LM）
H       0.400000 -> 0.079988 mm
alpha   0.008727 -> 0.024171 rad
beta    0.004363 -> 0.005666 rad
gamma   0.034907 -> 0.023450 rad
```

因此降阶显著改善了数值条件和 H/gamma，但 alpha 仍被 gamma 补偿。当前推荐用
`--order 1 --optimization_size 64` 做仿真研究，同时保留联合审计；不能因为
Dice/IoU 很高就把 alpha/beta/gamma 当作真实标签。
