# 消融实验与轨迹差异分析

## 1. 实验变量总表

### 1.1 四组已完成消融实验

| 组别 | 配置文件 | 划分策略 | 划分单位 | Weighted MSE | SSIM | Dice | clDice | Edge | Structure | Ink | 目的 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| A | `stroke10_v1_a_random_legacy_loss.yaml` | random | 单个笔画样本 | 1.0 | 0.5 | 0.5 | 0.3 | 0.2 | 0.2 | 0.1 | 旧损失和随机划分基线 |
| B | `stroke10_v1_b_grouped_legacy_loss.yaml` | group | `sample_id` 整字分组 | 1.0 | 0.5 | 0.5 | 0.3 | 0.2 | 0.2 | 0.1 | 只替换为无泄漏分组划分 |
| C | `stroke10_v1_c_grouped_full.yaml` | group | `sample_id` 整字分组 | 1.0 | 0.3 | 0.3 | 0.05 | 0.1 | 0.05 | 0.1 | 当前默认完整损失 |
| D | `stroke10_v1_d_no_topology.yaml` | group | `sample_id` 整字分组 | 1.0 | 0.3 | 0.3 | 0 | 0 | 0 | 0.1 | 去掉拓扑、边缘与结构约束 |

`positive_weight=4.0` 在四组中保持一致。B、C、D 共用
`outputs/ablations/grouped_split_manifest.json`，因此三组之间的损失比较最干净。

### 1.2 控制变量

| 类别 | 当前值 | 是否变化 | 说明 |
|---|---|---|---|
| 训练数据 | `bbsmg_train_10d.npz`，112617 个样本 | 否 | 四组使用同一 NPZ |
| 输入 schema | `stroke10_v1`，10 维 | 否 | `h, heading, heading_copy, x0, y0, x1, y1, dx, dy, length` |
| 网络结构 | latent 128，base channels 64，输出 1 通道 | 否 | B-BSMG 结构不变 |
| 图像尺寸 | 128 x 128 | 否 | 目标和预测统一尺寸 |
| batch size | 16 | 否 | 适配 GTX 1660 6GB |
| epoch | 30 | 否 | `--epochs 30` 为总 epoch 数 |
| 优化器初始学习率 | 0.0003 | 否 | weight decay 为 `1e-6` |
| 学习率调度 | factor 0.5，patience 3，min `1e-6` | 否 | 四组一致 |
| 随机种子 | 42 | 否 | 当前只有单种子结果 |
| AMP | FP16，启用 | 否 | 训练运行变量，不是消融因素 |
| checkpoint | best 用于评估，last 用于恢复 | 否 | 每 10 epoch 额外保留快照 |
| 动态笔刷 | disabled | 否 | 尚无真实 `z -> width/drag/offset` 标定 |

### 1.3 数据、物理和轨迹优化变量

| 变量 | 单位/范围 | 当前语义 | 当前是否可辨识 |
|---|---|---|---|
| `x, y` | 很可能为 mm | 平面轨迹位置，未约定绝对坐标系 | 只能比较相对位移 |
| `z` | 很可能为 mm | 越大表示下压越深 | pose-v2 可用；旧 10D 实验不直接使用真实 z |
| `alpha, beta, gamma` | rad | 分别绕 x/y/z 轴旋转 | 当前数据角度基本为 0，无法有效学习 |
| `state` | 0/1/2/3 | 落笔/行笔/提笔/提笔移动 | 与 `PointState` 一致 |
| `timestamp` | 未固定 | 采样时间；采样频率可能不固定 | 重采样主要按空间弧长，不假定固定频率 |
| Chebyshev order | 3 到 4 | 每个轨迹分量的参数化阶数 | 默认取 4 |
| LM damping | 0.05 | LM 阻尼 | 轨迹优化控制量 |
| LM max steps | 20 | 每个样本最大优化步数 | 轨迹优化控制量 |
| XYZ 正则 | `1e-4` | 限制平面/位置改变量 | 与 z、角度正则共同防止漂移 |
| z 正则 | `1e-2` | 限制下压深度改变量 | 无笔刷标定时解释能力有限 |
| 角度正则 | `1e-2` | 限制姿态改变量 | 仅 pose-v2 和 `--use_6d` 有效 |

### 1.4 因变量，也就是评价指标

| 指标 | 趋势 | 含义 | 注意事项 |
|---|---|---|---|
| Plain MSE / MAE | 越低越好 | 全图像素误差 | 背景像素较多时容易掩盖前景错误 |
| Foreground MAE | 越低越好 | 真实墨迹区域误差 | 对缺笔和覆盖不足更敏感 |
| Dice score | 越高越好 | 连续灰度重合度 | 与 `evaluate_bbsmg.py` 口径一致 |
| Dice@0.5 / IoU@0.5 | 越高越好 | 二值轮廓重合度 | 会受阈值附近灰度影响 |
| SSIM score | 越高越好 | 11 x 11 高斯窗口结构相似度 | 为兼容项目现有实现，数值尺度不同于常见第三方 SSIM |
| Ink delta | 越接近 0 越好 | 生成平均墨量减真实平均墨量 | 正值多墨，负值缺墨 |
| XYZ/角度 RMSE | 视目标而定 | 优化轨迹相对输入轨迹的改变量 | 没有独立真实机器人轨迹时不能称为真值误差 |
| LM acceptance rate | 结合其他指标 | 优化候选是否改善前景加权目标 | 接受不保证每个单项指标都改善 |

## 2. 已完成结果记录

| 实验 | 验证样本 | MSE | MAE | Dice | IoU@0.5 |
|---|---:|---:|---:|---:|---:|
| A random + legacy loss | 11262 | 0.006156 | 0.007298 | 0.73921 | 0.62657 |
| B grouped + legacy loss | 11272 | 0.004105 | 0.004581 | 0.79149 | 0.68357 |
| C grouped + full loss | 11272 | **0.003749** | **0.004327** | 0.79823 | 0.69417 |
| D grouped + no topology | 11272 | 0.003823 | 0.004419 | **0.79919** | **0.69433** |

当前推荐 C 作为主配置。它相对 B 的 MSE 下降 8.68%，MAE 下降 5.53%，
Dice 提高 0.00674，IoU 提高 0.01060，说明降低旧版结构类损失权重后，像素误差和轮廓重合度同时改善。

D 与 C 基本持平：D 的 Dice 和 IoU 分别只高 0.00096 和 0.00016，但 MSE、MAE
分别差约 1.99% 和 2.12%。这不足以证明结构项无效。两组只有 seed=42 一次结果，建议至少补
3 个种子并报告均值和标准差。不同损失权重下的 `composite_loss` 数值不可直接横向比较。

### 2.1 分目标类型结果

| 实验 | 子集 | 样本 | MSE | MAE | Dice | IoU@0.5 | 全零基线 MSE |
|---|---|---:|---:|---:|---:|---:|---:|
| A | real | 2514 | 0.011273 | 0.014705 | 0.39325 | 0.23392 | 0.004970 |
| A | synthetic | 8748 | 0.004685 | 0.005169 | 0.83863 | 0.73941 | 0.013572 |
| B | real | 792 | 0.007843 | 0.010071 | 0.37901 | 0.22423 | 0.003315 |
| B | synthetic | 10480 | 0.003822 | 0.004166 | 0.82267 | 0.71829 | 0.009917 |
| C | real | 792 | **0.007477** | **0.009746** | 0.38002 | 0.22458 | 0.003315 |
| C | synthetic | 10480 | **0.003467** | **0.003918** | 0.82984 | 0.72966 | 0.009917 |
| D | real | 792 | 0.007602 | 0.009888 | **0.38279** | **0.22665** | 0.003315 |
| D | synthetic | 10480 | 0.003538 | 0.004006 | **0.83066** | **0.72967** | 0.009917 |

C 在真实和合成子集上都取得最低 MSE/MAE；D 在两个子集上都取得略高的 Dice/IoU，说明
C/D 的权衡不是由某一个目标类型单独造成。同步回来的 20 张 C/D 配对图中，C 有 13 张的
像素 MSE 更低，D 有 7 张更低；两组预测图的平均绝对像素差约为 0.00197。抽查未发现 D
稳定产生更多断笔或粘连，因此现阶段不应基于少数图片宣称拓扑项有明确视觉收益。

## 3. 原因分析

1. **最大问题是合成到真实的域差异。** B 的真实子集 MSE 为 0.007843、Dice 为
   0.3790、IoU 为 0.2242；合成子集分别为 0.003822、0.8227、0.7183。真实图包含不同的
   墨色、边缘抗锯齿、笔画宽度、书写风格和配准误差，而训练样本主要来自规则化合成过程。
2. **真实目标上模型可能倾向错误出墨。** B 的真实子集 MSE 0.007843 高于全零预测基线
   0.003315，A 也有同样现象。这表示在部分真实图上，错误位置产生的墨迹比不生成更受罚。
   后续优先测试真实样本过采样、真实子集损失加权，以及只用真实图进行短程微调。
3. **A/B 不是纯粹的划分策略对照。** A 的真实验证样本占 2514/11262，约 22.3%；B 为
   792/11272，约 7.0%。目标类型构成不同会显著影响总指标，因此不能把 A 到 B 的全部提升解释为
   分组划分本身。分组划分仍是正确方案，因为它阻止同一 `sample_id` 的相关笔画跨训练和验证集泄漏。
4. **C/D 的微小差距低于单种子不确定性。** 去掉 clDice、Edge、Structure 后轮廓指标微升、
   像素指标微降，可能是正则项带来轻微平滑，也可能只是随机波动。应补 seed 7、42、2026，且共享同一
   split manifest，再决定是否删除结构项。
5. **姿态和动态笔刷目前不是这四组结果的解释变量。** 旧 `stroke10_v1` 不消费真实
   `alpha/beta/gamma`，又没有 `z -> width/drag/offset` 标定。因此不能把当前图像误差归因于真实机器人
   压力或姿态控制；这些结论必须等 pose-v2 数据和标定实验后再验证。

运行 `python tools/summarize_experiments.py` 会更新 `summary.csv`，并额外生成
`subgroup_summary.csv` 与中文 `analysis.md`，用于保存全部、真实和合成子集结果。

## 4. 生成轨迹差异图

先复制并填写 manifest，每一行显式绑定轨迹 `sample_id` 与真实目标图：

```bash
cp configs/trajectory_comparison_manifest.example.csv \
  configs/trajectory_comparison_manifest.csv
```

`target_image` 相对路径优先按仓库运行目录解析；若该路径不存在，再按 manifest 所在目录解析。
建议从仓库根目录运行，并使用 `data/...` 路径。`target_kind` 可填 `real` 或 `synthetic`，便于分组汇总。

```bash
python -u tools/compare_trajectories.py \
  --config configs/ablations/stroke10_v1_c_grouped_full.yaml \
  --trajectory_csv data/raw/trajectories.csv \
  --manifest configs/trajectory_comparison_manifest.csv \
  --bbsmg_ckpt outputs/ablations/stroke10_v1_c_grouped_full/bbsmg_best.pt \
  --normalization_npz data/processed/bbsmg_train_10d.npz \
  --output_dir outputs/trajectory_comparison_c
```

如果 checkpoint 已包含输入归一化信息，可省略 `--normalization_npz`。LM 优化需要多次渲染和数值
Jacobian，建议先放 3 到 5 个样本试运行，再扩展到完整集合。

每个样本输出以下文件：

| 文件 | 内容 |
|---|---|
| `comparison.png` | 真实目标、初始渲染、优化渲染、绝对差、彩色差和轨迹叠加的总览 |
| `signed_diff.png` | 红=多出的墨，蓝=缺失的墨，绿=重合墨迹 |
| `trajectory_overlay.png` | 蓝=输入参考轨迹，红=优化轨迹 |
| `generated_trajectory.csv` | 可继续用于仿真或机器人侧执行的优化轨迹 |
| `metrics.json` | 单样本图像、轨迹和 LM 指标 |

输出根目录还包含 `sample_metrics.csv`、`summary.json` 和自动生成的中文 `analysis.md`。
同步整个目录回来即可继续做跨样本和真实/合成目标分析。

## 5. 下一轮建议实验

| 优先级 | 实验 | 自变量 | 建议设置 | 目的 |
|---:|---|---|---|---|
| 1 | 真实样本权重 | real sampling/loss weight | 1x、2x、4x | 缩小真实域差距 |
| 2 | 真实图微调 | 预训练后真实图 epoch | 0、3、5 | 验证低成本域适配 |
| 3 | C/D 多种子 | seed | 7、42、2026 | 判断结构损失差异是否稳定 |
| 4 | z 标定 | `z -> width/drag/offset` | 真实测量拟合 | 让下压深度具有物理可解释性 |
| 5 | pose-v2 | 是否使用姿态特征 | legacy、pose-v2 | 验证 6D 姿态是否提供有效信息 |

每组应保存配置、Git commit、split manifest、训练日志、best checkpoint、总指标和真实/合成子集指标。
