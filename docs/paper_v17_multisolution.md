# v17 多解反演与可信度评估（仿真原型）

v17 不重复训练 B‑BSMG。它冻结 `paper_bbsmg_general_v16_pose_dense`，先生成一个
带已知 `H/alpha/beta/gamma` 的合成足迹，再从多个初始姿态运行 PSOC/LM。每个初始
值都保留自己的 CSV、target/render/diff/comparison 图；`candidate_summary.json` 只把
它们排序为 Top‑K 候选，不把单张图像下的姿态伪装成唯一真值。

## Ubuntu 命令

```bash
cd /home/robot/coppeliasim/machine_learning/model
PY=/home/robot/miniconda3/envs/ddpm/bin/python

$PY -u tools/run_paper_v17_multisolution.py \
  --trajectory_csv data/raw/trajectories.csv \
  --bbsmg_ckpt outputs/paper_bbsmg_general_v16_pose_dense/bbsmg_best.pt \
  --character 武 --sample_id 武_fake_sim \
  --output_dir outputs/wu_paper_multisolution_v17_v16 \
  --device cuda --seed 17017 --gamma_truth_mode random \
  --perturbation_scales -2 -1 -0.5 0.5 1 2 \
  --top_k 5 --copy_top_k --optimization_size 64 --max_steps 5 \
  --point_batch_size 64 --resume_completed
```

若要测试“`gamma` 由相邻 x/y 方向推导”的约定，把
`--gamma_truth_mode random` 换成 `--gamma_truth_mode heading`。汇总中会额外报告
每个候选的 `gamma_heading_metrics`（由相邻 x/y 得到的前进方向与 CSV gamma 的
环差）；这仍是纸面几何候选，不是机器人角度标定。

## 输出与判定

* `synthetic_truth.csv/.json`：本次随机种子产生的仿真真值（H 用 mm，角度用 rad）。
* `synthetic_target.png`：用冻结 v16 渲染器生成的合成目标。
* `m2/…`, `m1/…`, `p0p5/…` 等：每个初始值的候选轨迹和四张对比图。
* `top_k/rank_01/…`：按综合分复制的前 K 组候选，便于后续机械臂测试。
* `candidate_summary.json`：图像 IoU、局部拖拽/半宽/长宽比 RMSE、笔画内连续性、
  边界饱和、跨初值稳定性、合成真值恢复误差和每个字段置信度；同时给出经验性的
  多初值 p10–p90 区间（不是物理标定置信区间）。

综合分的权重默认是 IoU 0.45、局部足迹 0.25、连续性 0.15、边界安全 0.10、
跨初值稳定性 0.05。导出条件默认 IoU≥0.95、边界近限比例≤0.05、二阶连续性跳变
≤0.25；不满足的候选仍保留，但在 JSON 中标记 `withheld_reasons`。

α/β 的 `low` 置信度是预期结果：单张整字图的联合 Jacobian/SNR 不足或存在参数
补偿时，v17 会明确标记 `single_image_non_unique`，而不是输出假装唯一的角度。真实
且唯一的 α/β 仍需要局部足迹、压力/速度、多视角或真实毛笔—机器人标定数据。

## 通过标准

合成回归至少检查：每个候选 IoU≥0.95、H 归一化 RMSE≤0.05、姿态不贴边；多初值
一致性用各字段归一化 RMS 标准差≤0.02 判断。若某个字段未通过，只能降低该字段
置信度或保留多解，不能通过任意扭曲 x/y 或更换废弃的行楷目标来“修复”。
