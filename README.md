# Fusion Model：动态笔刷与 B-BSMG 笔触模型

本项目研究从目标楷书图像和给定二维轨迹，生成多组修正后的 `x/y/z/alpha/beta/gamma`，再通过动态笔刷与 B-BSMG 正向渲染检验结果。当前主流程是 **冻结通用 B-BSMG，执行 v17 多初值反演与可信度评估**。

`main` 统一维护笔触模型、数据构建、训练、渲染、反演、ROS 2 和 UR10 毛笔实机工具。机器人代码与模型代码在同一主分支集成，不再要求切换 `robot` 分支。模型权重、原始数据库、标定记录和运行输出不随代码上传，克隆仓库后需要另行准备。

入口导航：[模型流程](#1-流程与版本) · [机械臂写字命令](docs/机械臂写字.md) · [写后相机采集](docs/brush_write_capture.md) · [悬空力参考采样](docs/brush_unloaded_calibration.md) · [ROS 2 轨迹注册](docs/ros_pose_registry.md)

## 1. 流程与版本

```text
楷书数据库字符索引 + 姿态采样
  → 解析 B-BSM 局部足迹数据 → B-BSMG 训练与验证

目标楷书图 + 输入 x/y 轨迹 + 默认姿态
  → 目标兼容性筛选 / 有界骨架配准
  → 每笔 CGL 节点参数化 → PSOC/LM 多初值优化
  → 动态宽度、拖曳、偏移 → B-BSMG 足迹 → 整字渲染
  → 图像、局部足迹、连续性、触边和稳定性评价
  → Top-K 轨迹候选 + 字段可信度 + JSON 报告
```

| 名称 | 含义 | 当前用途 |
| --- | --- | --- |
| 通用 B-BSMG v15/v16 | 从姿态参数生成局部足迹的正向网络 | v17 使用 `paper_bbsmg_general_v16_pose_dense/bbsmg_best.pt` |
| 风格 v16 | 独立的楷书灰度/墨色细化器 | 改善外观；不产生物理姿态标签 |
| 反演 v17 | 多初值、Top-K、可观测性和可信度评估 | 当前单字与全库反演入口 |
| 历史 v42–v45 | H 融合推导、局部足迹与 x/y 修正实验 | 历史方案，不代表当前 checkpoint 版本 |
| 整字 U-Net v8 | 同源轨迹结构渲染基线 | 独立历史流程，不用于当前姿态反演 |

版本不能只看目录名。实际依据是运行命令的 `--bbsmg_ckpt`、checkpoint 元数据、JSON 中的格式/配置和 CSV 的 `prototype`。历史命令见 [旧 README 归档](docs/archive/readme_before_brush_rewrite.md)，v17 算法细节见 [多解反演说明](docs/paper_v17_multisolution.md)。

## 2. 数据与坐标语义

```text
data/raw/trajectories.csv             全字符输入轨迹
data/raw/data.csv                     图片元数据，chirography=楷
data/raw/images/                      数据库图片
data/raw/json_files/                  LabelMe 字符框标注
data/raw/targets/wu_kaishu_target.png  规范楷书“武”目标
data/processed/                      NPZ 与数据审计
outputs/                             权重、轨迹、图片、指标
```

只允许数据库中 `chirography=楷` 的目标参与当前训练与验证。规范武字目标固定为 `wu_kaishu_target.png`；旧 `武.png`、`wu_target_xingkai.png` 已废弃。

输入 CSV 通过 `sample_id` 分组，以 `stroke_id/point_id` 组织各笔点列，读取 `character,x,y,z,alpha,beta,gamma,state`。轨迹不含可靠物理时间戳或测力信息时，位移和高度只能作为速度/压力代理量。

| 导出字段 | 语义 |
| --- | --- |
| `character`, `sample_id` | 字符及轨迹样本标识 |
| `stroke_id`, `point_id`, `state` | 笔画、点编号及落笔/行笔/抬笔状态 |
| `x`, `y` | 输入轨迹坐标系中的修正坐标；不是机器人基坐标 |
| `z` | 本论文原型中的 H，单位 mm；不是 TCP 世界高度或测得的下压力 |
| `alpha`, `beta` | 论文回归使用的倾角/纸面旋转参数，CSV 单位 rad |
| `gamma` | 按所用模式解释的角度，CSV 单位 rad；不能直接当作机器人欧拉角 |
| `z_unit`, `angle_unit`, `pose_frame` | 单位和纸面模型坐标声明 |
| `prototype`, `regression_angle_basis` | 导出算法标识、回归角度基底 |
| `*_source`, `*_confidence` | 各字段来源及可信度 |

H 范围为 11–20 mm，α 为 0–10°，β 为 0–5°。网络的局部 γ 采样范围与反演 CSV 的绝对方向范围不同：默认 `relative_to_heading` 先计算 `wrap(gamma_csv - forward_xy_heading)` 再送入局部网络。`heading` 合成测试的 γ 由同一笔内的相邻 x/y 求得；当前 v17 实际反演仍允许优化 γ，不能把所有候选都解释成纯方向推导结果。正向复核必须复用反演的 gamma 模式与尺度。

论文回归式提供仿真几何，动态模型提供时序宽度/拖曳/偏移。解析足迹监督不是实测毛笔足迹，目标图也没有真实 H/α/β 标签。单张整字图可能存在参数互相补偿，因此保留多解和低置信字段。

## 3. Ubuntu 环境与工作区

在同步后的 `main` 项目根目录运行。需要保留现有本地修改并另建独立副本时，可使用只读起点的 worktree：

```bash
cd /home/robot/coppeliasim/machine_learning/model
git fetch origin main
git worktree add --detach ../model-brush-main origin/main
cd ../model-brush-main
conda activate ddpm
export PYTHONPATH="$PWD"
PY=/home/robot/miniconda3/envs/ddpm/bin/python
```

上述 worktree 仅带仓库中的规范目标图，不包含完整数据库和权重。已准备好的数据可链接进来：

```bash
ln -s /home/robot/coppeliasim/machine_learning/model/data/raw/trajectories.csv data/raw/trajectories.csv
ln -s /home/robot/coppeliasim/machine_learning/model/data/raw/data.csv data/raw/data.csv
ln -s /home/robot/coppeliasim/machine_learning/model/data/raw/images data/raw/images
ln -s /home/robot/coppeliasim/machine_learning/model/data/raw/json_files data/raw/json_files
ln -s /home/robot/coppeliasim/machine_learning/model/data/processed data/processed
ln -s /home/robot/coppeliasim/machine_learning/model/outputs outputs
```

仅在目标路径不存在时执行链接。若直接在已同步的 `main` 工作区运行，省略 worktree 和链接步骤。后面的命令均在项目根目录执行。

环境需要 PyTorch（与驱动兼容的 CUDA 构建）、NumPy、Pillow、PyYAML、pandas、SciPy、scikit-image，以及 `requirements-character.txt` 中的附加依赖。已有 `ddpm` 环境可先检查：

```bash
python -m pip install -r requirements-character.txt
nvidia-smi
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
python tools/run_paper_v17_multisolution.py --help
python tools/invert_kaishu_v17_batch.py --help
```

显卡不可用时先处理驱动/环境，确认训练配置的 `train.device` 和命令中的 `--device cuda`。不要把 CPU 运行时间与 GPU 实验混为一谈。

## 4. 构建通用 B-BSMG 数据并训练

已有经过验证的 v16 checkpoint 时，可直接进入第 5 节。需要重新训练时，使用独立目录，不覆盖已有权重：

```bash
$PY -u tools/build_general_paper_bbsmg_dataset.py \
  --trajectory_csv data/raw/trajectories.csv \
  --style_data_csv data/raw/data.csv \
  --style_image_dir data/raw/images \
  --style_json_dir data/raw/json_files \
  --chirography 楷 --holdout_character 武 \
  --samples_per_character 100 --seed 1501 \
  --output_npz data/processed/paper_bbsmg_general_kaishu_rebuild.npz

$PY tools/audit_paper_bbsmg_dataset.py \
  --npz_path data/processed/paper_bbsmg_general_kaishu_rebuild.npz \
  --output_json outputs/paper_bbsmg_general_kaishu_rebuild/dataset_audit.json

$PY -u tools/train_bbsmg.py \
  --config configs/paper_bbsmg_gamma_v13.yaml \
  --npz_path data/processed/paper_bbsmg_general_kaishu_rebuild.npz \
  --output_dir outputs/paper_bbsmg_general_kaishu_rebuild \
  --epochs 50 --val_ratio 0.1 --group_split \
  --lr_factor 0.5 --lr_patience 4 --min_lr 0.000001
```

网络输入为 `[H_mm, alpha_rad, beta_rad, gamma_relative_rad, x0_px, y0_px]`；目标是解析 B-BSM 抗锯齿足迹。数据库负责筛选楷书字符，真实数据库灰度图不是这一步的姿态监督。`武` 留出；训练使用 `group_ids` 按字符分组验证。NPZ 和 checkpoint 保存的归一化、角度基底必须一致。

配置名中的 `v13` 是兼容的 6D 网络配置名，不代表新训练模型必须叫 v13。批量大小在 YAML 的 `train.batch_size` 设置；`train_bbsmg.py` 没有 `--batch_size` 参数。

评估局部渲染并导出对比图：

```bash
$PY -u tools/evaluate_bbsmg.py \
  --config configs/paper_bbsmg_gamma_v13.yaml \
  --npz_path data/processed/paper_bbsmg_general_kaishu_rebuild.npz \
  --checkpoint outputs/paper_bbsmg_general_kaishu_rebuild/bbsmg_best.pt \
  --output_dir outputs/eval_paper_bbsmg_general_kaishu_rebuild \
  --val_ratio 0.1 --seed 42 --num_images 40
```

**评估限制：** 当前 `evaluate_bbsmg.py` 按样本随机选择子集，没有 `--group_split`；这份报告不能冒充字符留出验证。字符分组验证以训练阶段保存的验证指标为依据。局部足迹 IoU 高也不保证真实目标整字反演 IoU 高。

## 5. 仿真真值恢复与多初值检验

当前已使用的 checkpoint 路径：

```bash
CKPT=outputs/paper_bbsmg_general_v16_pose_dense/bbsmg_best.pt
test -f "$CKPT"

$PY -u tools/run_paper_v17_multisolution.py \
  --trajectory_csv data/raw/trajectories.csv --bbsmg_ckpt "$CKPT" \
  --character 武 --sample_id 武_fake_sim \
  --output_dir outputs/wu_v17_truth_heading \
  --device cuda --seed 17017 --gamma_truth_mode heading \
  --perturbation_scales -1 0 1 --top_k 3 --copy_top_k \
  --optimization_size 64 --max_steps 5 --point_batch_size 64
```

`sample_id` 必须对应实际数据；没有 `武_fake_sim` 时去掉它，由字符筛选样本。改为 `--gamma_truth_mode random` 可检验随机 γ 真值。生成的 `synthetic_truth.csv/.json` 和 `synthetic_target.png` 属于已知仿真真值，适合检查反演器能否恢复参数。

至少检查 IoU ≥ 0.95、H 归一化 RMSE ≤ 0.05、边界比例 ≤ 0.05、笔画内二阶跳变 ≤ 0.25；字段跨初值 normalized RMS 标准差以 ≤ 0.02 为参考。分别阅读 `pose_recovery`、`field_stability` 和字段置信度，不能只看排序第一名。

## 6. 对真实楷书目标做小规模反演

先执行单字，再检查跨字回归；不要直接重启全库。

```bash
$PY -u tools/invert_kaishu_v17_batch.py \
  --trajectory_csv data/raw/trajectories.csv --bbsmg_ckpt "$CKPT" \
  --output_dir outputs/kaishu_pose_v17_regression \
  --include_characters 武 一 乙 巳 它 污 戌 \
  --chirography 楷 --target_selection best_model_support \
  --model_support_width_px 7 \
  --target_override 武=data/raw/targets/wu_kaishu_target.png \
  --perturbation_scales -1 0 1 --top_k 3 \
  --order 1 --max_steps 3 --optimization_size 64 \
  --pixel_weight 12 --h_smoothness_weight 0.2 \
  --h_point_velocity_weight 5 --h_point_acceleration_weight 10 \
  --optimize_xy --xy_max_offset_px 3 \
  --xy_smoothness_weight 1 --xy_prior_weight 0.5 \
  --xy_segment_length_weight 0.10 --xy_segment_direction_weight 0.10 \
  --xy_target_skeleton_weight 0.20 --xy_target_skeleton_max_distance_px 8 \
  --device cuda --timeout_seconds 1800
```

这是一组快速回归参数，不保证通过验收。`max_steps=3` 的有限预算不足以代表最优解；改预算、checkpoint 或目标后换新的输出目录，以免复用旧候选。

目标先按模型支持和轨迹覆盖率筛选。缺少输入轨迹记录为 `missing_trajectory`；错标、污染或结构不兼容的目标记录为 `target_incompatible`。SVG/输入 x/y 是可追溯先验，可以有界、平滑修正，但不能靠扭曲笔画来掩盖错误目标。

典型输出：

```text
outputs/kaishu_pose_v17_regression/
  batch_summary.json
  manifest.jsonl
  char_u6b66/
    target.png
    pose_top1.csv
    v17_batch.log
    v17/
      candidate_summary.json
      top_k_candidates.csv
      top_k/rank_01/...
```

**`pose_top1.csv` 的存在仅代表候选已输出，不代表通过筛选。** 检查 `candidate_summary.json` 中的 `export_eligible`、`withheld_reasons`、`field_stability`、`field_confidence_intervals`，并查看候选 render、diff、comparison。多初值 p10–p90 区间是经验范围，不是物理标定置信区间。

## 7. 全楷书库反演与恢复

只有真值恢复和小规模跨字回归通过后，才启动全库。当前 Ubuntu 脚本固定进入 `/home/robot/coppeliasim/machine_learning/model`，并固定使用 v16 checkpoint；运行前确认**该目录**已含最新 `main` 模型代码。独立 worktree 下运行可使用上一节 Python 入口并去掉 `--include_characters`，按需要设置分片。

```bash
# 单进程依次执行 4 个分片（6 GB 显卡优先）
OUTPUT_DIR=outputs/kaishu_pose_v17_full_new \
  nohup bash tools/run_kaishu_v17_full.sh \
  > /tmp/kaishu_v17_full_new.log 2>&1 < /dev/null &

# 4 分片并行：确认显存和资源足够后，与上面的方案二选一
OUTPUT_DIR=outputs/kaishu_pose_v17_full_new \
  nohup bash tools/run_kaishu_v17_parallel.sh \
  > /tmp/kaishu_v17_parallel_new.log 2>&1 < /dev/null &
```

并行日志是 `/tmp/kaishu_v17_full_snap_shard_0.log` 至 `_3.log`（文件名固定）。4 个 GPU 进程共享显存，不保证比顺序执行更快。不要在同一输出目录重复启动两套任务。

```bash
pgrep -af 'invert_kaishu_v17_batch|run_paper_v17_multisolution'
nvidia-smi
tail -n 30 /tmp/kaishu_v17_full_new.log
```

`--resume_completed` 依据已有候选汇总跳过字符，并不表示恢复 LM 中间状态或确认质量合格。最后读取四份 `batch_summary_shard_*_of_4.json` 和 `manifest_shard_*_of_4.jsonl`，分别统计 completed、accepted、low_quality、failed、target_incompatible 和 missing_trajectory。不要把 completed 当作 accepted，也不要累加多份汇总中重复出现的全库 missing 列表。

## 8. 正向重放与筛选

针对选定候选重放，复用反演的动态参数：

```bash
$PY -u tools/render_paper_trajectory.py \
  --trajectory_csv outputs/kaishu_pose_v17_regression/char_u6b66/input_trajectory.csv \
  --pose_csv outputs/kaishu_pose_v17_regression/char_u6b66/pose_top1.csv \
  --bbsmg_ckpt "$CKPT" --character 武 \
  --target_image outputs/kaishu_pose_v17_regression/char_u6b66/target.png \
  --dynamic_profile wang2020_figure4_digitized_v1 \
  --gamma_mode relative_to_heading \
  --footprint_longitudinal_scale 0.22 --footprint_transverse_scale 0.258 \
  --padding 16 --render_max_step_px 2 --point_batch_size 64 \
  --output_image outputs/wu_v17_replay/render.png --device cuda
```

输入轨迹文件以单字 manifest 中的 `trajectory_input` 实际路径为准。若修改过 offset、惯性或尺度，按候选报告复用全部参数。输出图像及对应 JSON、states CSV 共同记录图像误差和动态状态。

筛选需要同时检查：

- 图像 IoU/Dice/MSE、骨架位置、局部笔宽和墨量；diff 只表示差异，不是生成笔画。
- x/y 位移、笔画长度/走向、端点及抬笔分隔；交叉点不可变成额外连线。
- H/角度连续性、边界饱和、联合可观测性及多初值稳定性。
- 正向重放能复现候选结果，参数单位、尺度和 γ 语义一致。

当前 v17 的 `export_eligible` 只覆盖图像/触边/连续性门限，仍需单独审查稳定性和字段可信度。低 IoU、严重触边或不稳定候选不能因名次靠前而采用。α/β 无法辨识时保留低置信多解；增加像素相似度不能证明姿态唯一。

筛选后的纸面轨迹可交给同一 `main` 分支内的 ROS 注册工具，转换为 `pose_refined.csv + manifest.jsonl`。机器人侧还需 IK、碰撞和奇异位姿检查。模型候选 manifest 不是 ROS 正式注册清单，模型导出的 α/β/γ 也不自动等于机械臂姿态约定。

V16 原始目标图六字段反演的新[绝对工具姿态代价与转动量对照](docs/tool_absolute_rotation.md)沿用当前 ROS 的局部 γ 转换，并分别评价图像质量、笔内工具转动和完整 UR10 规划腕部转动；提笔姿态过渡仍由 ROS 规划器处理。

## 9. 楷书风格细化（可选）

风格细化保持 B-BSMG 冻结，只改善灰度、笔锋、墨量与端点。已有 `kaishu_style_v27.npz` 时：

```bash
$PY -u tools/augment_kaishu_style_dataset_v16.py \
  --base_npz data/processed/kaishu_style_v27.npz \
  --trajectory_csv data/raw/trajectories.csv \
  --output_npz data/processed/kaishu_style_v16.npz \
  --heldout_character 武 --min_trajectory_coverage 0.30

$PY -u tools/train_kaishu_style_refiner.py \
  --npz data/processed/kaishu_style_v16.npz \
  --output_dir outputs/kaishu_style_v16_new \
  --heldout_character 武 --epochs 50 --batch_size 16 \
  --base_channels 24 --workers 2 --lr 0.0003 --val_ratio 0.15 \
  --device cuda --support_mode mask_or_soft
```

内部损失使用墨迹为 1、背景为 0；交付图片应采用黑墨白底。细化前几何指标和细化后外观指标分别报告。风格网络的目标结构条件与足迹真值恢复不同，不能把外观改善解释成 H/姿态恢复成功。

## 10. 排错与代码验证

| 现象 | 先检查 |
| --- | --- |
| 局部足迹很好，整字差 | 轨迹与目标兼容性、整体尺度、γ 语义、抬笔及足迹叠加 |
| 一等简单字差 | 目标框/标签、污染、笔宽支持范围；不要用无界 x/y 补偿 |
| α/β 全零或触边 | observability gate、先验、回归基底和边界报告；零值可能是保留默认 |
| 多余斜线 | 每笔状态分隔、点次序、x/y 回折及旧目标继承 |
| CUDA OOM | 减少并行进程与 point_batch_size；有限差分 Jacobian 不需要整图反向图 |
| 换参数后结果不变 | resume 是否跳过旧 summary；改输出目录重跑 |
| GPU 驱动错误 | nvidia-smi 和 torch.cuda.is_available，先修复环境 |

```bash
$PY -m pytest tests/test_paper_v17_multisolution.py \
  tests/test_kaishu_pose_trajectory_batch.py -q
bash -n tools/run_kaishu_v17_full.sh tools/run_kaishu_v17_parallel.sh
```

算法测试通过代表代码行为受到检查，不代表全库轨迹或真实机器人执行已验收。实机工作已推进到 UR10 初始换算试写；其标定来源、局部验证范围和未完成项见下面的实机入口说明。


## 11. UR10 毛笔实机工具（2026-10-23）

当前交付包括完整三笔“工”字初始换算试写、实测悬空起点接入、已通过路径证据复用、原始力与位姿记录，以及独立拍照位的离线设计和墨迹测量工具。默认入口为离线预览，实际运动需要显式传入 `--execute --accept-initial-trial`。

```bash
# 离线预览；不会发送运动，也不会连接机械臂。
/usr/bin/python3 -B tools/run_gong_model_trial.py
```

上述命令需要现场已有的 `outputs/brush_mapping_20261023/` 轨迹、零点、TCP、纸面、碰撞和力参考资料。该目录被 Git 忽略，未随本次代码发布；全新克隆不会自动具备实机运行资格。运行主机 MU、jx 的 SSH 配置及 jx 中现有 UR10/ROS 工具也属于现场环境。

| 组件 | 入口／说明 |
| --- | --- |
| 模型三笔试写 | `tools/run_gong_model_trial.py`、`tools/gong_model_trial_remote.py` |
| 当前位置接入 | `tools/brush_model_start.py`；保持原书写路径和零点，单独检查新增接入段 |
| 原生缓冲运动与保护 | `tools/brush_buffered_motion.py`、`tools/brush_buffered_guard.py`；保留笔画内圆滑过渡、抬落笔分隔与控制器看门狗 |
| 原始力数据读取 | `tools/brush_force_snapshot.py`；读取最新十帧，保存失效时的样本年龄诊断 |
| 悬空姿态参考 | `tools/brush_unloaded_calibration.py`、`tools/validate_brush_unloaded_sweep.py` |
| 写后固定相机采集与测量 | `tools/brush_write_capture.py`、`tools/brush_capture_measure.py`，见[采集说明](docs/brush_write_capture.md) |
| 既有标定工具依赖 | `integration/ur10_paper_calibration/`；保留源记录与各自验证状态 |

**当前实机结果：**完整路径及新增接入段曾通过检查；发布后只完成第一笔，在第二笔落笔前因力数据更新保护停止，第三笔未执行。不能将此写成完整三笔验收成功。之后按用户要求将本次试写的力数据接收与源时间戳允许延迟改为 **1.5 秒**，软件测试通过，尚未改后实机复验。位姿、主机心跳和控制器看门狗仍为 **150 ms**，力增量停机阈值仍为 **10 N**。

模型 H 使用显式初始换算 `H × 0.045`，得到约 0.495～0.9 mm 的指令下压量；α/β/γ 增益为 0.2/0.2/0.02。XY 保持原尺寸并在纸面平移，变姿位置按自然笔尖 TCP 完整补偿。H 不等同于实测压入量，模型 H 不裁剪，原始 6＋4＋6 个控制点及全部字段保留。局部悬空参考的跨位置和接触精度仍未独立验证。

全部现场命令、用途、是否驱动机械臂和验证范围集中维护在[机械臂写字](docs/机械臂写字.md)。现场主副本为 `/home/robot/Documents/机械臂写字.md`；后续验证可用的写字命令同步更新到两处。当前实机命令都从第一笔开始，不是断点续写命令。

相关代码验证可在不连接机械臂的情况下运行：

```bash
/usr/bin/python3 -B -m unittest discover -s tests -p 'test_brush_*.py'
/usr/bin/python3 -B -m unittest discover -s tests -p 'test_run_gong_model_trial.py'
/usr/bin/python3 -B -m unittest discover -s tests -p 'test_prepare_brush_capture_runtime.py'
/usr/bin/python3 -B -m unittest discover -s tests -p 'test_validate_brush_unloaded_sweep.py'
/usr/bin/python3 -B -m unittest discover -s tests -p 'test_write_gong_1mm.py'
```

## 12. ROS 2 与机器人仿真

`main` 同时保留笔触模型、ROS 2、UR10 标定及轨迹注册功能。
参见 [轨迹注册](docs/ros_pose_registry.md) 和 [腕部验证说明](integration/ros2_v16_rotation/wrist_joint_20261002/README_wrist_validation.md)。
完整旧 robot 分支说明保存在 [归档](docs/archive/robot_readme_before_merge_20261022.md)。
