# V16 原始目标图六字段反演：工具转动量对照

本目录是对已有 `/home/robot/ros2_ws/src/fusion_model_ros2_beta` 源码的窄补丁，不是完整 ROS 包。保留现有 UR10 工厂校准、IK、碰撞、限位、奇异性与提笔规划实现。它不重新训练 V16，不自动注册候选，也不向机械臂发送运动命令。

`robot` 分支的 `ros2_sim_ws/src/fusion_model_ros2_beta` 已直接包含同一功能。本目录供现有独立部署的 ROS 工作区更新使用；不要在已更新的仓库包上重复应用补丁。规划报告绑定实际使用的规划器、运动学、gamma 转换和代价源码 SHA256，不同碰撞包络版本的可行性结果不可混用。

模型端依赖：`optim/tool_orientation.py`、`utils/joint_rotation_metrics.py`，同时发布在 `main`。局部 gamma 按当前 ROS 规则转为绝对角，再计算 `Rz(gamma_abs) @ Ry(beta) @ Rx(alpha) @ Rx(pi)`。反演画布 y 向下、导出纸面 y 向上；新代价在该转换后计算。仅对同一笔内相邻接触点施加 SO(3) 弦长残差，不处理跨笔姿态转换；提笔阶段仍由原 ROS 规划器处理。

## 部署到已有 ROS 源码

```bash
cd /home/robot/coppeliasim/machine_learning/model
python3 integration/ros2_v16_rotation/install_overlay.py \
  --package-root /home/robot/ros2_ws/src/fusion_model_ros2_beta \
  --model-root "$PWD"

# 上一步只检查哈希；符合审核快照才允许应用。工具会备份原文件。
python3 integration/ros2_v16_rotation/install_overlay.py \
  --package-root /home/robot/ros2_ws/src/fusion_model_ros2_beta \
  --model-root "$PWD" --apply
```

遇到未知源码或独立修改时工具拒绝执行，不得强行覆盖。补丁是针对当前已审计 ROS 源版本，不应直接用于其他 UR10/UR5 实现。

## 配对实验（只反演和规划，不回放）

```bash
unset LD_LIBRARY_PATH
source /opt/ros/humble/setup.bash
source /home/robot/ros2_ws/install_actual_ur10/setup.bash
export PYTHONPATH=/home/robot/ros2_ws/src/fusion_model_ros2_beta:/home/robot/coppeliasim/machine_learning/model:${PYTHONPATH:-}

/usr/bin/python3 -u -m fusion_model_ros2_beta.evaluate_tool_rotation_tradeoff \
  --manifest /home/robot/ros2_ws/evaluation/original_target_20260923_codex_staging_copy/references_wu_verified.json \
  --output /home/robot/ros2_ws/evaluation/tool_absolute_rotation_v16 \
  --mode all --size 0.29 --steps 16 --order 11 \
  --rotation-weights 0 10 100 --hard-domain --device cuda \
  --max-iou-drop 0.01
```

`--mode all` 由模型 Python 生成，再由系统 ROS Python 规划。也可分开运行 `--mode generate` 和 `--mode plan`。每个权重从相同原始目标、初值和预算开始；不拿旧缓存充当新基线。GPU 模型环境默认 `/home/robot/miniconda3/envs/ddpm/bin/python`。

输出：

- `comparison.json/csv/png`：原始目标图 MSE/Dice/IoU、笔内绝对工具转动与完整规划腕部累计转动的对照，图像黑墨白底；
- 每个候选的 `inversion_report.json`：六字段变化、H 连续性、边界比例、可观测性及正向原图评价；
- `absolute_tool_rotation_audit.json`：代价所用姿态约定和笔内转动；
- `ur10_rotation_planning.json`：严格规划可行性、失败原因、CSV 哈希和统计；
- `ur10_planned_joints.csv`：含原规划器生成的提笔/落笔过渡、状态、时长和六关节目标。

腕部累计转动在最终展开的关节序列上统计，不对关节增量取模；分别统计笔内接触段和提笔/跨笔/初始接近。失败的半条规划不能按“转动小”推荐。选择规则只推荐完整规划可行、IoU 相对基线下降不超过指定值、腕部累计转动最小的版本用于进一步评价，不等于视觉验收或真实机器人安全认证。

现有 ROS 入口新增参数 `offline_joint_tool_absolute_rotation_weight`（默认 0，保持兼容）。其值进入原始目标反演配置、缓存标识、LM 残差和最优结果选择；不改变 B-BSMG 权重。经评价后如要在现有 fake-hardware launch 使用，可显式设置该参数，但本次脚本不会自动更改正式入口或发布轨迹。

## 验证

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/robot/miniconda3/envs/ddpm/bin/python -m pytest -q \
  tests/test_tool_orientation.py \
  /home/robot/ros2_ws/src/fusion_model_ros2_beta/test/test_tool_rotation_tradeoff.py

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /usr/bin/python3 -m pytest -q \
  /home/robot/ros2_ws/src/fusion_model_ros2_beta/test/test_gamma_semantics.py \
  /home/robot/ros2_ws/src/fusion_model_ros2_beta/test/test_joint_continuity.py \
  /home/robot/ros2_ws/src/fusion_model_ros2_beta/test/test_tool_rotation_tradeoff.py
```

ROS 侧无 torch 时模型矩阵对照测试会明确跳过；真实生成结果仍执行优化器统计与驱动实际姿态总转动的数值一致性检查。必须同时查看模型环境测试及规划结果，不能把跳过当成验证通过。
