# 模型轨迹筛选与 ROS 注册

每次完整模型迭代都必须保留原始输出，并把 ROS 可用轨迹写入独立的版本目录。ROS
只读取注册目录中的 `manifest.jsonl` 和 `char_uXXXX/pose_refined.csv`，不直接扫描
模型运行目录。这样可避免把仍在写入、低质量或由旧机器人报告验证过的文件混入
正式轨迹库。

流水线有两个门：

1. 模型门检查目标来源、Top-1、IoU、边界饱和、笔画内连续性、多初值稳定性、
   `z/alpha/beta/gamma` 置信度、CSV 字段、单位、坐标系和物理范围。
2. 机器人门检查同一 CSV 在指定机器人上的 IK、碰撞、奇异位姿、关节限位和轨迹
   连续性。报告必须包含该 CSV 的 SHA-256，旧迭代的检查结果不能复用。

输出文件：

```text
registry_summary.json             全局数量、策略和拒绝原因统计
screening_manifest.jsonl          所有字符，包括拒绝和等待项
robot_validation_queue.jsonl      模型合格、等待机器人检查的字符
manifest.jsonl                    两个门都通过的正式 ROS 字符
char_uXXXX/pose_refined.csv        ROS 兼容文件名
char_uXXXX/target.png
char_uXXXX/candidate_summary.json
char_uXXXX/registration.json
```

## v17 完成前审计

当前四个 shard 尚未全部结束时，只能生成审计结果：

```bash
cd /home/robot/coppeliasim/machine_learning/model
PY=/home/robot/miniconda3/envs/ddpm/bin/python

$PY -u tools/register_pose_library_for_ros.py \
  --input_root outputs/kaishu_pose_v17_full_snap \
  --output_root outputs/ros_trajectory_registry/kaishu_pose_v17_full_snap_ur10 \
  --mode audit --expected_shards 4 \
  --robot_model ur10 \
  --robot_validation_root outputs/ros_robot_validation/kaishu_pose_v17_full_snap_ur10
```

审计模式始终保持正式 `manifest.jsonl` 为空。`screening_manifest.jsonl` 可以提前
定位低 IoU、参数触边、跨初值不稳定和低置信度字符。

## v17 完成后的正式筛选

四份 `batch_summary_shard_*_of_4.json` 出现且每份
`processed == requested_characters` 后运行：

```bash
$PY -u tools/register_pose_library_for_ros.py \
  --input_root outputs/kaishu_pose_v17_full_snap \
  --output_root outputs/ros_trajectory_registry/kaishu_pose_v17_full_snap_ur10 \
  --expected_shards 4 \
  --robot_model ur10 \
  --robot_validation_root outputs/ros_robot_validation/kaishu_pose_v17_full_snap_ur10
```

默认模型门为：Top-1 自身 `export_eligible=true`、候选数至少 3、IoU ≥ 0.95、
最大边界比例 ≤ 0.05、二阶连续性跳变 ≤ 0.25、四字段跨初值归一化 RMS 标准差
≤ 0.02，且四字段置信度至少为 `medium_simulation`。CSV 必须使用 `z_unit=mm`、
`angle_unit=rad`、`pose_frame=paper_model`，H/z 位于 11–20 mm。

机器人检查工具应为队列中的每个字符写入：

```json
{
  "format": "ros_pose_validation_v1",
  "robot_model": "ur10",
  "trajectory_sha256": "与队列完全一致的 SHA-256",
  "ik_passed": true,
  "collision_free": true,
  "singularity_free": true,
  "joint_limits_passed": true,
  "trajectory_continuity_passed": true
}
```

保存位置为：

```text
outputs/ros_robot_validation/kaishu_pose_v17_full_snap_ur10/
  char_uXXXX/robot_validation_report.json
```

报告生成后重新执行正式筛选命令。注册器会核对机器人型号和轨迹哈希，只有全部
布尔检查为 `true` 的字符才复制进正式库。

## 后续模型版本

新版本继续使用同一候选摘要字段时，只需给它独立的输入和注册目录。如果摘要版本
或相对路径变化，显式传入：

```bash
--summary_format paper_pose_multisolution_v18 \
--candidate_summary_relpath v18/candidate_summary.json
```

训练或反演脚本不得直接改写现有注册目录。每个模型版本都生成自己的筛选统计、
机器人验证队列和正式清单，便于回退和比较。
