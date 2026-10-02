# V16 原图反演：UR10 真实腕部代价与提笔优化

本目录是离线验收，不是正式轨迹库，不启动控制器，不发布机械臂运动。
冻结 V16 权重，只以规范楷书 `wu_kaishu_target.png` 为字迹目标；原 CSV 仅作相同的初值。
不使用历史效果 A，也不通过修改 ROS 渲染字号改善图像分数。

## 实现

`source/fusion_model_ros2_beta/ur10_wrist_cost.py` 使用当前 ROS 的工厂标定 UR10 CB3
（serial 2022300602，calib_15120592593058779304）、328 mm TCP 和工具/相机几何。
每次 LM 残差/差分 Jacobian 计算都会从当前 x/y/H/alpha/beta/gamma 解连续 IK。
q4/q5/q6 的平滑 L1 累计运动残差进入 LM 正规方程和正则化结果选择，不是规划后挑选权重。
全链参数、工厂几何源文件和纸面布局固定且带 SHA256。
关节差使用实际连续角度相减，绝不对差值取模来隐藏整圈转动。

局部 gamma 到绝对工具姿态：

```text
gamma_abs = gamma_local + atan2(dy_physical, dx_physical)
R_tool = Rz(gamma_abs) Ry(beta) Rx(alpha) Rx(pi)
p_tool0 = p_tip - R_tool[:,2] * 0.328
z_tip = paper_z - H_mm / 1000
```

方向使用同笔前向差分，末点沿用该笔上一段方向；该约定与现有 ROS 相同。
canvas y 向下、纸面 y 向上；固定中心 64、比例 0.29/96 m/pixel，绝不按优化结果重拟合 bbox。
H 是模型高度/压入量字段，不是已测得的机器人力。

`wrist_aware_transitions.py` 只优化两笔之间的抬笔—转姿态—落笔。
接触点及其六个关节值逐项保持不变；在现有安全路径上做关节空间约束 shortcut，
并比较固定端点姿态的抬高 IK 桥接路径。新路径检查纸面上方转姿态、笔尖朝下、
限位、奇异位姿和工具/相机/纸桌碰撞，再重采样、限速。
每笔不回初始位置；若安全路径无法避免回零，结果明确拒绝。
初始接近运动单独计分，不混入接触或两笔之间提笔运动。

腕部反演残差包含接触采样和跨笔端点差；提笔轨迹绕行的额外运动由独立 ROS 提笔优化器处理。
它没有把整条有碰撞修复的提笔路径塞入每次像素差分求导。

## 固定对照

当前 520×320 mm 画布，根据未修正源轨迹长宽比和 `plan_canvas` 推导：

- 单格字号 290 mm。
- 三格字号 139.702440 mm，测试中央格。

各字号独立生成相同初值、原目标、冻结权重、16 步/order 11 的完整当前 ROS 预算。
16 步是当前 `offline_joint_max_steps` 默认值，不是此前 4 步短试验。

| 对照 | 工具绝对姿态权重 | 真实 q4/q5/q6 权重 | 主结果提笔路径 |
|---|---:|---:|---|
| A | 0 | 0 | 原规划 |
| B | 10 | 0 | 原规划 |
| C | 10 | 10 | 新提笔优化 |

不扫描权重。每个结果还保存相同接触轨迹下旧/新提笔规划的消融比较，
用来区分接触姿态优化的收益和单独提笔优化的收益。
现有几何、H 和角度连续性先验保持相同。

## 完整可复现命令

```bash
ssh mu
cd /home/robot/ros2_ws/evaluation/wrist_joint_20261002
bash run_tests.sh
bash run_full_abc.sh
bash build_optin.sh
bash run_final_acceptance.sh
sha256sum -c verification.sha256
/usr/bin/python3 source/verify_wrist_artifacts.py --output .
```

脚本保存在本目录，内含固定 Python、ROS source、模型、manifest、字号和预算。
`source` 是冻结的反演代码快照；`deployed_source` 是本轮显式 opt-in 补丁部署后的当前 ROS 源码快照。
最终脚本仅使用后者重新规划六组已完成的候选并跑测试，不启动新训练，不发布运动。
`build_optin.sh` 只构建 evaluation 内的隔离安装并执行 `--show-args`，不替换正式安装、不启动 launch。
运行最终脚本时不要把日志重定向到本 evaluation 目录内，否则写日志会改变最终 SHA 清单；可使用 `/tmp/ur10_wrist_final.log`。
已有完整缓存会校验身份与文件 SHA 后复用；有不完整缓存时 fail closed，不掩盖失败。
如需全新计算，应复制代码/脚本到新 evaluation 目录并改输出路径，不能覆盖原证据。

单组的等价命令（其余只改变 A/B/C 或 slots）：

```bash
source /opt/ros/humble/setup.bash
export PYTHONPATH=/home/robot/ros2_ws/evaluation/wrist_joint_20261002/source:/home/robot/coppeliasim/machine_learning/model:$PYTHONPATH
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
MANIFEST=/home/robot/ros2_ws/evaluation/original_target_20260923_codex_staging_copy/references_wu_verified.json
OUT=/home/robot/ros2_ws/evaluation/wrist_joint_20261002
/home/robot/miniconda3/envs/ddpm/bin/python -u -m fusion_model_ros2_beta.evaluate_actual_wrist_tradeoff \
  --mode generate --manifest "$MANIFEST" --output "$OUT" \
  --case C --slots 3 --steps 16 --order 11 --width .52 --height .32
/usr/bin/python3 -u -m fusion_model_ros2_beta.evaluate_actual_wrist_tradeoff \
  --mode plan --manifest "$MANIFEST" --output "$OUT" \
  --case C --slots 3 --steps 16 --order 11 --width .52 --height .32
```

## 输出与验收

`canvas_*_slots/{A,B,C}_result.json`：原图 IoU/SSIM/MSE、预算、模型 SHA、真实关节累计运动、
接触/纯提笔/初始接近分项、最大内部关节步长、时长、约束检查、接触 FK 残差和反馈 IK 与实际规划差。
`{A,B,C}/ur10_planned_joints.csv`：不取模的实际 q1–q6、点姿态、状态和时间。
其中 x/y/z（m）是根据标定 FK 计算的 `base` 坐标系笔尖位置，alpha/beta/gamma 为同一 ROS 姿态约定的弧度值；q1–q6 为未取模弧度。
`duration_s` 是相邻目标运动时间，`time_from_start_s` 是累计时间；`desired_*` 保留原规划目标，首行仅为指定初始关节种子，`cartesian_target_applicable=False`。
候选 `physical_trajectory.csv` 的 x/y 是纸面局部 m，z 是模型 H（mm），gamma 已转为绝对姿态；它不是机器人法兰的世界坐标，不能直接下发为笛卡尔命令。
`{A,B,C}/stroke_residuals.json`：按源笔画 Voronoi 分区的原图局部残差；这不是人工标注的真实笔画分割。
`comparison.png` 为黑墨白底 target/render/diff，不是 ROS 视觉渲染。
`C/feedback_ik.csv` 可独立核对进入反演的腕部关节。
`tests/*.xml` 保存测试结果；`verification.json` 保存各证据 SHA；`verification.sha256` 校验报告本身。
`source/verify_wrist_artifacts.py` 独立核验 SHA 清单中的全部文件，不会将不合格字迹提升为通过。
`frozen_input_verification.json` 还核验六份反演请求固定的原图、初值、V16权重、反演实现、画布上下文和完整预算；变更外部源文件会拒绝验收。

独立绝对字迹门槛 IoU≥0.95、SSIM≥0.90；C 相对 A 的 IoU 下降≤0.01 仅是附加门槛。
空白背景能使 SSIM 偏高，不能代替前景 IoU。
C 还需两个字号的腕部总转动均小于 A/B、全部机器人检查通过且测试证据齐全。
未达标一律 `accepted:false`，不能注册正式轨迹或宣称可执行。

碰撞验收范围是当前机器人连杆包络、纸/桌和已注册工具/相机 OBB；
不等于完整场景网格自碰撞或真实环境的硬件安全认证。
增加独立全链校准 Jacobian 奇异值审计（线性部分按1.1838 m归一化），
保留原 ROS 奇异/限位/碰撞/速度门槛。未验证真实加速度、力矩和实际接触力。

## ROS 接入参数（显式 opt-in，正式默认不变）

```yaml
offline_joint_tool_absolute_rotation_weight: 10.0
offline_joint_actual_wrist_weight: 10.0
offline_joint_robot_context: /home/robot/ros2_ws/evaluation/wrist_joint_20261002/canvas_3_slots/robot_context.json
offline_joint_max_steps: 16
optimize_penup_wrist_motion: true
use_exact_calibrated_ik: true
strict_ik: true
publish_on_start: false
```

纸面中心(0,-0.65)m、paper_z=0、TCP=0.328m必须匹配 context。
更改位置、TCP、标定或工具/相机几何时会拒绝复用 C，必须生成新的 context 并重新反演。
局部 gamma 已在物理 CSV 中转为 ROS 绝对姿态，不要重复加走向。
这些参数只供后续明确授权的接入；本次脚本只构建离线规划，硬禁止运动发布。
