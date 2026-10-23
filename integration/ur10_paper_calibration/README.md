# UR10 纸面标定：接续 jx 的 12 点采集

本目录接续 SSH 主机 `jx` 的“读取交接索引”对话；机械臂身份来自本机本项目的“ROS 2”对话及 `ur10_actual_kinematics.py`：UR10 CB3，序列号 `2022300602`，工厂标定 `calib_15120592593058779304`，控制器坐标系 `base`。

保留远程已经实现的流程：两个对角点示教（纸边与基座 XY 平行）、向内留边、自动 12 点、9 点拟合及 3 点独立验证、逐点保存、失败停止和续采。接触检测为相对静止基线的 1 N 三轴力变化，原始合力变化达到 5 N 停止。基线、姿态、实时帧、路径模型、关节余量门禁保持生效。

当前安装的是先前对话中的圆珠笔，使用 `~/UR10/outputs/tcp_calibration/ballpoint-pen-20261016.json` 的实测笔尖偏移约 `(0.114, -1.661, 329.069) mm`。这与 ROS 毛笔场景的名义 328 mm 长度用途不同。脚本不写控制器 TCP，也不修改 ROS 正式标定和 CoppeliaSim。

## 本次修复

- 最新实测失败发生在第 2 点：接触后固定等待 0.1 秒、只看一帧速度，报 `robot not stationary after contact stop`。改为最多 3 秒的受监控等待，连续至少 0.15 秒低线速度和低角速度、至少 9 帧位置稳定后才记录接触坐标；等待中仍检查力、状态流和关节余量。超时或异常停止，不强行抬笔。
- 保留上一版按速度/加速度计算有限运动超时、每段最多两次停机重连的修复。重连后若已到目标，不再重复发送已完成的运动指令。实时断流的现场根因尚未确定，不能据离线测试宣称已解决。
- 续采与拟合核对原布点计划、唯一编号、训练/验证角色及实测 XY。只读预检沿用原计划，避免重排已有点。`--host` 同时传给 Dashboard 检查，并核对机械臂型号、序列号及工厂标定身份。
- 接收层单独保存在 `paper_ur_support.py`；模型/碰撞依赖仅在真实路径检查时导入。本地可以不连接 ROS 和机械臂进行离线测试；真实执行仍必须通过原项目的 Pinocchio/碰撞检查。
- SSH 非交互环境未加载 `.bashrc` 时，路径检查会加载 jx 已安装在 `/opt/openrobots/lib/python3.10/site-packages` 的 Pinocchio；不会安装新依赖或跳过检查。
- 预检的两个静止位姿均改为九帧低速度窗口取平均，继续按 0.1 mm 比较前后位移。已在原 0.1 mm 到位范围内的段直接结束，避免对编码器波动反复做微小修正。首次静止连接等待首帧最多 3 秒，运动中的 250 ms 断流门限保持不变。
- 每个有限运动段开始和结束时保存采集进度，失败时额外保存基线噪声和预检实际位移，运行中能读取真实进度。
- 运动指令改为单个完整的 `def … end` 程序，以 `end` 结束；移除外层 `end` 后额外发送的函数调用。格式与 [UR 官方 URScript 手册第 2 节](https://www.universal-robots.com/manuals/EN/PDF/SW5_20/scriptmanualG5/script_directory_Poly5.pdf)的接收要求一致。
- 30002 指令连接改为复用连接并由独立接收线程持续读取控制器输出，按官方手册建议在关闭前读取至少 79 字节，避免发送后立即关闭未读取的连接。此传输层不自动重发运动；发送失败直接上报，由原有停机流程处理。父程序在触纸子进程接管前释放连接。
- 触纸时的 XY/姿态门禁保持 0.2 mm / 0.1°，触发时额外保存目标、实际位姿、速度、力和最近 32 帧误差，下一次实测可以直接辨别实际偏差。
- 竖直到位单独检查九帧均值的 Z 误差；XY 继续独立按 0.2 mm 门禁检查。第 5 点的 32 帧实测显示静止 Z 残差中位数为 −0.0504 mm、标准差 0.0106 mm、速度为零，50 μm 门限导致稳定计时反复重置；据此将 Z 到位容差设为 75 μm，仍小于 0.1 mm 单步距，并继续要求持续 0.15 秒低速度。运动中断流仍为 250 ms；静止预检只在总计 3 秒内等待新的九帧窗口，遇到帧间隔超过 150 ms 时清空旧窗口。
- `recover_paper_start.py` 处理 0–1.5 mm 的纸内近纸起点：先使用上次自由状态基线、保持当前 XY/姿态分段向上 4 mm。若手动姿态改变不超过 2°，继续竖直抬至已知纸面最高点上方至少 20 mm，确认自由状态后取基线，再分段恢复原姿态。每段均做原路径、力、实时帧和关节检查；触纸流程的姿态门限不变。该工具默认只读，`--execute` 才运动。

## 在 jx 上运行

先保持纸张、笔的夹持及控制器 TCP 设置与原采集一致。若笔尖仍接触纸面，先用示教器抬离，起点允许离纸 2–100 mm。

```bash
source /opt/ros/humble/setup.bash
source ~/colcon_ws/install/setup.bash

PYTHONNOUSERSITE=1 FASTDDS_BUILTIN_TRANSPORTS=UDPv4 \
python3 ~/.dsh-analysis/calibrate_paper_plane_auto.py \
  --resume-if-exists --check \
  --output ~/UR10/outputs/paper_calibration/paper-plane-20261016.json

PYTHONNOUSERSITE=1 FASTDDS_BUILTIN_TRANSPORTS=UDPv4 \
python3 ~/.dsh-analysis/calibrate_paper_plane_auto.py \
  --execute --resume-if-exists \
  --output ~/UR10/outputs/paper_calibration/paper-plane-20261016.json
```

第一条只读取机器人、力及下一段路径。第二条由现场用户核对范围后按回车才连续采集；会跳过已接受的第 1 点。环境变量由命令提供，脚本不硬编码 DDS 传输设置。

本次接手时仅有 1/12 个接受点；第 2 点失败的记录不会作为有效点加入平面拟合。离线测试中的完整 12 点来自模拟数据，实机标定尚未完成。只有全部点有效、训练 RMS ≤0.5 mm、留出最大误差 ≤1 mm、覆盖和几何门禁均通过，才会输出 `status=validated`。测得的是受力接触笔尖所在平面，包含笔尖 TCP 误差与笔的受压变形，不能宣称具有独立测量的绝对精度。

## 离线验证与来源

```bash
python3 -m unittest discover -s integration/ur10_paper_calibration -p test_paper_motion.py -v
python3 integration/ur10_paper_calibration/calibrate_paper_plane_auto.py --self-test
python3 integration/ur10_paper_calibration/probe_paper_height.py --self-test
```

覆盖停机减速/停稳、等待超时、力/状态故障、到位后不重复运动、同一端点重试、恢复次数上限、超过旧 5 秒限制的低加速度运动、机器人身份、重复/缺失点拒收，以及原有几何和完整失败续采集成测试。

2026-10-16 自动实测补充：修正静止预检和程序封装后，已实际完成下一次下降的第 1 个小段，未记录实时断流恢复；第 2 个小段检测到 7.024 N 原始三轴合力变化（当时五点中值约 0.590 N），按原 5 N 门限停止。停机后对照原基线的 1001 条只读样本，合力变化中位数 0.085 N、最大 0.419 N，速度为零。用户随后明确确认有人触碰、现已离开且路径空旷，因此在相同保护下继续自动续采。原第 1 点及全部失败日志保留；完整 12 点可靠性仍以最终实采和验证为准。

后续恢复时发现有线网卡 `enp0s31f6` 为 `NO-CARRIER`、无 IPv4 地址，机器人 IP 被路由至 `singbox_tun`；Dashboard/Primary/Realtime 均收不到响应。用户在现场恢复原 `Wired connection 1` 后，有线地址恢复 `192.168.1.10/24`，路由恢复直连 `enp0s31f6`，控制器状态可读。该原因与脚本的单段处理问题分别保留，不能把所有历史断流都归因于某一个原因。

`source_manifest.json` 保存远程原脚本与当前文件的 SHA256。部署前核对远程旧版 SHA，备份再替换，不修改采样 JSON。新增文件 `paper_ur_support.py` 必须与主脚本一起部署。

实时帧偏移沿用 jx 原始 `record_ur_trajectory.py` 的实际关节、TCP 位姿和速度字段，并对照 [UR 官方实时协议](https://docs.universal-robots.com/tutorials/communication-protocol-tutorials/realtime-guide.html)。停机等待依据 [UR 官方 stopl 说明](https://www.universal-robots.com/manuals/EN/HTML/SW5_19/Content/prod-scriptmanual/G5/stopl_a_aRot_a.htm)中的减速至零语义；后一个链接为较新手册，仅用于说明 stopl 的语义，CB3 3.15.8 的现场流程仍需实机验证。
