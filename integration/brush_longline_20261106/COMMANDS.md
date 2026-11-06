## 2026-11-06 新增：固定方向长线 H／α／β 标定代码

本节增加新测试入口，不改变此前实机成功、失败和停用记录。详细配置与操作步骤见仓库 `docs/毛笔长线标定_代码使用.md`。

**本轮验证范围：仅软件测试与合成数据。** MU 上 24 项新增测试和 12 项原守护器回归通过，找零／联合规划、合成 CSV 拟合与预测通过。没有访问 jx、触发真实相机或向机械臂发送运动；不存在本轮实机发布、部分执行或完整实机成功记录。

通用入口（以下命令在对应主机的本仓库根目录运行）：

```bash
PYTHONNOUSERSITE=1 /usr/bin/python3 -B tools/brush_longline_calibration.py --help
```

| 子命令 | 用途与运行主机 | 是否发送运动 | 关键参数与验证状态 |
| --- | --- | --- | --- |
| `init` | MU 离线生成实验 JSON，可导入明确选择的历史候选几何 | 否 | `--output`、可选 `--candidate`；示例初始化已软件验证，历史几何不自动认定为当前有效 |
| `plan` | MU 离线生成单条连续 H／α／β 轨迹、输入 CSV 和预览 | 否 | `--config --kind --trial-id --output`；支持 zero/h/alpha/beta/joint；本轮软件验证 |
| `demo` | MU 离线检查拟合与独立预测链 | 否 | `--output`；已运行，全部为 synthetic_demo |
| `measure` | MU 离线处理写前／写后图像，测笔宽与中心偏移 | 否 | `--before --after --plan --calibration --output`；可加反馈、手眼与实际拍照位姿；毫米图像和手眼逻辑经合成测试，真实采集待验证 |
| `side` | MU 离线提取侧视轮廓和笔尖—纸面投影间隙 | 否 | `--video --frames --calibration --output`；视频及时间对应经软件测试，真实毛束识别待验证 |
| `zero` | MU 离线从持续视觉接触候选估计纸面法向修正 | 否 | `--side-csv --feedback --plan --output`；只输出候选与时间／高度区间，不自动覆盖纸面；经软件边界测试 |
| `fit` | MU 离线拟合三变量静态墨迹响应 | 否 | `--csv --output`；只接收有可靠输入对应的 train 数据；CSV 命令链已合成验证 |
| `predict` | MU 离线对新数据预测并算域内误差 | 否 | `--model --csv --output`；标记训练组重用和域外点；CSV 命令链已合成验证 |
| `camera-check` | 相机所在主机测试侧视录像和写前／写后照片 | 否 | `--cameras --duration-s --output`；会调用相机，不访问机械臂；未在实际相机运行 |
| `record-air` | jx 在当前已稳定悬空姿态采集原始力和位姿 | 否 | `--robot-scripts --unloaded-confirmed --output`；只读采集，不移动、不清零；本轮未实采 |
| `fit-force` | MU 或 jx 对新空载样本拟合局部姿态参考并检查独立留出 | 否 | `--samples --holdout --maximum-error-n --maximum-translation-m --output`；合成记录和校验绑定已软件测试，真实参考待采 |
| `prepare` | jx 读取当前状态并调用既有完整路径／圆滑邻域审计 | 否 | `--plan --force-reference --watchdog --guard-host --robot-scripts --output`；导出绑定证据的准备包，本轮未做现场预检 |
| `run` | 无 `--execute` 时只预览准备包；jx 显式执行时进行一条守护实验和相机采集 | **仅 `--execute` 发送** | 实机另需 `--cameras --robot-scripts`；原守护器保持不变；本轮仅验证默认不访问设备及模拟失败分支 |

可直接在 MU 运行的无设备检查：

```bash
PYTHONNOUSERSITE=1 /usr/bin/python3 -B tools/brush_longline_calibration.py init \
  --output outputs/longline_session/experiment.json

PYTHONNOUSERSITE=1 /usr/bin/python3 -B tools/brush_longline_calibration.py plan \
  --config outputs/longline_session/experiment.json --kind h --trial-id H01 \
  --repeat-group H_repeat --output outputs/longline_session/H01_plan

PYTHONNOUSERSITE=1 /usr/bin/python3 -B tools/brush_longline_calibration.py demo \
  --output outputs/longline_demo_01

PYTHONNOUSERSITE=1 /usr/bin/python3 -B -m unittest discover \
  -s tests -p 'test_brush_longline_calibration.py' -v
```

现场执行的命令结构如下，**此命令本轮未在实机运行，不能据此写成已成功**：

```bash
PYTHONNOUSERSITE=1 /usr/bin/python3 -B tools/brush_longline_calibration.py run \
  --prepared outputs/longline_session/H01_prepared/prepared.json \
  --cameras outputs/longline_session/cameras.json \
  --robot-scripts /home/jx/UR10/scripts --execute \
  --output outputs/longline_session/H01_run
```

准备包需先经过 `prepare`，每个实验独立生成，并使用新鲜的姿态空载参考与当前编译器匹配的原看门狗硬件记录。示例安装、纸面、角度、下压、相机地址和阈值必须替换为本轮真实配置。实机异常不自动重试或进入下一条线。

验证日期：2026-11-06（北京时间）。证据：`integration/brush_longline_20261106/software_tests.txt`、`existing_guard_tests.txt`、`synthetic_demo_result.json` 及本目录 `README.md`。后续硬件测试应追加其真实采集目录和实际执行范围，不覆盖本轮软件验证状态。
