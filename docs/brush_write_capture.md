# UR10 毛笔：书写位与独立拍照位

本文主体记录最初关机状态下的离线设计与程序准备，当时没有上电、运动、实时状态读取或相机触发。后续实机试写状态与运行命令见 [机械臂写字](机械臂写字.md)：截至 2026-10-23，完整三笔仍未验收成功；独立自动拍照转移的原验证状态没有因此扩大。

入口：`tools/brush_write_capture.py`。默认保存预览结果；只有显式 `--execute` 才会进入现有守护执行器。独立拍照位不要求看见笔头，目标是无遮挡观察纸上墨迹。

## 本轮设计

- 纸面：复用 `paper-frame-reuse-20261022.json`，保留“历史三角点结合新 TCP 换算、绝对高度未独立验证”的状态。
- TCP：使用本轮 4 mm 验收记录的平移 `[-0.398, -4.615, 474.199] mm`；原始求解器旧门限失败结论保留。
- 笔杆方向：CAD 与现场描述确定的名义方向，角度误差未独立测量。
- 手眼：`config/handeye_eye_in_hand_20260921.yaml` 的 `tool0_from_camera`；保留其原有数值验证范围，不扩展为运动验收。
- CAD：保持装配高 Y 传感器安装面、CAD −Y 向外、270° 安装角的原有约定；初步 CAD 包络仍非现场完整障碍物模型。

相机光学位姿按纸面中心附近的 10 mm 标定短线设计。`base_from_tool0 = base_from_camera @ inverse(tool0_from_camera)`，没有反用手眼变换。初步距离 450 mm、检查区域 50 × 50 mm，实际工作距离范围依赖现有相机镜头版本。官方 PRO XS 350/550 mm 焦距版本分别推荐 300–450/450–600 mm；450 mm 是名义边界，不是镜头版本与对焦效果已验证。实际采集以 CameraInfo、图像清晰度和点云有效区域复核。

官方来源：https://docs.mech-mind.net/en/eye-3d-camera/2.3.1/hardware/specifications-pro-xs.html

最后采用相机画面滚转 270° 的候选；这是拍照方向参数，和 CAD 装配的 270° 安装角是两个不同参数。另一数值可行候选在自碰撞模型中失败，保留其报告。

## 离线使用

在 `/home/robot/coppeliasim/machine_learning/model` 下，直接预览本轮已检查样本：

```bash
python3 -B tools/brush_write_capture.py \
  --sample outputs/brush_mapping_20261023/write_capture_delivery/sample_preview_checked.json \
  --output /tmp/brush_capture_preview_01
```

输出目录必须不存在。命令不导入 ROS、不触发相机、不打开机器人网络连接。

重新生成全部名义候选和稠密数值检查：

```bash
OPENBLAS_NUM_THREADS=1 python3 -B tools/brush_write_capture.py --design \
  --output /tmp/brush_capture_design_01
```

`--design` 只挑选数值/光学候选，不将自碰撞或现场状态自动标记通过。额外自碰撞检查用 `check_brush_capture_self_collision.py`，在 jx 已安装的 Pinocchio/hppfcl 中只处理给定关节数组；不调用机器人或相机。将对应报告用 `--select-candidate ... --collision-report ... --sample .../sample.json --output ...` 绑定到最终样本。该报告保存输入数组并逐项核对，避免混用路径。

## 多档深度和姿态配置

`outputs/brush_mapping_20261023/write_capture_delivery/trial_specs/` 保存 15 份独立计划：0.3/0.6/1.0 mm 指令深度、两个纸面轴的正负倾角、正负轴向角，以及独立留出 trial。它们是待检查的物理命令候选，不是已测毛束压入量或模型 H。各位置按名义 A4 范围分开安排，是否为空白仍须用固定相机观察；没有自动递增深度或批量执行入口。

改变深度、纸面位置或姿态时，给设计命令传入一个配置：

```bash
OPENBLAS_NUM_THREADS=1 python3 -B tools/brush_write_capture.py --design \
  --trial-spec outputs/brush_mapping_20261023/write_capture_delivery/trial_specs/tilt-5-0.json \
  --output /tmp/brush_capture_tilt_design_01
```

配置中 `commanded_tilt_delta_uv_rad` 是相对于已记录书写姿态、绕纸面 u/v 轴的指令变化；先 u 后 v。`commanded_axis_delta_rad` 在名义毛笔局部 Z 轴上追加旋转。位置与姿态联合用完整 TCP 换算，使指定自然笔尖的纸面位置与名义下压量保持不变。名义工具轴的未独立测量状态保留。

这只检查当前独立 trial，不复用第一组样本的路径通过结论。`--data-dir` 可指定包含纸面/TCP记录的目录，`--upstream-dir` 可指向已有 UR10 仓库中的配置和 CAD，均无需联网或安装新依赖。

## 一次样本的实际流程

1. 独立样本编号，同时指定 `train`/`holdout`、重复组、指令深度、指令姿态变化、速度与空白纸面位置。不同档位由操作者逐次决定，不自动增加深度。
2. 取得新零压和悬空参考后，以同一套纸面、TCP、名义工具姿态生成 `planned_write` 与 `planned_transfer` 的证据包。指令深度是沿纸面法向的位移，`model_H_mm` 和未独立测量的压入量保持 null。
3. 一次受保护书写 → 沿纸面法向抬笔 → 在离纸高度改变姿态、平移、进入拍照位。书写结束、拍摄开始必须停稳；书写路径内部仍由原生 buffered movel 混合过渡控制。
4. 原始二维图、普通点云、带纹理点云依次采集，保存各自的请求时间和消息时间。不把三次服务触发声称为同一次同步曝光。
5. 拍摄全程读取位姿并锁存移动/断流失败；采集结束再次核验停稳。任一运动阶段失败均不继续拍摄，不自动重试或返回书写位。

`--execute` 不上电、不解锁、不设 TCP 或负载。它要求真实新参考、两个已有运动工具认可的完整证据包、相同样本编号、精确路点/速度/加速度一致，以及拍照末位姿一致。本轮输出保留待补项，因此不能直接用于执行。原有 force、watchdog、路径证据和运动包验证没有被放宽或换成新的通过标志。

当前还没有可供拍照变姿路径使用的独立有效悬空力参考；历史重力补偿仍仅静态展示候选。程序不会把它代替接触保护的参考。

## 保存的实际文件

每个样本目录保存：

- `sample.json`：同一样本编号、深度、速度、姿态、纸面/TCP/手眼原始配置快照及各自状态。
- `used_bundles.json`、`write_execution.json`、`transfer_execution.json`：原始指令与现有执行器结果。
- `write_raw_force.jsonl`、`transfer_raw_force.jsonl`：完整原始力时序；不受现有 50000 行内存滑窗截断。
- `*_controller_feedback.jsonl`：关节、实际 tool0 位姿、速度、主机/控制器时钟。拍摄阶段另外保存连续状态。
- `photo_stationary_1.json`、`photo_stationary_2.json`：拍摄前后位姿窗口。
- `camera/image.cdr`、`.bin`、`.png`、`image_mono.png`：真实 ROS 消息、原始像素和无损单色图。
- `camera/camera_info.json`、`.cdr`：图像实际内参、畸变、尺寸、frame、时间。
- `camera/cloud.*`、`textured_cloud.*`：实际 CDR、二进制点云、完整组织 XYZ 米制数组及其 Z 深度数组；每个文件有字段/行步长/时间元数据。
- `camera/capture.json`：采集请求、曝光消息时间、图像/点云坐标变换。不同 frame 无 TF 时明确缺失，不擅自视为已对齐。
- `result.json`：最终采集状态；失败时保留已获得原始文件。

## 毫米测量和数据集

```bash
python3 -B tools/brush_capture_measure.py --sample SAMPLE.json --run SAMPLE_DIR \
  --output MEASUREMENT_DIR --mm-per-pixel 0.1

python3 -B tools/brush_capture_measure.py --dataset MEASUREMENT_DIR/measurement.json \
  --output DATASET_DIR
```

测量使用拍摄反馈位姿与手眼标定，按原始内参/畸变把纸面毫米网格投影到原始单色图，输出正射图、墨迹掩膜、毫米轮廓及最小包围矩形的宽、长、方向。长度定义为长轴跨度，不称为弯曲线段的骨架长度。ROI 越界、墨迹碰边或出现多个主要连通块均停止测量，避免把遮挡或相邻墨迹计作当前样本。

默认测量 ROI 随样本的纸面位置和短线长度移动；也可用 `--roi-mm xmin xmax ymin ymax` 指定纸面毫米范围。传入的配置必须与采集目录中的 `sample.json` 相同，图像、结果和配置的编号也必须一致。

深度/点云保留原始值；存在图像到点云的已知变换时，报告目标区域点云相对纸面的残差。缺少 TF 不伪造配准。当前纸面绝对高度未独立验证，所以生成尺寸保留估计状态，不自动批准实机 H 映射。

数据集索引要求真实图像、点云、力与执行记录存在且编号一致；相同重复组不得跨训练/留出集。`dataset_plan.csv` 仅安排未来深度、倾角与独立重复，未采样行不写成真实样本。目前实测数据集为空。

模型 H、真实压入量、指令下压量分列保存。单档 10 mm 的旧照片不能替代多档响应或留出验证。

## 验证

```bash
python3 -B -m unittest discover -s tests -p test_brush_write_capture.py -v
```

测试涵盖默认无硬件调用、阶段顺序、转移失败不拍摄、完整力流、原始消息落盘、带行填充/大小端的图像点云、已知尺寸的毫米恢复、错误 ROI 与无真实原始数据的样本拒绝。

2026-10-23：13 项测试通过。新增配置入口和测量代码没有改变默认样本的已检查几何；13 项包含默认几何一致性、倾角变化后的笔尖位置保持，以及从保存的图像/位姿/内参完整恢复毫米尺寸。实际相机仍未触发，测得的真实样本数为 0。


## 2026-10-23 上电后当前姿态诊断

已在当前停稳姿态完成独立相机诊断，不执行书写或运动。记录位于 `outputs/brush_mapping_20261023/camera-current-20261023T020939590153Z/ONLINE_CHECK.md`。原始图、点云和连续反馈均已落盘；当前画面没有目标纸面，仍需进入经检查的独立拍照位验证。点云到图像的 TF 尚缺失，保留未配准状态。现场确认存在接触，不能将该姿态当作悬空参考。

实采暴露了连续状态首帧尚未到达就套用 150 ms 运行门限的问题。现先等待既有初始停稳窗口；失败不触发相机，采集期间门限不变。14 项测试通过，修正后的诊断采集也通过连续监测。失败首轮保留原记录，真实书写样本仍为 0。
