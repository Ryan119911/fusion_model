# 轨迹预处理、抬笔与安全检查

`utils/trajectory_processing.py` 是机器人无关的轨迹后处理层。它不依赖
UR5、ZYArm 或任何具体机械臂，输入已有的 `character/sample_id/stroke_id/
point_id/x/y/z/alpha/beta/gamma/state` CSV，输出可供仿真或未来机器人适配器
使用的轨迹。

## 默认处理规则

每个 `stroke_id` 都是一个硬边界：

```text
每笔第一个点: DOWN
每笔中间点:  MOVE
每笔最后一个点: UP
```

渲染器只在同一笔内部插值，跨笔不插值、不渲染连接线。即使外部 CSV 错误地
复用了一个 `stroke_id`，渲染器也按连续段切分，避免重新产生斜向误连接。

## 生成处理结果和预览图

```bash
python tools/process_trajectory.py \
  --input_csv data/raw/trajectories.csv \
  --character 武 \
  --output_dir outputs/trajectory_processed \
  --smooth_passes 2 \
  --smooth_strength 0.25 \
  --max_step_xy 2.0 \
  --safety_max_step_xy 4.0 \
  --safety_max_angle_step_rad 1.57 \
  --fail_on_unsafe
```

输出包括：

```text
trajectory_processed.csv
trajectory_raw_preview.png
trajectory_processed_preview.png
trajectory_overlay.png
trajectory_report.json
```

预览图中绿色圆点表示落笔端，红色圆点表示抬笔端，彩色线段按笔画分别
绘制。`trajectory_report.json` 记录跨笔连接数、每笔最大步长、最大姿态跳变、
越界点和所有错误/警告。

`--max_step_xy` 使用输入 CSV 的 XY 坐标单位，只在导出执行轨迹时插入同一笔
内部的插值点，会改变点数和 `point_id`；因此已有 pose CSV 需要重新按处理后的
CSV 生成。仅想平滑而保持
pose CSV 键不变时，把 `--max_step_xy` 保持为 `0`，平滑会保留每笔点数和原有
`point_id`。

如果要直接给未来的机器人适配器使用显式抬笔点，可以增加：

```bash
python tools/process_trajectory.py ... \
  --lift_z 20.0 \
  --expand_pen_up
```

每笔末尾会增加同一 XY 的抬笔点；下一笔先在 `clearance_z` 高度移动，再
垂直下降到 `DOWN` 点。这个执行格式会重新编号 `point_id`，只用于机器人
执行，不应再拿它直接训练或渲染B-BSMG。

## 直接用于纸面渲染

`tools/render_paper_trajectory.py` 默认自动修复状态，但不改变点数和点 ID：

```bash
python tools/render_paper_trajectory.py \
  --trajectory_csv outputs/trajectory_processed/trajectory_processed.csv \
  --bbsmg_ckpt outputs/paper_bbsmg_v1/bbsmg_best.pt \
  --pose_csv outputs/wu_paper_inverse/wu_trajectory.csv \
  --character 武 \
  --smooth_passes 0 \
  --output_image outputs/paper_forward/rendered.png
```

需要强制执行安全门时加 `--fail_on_unsafe`。渲染 JSON 会包含
`trajectory_safety` 和 `cross_stroke_segments_rendered=0`。

## 真实机械臂适配注意

`state` 只描述接触语义，不等于具体控制器的动作命令。机器人适配器应将
`UP` 转换为抬到安全高度，将下一笔的 `DOWN` 转换为下降到纸面，再发送该笔
内部的 `MOVE` 点。任何机器人执行前都必须检查速度、加速度、关节限位、TCP
和纸面穿透；这些检查不由本模块伪造完成。
