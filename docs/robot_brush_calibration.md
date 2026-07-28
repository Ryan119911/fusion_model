# 真实毛笔与机器人标定接口

单张整字图像不能唯一确定六自由度轨迹。真实阶段应先采集同步的机器人 TCP
姿态和笔触观测，再替换论文曲线与当前仿真尺度。仓库定义
`robot_brush_calibration_v1` CSV，模板为
`examples/robot_brush_calibration_v1.csv`。

字段单位直接写在列名中：

```text
trial_id, point_id, timestamp_s
x_mm, y_mm, z_mm
alpha_rad, beta_rad, gamma_rad
contact
footprint_width_mm, footprint_length_mm, footprint_angle_rad, image_path
```

每组标定应只改变一个主要变量，并覆盖多个重复试验。`timestamp_s` 在每个
trial 内必须严格递增，角度统一使用弧度。先运行：

```bash
python -u tools/validate_robot_brush_calibration.py \
  --calibration_csv data/raw/robot_brush_calibration.csv \
  --output_json data/processed/robot_brush_calibration.summary.json
```

校验器检查键、单位、数值范围、接触样本、测量字段和各姿态变量的激励跨度。
`excitation_ready=true` 只说明实验改变了该变量，不代表图像能够辨识它。随后仍须
用独立留出 trial 做 Jacobian/灵敏度审计。只有输入激励和输出可辨识性同时通过，
才开放对应变量：

```text
z/H      需要压力或高度扫描以及宽度/长度观测
alpha    需要独立倾角扫描
beta     需要独立纸面倾角扫描
gamma    需要非轴对称笔锋、footprint_angle 或外部姿态观测
```

在完成上述标定前，v11 导出的 H/alpha/beta/gamma 是仿真候选值，不是可以直接
下发真实机器人的物理真值。
