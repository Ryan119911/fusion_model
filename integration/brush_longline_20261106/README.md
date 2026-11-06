# 2026-11-06 毛笔固定方向长线标定软件验证

本轮主机：MU，工作区 `/tmp/codex-brush-longline-calibration`。使用 `PYTHONNOUSERSITE=1 /usr/bin/python3 -B`，系统 NumPy 1.21.5、SciPy 1.8.0、OpenCV 4.5.4。默认用户包环境有 NumPy/OpenCV ABI 冲突，未修改系统或用户依赖。

## 结果

| 检查 | 实际结果 | 证据 |
| --- | --- | --- |
| 新增长线软件测试 | 24 项通过 | `software_tests.txt` |
| 原运动守护器回归 | 12 项通过 | `existing_guard_tests.txt` |
| CLI 配置初始化 | 成功生成软件示例配置，无设备访问 | `/tmp/brush-longline-init-final/experiment.json` |
| CLI 找零／联合规划 | 各生成 123 个路点，无运动 | `/tmp/brush-longline-delivery-zero/plan.json`、`/tmp/brush-longline-delivery-joint/plan.json` |
| 合成数据拟合与独立预测 | 300 行训练、100 行新输入，90 行位于训练凸包内；训练实验与重复组未混入验证 | `synthetic_demo_result.json` |
| CLI CSV 拟合与预测 | 合成训练 CSV 经独立 fit、predict 入口完成 | `/tmp/brush-longline-delivery-fit/RESULT.json`、`/tmp/brush-longline-delivery-predict/metrics.json` |
| 机器人和相机 | 未连接、未触发、未发运动 | 本轮没有运行硬件入口 |

新测试覆盖旋转约定、倾斜纸面、自然笔尖补偿、五类输入、几何与速度限制、位姿重建、墨迹毫米测量、手上相机两次位姿、侧视帧时间、零点区间、反馈小抖动／大回退、联合模型秩、留出数据隔离、相机失败及执行故障状态归档。

合成误差只检查已知二次函数能否经接口恢复，不代表真实毛笔精度。调用硬件适配器的异常路径采用注入的模拟后端，不是真实机器人故障实验。

## 现场仍需验证

当前纸面与工具几何、实际相机主题和触发服务、ROI 和尺度、原始曝光与位姿对应、姿态空载参考、实际路径审核、单条程序执行与真实笔触重复性。静态空载留出检查不等同于动态接触力标定。

主流程和逐条命令见 [代码使用说明](../../docs/毛笔长线标定_代码使用.md)。执行模式必须显式使用 `run --execute`，且通过既有路径与看门狗检查。
