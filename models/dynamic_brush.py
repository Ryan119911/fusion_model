# 中文注释：本文件定义动态笔刷物理近似模型，用压力和姿态推导笔根、宽度和偏移。
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import math

from utils.types import TrajectoryPoint, StrokeTrajectory, CharacterTrajectory, DynamicBrushState


# 中文注释：用多项式系数表示一维标量函数。
@dataclass
class PolyFit1D:
    coeffs: List[float]

    # 中文注释：让对象可以像函数一样调用，计算当前输入对应的多项式值。
    def __call__(self, x: float) -> float:
        y = 0.0
        for i, c in enumerate(self.coeffs):
            y += c * (x ** i)
        return float(y)


# 中文注释：保存动态笔刷模型的可学习或标定参数。
@dataclass
class DynamicBrushParams:
    kw: float = 0.02
    kd: float = 0.02
    dt: float = 0.01
    width_fn: Optional[PolyFit1D] = None
    drag_fn: Optional[PolyFit1D] = None
    offset_fn: Optional[PolyFit1D] = None
    snap_clip_min: float = 0.0


# 中文注释：默认压力到笔宽的映射函数。
def default_width_fn() -> PolyFit1D:
    # 首版占位：可由标定脚本替换
    return PolyFit1D([0.0, 0.8])


# 中文注释：默认压力到阻尼/拖拽的映射函数。
def default_drag_fn() -> PolyFit1D:
    return PolyFit1D([0.0, 1.2])


# 中文注释：默认姿态到笔尖偏移的映射函数。
def default_offset_fn() -> PolyFit1D:
    # 三阶占位，允许出现局部峰值
    return PolyFit1D([0.0, 0.5, -0.02, 0.0005])


# 中文注释：根据机器末端轨迹模拟毛笔笔尖动态响应。
class DynamicBrushModel:
    # 中文注释：初始化对象并保存后续处理所需的配置和成员变量。
    def __init__(self, params: Optional[DynamicBrushParams] = None):
        self.params = params or DynamicBrushParams(
            width_fn=default_width_fn(),
            drag_fn=default_drag_fn(),
            offset_fn=default_offset_fn(),
        )
        if self.params.width_fn is None:
            self.params.width_fn = default_width_fn()
        if self.params.drag_fn is None:
            self.params.drag_fn = default_drag_fn()
        if self.params.offset_fn is None:
            self.params.offset_fn = default_offset_fn()

    # 中文注释：从状态向量中取出笔根二维坐标。
    @staticmethod
    def _root_xy(state: DynamicBrushState) -> Tuple[float, float]:
        return (
            state.x - state.o * math.cos(state.theta),
            state.y - state.o * math.sin(state.theta),
        )

    # 中文注释：根据下压深度估计当前笔触宽度。
    def width(self, z: float) -> float:
        return max(float(self.params.width_fn(z)), 0.0)

    # 中文注释：根据下压深度估计笔尖跟随的阻尼系数。
    def drag(self, z: float) -> float:
        return max(float(self.params.drag_fn(z)), 0.0)

    # 中文注释：根据姿态角估计笔尖相对笔根的偏移。
    def offset(self, z: float) -> float:
        return max(float(self.params.offset_fn(z)), self.params.snap_clip_min)

    # 中文注释：在抬笔或过渡时保持当前偏移，避免状态突变。
    def offset_hold(self, prev_root_xy: Tuple[float, float], next_xy: Tuple[float, float], theta_guess: float) -> float:
        dx = next_xy[0] - prev_root_xy[0]
        dy = next_xy[1] - prev_root_xy[1]
        return max(dx * math.cos(theta_guess) + dy * math.sin(theta_guess), self.params.snap_clip_min)

    # 中文注释：根据首个轨迹点初始化动态笔刷状态。
    def init_state(self, first_point: TrajectoryPoint, theta0: Optional[float] = None, reset_brush: bool = True) -> DynamicBrushState:
        if theta0 is None:
            theta0 = float(first_point.gamma)
        if reset_brush:
            w0 = 0.0
            d0 = 0.0
            o0 = 0.0
        else:
            w0 = self.width(first_point.z)
            d0 = self.drag(first_point.z)
            o0 = self.offset(first_point.z)
        return DynamicBrushState(x=first_point.x, y=first_point.y, z=first_point.z, w=w0, d=d0, o=o0, theta=float(theta0))

    # 中文注释：推进一个时间步，更新笔尖位置、宽度和偏移。
    def step(self, prev_state: DynamicBrushState, prev_point: TrajectoryPoint, next_point: TrajectoryPoint) -> DynamicBrushState:
        z_next = float(next_point.z)
        x_next = float(next_point.x)
        y_next = float(next_point.y)

        w_target = self.width(z_next)
        d_target = self.drag(z_next)
        w_next = prev_state.w * self.params.kw + w_target * (1.0 - self.params.kw)
        d_next = prev_state.d * self.params.kd + d_target * (1.0 - self.params.kd)

        prev_root = self._root_xy(prev_state)
        motion_dx = x_next - prev_point.x
        motion_dy = y_next - prev_point.y
        theta_guess = math.atan2(motion_dy, motion_dx) if abs(motion_dx) + abs(motion_dy) > 1e-9 else prev_state.theta

        o_free = self.offset(z_next)
        o_hold = self.offset_hold(prev_root, (x_next, y_next), theta_guess)
        o_next = min(o_free, o_hold)

        root_x = x_next - o_next * math.cos(theta_guess)
        root_y = y_next - o_next * math.sin(theta_guess)
        theta_next = math.atan2(y_next - root_y, x_next - root_x) if abs(x_next - root_x) + abs(y_next - root_y) > 1e-9 else theta_guess

        return DynamicBrushState(x=x_next, y=y_next, z=z_next, w=w_next, d=d_next, o=o_next, theta=theta_next)

    # 中文注释：逐点模拟一条笔画的动态笔刷状态。
    def simulate_stroke(self, stroke: StrokeTrajectory, theta0: Optional[float] = None, reset_brush: bool = True) -> List[DynamicBrushState]:
        points = stroke.sorted_points()
        if len(points) == 0:
            return []
        states: List[DynamicBrushState] = [self.init_state(points[0], theta0=theta0, reset_brush=reset_brush)]
        for i in range(1, len(points)):
            states.append(self.step(states[-1], points[i - 1], points[i]))
        return states

    # 中文注释：对字符中的每一笔分别执行动态笔刷模拟。
    def simulate_character(self, sample: CharacterTrajectory, reset_each_stroke: bool = True) -> Dict[int, List[DynamicBrushState]]:
        all_states: Dict[int, List[DynamicBrushState]] = {}
        prev_last_state: Optional[DynamicBrushState] = None
        for stroke in sample.sorted_strokes():
            if reset_each_stroke or prev_last_state is None:
                states = self.simulate_stroke(stroke, theta0=None, reset_brush=True)
            else:
                points = stroke.sorted_points()
                if len(points) == 0:
                    states = []
                else:
                    init = DynamicBrushState(x=points[0].x, y=points[0].y, z=points[0].z, w=prev_last_state.w, d=prev_last_state.d, o=prev_last_state.o, theta=prev_last_state.theta)
                    states = [init]
                    for i in range(1, len(points)):
                        states.append(self.step(states[-1], points[i - 1], points[i]))
            all_states[stroke.stroke_id] = states
            if len(states) > 0:
                prev_last_state = states[-1]
        return all_states


# 中文注释：根据输入输出样本拟合一维多项式。
def fit_poly_from_pairs(xs: List[float], ys: List[float], degree: int) -> PolyFit1D:
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have the same length")
    if len(xs) == 0:
        raise ValueError("empty calibration pairs")
    try:
        import numpy as np
        coef_desc = np.polyfit(np.array(xs, dtype=float), np.array(ys, dtype=float), deg=degree)
        coeffs = list(reversed([float(c) for c in coef_desc.tolist()]))
        return PolyFit1D(coeffs)
    except Exception as e:
        raise RuntimeError(f"Polynomial fitting failed: {e}")


# 中文注释：从标定系数构建动态笔刷模型。
def build_dynamic_brush_from_calibration(width_pairs: Tuple[List[float], List[float]], drag_pairs: Tuple[List[float], List[float]], offset_pairs: Tuple[List[float], List[float]], kw: float = 0.02, kd: float = 0.02, dt: float = 0.01, width_degree: int = 2, drag_degree: int = 2, offset_degree: int = 3) -> DynamicBrushModel:
    width_fn = fit_poly_from_pairs(width_pairs[0], width_pairs[1], degree=width_degree)
    drag_fn = fit_poly_from_pairs(drag_pairs[0], drag_pairs[1], degree=drag_degree)
    offset_fn = fit_poly_from_pairs(offset_pairs[0], offset_pairs[1], degree=offset_degree)
    params = DynamicBrushParams(kw=kw, kd=kd, dt=dt, width_fn=width_fn, drag_fn=drag_fn, offset_fn=offset_fn)
    return DynamicBrushModel(params=params)
# 使用说明：该模块实现了动态笔触模型的首版物理更新逻辑。
# DynamicBrushState 中的 7 维状态分别对应位置 (x,y,z)、笔触宽度 w、拖拽长度 d、偏移量 o 和真实朝向 theta；
# PolyFit1D 用于承载 Width(z)、Drag(z)、Offset(z) 的一维多项式拟合函数；
# DynamicBrushModel.step() 实现离散状态更新，其中 w 和 d 带有惯性项，o 
# 通过 min(自由偏移, 保持原地所需偏移) 近似实现 friction/snap 机制；
# simulate_stroke() 和 simulate_character() 分别用于单笔画与整字级的正向仿真。
# build_dynamic_brush_from_calibration() 则提供了从标定数据拟合多项式并构造模型的入口，后续可以由 fit_dynamic_model.py 调用，把真实实验数据替换掉当前的占位函数。
