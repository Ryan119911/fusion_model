# 中文注释：本文件提供坐标归一化、轨迹重采样和书写几何转换工具。
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import math

from utils.types import StrokeTrajectory, CharacterTrajectory, DynamicBrushState, BBSMGInput


# 中文注释：描述从源坐标到目标画布坐标的缩放与平移关系。
@dataclass
class CanvasTransform:
    src_min_x: float
    src_max_x: float
    src_min_y: float
    src_max_y: float
    dst_size: int = 128
    padding: int = 4

    # 中文注释：把单个点映射到目标画布坐标系。
    def map_point(self, x: float, y: float) -> Tuple[float, float]:
        w = max(self.src_max_x - self.src_min_x, 1e-6)
        h = max(self.src_max_y - self.src_min_y, 1e-6)
        avail = self.dst_size - 2 * self.padding
        scale = min(avail / w, avail / h)

        # 居中：把缩放后的内容在 avail 区域内水平/垂直居中
        off_x = self.padding + (avail - w * scale) / 2.0
        off_y = self.padding + (avail - h * scale) / 2.0

        nx = (x - self.src_min_x) * scale + off_x

        # 关键修正：y 轴翻转
        # 原始轨迹坐标和图像坐标 y 方向相反；
        # 图像/PIL 坐标系中 y 向下增大，所以这里用 src_max_y - y。
        ny = (self.src_max_y - y) * scale + off_y

        return nx, ny


# 中文注释：将 MakeHanzi 坐标转换为常见显示坐标方向。
def makehanzi_to_display(x: float, y: float) -> Tuple[float, float]:
    # MakeMeAHanzi: upper-left=(0,900), lower-right=(1024,-124), y decreases downward in source definition
    return x, 900 - y


# 中文注释：将 MakeHanzi 点归一化到 0 到 1 范围。
def makehanzi_to_normalized(x: float, y: float, canvas_size: int = 128) -> Tuple[float, float]:
    dx, dy = makehanzi_to_display(x, y)
    tx = dx / 1024.0 * (canvas_size - 1)
    ty = dy / 1024.0 * (canvas_size - 1)
    return tx, ty


# 中文注释：按给定边界框把点集归一化到画布尺寸。
def normalize_points(points: List[Tuple[float, float]], canvas_size: int = 128, padding: int = 4) -> List[Tuple[float, float]]:
    if len(points) == 0:
        return []
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    tfm = CanvasTransform(min(xs), max(xs), min(ys), max(ys), dst_size=canvas_size, padding=padding)
    return [tfm.map_point(x, y) for x, y in points]


# 中文注释：归一化单条 MakeHanzi 笔画中线。
def normalize_makehanzi_median(median: List[Tuple[int, int]], canvas_size: int = 128) -> List[Tuple[float, float]]:
    return [makehanzi_to_normalized(float(x), float(y), canvas_size=canvas_size) for x, y in median]


# 中文注释：从笔画中线提取起笔点。
def stroke_start_point_from_median(median: List[Tuple[int, int]], canvas_size: int = 128) -> Tuple[float, float]:
    if len(median) == 0:
        return 0.0, 0.0
    return normalize_makehanzi_median(median, canvas_size=canvas_size)[0]


# 中文注释：根据中线前两个点估计初始书写方向。
def estimate_initial_theta_from_median(median: List[Tuple[int, int]]) -> float:
    if len(median) < 2:
        return 0.0
    (x0, y0), (x1, y1) = median[0], median[1]
    dx = float(x1 - x0)
    dy = float(y1 - y0)
    return math.atan2(dy, dx)


# 中文注释：计算相邻点形成的方向角序列。
def compute_heading(points: List[Tuple[float, float]]) -> List[float]:
    if len(points) == 0:
        return []
    if len(points) == 1:
        return [0.0]
    headings: List[float] = []
    for i in range(len(points)):
        if i == len(points) - 1:
            x0, y0 = points[i - 1]
            x1, y1 = points[i]
        else:
            x0, y0 = points[i]
            x1, y1 = points[i + 1]
        headings.append(math.atan2(y1 - y0, x1 - x0))
    return headings


# 中文注释：计算折线总长度。
def _polyline_length(points: List[Tuple[float, float]]) -> float:
    total = 0.0
    for i in range(1, len(points)):
        dx = points[i][0] - points[i - 1][0]
        dy = points[i][1] - points[i - 1][1]
        total += math.sqrt(dx * dx + dy * dy)
    return total


# 中文注释：按等弧长方式重采样折线。
def resample_polyline(points: List[Tuple[float, float]], num_samples: int) -> List[Tuple[float, float]]:
    if len(points) == 0:
        return []
    if len(points) == 1 or num_samples <= 1:
        return [points[0]]
    total_len = _polyline_length(points)
    if total_len < 1e-8:
        return [points[0] for _ in range(num_samples)]
    step = total_len / (num_samples - 1)
    out = [points[0]]
    acc = 0.0
    cur = 1
    prev = points[0]
    target = step
    while len(out) < num_samples - 1 and cur < len(points):
        p1 = points[cur]
        seg_dx = p1[0] - prev[0]
        seg_dy = p1[1] - prev[1]
        seg_len = math.sqrt(seg_dx * seg_dx + seg_dy * seg_dy)
        if acc + seg_len >= target and seg_len > 1e-8:
            r = (target - acc) / seg_len
            nx = prev[0] + r * seg_dx
            ny = prev[1] + r * seg_dy
            out.append((nx, ny))
            prev = (nx, ny)
            target += step
        else:
            acc += seg_len
            prev = p1
            cur += 1
    out.append(points[-1])
    return out[:num_samples]


# 中文注释：计算轨迹点的二维外接边界。
def trajectory_bounds(sample: CharacterTrajectory) -> Tuple[float, float, float, float]:
    pts = sample.all_points()
    xs = [p.x for p in pts] if pts else [0.0]
    ys = [p.y for p in pts] if pts else [0.0]
    return min(xs), max(xs), min(ys), max(ys)


# 中文注释：把轨迹 xy 坐标归一化到指定画布范围。
def normalize_trajectory_xy(sample: CharacterTrajectory, canvas_size: int = 128, padding: int = 4) -> List[List[Tuple[float, float]]]:
    min_x, max_x, min_y, max_y = trajectory_bounds(sample)
    tfm = CanvasTransform(min_x, max_x, min_y, max_y, dst_size=canvas_size, padding=padding)
    normalized: List[List[Tuple[float, float]]] = []
    for stroke in sample.sorted_strokes():
        pts = [(p.x, p.y) for p in stroke.sorted_points()]
        normalized.append([tfm.map_point(x, y) for x, y in pts])
    return normalized

# 中文注释：使用指定边界归一化轨迹 xy 坐标。
def normalize_trajectory_xy_with_bounds(
    sample: CharacterTrajectory,
    bounds: Tuple[float, float, float, float],
    canvas_size: int = 128,
    padding: int = 4,
) -> List[List[Tuple[float, float]]]:
    min_x, max_x, min_y, max_y = bounds
    tfm = CanvasTransform(
        min_x,
        max_x,
        min_y,
        max_y,
        dst_size=canvas_size,
        padding=padding,
    )

    normalized: List[List[Tuple[float, float]]] = []

    for stroke in sample.sorted_strokes():
        pts = [(p.x, p.y) for p in stroke.sorted_points()]
        normalized.append([tfm.map_point(x, y) for x, y in pts])

    return normalized

# 中文注释：把笔画点序列转换为书写方向角序列。
def stroke_to_headings(stroke: StrokeTrajectory) -> List[float]:
    pts = [(p.x, p.y) for p in stroke.sorted_points()]
    return compute_heading(pts)


# 中文注释：把动态笔刷状态转换为 B-BSMG 使用的曲线控制描述。
def dynamic_state_to_bezier(state: DynamicBrushState) -> Tuple[float, float, float]:
    # 近似桥接：w -> Lr, d -> Lt+Lh。此处给出首版可替换实现。
    lt = max(state.d * 0.7, 0.0)
    lh = max(state.d * 0.3, 0.0)
    lr = max(state.w, 0.0)
    return lt, lh, lr


# 中文注释：把动态笔刷状态整理成 B-BSMG 输入特征。
def dynamic_state_to_bbsmg_input(state: DynamicBrushState, x0: Optional[float] = None, y0: Optional[float] = None) -> BBSMGInput:
    # 首版桥接：用 z, theta, theta 近似映射到 (h, alpha, beta)，起点默认取当前 x,y
    return BBSMGInput(
        h=float(state.z),
        alpha=float(state.theta),
        beta=float(state.theta),
        x0=float(state.x if x0 is None else x0),
        y0=float(state.y if y0 is None else y0),
    )


# 中文注释：按顺序配对真实轨迹笔画与 MakeHanzi 中线。
def pair_trajectory_strokes_with_medians(sample: CharacterTrajectory, medians: List[List[Tuple[int, int]]]) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    strokes = sample.sorted_strokes()
    n = min(len(strokes), len(medians))
    for i in range(n):
        pairs.append({
            "stroke": strokes[i],
            "median_raw": medians[i],
            "median_norm": normalize_makehanzi_median(medians[i]),
            "start_point": stroke_start_point_from_median(medians[i]),
            "theta0": estimate_initial_theta_from_median(medians[i]),
        })
    return pairs
# 使用说明：该模块负责统一处理 SVG/median、整字轨迹和神经渲染输入之间的几何关系。
# CanvasTransform、normalize_points() 与 normalize_trajectory_xy() 用于把毫米坐标或任意平面坐标映射到统一的训练画布；
# makehanzi_to_display() 和 makehanzi_to_normalized() 负责处理 MakeMeAHanzi 的特殊坐标系；
# normalize_makehanzi_median()、stroke_start_point_from_median() 和 estimate_initial_theta_from_median() 用于从笔画中轴提取起点与初始方向；
# resample_polyline() 和 compute_heading() 用于对轨迹进行稠密采样和方向估计；
# dynamic_state_to_bezier() 与 dynamic_state_to_bbsmg_input() 则作为动态笔触模型和 B-BSMG 之间的首版桥接接口，后续可在标定完成后替换为更精确的映射公式。
