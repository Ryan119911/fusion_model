# 中文注释：本文件定义轨迹、MakeHanzi 字形和动态笔刷等核心数据结构。
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Dict, Any, Tuple

# 中文注释：枚举单个轨迹点的落笔、移动、抬笔和过渡状态。
class PointState(IntEnum):
    DOWN = 0
    MOVE = 1
    UP = 2
    TRANSITION = 3

    # 中文注释：将整数或可转整数的值转换为 PointState。
    @classmethod
    def from_value(cls, value: int) -> "PointState":
        return cls(int(value))

    # 中文注释：返回状态的可读字符串名称。
    def to_name(self) -> str:
        mapping = {
            PointState.DOWN: "down",
            PointState.MOVE: "move",
            PointState.UP: "up",
            PointState.TRANSITION: "transition",
        }
        return mapping[self]

# 中文注释：表示一个带位置、姿态和状态的轨迹采样点。
@dataclass
class TrajectoryPoint:
    stroke_id: int
    point_id: int
    x: float
    y: float
    z: float
    alpha: float
    beta: float
    gamma: float
    state: PointState
    timestamp: Optional[float] = None

    # 中文注释：将点的位置和姿态转换为固定顺序元组。
    def as_tuple(self) -> Tuple[float, float, float, float, float, float]:
        return (self.x, self.y, self.z, self.alpha, self.beta, self.gamma)

    # 中文注释：将轨迹点转换为便于序列化的字典。
    def as_dict(self) -> Dict[str, Any]:
        return {
            "stroke_id": self.stroke_id,
            "point_id": self.point_id,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "state": int(self.state),
            "timestamp": self.timestamp,
        }

# 中文注释：表示同一笔画内的全部轨迹点。
@dataclass
class StrokeTrajectory:
    stroke_id: int
    points: List[TrajectoryPoint] = field(default_factory=list)

    # 中文注释：按 point_id 排序笔画内的采样点。
    def sorted_points(self) -> List[TrajectoryPoint]:
        return sorted(self.points, key=lambda p: p.point_id)

    # 中文注释：提取笔画内每个采样点的状态序列。
    def states(self) -> List[PointState]:
        return [p.state for p in self.sorted_points()]

# 中文注释：表示一个字符由多条笔画轨迹组成的完整轨迹。
@dataclass
class CharacterTrajectory:
    character: Optional[str] = None
    strokes: List[StrokeTrajectory] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    # 中文注释：按 stroke_id 排序字符中的笔画。
    def sorted_strokes(self) -> List[StrokeTrajectory]:
        return sorted(self.strokes, key=lambda s: s.stroke_id)

    # 中文注释：按笔画顺序展开字符中的所有轨迹点。
    def all_points(self) -> List[TrajectoryPoint]:
        pts: List[TrajectoryPoint] = []
        for stroke in self.sorted_strokes():
            pts.extend(stroke.sorted_points())
        return pts

# 中文注释：保存 MakeMeAHanzi 字典记录中的字符、定义和拼音。
@dataclass
class MakeHanziDictionaryRecord:
    character: str
    pinyin: str = ""
    definition: Optional[str] = None
    decomposition: Optional[str] = None
    radical: Optional[str] = None
    matches: Optional[List[Any]] = None
    etymology: Optional[Dict[str, Any]] = None

# 中文注释：保存 MakeMeAHanzi 图形记录中的 SVG 路径和中线。
@dataclass
class MakeHanziGraphicsRecord:
    character: str
    strokes: List[str] = field(default_factory=list)
    medians: List[List[Tuple[int, int]]] = field(default_factory=list)

# 中文注释：封装 B-BSMG 的十维输入特征。
@dataclass
class BBSMGInput:
    h: float
    alpha: float
    beta: float
    x0: float
    y0: float

    # 中文注释：按模型输入要求把字段展开为列表。
    def as_list(self) -> List[float]:
        return [self.h, self.alpha, self.beta, self.x0, self.y0]

# 中文注释：封装 B-BSMG 单个训练样本的输入和目标图像。
@dataclass
class BBSMGSample:
    inputs: BBSMGInput
    target_image_path: Optional[str] = None
    target_mask_path: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

# 中文注释：保存动态笔刷在某一时刻的位置、姿态、宽度和偏移状态。
@dataclass
class DynamicBrushState:
    x: float
    y: float
    z: float
    w: float
    d: float
    o: float
    theta: float

    # 中文注释：按模型输入要求把字段展开为列表。
    def as_list(self) -> List[float]:
        return [self.x, self.y, self.z, self.w, self.d, self.o, self.theta]

# 中文注释：将无序轨迹点按 stroke_id 分组为笔画。
def group_points_by_stroke(points: List[TrajectoryPoint]) -> List[StrokeTrajectory]:
    bucket: Dict[int, List[TrajectoryPoint]] = {}
    for point in points:
        bucket.setdefault(point.stroke_id, []).append(point)
    strokes = [StrokeTrajectory(stroke_id=sid, points=pts) for sid, pts in bucket.items()]
    return sorted(strokes, key=lambda s: s.stroke_id)

# 中文注释：根据轨迹点列表构建字符级轨迹对象。
def build_character_trajectory(points: List[TrajectoryPoint], character: Optional[str] = None, meta: Optional[Dict[str, Any]] = None) -> CharacterTrajectory:
    return CharacterTrajectory(character=character, strokes=group_points_by_stroke(points), meta=meta or {})

# 字段说明：PointState 统一管理 0=落笔、1=行笔、2=提笔、3=提笔移动 的编码；
# TrajectoryPoint、StrokeTrajectory 和 CharacterTrajectory 用于表达整字级轨迹及其按笔画分组后的结构；
# MakeHanziDictionaryRecord 与 MakeHanziGraphicsRecord 对应 MakeMeAHanzi 的两类 JSON 行记录；
# BBSMGInput 和 BBSMGSample 用于神经生成器训练样本组织；DynamicBrushState 则承载动态笔触模型中的 7 维状态。
# 后续的数据集读取、正向渲染、参数映射和轨迹优化模块都会直接复用这些类型。
