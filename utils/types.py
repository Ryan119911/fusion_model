from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Dict, Any, Tuple

class PointState(IntEnum):
    DOWN = 0
    MOVE = 1
    UP = 2
    TRANSITION = 3

    @classmethod
    def from_value(cls, value: int) -> "PointState":
        return cls(int(value))

    def to_name(self) -> str:
        mapping = {
            PointState.DOWN: "down",
            PointState.MOVE: "move",
            PointState.UP: "up",
            PointState.TRANSITION: "transition",
        }
        return mapping[self]

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

    def as_tuple(self) -> Tuple[float, float, float, float, float, float]:
        return (self.x, self.y, self.z, self.alpha, self.beta, self.gamma)

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
        }

@dataclass
class StrokeTrajectory:
    stroke_id: int
    points: List[TrajectoryPoint] = field(default_factory=list)

    def sorted_points(self) -> List[TrajectoryPoint]:
        return sorted(self.points, key=lambda p: p.point_id)

    def states(self) -> List[PointState]:
        return [p.state for p in self.sorted_points()]

@dataclass
class CharacterTrajectory:
    character: Optional[str] = None
    strokes: List[StrokeTrajectory] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def sorted_strokes(self) -> List[StrokeTrajectory]:
        return sorted(self.strokes, key=lambda s: s.stroke_id)

    def all_points(self) -> List[TrajectoryPoint]:
        pts: List[TrajectoryPoint] = []
        for stroke in self.sorted_strokes():
            pts.extend(stroke.sorted_points())
        return pts

@dataclass
class MakeHanziDictionaryRecord:
    character: str
    pinyin: str = ""
    definition: Optional[str] = None
    decomposition: Optional[str] = None
    radical: Optional[str] = None
    matches: Optional[List[Any]] = None
    etymology: Optional[Dict[str, Any]] = None

@dataclass
class MakeHanziGraphicsRecord:
    character: str
    strokes: List[str] = field(default_factory=list)
    medians: List[List[Tuple[int, int]]] = field(default_factory=list)

@dataclass
class BBSMGInput:
    h: float
    alpha: float
    beta: float
    x0: float
    y0: float

    def as_list(self) -> List[float]:
        return [self.h, self.alpha, self.beta, self.x0, self.y0]

@dataclass
class BBSMGSample:
    inputs: BBSMGInput
    target_image_path: Optional[str] = None
    target_mask_path: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DynamicBrushState:
    x: float
    y: float
    z: float
    w: float
    d: float
    o: float
    theta: float

    def as_list(self) -> List[float]:
        return [self.x, self.y, self.z, self.w, self.d, self.o, self.theta]

def group_points_by_stroke(points: List[TrajectoryPoint]) -> List[StrokeTrajectory]:
    bucket: Dict[int, List[TrajectoryPoint]] = {}
    for point in points:
        bucket.setdefault(point.stroke_id, []).append(point)
    strokes = [StrokeTrajectory(stroke_id=sid, points=pts) for sid, pts in bucket.items()]
    return sorted(strokes, key=lambda s: s.stroke_id)

def build_character_trajectory(points: List[TrajectoryPoint], character: Optional[str] = None, meta: Optional[Dict[str, Any]] = None) -> CharacterTrajectory:
    return CharacterTrajectory(character=character, strokes=group_points_by_stroke(points), meta=meta or {})

# 字段说明：PointState 统一管理 0=落笔、1=行笔、2=提笔、3=提笔移动 的编码；
# TrajectoryPoint、StrokeTrajectory 和 CharacterTrajectory 用于表达整字级轨迹及其按笔画分组后的结构；
# MakeHanziDictionaryRecord 与 MakeHanziGraphicsRecord 对应 MakeMeAHanzi 的两类 JSON 行记录；
# BBSMGInput 和 BBSMGSample 用于神经生成器训练样本组织；DynamicBrushState 则承载动态笔触模型中的 7 维状态。
# 后续的数据集读取、正向渲染、参数映射和轨迹优化模块都会直接复用这些类型。