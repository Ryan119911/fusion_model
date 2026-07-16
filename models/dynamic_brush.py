from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.types import (
    CharacterTrajectory,
    DynamicBrushState,
    StrokeTrajectory,
    TrajectoryPoint,
)


MODEL_VERSION = 1


@dataclass
class PolyFit1D:
    coeffs: List[float]

    def __call__(self, value: float) -> float:
        return float(sum(coefficient * value ** index for index, coefficient in enumerate(self.coeffs)))


@dataclass
class DynamicBrushParams:
    mode: str = "disabled"
    kw: float = 0.02
    kd: float = 0.02
    dt: float = 0.01
    width_fn: Optional[PolyFit1D] = None
    drag_fn: Optional[PolyFit1D] = None
    offset_fn: Optional[PolyFit1D] = None
    snap_clip_min: float = 0.0


def default_width_fn() -> PolyFit1D:
    return PolyFit1D([0.0, 0.8])


def default_drag_fn() -> PolyFit1D:
    return PolyFit1D([0.0, 1.2])


def default_offset_fn() -> PolyFit1D:
    return PolyFit1D([0.0, 0.5, -0.02, 0.0005])


class DynamicBrushModel:
    def __init__(self, params: Optional[DynamicBrushParams] = None):
        self.params = params or DynamicBrushParams()
        if self.params.mode not in {"disabled", "heuristic", "calibrated"}:
            raise ValueError(f"Unsupported dynamic brush mode: {self.params.mode}")
        if self.params.mode != "disabled":
            self.params.width_fn = self.params.width_fn or default_width_fn()
            self.params.drag_fn = self.params.drag_fn or default_drag_fn()
            self.params.offset_fn = self.params.offset_fn or default_offset_fn()

    def width(self, z: float) -> float:
        if self.params.mode == "disabled":
            return 0.0
        return max(float(self.params.width_fn(z)), 0.0)

    def drag(self, z: float) -> float:
        if self.params.mode == "disabled":
            return 0.0
        return max(float(self.params.drag_fn(z)), 0.0)

    def offset(self, z: float) -> float:
        if self.params.mode == "disabled":
            return 0.0
        return max(float(self.params.offset_fn(z)), self.params.snap_clip_min)

    def init_state(
        self,
        first_point: TrajectoryPoint,
        theta0: Optional[float] = None,
        reset_brush: bool = True,
    ) -> DynamicBrushState:
        theta = float(first_point.gamma if theta0 is None else theta0)
        return DynamicBrushState(
            x=float(first_point.x),
            y=float(first_point.y),
            z=float(first_point.z),
            w=0.0 if reset_brush else self.width(first_point.z),
            d=0.0 if reset_brush else self.drag(first_point.z),
            o=0.0 if reset_brush else self.offset(first_point.z),
            theta=theta,
        )

    def step(
        self,
        previous: DynamicBrushState,
        previous_point: TrajectoryPoint,
        next_point: TrajectoryPoint,
    ) -> DynamicBrushState:
        dx = float(next_point.x - previous_point.x)
        dy = float(next_point.y - previous_point.y)
        heading = math.atan2(dy, dx) if abs(dx) + abs(dy) > 1e-12 else previous.theta
        if self.params.mode == "disabled":
            return DynamicBrushState(
                x=float(next_point.x),
                y=float(next_point.y),
                z=float(next_point.z),
                w=0.0,
                d=0.0,
                o=0.0,
                theta=heading,
            )
        width = previous.w * self.params.kw + self.width(next_point.z) * (1.0 - self.params.kw)
        drag = previous.d * self.params.kd + self.drag(next_point.z) * (1.0 - self.params.kd)
        offset = self.offset(next_point.z)
        return DynamicBrushState(
            x=float(next_point.x),
            y=float(next_point.y),
            z=float(next_point.z),
            w=width,
            d=drag,
            o=offset,
            theta=heading,
        )

    def simulate_stroke(
        self,
        stroke: StrokeTrajectory,
        theta0: Optional[float] = None,
        reset_brush: bool = True,
    ) -> List[DynamicBrushState]:
        points = stroke.sorted_points()
        if not points:
            return []
        states = [self.init_state(points[0], theta0, reset_brush)]
        for index in range(1, len(points)):
            states.append(self.step(states[-1], points[index - 1], points[index]))
        return states

    def simulate_character(
        self,
        sample: CharacterTrajectory,
        reset_each_stroke: bool = True,
    ) -> Dict[int, List[DynamicBrushState]]:
        return {
            stroke.stroke_id: self.simulate_stroke(stroke, reset_brush=reset_each_stroke)
            for stroke in sample.sorted_strokes()
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": MODEL_VERSION,
            "mode": self.params.mode,
            "kw": self.params.kw,
            "kd": self.params.kd,
            "dt": self.params.dt,
            "snap_clip_min": self.params.snap_clip_min,
            "width_coeffs": None if self.params.width_fn is None else self.params.width_fn.coeffs,
            "drag_coeffs": None if self.params.drag_fn is None else self.params.drag_fn.coeffs,
            "offset_coeffs": None if self.params.offset_fn is None else self.params.offset_fn.coeffs,
        }

    def save_json(self, path: str) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DynamicBrushModel":
        mode = str(data.get("mode", "calibrated"))
        return cls(
            DynamicBrushParams(
                mode=mode,
                kw=float(data.get("kw", 0.02)),
                kd=float(data.get("kd", 0.02)),
                dt=float(data.get("dt", 0.01)),
                snap_clip_min=float(data.get("snap_clip_min", 0.0)),
                width_fn=_poly_from_data(data.get("width_coeffs")),
                drag_fn=_poly_from_data(data.get("drag_coeffs")),
                offset_fn=_poly_from_data(data.get("offset_coeffs")),
            )
        )

    @classmethod
    def load_json(cls, path: str) -> "DynamicBrushModel":
        with open(path, "r", encoding="utf-8") as file:
            return cls.from_dict(json.load(file))


def _poly_from_data(coefficients: Any) -> Optional[PolyFit1D]:
    return None if coefficients is None else PolyFit1D([float(value) for value in coefficients])


def build_dynamic_brush(config: Any) -> DynamicBrushModel:
    mode = str(config.mode)
    if mode == "calibrated":
        model = DynamicBrushModel.load_json(config.calibration_path)
        model.params.mode = "calibrated"
        return model
    return DynamicBrushModel(
        DynamicBrushParams(
            mode=mode,
            kw=float(config.kw),
            kd=float(config.kd),
            dt=float(config.dt),
            snap_clip_min=float(config.snap_clip_min),
            width_fn=default_width_fn() if mode == "heuristic" else None,
            drag_fn=default_drag_fn() if mode == "heuristic" else None,
            offset_fn=default_offset_fn() if mode == "heuristic" else None,
        )
    )


def fit_poly_from_pairs(
    xs: List[float], ys: List[float], degree: int
) -> PolyFit1D:
    if len(xs) != len(ys) or not xs:
        raise ValueError("Calibration x/y pairs must be non-empty and equally sized")
    import numpy as np

    degree = min(int(degree), len(xs) - 1)
    descending = np.polyfit(
        np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), degree
    )
    return PolyFit1D(list(reversed([float(value) for value in descending])))


def build_dynamic_brush_from_calibration(
    width_pairs: Tuple[List[float], List[float]],
    drag_pairs: Tuple[List[float], List[float]],
    offset_pairs: Tuple[List[float], List[float]],
    kw: float = 0.02,
    kd: float = 0.02,
    dt: float = 0.01,
    width_degree: int = 2,
    drag_degree: int = 2,
    offset_degree: int = 3,
) -> DynamicBrushModel:
    return DynamicBrushModel(
        DynamicBrushParams(
            mode="calibrated",
            kw=kw,
            kd=kd,
            dt=dt,
            width_fn=fit_poly_from_pairs(*width_pairs, degree=width_degree),
            drag_fn=fit_poly_from_pairs(*drag_pairs, degree=drag_degree),
            offset_fn=fit_poly_from_pairs(*offset_pairs, degree=offset_degree),
        )
    )
