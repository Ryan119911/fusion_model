import csv
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from models.geometry import resample_character_trajectory
from utils.types import (
    CharacterTrajectory,
    PointState,
    TrajectoryPoint,
    build_character_trajectory,
)


def _to_int(value: Any, default: int = 0) -> int:
    return default if value in (None, "") else int(float(value))


def _to_float(value: Any, default: float = 0.0) -> float:
    return default if value in (None, "") else float(value)


def parse_state(value: Any) -> PointState:
    if isinstance(value, PointState):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        names = {
            "down": PointState.DOWN,
            "move": PointState.MOVE,
            "up": PointState.UP,
            "transition": PointState.TRANSITION,
        }
        if text in names:
            return names[text]
    return PointState.from_value(int(value))


def row_to_point(
    row: Dict[str, Any],
    timestamp_column: Optional[str] = None,
) -> TrajectoryPoint:
    timestamp = None
    if timestamp_column and row.get(timestamp_column) not in (None, ""):
        timestamp = float(row[timestamp_column])
    return TrajectoryPoint(
        stroke_id=_to_int(row.get("stroke_id")),
        point_id=_to_int(row.get("point_id")),
        x=_to_float(row.get("x")),
        y=_to_float(row.get("y")),
        z=_to_float(row.get("z")),
        alpha=_to_float(row.get("alpha")),
        beta=_to_float(row.get("beta")),
        gamma=_to_float(row.get("gamma")),
        state=parse_state(row.get("state", 1)),
        timestamp=timestamp,
    )


def _sample_key(row: Dict[str, Any]) -> str:
    for key in ("sample_id", "character", "char_id", "file_stem"):
        if row.get(key) not in (None, ""):
            return str(row[key])
    return "default_sample"


def _validate_sample(sample: CharacterTrajectory) -> None:
    points = sample.all_points()
    if not points:
        raise ValueError(f"Trajectory sample {sample.meta.get('sample_id')} is empty")
    for point in points:
        values = (point.x, point.y, point.z, point.alpha, point.beta, point.gamma)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(
                f"Non-finite trajectory value in sample={sample.meta.get('sample_id')}, "
                f"stroke={point.stroke_id}, point={point.point_id}"
            )
        if int(point.state) not in (0, 1, 2, 3):
            raise ValueError(f"Unsupported point state: {point.state}")


def load_trajectory_csv(
    csv_path: str,
    timestamp_column: Optional[str] = None,
    validate: bool = True,
    points_per_stroke: Optional[int] = None,
) -> List[CharacterTrajectory]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Trajectory CSV not found: {csv_path}")

    grouped_rows: Dict[str, List[Dict[str, Any]]] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"stroke_id", "point_id", "x", "y", "z", "state"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Trajectory CSV is missing columns: {sorted(missing)}")
        for row in reader:
            grouped_rows.setdefault(_sample_key(row), []).append(row)

    samples: List[CharacterTrajectory] = []
    for sample_id, rows in grouped_rows.items():
        first = rows[0]
        metadata = {
            key: value
            for key, value in first.items()
            if key not in {
                "stroke_id", "point_id", "x", "y", "z",
                "alpha", "beta", "gamma", "state",
            }
        }
        metadata["sample_id"] = sample_id
        sample = build_character_trajectory(
            [row_to_point(row, timestamp_column) for row in rows],
            character=str(first["character"]) if first.get("character") else None,
            meta=metadata,
        )
        if validate:
            _validate_sample(sample)
        if points_per_stroke is not None:
            sample = resample_character_trajectory(sample, points_per_stroke)
        samples.append(sample)
    return samples


def trajectory_to_tensor(sample: CharacterTrajectory) -> Dict[str, torch.Tensor]:
    points = sample.all_points()
    timestamps = [
        float(point.timestamp) if point.timestamp is not None else float("nan")
        for point in points
    ]
    return {
        "xyz": torch.tensor([[p.x, p.y, p.z] for p in points], dtype=torch.float32),
        "angles": torch.tensor(
            [[p.alpha, p.beta, p.gamma] for p in points], dtype=torch.float32
        ),
        "states": torch.tensor([int(p.state) for p in points], dtype=torch.long),
        "stroke_ids": torch.tensor([p.stroke_id for p in points], dtype=torch.long),
        "point_ids": torch.tensor([p.point_id for p in points], dtype=torch.long),
        "timestamps": torch.tensor(timestamps, dtype=torch.float64),
    }


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        transform: Optional[Callable[[CharacterTrajectory], Any]] = None,
        timestamp_column: Optional[str] = None,
        validate: bool = True,
        points_per_stroke: Optional[int] = None,
    ):
        self.samples = load_trajectory_csv(
            csv_path,
            timestamp_column=timestamp_column,
            validate=validate,
            points_per_stroke=points_per_stroke,
        )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        return self.transform(sample) if self.transform is not None else sample


def _pad_2d(sequences: List[torch.Tensor], value: float = 0.0) -> torch.Tensor:
    max_len = max(sequence.shape[0] for sequence in sequences)
    output = torch.full(
        (len(sequences), max_len, sequences[0].shape[1]),
        value,
        dtype=sequences[0].dtype,
    )
    for index, sequence in enumerate(sequences):
        output[index, : sequence.shape[0]] = sequence
    return output


def _pad_1d(sequences: List[torch.Tensor], value: float) -> torch.Tensor:
    max_len = max(sequence.shape[0] for sequence in sequences)
    output = torch.full(
        (len(sequences), max_len), value, dtype=sequences[0].dtype
    )
    for index, sequence in enumerate(sequences):
        output[index, : sequence.shape[0]] = sequence
    return output


def collate_trajectory_batch(batch: List[CharacterTrajectory]) -> Dict[str, Any]:
    tensors = [trajectory_to_tensor(sample) for sample in batch]
    return {
        "xyz": _pad_2d([item["xyz"] for item in tensors]),
        "angles": _pad_2d([item["angles"] for item in tensors]),
        "states": _pad_1d([item["states"] for item in tensors], -1),
        "stroke_ids": _pad_1d([item["stroke_ids"] for item in tensors], -1),
        "point_ids": _pad_1d([item["point_ids"] for item in tensors], -1),
        "timestamps": _pad_1d([item["timestamps"] for item in tensors], float("nan")),
        "lengths": torch.tensor(
            [item["xyz"].shape[0] for item in tensors], dtype=torch.long
        ),
        "raw": batch,
    }
