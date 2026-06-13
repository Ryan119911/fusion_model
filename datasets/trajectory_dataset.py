import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

import torch
from torch.utils.data import Dataset

from utils.types import PointState, TrajectoryPoint, CharacterTrajectory, build_character_trajectory


def _to_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    return int(value)


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)


def parse_state(value: Any) -> PointState:
    if isinstance(value, PointState):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        name_map = {
            "down": PointState.DOWN,
            "move": PointState.MOVE,
            "up": PointState.UP,
            "transition": PointState.TRANSITION,
            "落笔": PointState.DOWN,
            "行笔": PointState.MOVE,
            "提笔": PointState.UP,
            "提笔移动": PointState.TRANSITION,
        }
        if text in name_map:
            return name_map[text]
    return PointState.from_value(int(value))


def row_to_point(row: Dict[str, Any]) -> TrajectoryPoint:
    return TrajectoryPoint(
        stroke_id=_to_int(row.get("stroke_id"), 0),
        point_id=_to_int(row.get("point_id"), 0),
        x=_to_float(row.get("x"), 0.0),
        y=_to_float(row.get("y"), 0.0),
        z=_to_float(row.get("z"), 0.0),
        alpha=_to_float(row.get("alpha"), 0.0),
        beta=_to_float(row.get("beta"), 0.0),
        gamma=_to_float(row.get("gamma"), 0.0),
        state=parse_state(row.get("state", 1)),
    )


def _sample_key(row: Dict[str, Any]) -> str:
    for key in ["sample_id", "character", "char_id", "file_stem"]:
        if key in row and row[key] not in (None, ""):
            return str(row[key])
    return "default_sample"


def load_trajectory_csv(csv_path: str) -> List[CharacterTrajectory]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Trajectory CSV not found: {csv_path}")

    grouped_rows: Dict[str, List[Dict[str, Any]]] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            grouped_rows.setdefault(_sample_key(row), []).append(row)

    samples: List[CharacterTrajectory] = []
    for sample_id, rows in grouped_rows.items():
        points = [row_to_point(r) for r in rows]
        character = None
        if len(rows) > 0 and rows[0].get("character"):
            character = str(rows[0]["character"])         
            meta = {"sample_id": sample_id}
        samples.append(build_character_trajectory(points, character=character, meta=meta))
    return samples


def trajectory_to_tensor(sample: CharacterTrajectory) -> Dict[str, torch.Tensor]:
    points = sample.all_points()
    xyz = []
    angles = []
    states = []
    stroke_ids = []
    point_ids = []
    for p in points:
        xyz.append([p.x, p.y, p.z])
        angles.append([p.alpha, p.beta, p.gamma])
        states.append(int(p.state))
        stroke_ids.append(p.stroke_id)
        point_ids.append(p.point_id)

    return {
        "xyz": torch.tensor(xyz, dtype=torch.float32),
        "angles": torch.tensor(angles, dtype=torch.float32),
        "states": torch.tensor(states, dtype=torch.long),
        "stroke_ids": torch.tensor(stroke_ids, dtype=torch.long),
        "point_ids": torch.tensor(point_ids, dtype=torch.long),
    }


class TrajectoryDataset(Dataset):
    def __init__(self, csv_path: str, transform: Optional[Callable[[CharacterTrajectory], Any]] = None):
        self.csv_path = csv_path
        self.samples = load_trajectory_csv(csv_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        if self.transform is not None:
            return self.transform(sample)
        return sample


def pad_sequence_2d(sequences: List[torch.Tensor], pad_value: float = 0.0) -> torch.Tensor:
    max_len = max(seq.shape[0] for seq in sequences)
    feat_dim = sequences[0].shape[1]
    out = torch.full((len(sequences), max_len, feat_dim), pad_value, dtype=sequences[0].dtype)
    for i, seq in enumerate(sequences):
        out[i, : seq.shape[0]] = seq
    return out


def pad_sequence_1d(sequences: List[torch.Tensor], pad_value: int = -1) -> torch.Tensor:
    max_len = max(seq.shape[0] for seq in sequences)
    out = torch.full((len(sequences), max_len), pad_value, dtype=sequences[0].dtype)
    for i, seq in enumerate(sequences):
        out[i, : seq.shape[0]] = seq
    return out


def collate_trajectory_batch(batch: List[CharacterTrajectory]) -> Dict[str, Any]:
    tensor_batch = [trajectory_to_tensor(sample) for sample in batch]
    xyz_list = [item["xyz"] for item in tensor_batch]
    angles_list = [item["angles"] for item in tensor_batch]
    states_list = [item["states"] for item in tensor_batch]
    stroke_ids_list = [item["stroke_ids"] for item in tensor_batch]
    point_ids_list = [item["point_ids"] for item in tensor_batch]
    lengths = torch.tensor([x.shape[0] for x in xyz_list], dtype=torch.long)

    return {
        "xyz": pad_sequence_2d(xyz_list, pad_value=0.0),
        "angles": pad_sequence_2d(angles_list, pad_value=0.0),
        "states": pad_sequence_1d(states_list, pad_value=-1),
        "stroke_ids": pad_sequence_1d(stroke_ids_list, pad_value=-1),
        "point_ids": pad_sequence_1d(point_ids_list, pad_value=-1),
        "lengths": lengths,
        "raw": batch,
    }


if __name__ == "__main__":
    # 简单自检
    example_path = "data/raw/trajectories.csv"
    if Path(example_path).exists():
        dataset = TrajectoryDataset(example_path)
        print(f"Loaded trajectory samples: {len(dataset)}")
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Character: {sample.character}, strokes: {len(sample.strokes)}, points: {len(sample.all_points())}")

# 使用说明：该模块默认读取一个包含表头的 CSV 文件，至少支持字段 stroke_id、point_id、x、y、z、alpha、beta、gamma、state；
# 如果 CSV 里还包含 character、sample_id、char_id 或 file_stem，则会优先用这些字段把整字样本分组。state 既支持数字编码（0/1/2/3），也支持中英文名称（如 down、move、落笔、提笔移动）。
# 其中 load_trajectory_csv() 用于把 CSV 解析为 CharacterTrajectory 列表；
# TrajectoryDataset 提供 PyTorch Dataset 接口；
# collate_trajectory_batch() 会自动把不同长度的整字轨迹补齐成批量张量，方便后续训练、拟合和优化模块直接调用。
