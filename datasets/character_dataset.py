from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.character_features import SPATIAL_CHANNEL_NAMES


CHARACTER_DATA_FORMAT = "character_spatial_v2"


class CharacterTrainDataset(Dataset):
    """Whole-character spatial maps paired with complete target glyphs."""

    def __init__(self, npz_path: str):
        path = Path(npz_path)
        if not path.exists():
            raise FileNotFoundError(f"Character training NPZ not found: {npz_path}")
        data = np.load(path, allow_pickle=True)
        required = {"inputs", "targets", "format_version"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(
                f"Character NPZ is missing keys {sorted(missing)}. "
                "Rebuild it with build_character_pairs.py."
            )

        data_format = str(np.asarray(data["format_version"]).item())
        if data_format != CHARACTER_DATA_FORMAT:
            raise ValueError(
                f"Unsupported character NPZ format {data_format!r}; expected "
                f"{CHARACTER_DATA_FORMAT!r}. Transformer-era NPZ files are incompatible "
                "and must be rebuilt."
            )

        self.inputs = np.asarray(data["inputs"], dtype=np.float16)
        self.targets = np.asarray(data["targets"], dtype=np.float16)
        self.metadata = (
            np.asarray(data["meta"], dtype=object)
            if "meta" in data.files
            else np.asarray([{} for _ in range(self.inputs.shape[0])], dtype=object)
        )
        self.channel_names = (
            tuple(str(value) for value in data["channel_names"].tolist())
            if "channel_names" in data.files
            else tuple(SPATIAL_CHANNEL_NAMES)
        )

        if self.inputs.ndim != 4:
            raise ValueError(f"inputs must have shape [N,C,H,W], got {self.inputs.shape}")
        if self.inputs.shape[1] != len(SPATIAL_CHANNEL_NAMES):
            raise ValueError(
                f"Expected {len(SPATIAL_CHANNEL_NAMES)} trajectory channels, "
                f"got {self.inputs.shape[1]}"
            )
        if self.channel_names != tuple(SPATIAL_CHANNEL_NAMES):
            raise ValueError(
                f"Spatial channel schema mismatch: {self.channel_names} != "
                f"{SPATIAL_CHANNEL_NAMES}"
            )
        if self.targets.ndim == 3:
            self.targets = self.targets[:, None, :, :]
        if self.targets.ndim != 4 or self.targets.shape[1] != 1:
            raise ValueError(f"targets must have shape [N,1,H,W], got {self.targets.shape}")
        if self.inputs.shape[0] != self.targets.shape[0] or len(self.metadata) != len(self.inputs):
            raise ValueError("Character NPZ arrays have inconsistent sample counts")
        if self.inputs.shape[-2:] != self.targets.shape[-2:]:
            raise ValueError("Input maps and target images must share the same spatial size")
        if not np.isfinite(self.inputs).all() or not np.isfinite(self.targets).all():
            raise ValueError("Character NPZ contains NaN or Inf")

        print("[CHECK] spatial trajectory inputs shape:", self.inputs.shape)
        print("[CHECK] whole-character targets shape:", self.targets.shape)
        print("[CHECK] input channels:", ", ".join(self.channel_names))

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        meta = self.metadata[index]
        if hasattr(meta, "item"):
            meta = meta.item()
        return {
            "inputs": torch.from_numpy(self.inputs[index]).float(),
            "targets": torch.from_numpy(self.targets[index]).float(),
            "meta": meta if isinstance(meta, dict) else {},
            "index": int(index),
        }


def collate_character_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "inputs": torch.stack([item["inputs"] for item in batch], dim=0),
        "targets": torch.stack([item["targets"] for item in batch], dim=0),
        "meta": [item["meta"] for item in batch],
        "indices": [item["index"] for item in batch],
    }


def deterministic_split_indices(
    length: int,
    val_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    if length < 1:
        raise ValueError("Dataset is empty")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must satisfy 0 <= val_ratio < 1")
    if length == 1 or val_ratio == 0.0:
        return list(range(length)), []
    val_len = max(1, int(length * val_ratio))
    val_len = min(val_len, length - 1)
    permutation = torch.randperm(length, generator=torch.Generator().manual_seed(seed)).tolist()
    return permutation[val_len:], permutation[:val_len]
