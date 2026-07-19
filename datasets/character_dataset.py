from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.character_features import (
    compute_character_normalization,
    normalize_character_features,
)


class CharacterTrainDataset(Dataset):
    """NPZ dataset containing one variable-length stroke sequence per character."""

    def __init__(
        self,
        npz_path: str,
        coordinate_scale: float = 128.0,
        normalization: Optional[Dict[str, Any]] = None,
    ):
        path = Path(npz_path)
        if not path.exists():
            raise FileNotFoundError(f"Character training NPZ not found: {npz_path}")
        data = np.load(path, allow_pickle=True)
        required = {"inputs", "stroke_masks", "targets"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"Character NPZ is missing keys: {sorted(missing)}")

        raw_inputs = np.asarray(data["inputs"], dtype=np.float32)
        self.stroke_masks = np.asarray(data["stroke_masks"], dtype=np.bool_)
        self.targets = np.asarray(data["targets"], dtype=np.float32)
        self.metadata = (
            np.asarray(data["meta"], dtype=object)
            if "meta" in data.files
            else np.asarray([{} for _ in range(raw_inputs.shape[0])], dtype=object)
        )

        if raw_inputs.ndim != 3:
            raise ValueError(f"inputs must have shape [N,S,D], got {raw_inputs.shape}")
        if self.stroke_masks.shape != raw_inputs.shape[:2]:
            raise ValueError(
                f"stroke_masks {self.stroke_masks.shape} do not match inputs {raw_inputs.shape[:2]}"
            )
        if self.targets.ndim == 3:
            self.targets = self.targets[:, None, :, :]
        if self.targets.ndim != 4 or self.targets.shape[1] != 1:
            raise ValueError(f"targets must have shape [N,1,H,W], got {self.targets.shape}")
        if not (
            raw_inputs.shape[0]
            == self.stroke_masks.shape[0]
            == self.targets.shape[0]
            == len(self.metadata)
        ):
            raise ValueError("Character NPZ arrays have inconsistent sample counts")
        if np.any(self.stroke_masks.sum(axis=1) == 0):
            raise ValueError("Every character sample must contain at least one valid stroke")

        self.input_normalization = normalization or compute_character_normalization(
            raw_inputs,
            self.stroke_masks,
            coordinate_scale=coordinate_scale,
        )
        self.inputs = normalize_character_features(
            raw_inputs,
            self.stroke_masks,
            self.input_normalization,
        )
        print("[CHECK] character inputs shape:", self.inputs.shape)
        print("[CHECK] stroke masks shape:", self.stroke_masks.shape)
        print("[CHECK] character targets shape:", self.targets.shape)

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        meta = self.metadata[index]
        if hasattr(meta, "item"):
            meta = meta.item()
        return {
            "inputs": torch.from_numpy(self.inputs[index]),
            "stroke_mask": torch.from_numpy(self.stroke_masks[index]),
            "targets": torch.from_numpy(self.targets[index]),
            "meta": meta if isinstance(meta, dict) else {},
            "index": int(index),
        }


def collate_character_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "inputs": torch.stack([item["inputs"] for item in batch], dim=0),
        "stroke_mask": torch.stack([item["stroke_mask"] for item in batch], dim=0),
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
