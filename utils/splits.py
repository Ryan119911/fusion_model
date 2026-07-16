import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


MANIFEST_VERSION = 1


def _meta_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "item"):
        item = value.item()
        if isinstance(item, dict):
            return item
    return {}


def group_value(meta: Dict[str, Any], key: str, index: int) -> str:
    value = meta.get(key)
    if value not in (None, ""):
        return str(value)
    for fallback in ("sample_id", "trajectory_sample_index", "character"):
        value = meta.get(fallback)
        if value not in (None, ""):
            return f"{fallback}:{value}"
    return f"row:{index}"


def build_split(
    meta: Optional[Sequence[Any]],
    length: int,
    val_ratio: float,
    seed: int,
    strategy: str = "group",
    group_key: str = "sample_id",
) -> Tuple[List[int], List[int], Dict[str, Any]]:
    if length < 1:
        raise ValueError("Cannot split an empty dataset")
    val_count = 0 if length == 1 or val_ratio <= 0 else max(1, int(round(length * val_ratio)))
    rng = random.Random(seed)

    if strategy == "random":
        indices = list(range(length))
        rng.shuffle(indices)
        val_indices = sorted(indices[:val_count])
        train_indices = sorted(indices[val_count:])
        train_groups = [f"row:{i}" for i in train_indices]
        val_groups = [f"row:{i}" for i in val_indices]
    elif strategy == "group":
        if meta is None or len(meta) != length:
            raise ValueError("Grouped splitting requires NPZ metadata for every sample")
        buckets: Dict[str, List[int]] = {}
        for index, raw in enumerate(meta):
            key = group_value(_meta_dict(raw), group_key, index)
            buckets.setdefault(key, []).append(index)
        groups = list(buckets)
        rng.shuffle(groups)
        groups.sort(key=lambda key: len(buckets[key]), reverse=True)
        selected: List[str] = []
        selected_count = 0
        for key in groups:
            if selected_count >= val_count and selected:
                break
            selected.append(key)
            selected_count += len(buckets[key])
        val_group_set = set(selected)
        val_indices = sorted(i for key in selected for i in buckets[key])
        train_indices = sorted(i for key, values in buckets.items() if key not in val_group_set for i in values)
        train_groups = sorted(key for key in buckets if key not in val_group_set)
        val_groups = sorted(val_group_set)
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")

    if not train_indices:
        raise ValueError("Split produced an empty training set")
    manifest = {
        "version": MANIFEST_VERSION,
        "dataset_length": length,
        "strategy": strategy,
        "group_key": group_key,
        "seed": seed,
        "requested_val_ratio": val_ratio,
        "actual_val_ratio": len(val_indices) / length,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "train_groups": train_groups,
        "val_groups": val_groups,
    }
    return train_indices, val_indices, manifest


def save_manifest(path: str, manifest: Dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)


def load_manifest(path: str, expected_length: Optional[int] = None) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        manifest = json.load(file)
    if manifest.get("version") != MANIFEST_VERSION:
        raise ValueError(f"Unsupported split manifest version: {manifest.get('version')}")
    if expected_length is not None and manifest.get("dataset_length") != expected_length:
        raise ValueError(
            f"Split manifest dataset_length={manifest.get('dataset_length')} "
            f"does not match dataset length {expected_length}"
        )
    train = set(manifest["train_indices"])
    val = set(manifest["val_indices"])
    if train & val:
        raise ValueError("Split manifest contains train/validation index overlap")
    if manifest.get("strategy") == "group":
        overlap = set(manifest.get("train_groups", [])) & set(manifest.get("val_groups", []))
        if overlap:
            raise ValueError(f"Split manifest contains group leakage: {sorted(overlap)[:5]}")
    return manifest
