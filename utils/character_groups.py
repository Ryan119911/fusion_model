import random
from collections import OrderedDict
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

try:
    import torch
    from torch.utils.data import Sampler
except ModuleNotFoundError:
    torch = None
    Sampler = object

from utils.splits import group_value


def metadata_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "item"):
        item = value.item()
        if isinstance(item, dict):
            return item
    return {}


def group_indices(
    metadata: Sequence[Any],
    indices: Optional[Sequence[int]] = None,
    group_key: str = "sample_id",
) -> List[Tuple[str, List[int]]]:
    buckets: "OrderedDict[str, List[int]]" = OrderedDict()
    selected = range(len(metadata)) if indices is None else indices
    for index in selected:
        meta = metadata_dict(metadata[index])
        key = group_value(meta, group_key, index)
        buckets.setdefault(key, []).append(int(index))
    for values in buckets.values():
        values.sort(key=lambda index: (
            int(metadata_dict(metadata[index]).get("stroke_order", index)),
            index,
        ))
    return list(buckets.items())


def validate_group_consistency(
    metadata: Sequence[Any],
    group_key: str = "sample_id",
) -> List[str]:
    errors: List[str] = []
    for key, indices in group_indices(metadata, group_key=group_key):
        items = [metadata_dict(metadata[index]) for index in indices]
        characters = {str(item.get("character")) for item in items}
        if len(characters) != 1:
            errors.append(f"{key}: multiple characters {sorted(characters)}")
        real_flags = {bool(item.get("used_real_image")) for item in items}
        if len(real_flags) != 1:
            errors.append(f"{key}: mixes real and synthetic targets")
        if real_flags == {True}:
            sources = {str(item.get("image_path")) for item in items}
            if len(sources) != 1 or sources == {"None"}:
                errors.append(f"{key}: real strokes use different or missing image_path values")
            transforms = {repr(item.get("canvas_transform")) for item in items}
            if len(transforms) != 1:
                errors.append(f"{key}: strokes use different canvas transforms")
        orders = [item.get("stroke_order") for item in items]
        if len(set(orders)) != len(orders):
            errors.append(f"{key}: duplicate stroke_order values")
        expected = {int(item.get("num_strokes_in_traj", len(items))) for item in items}
        if len(expected) != 1 or next(iter(expected)) != len(items):
            errors.append(
                f"{key}: contains {len(items)} records but num_strokes_in_traj={sorted(expected)}"
            )
    return errors


def validate_character_target_mapping(
    metadata: Sequence[Any],
    target_indices: Sequence[Any],
    target_count: int,
    group_key: str = "sample_id",
) -> List[str]:
    errors: List[str] = []
    if len(target_indices) != len(metadata):
        return [
            "character_target_indices length "
            f"{len(target_indices)} does not match metadata length {len(metadata)}"
        ]
    if target_count < 1:
        return ["character_targets is empty"]

    parsed_indices: List[Optional[int]] = []
    for position, value in enumerate(target_indices):
        try:
            target_index = int(value)
        except (TypeError, ValueError):
            errors.append(f"record {position}: invalid character_target_index {value!r}")
            parsed_indices.append(None)
            continue
        if not 0 <= target_index < target_count:
            errors.append(
                f"record {position}: character_target_index {target_index} is outside "
                f"[0, {target_count})"
            )
        parsed_indices.append(target_index)

    target_owners: Dict[int, str] = {}
    for key, indices in group_indices(metadata, group_key=group_key):
        group_targets = {
            parsed_indices[index]
            for index in indices
            if parsed_indices[index] is not None
        }
        if len(group_targets) != 1:
            errors.append(f"{key}: maps to character targets {sorted(group_targets)}")
            continue
        target_index = next(iter(group_targets))
        if target_index is None or not 0 <= target_index < target_count:
            continue
        previous_owner = target_owners.setdefault(target_index, key)
        if previous_owner != key:
            errors.append(
                f"character target {target_index} is shared by groups "
                f"{previous_owner!r} and {key!r}"
            )
        for index in indices:
            recorded = metadata_dict(metadata[index]).get("character_target_index")
            if recorded is None:
                errors.append(f"{key}: metadata record {index} has no character_target_index")
                continue
            try:
                recorded_index = int(recorded)
            except (TypeError, ValueError):
                errors.append(
                    f"{key}: metadata record {index} has invalid "
                    f"character_target_index {recorded!r}"
                )
                continue
            if recorded_index != target_index:
                errors.append(
                    f"{key}: metadata record {index} maps to {recorded_index}, "
                    f"array maps to {target_index}"
                )
    return errors


class CharacterBatchSampler(Sampler):
    def __init__(
        self,
        metadata: Sequence[Any],
        indices: Sequence[int],
        max_batch_size: int,
        group_key: str = "sample_id",
        shuffle: bool = False,
        seed: int = 42,
    ):
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        self.groups = [values for _, values in group_indices(metadata, indices, group_key)]
        self.max_batch_size = max_batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _batches(self) -> List[List[int]]:
        groups = [list(group) for group in self.groups]
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(groups)
        batches: List[List[int]] = []
        batch: List[int] = []
        for group in groups:
            if batch and len(batch) + len(group) > self.max_batch_size:
                batches.append(batch)
                batch = []
            batch.extend(group)
            if len(batch) >= self.max_batch_size:
                batches.append(batch)
                batch = []
        if batch:
            batches.append(batch)
        return batches

    def __iter__(self) -> Iterator[List[int]]:
        yield from self._batches()

    def __len__(self) -> int:
        return len(self._batches())


def fuse_character_tensors(
    images: Any,
    metadata: Sequence[Any],
    group_key: str = "sample_id",
) -> Tuple[Any, List[str]]:
    if torch is None:
        raise RuntimeError("PyTorch is required to fuse character tensors")
    if images.shape[0] != len(metadata):
        raise ValueError("Image batch and metadata length differ")
    positions: "OrderedDict[str, List[int]]" = OrderedDict()
    for position, value in enumerate(metadata):
        key = group_value(metadata_dict(value), group_key, position)
        positions.setdefault(key, []).append(position)
    fused = [torch.amax(images[indexes], dim=0) for indexes in positions.values()]
    return torch.stack(fused, dim=0), list(positions)
