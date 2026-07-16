import argparse
from contextlib import nullcontext
import csv
from dataclasses import asdict
import json
from pathlib import Path
import random
import shutil
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from config import ensure_dirs, load_config
from models.bbsmg import build_bbsmg
from utils.feature_schema import (
    checkpoint_schema,
    get_feature_schema,
    normalization_for_inputs,
    read_npz_schema,
)
from utils.losses import CompositeStrokeLoss
from utils.splits import build_split, load_manifest, save_manifest


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BBSMGTrainDataset(Dataset):
    def __init__(
        self,
        npz_path: str,
        coordinate_scale: float = 128.0,
        expected_schema: Optional[str] = None,
        target_cache_dir: Optional[str] = "data/cache/npz_arrays",
    ):
        self.path = Path(npz_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Training NPZ not found: {npz_path}")
        data = np.load(self.path, allow_pickle=True)
        raw_inputs = np.asarray(data["inputs"])
        self.targets = _load_targets(self.path, data, target_cache_dir)
        self.meta = np.asarray(data["meta"], dtype=object) if "meta" in data.files else None
        if raw_inputs.ndim != 2:
            raise ValueError(f"inputs must have shape [N,D], got {raw_inputs.shape}")
        if self.targets.shape[0] != raw_inputs.shape[0]:
            raise ValueError("inputs and targets contain different sample counts")
        if self.meta is not None and len(self.meta) != len(raw_inputs):
            raise ValueError("meta length does not match inputs")
        if self.targets.ndim == 3:
            self.targets = self.targets[:, None, :, :]
        if self.targets.ndim != 4 or self.targets.shape[1] != 1:
            raise ValueError(f"targets must have shape [N,1,H,W] or [N,H,W], got {self.targets.shape}")

        self.feature_schema = read_npz_schema(data, raw_inputs.shape[1])
        schema = get_feature_schema(self.feature_schema)
        if expected_schema and self.feature_schema != expected_schema:
            raise ValueError(
                f"NPZ schema {self.feature_schema} does not match configured schema {expected_schema}"
            )
        if schema.input_dim != raw_inputs.shape[1]:
            raise ValueError("NPZ feature dimension does not match its schema")
        if not np.isfinite(raw_inputs).all():
            raise ValueError("NPZ inputs contain NaN or Inf")

        self.input_normalization = normalization_for_inputs(
            raw_inputs.astype(np.float32, copy=False),
            self.feature_schema,
            coordinate_scale,
        )
        scales = np.asarray(self.input_normalization["scales"], dtype=np.float32)
        self.inputs = raw_inputs.astype(np.float32, copy=False) / scales[None, :]
        data.close()

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        metadata = self.meta[index] if self.meta is not None else {}
        if hasattr(metadata, "item") and not isinstance(metadata, dict):
            try:
                metadata = metadata.item()
            except ValueError:
                pass
        return {
            "inputs": torch.from_numpy(self.inputs[index]),
            # targets.npy is intentionally read with mmap_mode="r"; copy one
            # sample so PyTorch never receives a tensor backed by read-only memory.
            "targets": torch.from_numpy(
                np.array(self.targets[index], dtype=np.float32, copy=True)
            ),
            "meta": metadata if isinstance(metadata, dict) else {},
            "index": index,
        }


def _load_targets(
    npz_path: Path,
    npz: Any,
    target_cache_dir: Optional[str],
) -> np.ndarray:
    if not target_cache_dir:
        return np.asarray(npz["targets"])
    cache_root = Path(target_cache_dir)
    fingerprint = f"{npz_path.stem}-{npz_path.stat().st_size}-{npz_path.stat().st_mtime_ns}"
    cache_dir = cache_root / fingerprint
    target_path = cache_dir / "targets.npy"
    if not target_path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = target_path.with_suffix(".npy.tmp")
        print(f"[CACHE] Extracting targets.npy to {target_path}")
        with zipfile.ZipFile(npz_path) as archive:
            try:
                source = archive.open("targets.npy")
            except KeyError as error:
                raise ValueError(f"{npz_path} does not contain targets.npy") from error
            with source, open(temporary, "wb") as destination:
                shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)
        temporary.replace(target_path)
    return np.load(target_path, mmap_mode="r", allow_pickle=False)


def collate_bbsmg_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "inputs": torch.stack([item["inputs"] for item in batch]),
        "targets": torch.stack([item["targets"] for item in batch]),
        "meta": [item["meta"] for item in batch],
        "indices": torch.tensor([item["index"] for item in batch], dtype=torch.long),
    }


def _loader(
    dataset: Dataset,
    indices: Sequence[int],
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    pin_memory: bool,
    persistent_workers: bool,
) -> DataLoader:
    return DataLoader(
        Subset(dataset, list(indices)),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        collate_fn=collate_bbsmg_batch,
    )


def build_dataloaders(
    npz_path: str,
    batch_size: int,
    num_workers: int,
    val_ratio: float = 0.1,
    coordinate_scale: float = 128.0,
    split_strategy: str = "group",
    split_group_key: str = "sample_id",
    seed: int = 42,
    split_manifest_path: Optional[str] = None,
    expected_schema: Optional[str] = None,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    target_cache_dir: Optional[str] = "data/cache/npz_arrays",
) -> Tuple[DataLoader, Optional[DataLoader], BBSMGTrainDataset, Dict[str, Any]]:
    dataset = BBSMGTrainDataset(
        npz_path,
        coordinate_scale=coordinate_scale,
        expected_schema=expected_schema,
        target_cache_dir=target_cache_dir,
    )
    if split_manifest_path and Path(split_manifest_path).exists():
        manifest = load_manifest(split_manifest_path, expected_length=len(dataset))
    else:
        train_indices, val_indices, manifest = build_split(
            dataset.meta,
            len(dataset),
            val_ratio,
            seed,
            strategy=split_strategy,
            group_key=split_group_key,
        )
        if split_manifest_path:
            save_manifest(split_manifest_path, manifest)
    train_indices = manifest["train_indices"]
    val_indices = manifest["val_indices"]
    train_loader = _loader(
        dataset, train_indices, batch_size, num_workers, True,
        pin_memory, persistent_workers,
    )
    val_loader = (
        _loader(
            dataset, val_indices, batch_size, num_workers, False,
            pin_memory, persistent_workers,
        )
        if val_indices
        else None
    )
    return train_loader, val_loader, dataset, manifest


def _autocast_context(device: torch.device, enabled: bool, dtype: torch.dtype):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def _compute_loss_fp32(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    criterion: CompositeStrokeLoss,
) -> torch.Tensor:
    """Keep numerically sensitive image losses out of the FP16 autocast path."""
    with torch.autocast(device_type=predictions.device.type, enabled=False):
        return criterion(predictions.float(), targets.float())


def create_grad_scaler(
    device: torch.device,
    enabled: bool,
    init_scale: float,
    growth_interval: int,
    backoff_factor: float,
) -> Any:
    kwargs = {
        "enabled": enabled,
        "init_scale": init_scale,
        "growth_interval": growth_interval,
        "backoff_factor": backoff_factor,
    }
    try:
        return torch.amp.GradScaler(device.type, **kwargs)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(**kwargs)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: CompositeStrokeLoss,
    device: torch.device,
    scaler: Any,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    gradient_clip_norm: float,
    log_interval: int,
    max_consecutive_nonfinite_steps: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    consecutive_nonfinite_steps = 0
    for step, batch in enumerate(loader, start=1):
        inputs = batch["inputs"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, amp_enabled, amp_dtype):
            predictions = model(inputs)
        loss = _compute_loss_fp32(predictions, targets, criterion)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss at step {step}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), gradient_clip_norm
        )
        if not torch.isfinite(gradient_norm):
            if not amp_enabled:
                raise RuntimeError(f"Non-finite gradient norm at step {step}")
            consecutive_nonfinite_steps += 1
            scale_before = float(scaler.get_scale())
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            if (
                consecutive_nonfinite_steps <= 3
                or consecutive_nonfinite_steps % 5 == 0
            ):
                print(
                    f"[AMP] skipped step={step} for non-finite gradients; "
                    f"scale={scale_before:.6g}->{float(scaler.get_scale()):.6g}"
                )
            if consecutive_nonfinite_steps >= max_consecutive_nonfinite_steps:
                raise RuntimeError(
                    "AMP gradients remained non-finite for "
                    f"{consecutive_nonfinite_steps} consecutive steps. "
                    "Set train.amp: false to diagnose full-precision training."
                )
            continue
        scaler.step(optimizer)
        scaler.update()
        consecutive_nonfinite_steps = 0
        batch_size = inputs.shape[0]
        total_loss += float(loss.detach()) * batch_size
        total_count += batch_size
        if log_interval > 0 and step % log_interval == 0:
            print(
                f"[TRAIN] step={step}/{len(loader)} "
                f"loss={total_loss / max(total_count, 1):.6f}"
            )
    return total_loss / max(total_count, 1)


@torch.no_grad()
def validate_detailed(
    model: nn.Module,
    loader: Optional[DataLoader],
    criterion: CompositeStrokeLoss,
    device: torch.device,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.float16,
) -> Optional[Dict[str, float]]:
    if loader is None:
        return None
    model.eval()
    totals: Dict[str, float] = {}
    count = 0
    for batch in loader:
        inputs = batch["inputs"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True).clamp(0.0, 1.0)
        with _autocast_context(device, amp_enabled, amp_dtype):
            predictions = model(inputs)
        predictions = predictions.float().clamp(0.0, 1.0)
        components = criterion.compute_components(predictions, targets.float())
        total = criterion.combine_components(components)
        pred_binary = predictions >= 0.5
        target_binary = targets >= 0.5
        intersection = (pred_binary & target_binary).sum((1, 2, 3)).float()
        union = (pred_binary | target_binary).sum((1, 2, 3)).float()
        values = {name: float(value) for name, value in components.items()}
        values.update(
            {
                "composite_loss": float(total),
                "plain_mse": float(F.mse_loss(predictions, targets)),
                "mae": float(F.l1_loss(predictions, targets)),
                "ssim_score": 1.0 - values["ssim_loss"],
                "dice_score": 1.0 - values["dice_loss"],
                "iou_at_0.5": float(
                    ((intersection + 1e-6) / (union + 1e-6)).mean()
                ),
            }
        )
        batch_size = inputs.shape[0]
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + value * batch_size
        count += batch_size
    return {name: value / max(count, 1) for name, value in totals.items()}


def append_metrics_csv(
    path: Path,
    epoch: int,
    train_loss: float,
    learning_rate: float,
    val_metrics: Optional[Dict[str, float]],
) -> None:
    row: Dict[str, Any] = {
        "epoch": epoch,
        "learning_rate": learning_rate,
        "train_loss": train_loss,
    }
    if val_metrics:
        row.update({f"val_{name}": value for name, value in val_metrics.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    if path.exists():
        with open(path, "r", encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        rows = [existing for existing in rows if int(existing["epoch"]) != epoch]
    rows.append(row)
    fieldnames = list(dict.fromkeys(
        [key for existing in rows for key in existing] + list(row)
    ))
    with open(path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    train_loss: float,
    val_metrics: Optional[Dict[str, float]],
    best_val: Optional[float],
    dataset: BBSMGTrainDataset,
    split_manifest: Dict[str, Any],
    resolved_config: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 2,
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "train_loss": train_loss,
            "val_loss": None if val_metrics is None else val_metrics["composite_loss"],
            "val_metrics": val_metrics,
            "best_val": best_val,
            "feature_schema": dataset.feature_schema,
            "input_normalization": dataset.input_normalization,
            "split_manifest": split_manifest,
            "npz_path": str(dataset.path),
            "config": resolved_config,
        },
        path,
    )


def _load_model_state(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state", "model_state_dict", "state_dict"):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def load_torch_checkpoint(path: str, map_location: Any) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_resume_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    device: torch.device,
    dataset: BBSMGTrainDataset,
) -> Tuple[int, Optional[float]]:
    checkpoint = load_torch_checkpoint(path, map_location=device)
    model.load_state_dict(_load_model_state(checkpoint))
    if not isinstance(checkpoint, dict):
        return 1, None
    if checkpoint.get("optimizer_state"):
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if checkpoint.get("scheduler_state"):
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    if checkpoint.get("scaler_state"):
        scaler.load_state_dict(checkpoint["scaler_state"])
    schema = checkpoint_schema(checkpoint, dataset.inputs.shape[1])
    if schema != dataset.feature_schema:
        raise ValueError(
            f"Checkpoint schema {schema} does not match NPZ schema {dataset.feature_schema}"
        )
    recorded = checkpoint.get("input_normalization")
    if recorded:
        expected = np.asarray(dataset.input_normalization["scales"])
        actual = np.asarray(recorded["scales"])
        if expected.shape != actual.shape or not np.allclose(expected, actual):
            raise ValueError("Resume checkpoint normalization does not match the current NPZ")
    best_val = checkpoint.get("best_val", checkpoint.get("val_loss"))
    return int(checkpoint.get("epoch", 0)) + 1, best_val


def main(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.val_ratio is not None:
        cfg.train.val_ratio = args.val_ratio
    if args.lr_factor is not None:
        cfg.train.lr_factor = args.lr_factor
    if args.lr_patience is not None:
        cfg.train.lr_patience = args.lr_patience
    if args.min_lr is not None:
        cfg.train.min_lr = args.min_lr
    if args.output_dir:
        cfg.train.output_dir = args.output_dir
    elif args.resume:
        cfg.train.output_dir = str(Path(args.resume).parent)
    ensure_dirs(cfg)
    set_seed(cfg.train.seed)

    device = torch.device(
        cfg.train.device
        if cfg.train.device == "cpu" or torch.cuda.is_available()
        else "cpu"
    )
    amp_enabled = bool(cfg.train.amp and device.type == "cuda")
    amp_dtype = torch.float16 if cfg.train.amp_dtype == "float16" else torch.bfloat16
    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = (
        Path(cfg.train.split_manifest)
        if cfg.train.split_manifest
        else output_dir / "split_manifest.json"
    )

    train_loader, val_loader, dataset, manifest = build_dataloaders(
        args.npz_path,
        cfg.train.batch_size,
        cfg.train.num_workers,
        cfg.train.val_ratio,
        cfg.bbsmg.image_size,
        cfg.train.split_strategy,
        cfg.train.split_group_key,
        cfg.train.seed,
        str(manifest_path),
        cfg.bbsmg.feature_schema,
        cfg.train.pin_memory and device.type == "cuda",
        cfg.train.persistent_workers,
        cfg.train.target_cache_dir,
    )
    if dataset.inputs.shape[1] != cfg.bbsmg.input_dim:
        raise ValueError(
            f"NPZ input dimension {dataset.inputs.shape[1]} does not match "
            f"config bbsmg.input_dim={cfg.bbsmg.input_dim}"
        )

    model = build_bbsmg(
        input_dim=cfg.bbsmg.input_dim,
        latent_dim=cfg.bbsmg.latent_dim,
        base_channels=cfg.bbsmg.base_channels,
        out_channels=cfg.bbsmg.out_channels,
        image_size=cfg.bbsmg.image_size,
        use_tanh=cfg.bbsmg.use_tanh,
    ).to(device)
    criterion = CompositeStrokeLoss(**asdict(cfg.train.loss)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.train.lr_factor,
        patience=cfg.train.lr_patience,
        min_lr=cfg.train.min_lr,
    )
    scaler = create_grad_scaler(
        device,
        amp_enabled,
        cfg.train.amp_init_scale,
        cfg.train.amp_growth_interval,
        cfg.train.amp_backoff_factor,
    )

    resolved_config = asdict(cfg)
    with open(output_dir / "resolved_config.json", "w", encoding="utf-8") as file:
        json.dump(resolved_config, file, ensure_ascii=False, indent=2)
    print(
        f"[DATA] samples={len(dataset)}, train={len(manifest['train_indices'])}, "
        f"val={len(manifest['val_indices'])}, schema={dataset.feature_schema}"
    )
    print(f"[RUNTIME] device={device}, amp={amp_enabled}, amp_dtype={cfg.train.amp_dtype}")

    start_epoch, best_val = 1, None
    if args.resume:
        start_epoch, best_val = load_resume_checkpoint(
            args.resume, model, optimizer, scheduler, scaler, device, dataset
        )
        print(
            f"[RESUME] checkpoint={args.resume}, start_epoch={start_epoch}, "
            f"target_epoch={cfg.train.epochs}, best_val={best_val}"
        )
    if start_epoch > cfg.train.epochs:
        print(f"[DONE] Checkpoint already reached target epoch {cfg.train.epochs}.")
        return

    for epoch in range(start_epoch, cfg.train.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler,
            amp_enabled, amp_dtype, cfg.train.gradient_clip_norm,
            cfg.train.log_interval, cfg.train.amp_max_consecutive_nonfinite_steps,
        )
        val_metrics = validate_detailed(
            model, val_loader, criterion, device, amp_enabled, amp_dtype
        )
        monitor = (
            val_metrics["composite_loss"] if val_metrics is not None else train_loss
        )
        scheduler.step(monitor)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        is_best = best_val is None or monitor < best_val
        if is_best:
            best_val = monitor
        print(
            f"[EPOCH {epoch:03d}] train={train_loss:.6f}, "
            f"val={monitor:.6f}, lr={learning_rate:.8g}"
        )
        append_metrics_csv(
            output_dir / "training_metrics.csv",
            epoch,
            train_loss,
            learning_rate,
            val_metrics,
        )
        save_args = (
            model, optimizer, scheduler, scaler, epoch, train_loss, val_metrics,
            best_val, dataset, manifest, resolved_config,
        )
        save_checkpoint(output_dir / "bbsmg_last.pt", *save_args)
        if is_best:
            save_checkpoint(output_dir / "bbsmg_best.pt", *save_args)
        if epoch % cfg.train.save_interval == 0:
            save_checkpoint(output_dir / f"bbsmg_epoch_{epoch:03d}.pt", *save_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--npz_path", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--val_ratio", type=float)
    parser.add_argument("--output_dir")
    parser.add_argument("--lr_factor", type=float)
    parser.add_argument("--lr_patience", type=int)
    parser.add_argument("--min_lr", type=float)
    main(parser.parse_args())
