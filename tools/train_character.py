import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ensure_dirs, load_config
from datasets.character_dataset import (
    CHARACTER_DATA_FORMAT,
    CharacterTrainDataset,
    collate_character_batch,
    deterministic_character_split_indices,
    deterministic_split_indices,
)
from models.character_generator import (
    CHARACTER_CHECKPOINT_FORMAT,
    build_character_generator,
)
from tools.train_bbsmg import set_seed
from utils.structure_mask import STRUCTURE_TARGET_MODE


def binary_boundary_f1(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    tolerance: int = 1,
) -> torch.Tensor:
    """Per-sample boundary F1 with a small spatial matching tolerance."""
    pred_mask = (predictions >= threshold).float()
    target_mask = (targets >= 0.5).float()
    kernel_size = 2 * tolerance + 1

    def boundary(mask):
        dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
        eroded = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
        return (dilated - eroded) > 0

    pred_boundary = boundary(pred_mask)
    target_boundary = boundary(target_mask)
    pred_neighborhood = F.max_pool2d(
        pred_boundary.float(),
        kernel_size=kernel_size,
        stride=1,
        padding=tolerance,
    ) > 0
    target_neighborhood = F.max_pool2d(
        target_boundary.float(),
        kernel_size=kernel_size,
        stride=1,
        padding=tolerance,
    ) > 0
    dims = (1, 2, 3)
    precision = (
        (pred_boundary & target_neighborhood).sum(dim=dims).float() + 1e-6
    ) / (pred_boundary.sum(dim=dims).float() + 1e-6)
    recall = (
        (target_boundary & pred_neighborhood).sum(dim=dims).float() + 1e-6
    ) / (target_boundary.sum(dim=dims).float() + 1e-6)
    return (2.0 * precision * recall + 1e-6) / (precision + recall + 1e-6)


class StructureMaskLoss(nn.Module):
    """Binary structure objective that penalizes over-ink and uncertain gray haze."""

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 0.75,
        tversky_weight: float = 0.5,
        boundary_weight: float = 0.25,
        background_weight: float = 0.5,
        confidence_weight: float = 0.1,
        ink_weight: float = 0.25,
        trajectory_weight: float = 0.02,
        bce_pos_weight: float = 1.5,
        tversky_alpha: float = 0.7,
        tversky_beta: float = 0.3,
    ):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.tversky_weight = float(tversky_weight)
        self.boundary_weight = float(boundary_weight)
        self.background_weight = float(background_weight)
        self.confidence_weight = float(confidence_weight)
        self.ink_weight = float(ink_weight)
        self.trajectory_weight = float(trajectory_weight)
        self.bce_pos_weight = float(bce_pos_weight)
        self.tversky_alpha = float(tversky_alpha)
        self.tversky_beta = float(tversky_beta)
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(-1, -2)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _edge_magnitude(self, values):
        grad_x = F.conv2d(values, self.sobel_x, padding=1)
        grad_y = F.conv2d(values, self.sobel_y, padding=1)
        return torch.sqrt(grad_x.square() + grad_y.square() + 1e-6) / 4.0

    def compute_components(self, preds, targets, inputs=None):
        preds = preds.clamp(1e-6, 1.0 - 1e-6)
        targets = (targets >= 0.5).float()
        dims = (1, 2, 3)
        pixel_weights = 1.0 + self.bce_pos_weight * targets
        weighted_bce = (
            F.binary_cross_entropy(preds, targets, reduction="none") * pixel_weights
        ).mean()
        intersection = (preds * targets).sum(dim=dims)
        dice_loss = 1.0 - (
            (2.0 * intersection + 1e-6)
            / (preds.sum(dim=dims) + targets.sum(dim=dims) + 1e-6)
        ).mean()
        false_positive = (preds * (1.0 - targets)).sum(dim=dims)
        false_negative = ((1.0 - preds) * targets).sum(dim=dims)
        tversky_loss = 1.0 - (
            (intersection + 1e-6)
            / (
                intersection
                + self.tversky_alpha * false_positive
                + self.tversky_beta * false_negative
                + 1e-6
            )
        ).mean()
        boundary_loss = F.l1_loss(
            self._edge_magnitude(preds),
            self._edge_magnitude(targets),
        )
        background = 1.0 - targets
        background_loss = (
            (preds.square() * background).sum() / background.sum().clamp_min(1.0)
        )
        confidence_loss = (4.0 * preds * (1.0 - preds)).mean()
        ink_loss = F.l1_loss(
            preds.mean(dim=dims),
            targets.mean(dim=dims),
        )
        if inputs is None:
            raise ValueError("Whole-character loss requires trajectory input maps")
        centerline = inputs[:, 0:1].clamp(0.0, 1.0)
        guided_centerline = centerline * targets
        trajectory_loss = (
            ((1.0 - preds) * guided_centerline).sum()
            / guided_centerline.sum().clamp_min(1.0)
        )
        return {
            "weighted_bce": weighted_bce,
            "dice_loss": dice_loss,
            "tversky_loss": tversky_loss,
            "boundary_loss": boundary_loss,
            "background_loss": background_loss,
            "confidence_loss": confidence_loss,
            "ink_loss": ink_loss,
            "trajectory_loss": trajectory_loss,
        }

    def combine_components(self, components):
        return (
            self.bce_weight * components["weighted_bce"]
            + self.dice_weight * components["dice_loss"]
            + self.tversky_weight * components["tversky_loss"]
            + self.boundary_weight * components["boundary_loss"]
            + self.background_weight * components["background_loss"]
            + self.confidence_weight * components["confidence_loss"]
            + self.ink_weight * components["ink_loss"]
            + self.trajectory_weight * components["trajectory_loss"]
        )

    def forward(self, preds, targets, inputs=None):
        return self.combine_components(self.compute_components(preds, targets, inputs))


def build_loss(device: torch.device) -> StructureMaskLoss:
    return StructureMaskLoss().to(device)


def compute_batch_metrics(
    predictions,
    targets,
    inputs,
    criterion,
    threshold: float = 0.5,
) -> Dict[str, float]:
    components = criterion.compute_components(predictions, targets, inputs)
    total = criterion.combine_components(components)
    pred_binary = predictions >= threshold
    target_binary = targets >= 0.5
    intersection = (pred_binary & target_binary).sum(dim=(1, 2, 3)).float()
    union = (pred_binary | target_binary).sum(dim=(1, 2, 3)).float()
    values = {name: float(value.item()) for name, value in components.items()}
    values.update({
        "composite_loss": float(total.item()),
        "plain_mse": float(F.mse_loss(predictions, targets).item()),
        "mae": float(F.l1_loss(predictions, targets).item()),
        "dice_at_threshold": float(
            ((2.0 * intersection + 1e-6) /
             (pred_binary.sum(dim=(1, 2, 3)).float()
              + target_binary.sum(dim=(1, 2, 3)).float() + 1e-6)).mean().item()
        ),
        "binary_threshold": float(threshold),
        "iou_at_threshold": float(
            ((intersection + 1e-6) / (union + 1e-6)).mean().item()
        ),
        "boundary_f1": float(
            binary_boundary_f1(predictions, targets, threshold=threshold).mean().item()
        ),
        "uncertain_fraction": float(
            ((predictions > 0.1) & (predictions < 0.9)).float().mean().item()
        ),
        "background_mean": float(
            ((predictions * (1.0 - targets)).sum() / (1.0 - targets).sum().clamp_min(1.0)).item()
        ),
        "trajectory_ink_mean": float(
            ((predictions * inputs[:, 0:1]).sum() / inputs[:, 0:1].sum().clamp_min(1.0)).item()
        ),
        "trajectory_target_mean": float(
            ((targets * inputs[:, 0:1]).sum() / inputs[:, 0:1].sum().clamp_min(1.0)).item()
        ),
        "trajectory_prediction_coverage": float(
            (((predictions >= 0.5).float() * (inputs[:, 0:1] >= 0.5).float()).sum()
             / (inputs[:, 0:1] >= 0.5).sum().clamp_min(1.0)).item()
        ),
        "trajectory_target_coverage": float(
            (((targets >= 0.5).float() * (inputs[:, 0:1] >= 0.5).float()).sum()
             / (inputs[:, 0:1] >= 0.5).sum().clamp_min(1.0)).item()
        ),
        "target_ink": float(targets.mean().item()),
        "prediction_ink": float(predictions.mean().item()),
    })
    return values


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total = 0.0
    count = 0
    for batch in loader:
        inputs = batch["inputs"].to(device)
        targets = batch["targets"].to(device).clamp(0.0, 1.0)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(inputs)
        loss = criterion(predictions, targets, inputs)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite whole-character loss")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if not torch.isfinite(grad_norm):
            raise RuntimeError("Non-finite whole-character gradient norm")
        optimizer.step()
        total += float(loss.item()) * inputs.shape[0]
        count += inputs.shape[0]
    return total / max(count, 1)


@torch.no_grad()
def validate(model, loader, criterion, device) -> Optional[Dict[str, float]]:
    if loader is None:
        return None
    model.eval()
    totals: Dict[str, float] = {}
    count = 0
    for batch in loader:
        inputs = batch["inputs"].to(device)
        targets = batch["targets"].to(device).clamp(0.0, 1.0)
        predictions = model(inputs).clamp(0.0, 1.0)
        values = compute_batch_metrics(predictions, targets, inputs, criterion)
        batch_size = inputs.shape[0]
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + value * batch_size
        count += batch_size
    return {name: value / max(count, 1) for name, value in totals.items()}


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    epoch: int,
    train_loss: float,
    val_metrics: Optional[Dict[str, float]],
    best_val: float,
    model_config: Dict[str, Any],
    channel_names,
    trajectory_padding,
    trajectory_width,
    target_mode,
    structure_threshold,
    min_component_pixels,
    train_indices,
    val_indices,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": CHARACTER_CHECKPOINT_FORMAT,
        "data_format": CHARACTER_DATA_FORMAT,
        "target_mode": target_mode,
        "structure_threshold": float(structure_threshold),
        "min_component_pixels": int(min_component_pixels),
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "train_loss": train_loss,
        "val_metrics": val_metrics,
        "val_loss": val_metrics.get("composite_loss") if val_metrics else None,
        "best_val": best_val,
        "model_config": model_config,
        "channel_names": list(channel_names),
        "trajectory_padding": int(trajectory_padding),
        "trajectory_width": int(trajectory_width),
        "train_indices": list(train_indices),
        "val_indices": list(val_indices),
    }, path)


def append_metrics(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def metadata_dict(value) -> Dict[str, Any]:
    if hasattr(value, "item"):
        value = value.item()
    return value if isinstance(value, dict) else {}


def metadata_character(value) -> str:
    return str(metadata_dict(value).get("character") or "unknown")


def main(args) -> None:
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.output_dir is not None:
        cfg.train.output_dir = args.output_dir
    elif args.resume:
        cfg.train.output_dir = str(Path(args.resume).parent)
    ensure_dirs(cfg)
    set_seed(cfg.train.seed)
    device = torch.device(
        cfg.train.device if torch.cuda.is_available() or cfg.train.device == "cpu" else "cpu"
    )

    resume_checkpoint = torch.load(args.resume, map_location="cpu") if args.resume else None
    character_init_checkpoint = (
        torch.load(args.init_character_checkpoint, map_location="cpu")
        if args.init_character_checkpoint
        else None
    )
    if resume_checkpoint is not None and resume_checkpoint.get("format") != CHARACTER_CHECKPOINT_FORMAT:
        raise ValueError(
            f"--resume requires a {CHARACTER_CHECKPOINT_FORMAT} checkpoint"
        )
    if (
        character_init_checkpoint is not None
        and character_init_checkpoint.get("format") != CHARACTER_CHECKPOINT_FORMAT
    ):
        raise ValueError(
            f"--init_character_checkpoint requires {CHARACTER_CHECKPOINT_FORMAT}; "
            "v6 must not initialize from grayscale-target checkpoints."
        )

    dataset = CharacterTrainDataset(args.npz_path)
    for label, checkpoint in (
        ("resume", resume_checkpoint),
        ("initialization", character_init_checkpoint),
    ):
        if checkpoint is None:
            continue
        if checkpoint.get("data_format") != CHARACTER_DATA_FORMAT:
            raise ValueError(
                f"{label} checkpoint data format does not match {CHARACTER_DATA_FORMAT}"
            )
        if checkpoint.get("target_mode") != STRUCTURE_TARGET_MODE:
            raise ValueError(
                f"{label} checkpoint does not use {STRUCTURE_TARGET_MODE}"
            )
        if abs(
            float(checkpoint.get("structure_threshold", -1.0))
            - dataset.structure_threshold
        ) > 1e-6:
            raise ValueError(
                f"{label} checkpoint structure threshold differs from this NPZ"
            )
        if (
            int(checkpoint.get("min_component_pixels", -1))
            != dataset.min_component_pixels
        ):
            raise ValueError(
                f"{label} checkpoint component cleanup differs from this NPZ"
            )
    model_config = asdict(cfg.character_generator)
    if resume_checkpoint and resume_checkpoint.get("model_config"):
        model_config = dict(resume_checkpoint["model_config"])
    elif character_init_checkpoint and character_init_checkpoint.get("model_config"):
        model_config = dict(character_init_checkpoint["model_config"])
    if dataset.inputs.shape[1] != int(model_config["input_channels"]):
        raise ValueError("NPZ channels do not match character_generator.input_channels")
    if dataset.inputs.shape[-1] != int(model_config["image_size"]):
        raise ValueError("NPZ spatial size does not match character_generator.image_size")

    if resume_checkpoint and resume_checkpoint.get("train_indices") is not None:
        train_indices = list(resume_checkpoint["train_indices"])
        val_indices = list(resume_checkpoint.get("val_indices", []))
        if train_indices + val_indices and max(train_indices + val_indices) >= len(dataset):
            raise ValueError(
                "Resume checkpoint split indices do not fit this NPZ. "
                "Use --init_character_checkpoint to fine-tune on a different dataset."
            )
    else:
        if args.split_mode == "character":
            train_indices, val_indices = deterministic_character_split_indices(
                dataset.metadata, args.val_ratio, cfg.train.seed
            )
        else:
            train_indices, val_indices = deterministic_split_indices(
                len(dataset), args.val_ratio, cfg.train.seed
            )
        # An explicitly supplied target (for example the user's 武 image) is a
        # fitting target, so keep it in training instead of losing it to a
        # random validation split. Validation still measures the other glyphs.
        external_characters = set()
        for index in range(len(dataset)):
            value = metadata_dict(dataset.metadata[index])
            if isinstance(value, dict) and value.get("target_source") == "external":
                external_characters.add(str(value.get("character") or ""))
        if external_characters:
            moved_indices = []
            for index in list(val_indices):
                value = metadata_dict(dataset.metadata[index])
                if (
                    isinstance(value, dict)
                    and str(value.get("character") or "") in external_characters
                ):
                    moved_indices.append(index)
            if moved_indices:
                val_indices = [index for index in val_indices if index not in moved_indices]
                train_indices.extend(moved_indices)
                print(
                    f"[SPLIT] Kept {len(moved_indices)} sample(s) from external-target "
                    "characters in training"
                )
    batch_size = args.batch_size or cfg.train.batch_size
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_character_batch,
    )
    val_loader = None
    if val_indices:
        val_loader = DataLoader(
            Subset(dataset, val_indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=cfg.train.num_workers,
            collate_fn=collate_character_batch,
        )
    else:
        print("[WARN] No validation samples; checkpoint selection will use training loss")

    model = build_character_generator(model_config).to(device)
    if character_init_checkpoint is not None:
        model.load_state_dict(character_init_checkpoint["model_state"])
        print(f"[INIT] Loaded U-Net character model from {args.init_character_checkpoint}")
    criterion = build_loss(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )

    start_epoch = 1
    best_val = float("inf")
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        if resume_checkpoint.get("scheduler_state"):
            scheduler.load_state_dict(resume_checkpoint["scheduler_state"])
        start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1
        best_val = float(resume_checkpoint.get("best_val", best_val))
        print(f"[RESUME] {args.resume}; start_epoch={start_epoch}")

    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "split_manifest.json", "w", encoding="utf-8") as file:
        json.dump({
            "seed": cfg.train.seed,
            "val_ratio": args.val_ratio,
            "split_mode": args.split_mode,
            "train_indices": train_indices,
            "val_indices": val_indices,
            "train_characters": sorted({
                metadata_character(dataset.metadata[index])
                for index in train_indices
            }),
            "val_characters": sorted({
                metadata_character(dataset.metadata[index])
                for index in val_indices
            }),
        }, file, ensure_ascii=False, indent=2)

    if start_epoch > cfg.train.epochs:
        print(f"[DONE] Checkpoint already reached requested epoch {cfg.train.epochs}")
        return

    for epoch in range(start_epoch, cfg.train.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = validate(model, val_loader, criterion, device)
        monitor = val_metrics["composite_loss"] if val_metrics else train_loss
        scheduler.step(monitor)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        message = f"[Epoch {epoch:03d}] train_loss={train_loss:.6f}, lr={learning_rate:.8g}"
        if val_metrics:
            message += f", val_loss={val_metrics['composite_loss']:.6f}"
        print(message)
        if val_metrics:
            print("[VAL CHARACTER] " + ", ".join(
                f"{name}={value:.6f}" for name, value in val_metrics.items()
            ))

        row: Dict[str, Any] = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": train_loss,
        }
        if val_metrics:
            row.update({f"val_{name}": value for name, value in val_metrics.items()})
        append_metrics(output_dir / "training_metrics.csv", row)

        is_best = monitor < best_val
        if is_best:
            best_val = monitor
        save_args = (
            model,
            optimizer,
            scheduler,
            epoch,
            train_loss,
            val_metrics,
            best_val,
            model_config,
            dataset.channel_names,
            dataset.trajectory_padding,
            dataset.trajectory_width,
            dataset.target_mode,
            dataset.structure_threshold,
            dataset.min_component_pixels,
            train_indices,
            val_indices,
        )
        save_checkpoint(output_dir / "character_last.pt", *save_args)
        if epoch % cfg.train.save_interval == 0:
            save_checkpoint(output_dir / f"character_epoch_{epoch:03d}.pt", *save_args)
        if is_best:
            save_checkpoint(output_dir / "character_best.pt", *save_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the spatial whole-character U-Net")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--npz_path", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--split_mode", choices=("character", "sample"), default="character")
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--init_character_checkpoint",
        default=None,
        help="Fine-tune model weights on a new NPZ without restoring optimizer/split state",
    )
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=3)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parsed = parser.parse_args()
    initialization_modes = [
        parsed.resume,
        parsed.init_character_checkpoint,
    ]
    if sum(value is not None for value in initialization_modes) > 1:
        parser.error(
            "--resume and --init_character_checkpoint are mutually exclusive"
        )
    main(parsed)
