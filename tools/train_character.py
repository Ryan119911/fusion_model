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
from utils.trajectory_target import TRAJECTORY_TARGET_MODE


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


def compute_structure_quality(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
) -> Dict[str, torch.Tensor]:
    pred_binary = predictions >= threshold
    target_binary = targets >= 0.5
    dims = (1, 2, 3)
    intersection = (pred_binary & target_binary).sum(dim=dims).float()
    union = (pred_binary | target_binary).sum(dim=dims).float()
    pred_count = pred_binary.sum(dim=dims).float()
    target_count = target_binary.sum(dim=dims).float()
    dice = ((2.0 * intersection + 1e-6) / (pred_count + target_count + 1e-6)).mean()
    iou = ((intersection + 1e-6) / (union + 1e-6)).mean()
    boundary_f1 = binary_boundary_f1(
        predictions, targets, threshold=threshold
    ).mean()
    mask_ink_ratio = (
        pred_binary.float().mean(dim=dims)
        / target_binary.float().mean(dim=dims).clamp_min(1e-6)
    )
    ink_balance = torch.exp(
        -torch.abs(torch.log(mask_ink_ratio.clamp_min(1e-6)))
    ).mean()
    selection_score = (
        0.40 * iou
        + 0.30 * boundary_f1
        + 0.20 * dice
        + 0.10 * ink_balance
    )
    return {
        "dice": dice,
        "iou": iou,
        "boundary_f1": boundary_f1,
        "prediction_mask_ink": pred_binary.float().mean(),
        "target_mask_ink": target_binary.float().mean(),
        "mask_ink_ratio": mask_ink_ratio.mean(),
        "ink_balance_score": ink_balance,
        "selection_score": selection_score,
    }


class StructureMaskLoss(nn.Module):
    """Structure objective balancing topology, boundary accuracy, and ink amount."""

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 0.75,
        tversky_weight: float = 0.75,
        cldice_weight: float = 0.35,
        boundary_weight: float = 0.5,
        background_weight: float = 0.75,
        confidence_weight: float = 0.2,
        ink_weight: float = 0.25,
        local_ink_weight: float = 0.75,
        trajectory_weight: float = 0.25,
        bce_pos_weight: float = 1.0,
        tversky_alpha: float = 0.8,
        tversky_beta: float = 0.2,
        skeleton_iterations: int = 5,
    ):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.tversky_weight = float(tversky_weight)
        self.cldice_weight = float(cldice_weight)
        self.boundary_weight = float(boundary_weight)
        self.background_weight = float(background_weight)
        self.confidence_weight = float(confidence_weight)
        self.ink_weight = float(ink_weight)
        self.local_ink_weight = float(local_ink_weight)
        self.trajectory_weight = float(trajectory_weight)
        self.bce_pos_weight = float(bce_pos_weight)
        self.tversky_alpha = float(tversky_alpha)
        self.tversky_beta = float(tversky_beta)
        self.skeleton_iterations = int(skeleton_iterations)
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

    @staticmethod
    def _soft_erode(values):
        eroded_horizontal = -F.max_pool2d(
            -values, kernel_size=(3, 1), stride=1, padding=(1, 0)
        )
        eroded_vertical = -F.max_pool2d(
            -values, kernel_size=(1, 3), stride=1, padding=(0, 1)
        )
        return torch.minimum(eroded_horizontal, eroded_vertical)

    @staticmethod
    def _soft_dilate(values):
        return F.max_pool2d(values, kernel_size=3, stride=1, padding=1)

    def _soft_skeleton(self, values):
        opened = self._soft_dilate(self._soft_erode(values))
        skeleton = F.relu(values - opened)
        for _ in range(self.skeleton_iterations):
            values = self._soft_erode(values)
            opened = self._soft_dilate(self._soft_erode(values))
            delta = F.relu(values - opened)
            skeleton = skeleton + F.relu(delta - skeleton * delta)
        return skeleton

    def _cldice_loss(self, preds, targets):
        pred_skeleton = self._soft_skeleton(preds)
        target_skeleton = self._soft_skeleton(targets)
        dims = (1, 2, 3)
        topology_precision = (
            (pred_skeleton * targets).sum(dim=dims) + 1e-6
        ) / (pred_skeleton.sum(dim=dims) + 1e-6)
        topology_sensitivity = (
            (target_skeleton * preds).sum(dim=dims) + 1e-6
        ) / (target_skeleton.sum(dim=dims) + 1e-6)
        cldice = (
            2.0 * topology_precision * topology_sensitivity + 1e-6
        ) / (topology_precision + topology_sensitivity + 1e-6)
        return 1.0 - cldice.mean()

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
        cldice_loss = self._cldice_loss(preds, targets)
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
        local_ink_loss = 0.5 * (
            F.l1_loss(
                F.avg_pool2d(preds, kernel_size=5, stride=1, padding=2),
                F.avg_pool2d(targets, kernel_size=5, stride=1, padding=2),
            )
            + F.l1_loss(
                F.avg_pool2d(preds, kernel_size=11, stride=1, padding=5),
                F.avg_pool2d(targets, kernel_size=11, stride=1, padding=5),
            )
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
            "cldice_loss": cldice_loss,
            "boundary_loss": boundary_loss,
            "background_loss": background_loss,
            "confidence_loss": confidence_loss,
            "ink_loss": ink_loss,
            "local_ink_loss": local_ink_loss,
            "trajectory_loss": trajectory_loss,
        }

    def combine_components(self, components):
        return (
            self.bce_weight * components["weighted_bce"]
            + self.dice_weight * components["dice_loss"]
            + self.tversky_weight * components["tversky_loss"]
            + self.cldice_weight * components["cldice_loss"]
            + self.boundary_weight * components["boundary_loss"]
            + self.background_weight * components["background_loss"]
            + self.confidence_weight * components["confidence_loss"]
            + self.ink_weight * components["ink_loss"]
            + self.local_ink_weight * components["local_ink_loss"]
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
    quality = compute_structure_quality(predictions, targets, threshold)
    values = {name: float(value.item()) for name, value in components.items()}
    values.update({
        "composite_loss": float(total.item()),
        "plain_mse": float(F.mse_loss(predictions, targets).item()),
        "mae": float(F.l1_loss(predictions, targets).item()),
        "dice_at_threshold": float(quality["dice"].item()),
        "binary_threshold": float(threshold),
        "iou_at_threshold": float(quality["iou"].item()),
        "boundary_f1": float(quality["boundary_f1"].item()),
        "prediction_mask_ink": float(quality["prediction_mask_ink"].item()),
        "target_mask_ink": float(quality["target_mask_ink"].item()),
        "mask_ink_ratio": float(quality["mask_ink_ratio"].item()),
        "ink_balance_score": float(quality["ink_balance_score"].item()),
        "selection_score": float(quality["selection_score"].item()),
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
    selection_thresholds = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
    quality_totals = {
        threshold: {
            "dice": 0.0,
            "iou": 0.0,
            "boundary_f1": 0.0,
            "mask_ink_ratio": 0.0,
            "ink_balance_score": 0.0,
            "selection_score": 0.0,
        }
        for threshold in selection_thresholds
    }
    count = 0
    for batch in loader:
        inputs = batch["inputs"].to(device)
        targets = batch["targets"].to(device).clamp(0.0, 1.0)
        predictions = model(inputs).clamp(0.0, 1.0)
        values = compute_batch_metrics(predictions, targets, inputs, criterion)
        batch_size = inputs.shape[0]
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + value * batch_size
        for threshold in selection_thresholds:
            quality = compute_structure_quality(predictions, targets, threshold)
            for name in quality_totals[threshold]:
                quality_totals[threshold][name] += (
                    float(quality[name].item()) * batch_size
                )
        count += batch_size
    result = {name: value / max(count, 1) for name, value in totals.items()}
    threshold_rows = {
        threshold: {
            name: value / max(count, 1)
            for name, value in values.items()
        }
        for threshold, values in quality_totals.items()
    }
    best_threshold, best_quality = max(
        threshold_rows.items(),
        key=lambda item: item[1]["selection_score"],
    )
    result.update({
        "selection_threshold": float(best_threshold),
        "selection_score": best_quality["selection_score"],
        "selection_dice": best_quality["dice"],
        "selection_iou": best_quality["iou"],
        "selection_boundary_f1": best_quality["boundary_f1"],
        "selection_mask_ink_ratio": best_quality["mask_ink_ratio"],
        "selection_ink_balance_score": best_quality["ink_balance_score"],
    })
    return result


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    epoch: int,
    train_loss: float,
    val_metrics: Optional[Dict[str, float]],
    best_score: float,
    model_config: Dict[str, Any],
    channel_names,
    trajectory_padding,
    trajectory_width,
    target_mode,
    structure_threshold,
    min_component_pixels,
    opening_iterations,
    skeleton_tolerance,
    render_min_width,
    render_max_width,
    render_pressure_gamma,
    render_pressure_invert,
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
        "opening_iterations": int(opening_iterations),
        "skeleton_tolerance": int(skeleton_tolerance),
        "render_min_width": float(render_min_width),
        "render_max_width": float(render_max_width),
        "render_pressure_gamma": float(render_pressure_gamma),
        "render_pressure_invert": bool(render_pressure_invert),
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "train_loss": train_loss,
        "val_metrics": val_metrics,
        "val_loss": val_metrics.get("composite_loss") if val_metrics else None,
        "selection_score": (
            val_metrics.get("selection_score") if val_metrics else -train_loss
        ),
        "best_score": best_score,
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
            "v8 must not initialize from older or unpaired-target checkpoints."
        )

    dataset = CharacterTrainDataset(args.npz_path)
    if dataset.data_format != CHARACTER_DATA_FORMAT:
        raise ValueError(
            f"Training requires {CHARACTER_DATA_FORMAT}; rebuild with "
            "tools/build_trajectory_character_pairs.py"
        )
    if dataset.target_mode != TRAJECTORY_TARGET_MODE:
        raise ValueError(
            f"Training requires target_mode={TRAJECTORY_TARGET_MODE!r}"
        )
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
        if checkpoint.get("target_mode") != TRAJECTORY_TARGET_MODE:
            raise ValueError(
                f"{label} checkpoint does not use {TRAJECTORY_TARGET_MODE}"
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
        if (
            int(checkpoint.get("opening_iterations", -1))
            != dataset.opening_iterations
        ):
            raise ValueError(
                f"{label} checkpoint morphology cleanup differs from this NPZ"
            )
        if (
            int(checkpoint.get("skeleton_tolerance", -1))
            != dataset.skeleton_tolerance
        ):
            raise ValueError(
                f"{label} checkpoint skeleton tolerance differs from this NPZ"
            )
        for key, dataset_value in (
            ("render_min_width", dataset.render_min_width),
            ("render_max_width", dataset.render_max_width),
            ("render_pressure_gamma", dataset.render_pressure_gamma),
        ):
            if abs(float(checkpoint.get(key, -1.0)) - dataset_value) > 1e-6:
                raise ValueError(
                    f"{label} checkpoint {key} differs from this NPZ"
                )
        if (
            bool(checkpoint.get("render_pressure_invert", False))
            != dataset.render_pressure_invert
        ):
            raise ValueError(
                f"{label} checkpoint pressure direction differs from this NPZ"
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
    if float(model_config.get("geometry_gate_threshold", 0.0)) <= 0.0:
        raise ValueError(
            "v8 training requires character_generator.geometry_gate_threshold > 0"
        )

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
    best_score = float("-inf")
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        if resume_checkpoint.get("scheduler_state"):
            scheduler.load_state_dict(resume_checkpoint["scheduler_state"])
        start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1
        best_score = float(resume_checkpoint.get("best_score", best_score))
        print(f"[RESUME] {args.resume}; start_epoch={start_epoch}")

    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "split_manifest.json", "w", encoding="utf-8") as file:
        json.dump({
            "seed": cfg.train.seed,
            "val_ratio": args.val_ratio,
            "split_mode": args.split_mode,
            "data_format": dataset.data_format,
            "target_mode": dataset.target_mode,
            "render_min_width": dataset.render_min_width,
            "render_max_width": dataset.render_max_width,
            "render_pressure_gamma": dataset.render_pressure_gamma,
            "render_pressure_invert": dataset.render_pressure_invert,
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
        loss_monitor = val_metrics["composite_loss"] if val_metrics else train_loss
        selection_score = (
            val_metrics["selection_score"] if val_metrics else -train_loss
        )
        scheduler.step(loss_monitor)
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

        is_best = selection_score > best_score
        if is_best:
            best_score = selection_score
        save_args = (
            model,
            optimizer,
            scheduler,
            epoch,
            train_loss,
            val_metrics,
            best_score,
            model_config,
            dataset.channel_names,
            dataset.trajectory_padding,
            dataset.trajectory_width,
            dataset.target_mode,
            dataset.structure_threshold,
            dataset.min_component_pixels,
            dataset.opening_iterations,
            dataset.skeleton_tolerance,
            dataset.render_min_width,
            dataset.render_max_width,
            dataset.render_pressure_gamma,
            dataset.render_pressure_invert,
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
