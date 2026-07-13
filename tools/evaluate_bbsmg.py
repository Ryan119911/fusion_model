import argparse
import csv
import json
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from config import load_config
from models.bbsmg import build_bbsmg
from tools.train_bbsmg import (
    BBSMGTrainDataset,
    DiceLoss,
    InkMeanLoss,
    LocalStructureLoss,
    SSIMLoss,
    SobelEdgeLoss,
    SoftCLDiceLoss,
    WeightedMSELoss,
    collate_bbsmg_batch,
    set_seed,
)


LOSS_WEIGHTS = {
    "weighted_mse": 1.0,
    "ssim_loss": 0.3,
    "dice_loss": 0.3,
    "cldice_loss": 0.05,
    "edge_loss": 0.1,
    "structure_loss": 0.05,
    "ink_loss": 0.1,
}


def load_model(cfg, checkpoint_path: str, device: torch.device):
    model = build_bbsmg(
        input_dim=cfg.bbsmg.input_dim,
        latent_dim=cfg.bbsmg.latent_dim,
        base_channels=cfg.bbsmg.base_channels,
        out_channels=cfg.bbsmg.out_channels,
        image_size=cfg.bbsmg.image_size,
        use_tanh=cfg.bbsmg.use_tanh,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state = checkpoint["model_state"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint


def fixed_validation_indices(length: int, val_ratio: float, seed: int):
    if length <= 1:
        return list(range(length))
    val_len = max(1, int(length * val_ratio))
    train_len = length - val_len
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(length, generator=generator).tolist()
    return permutation[train_len:]


def save_gray(array: np.ndarray, path: Path) -> None:
    image = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(image, mode="L").save(path)


def add_average(acc: Dict[str, float], values: Dict[str, float], batch_size: int) -> None:
    for key, value in values.items():
        acc[key] = acc.get(key, 0.0) + float(value) * batch_size


@torch.no_grad()
def evaluate(model, loader, device, output_dir: Path, image_count: int):
    losses = {
        "weighted_mse": WeightedMSELoss(pos_weight=4.0).to(device),
        "ssim_loss": SSIMLoss().to(device),
        "dice_loss": DiceLoss().to(device),
        "cldice_loss": SoftCLDiceLoss(iters=10).to(device),
        "edge_loss": SobelEdgeLoss().to(device),
        "structure_loss": LocalStructureLoss().to(device),
        "ink_loss": InkMeanLoss().to(device),
    }
    totals: Dict[str, float] = {}
    count = 0
    saved = 0

    for batch in loader:
        inputs = batch["inputs"].to(device)
        targets = batch["targets"].to(device).clamp(0.0, 1.0)
        predictions = model(inputs).clamp(0.0, 1.0)
        batch_size = inputs.shape[0]

        component_values = {
            name: float(fn(predictions, targets).item())
            for name, fn in losses.items()
        }
        composite = sum(
            LOSS_WEIGHTS[name] * component_values[name]
            for name in LOSS_WEIGHTS
        )

        pred_binary = predictions >= 0.5
        target_binary = targets >= 0.5
        intersection = (pred_binary & target_binary).sum(dim=(1, 2, 3)).float()
        union = (pred_binary | target_binary).sum(dim=(1, 2, 3)).float()
        iou = ((intersection + 1e-6) / (union + 1e-6)).mean()

        zero_predictions = torch.zeros_like(targets)
        zero_intersection = (zero_predictions.bool() & target_binary).sum(dim=(1, 2, 3)).float()
        zero_union = (zero_predictions.bool() | target_binary).sum(dim=(1, 2, 3)).float()

        values = {
            **component_values,
            "composite_loss": composite,
            "plain_mse": float(F.mse_loss(predictions, targets).item()),
            "mae": float(F.l1_loss(predictions, targets).item()),
            "ssim_score": 1.0 - component_values["ssim_loss"],
            "dice_score": 1.0 - component_values["dice_loss"],
            "iou_at_0.5": float(iou.item()),
            "zero_baseline_mse": float(F.mse_loss(zero_predictions, targets).item()),
            "zero_baseline_mae": float(F.l1_loss(zero_predictions, targets).item()),
            "zero_baseline_dice": 1.0 - float(losses["dice_loss"](zero_predictions, targets).item()),
            "zero_baseline_iou_at_0.5": float(
                ((zero_intersection + 1e-6) / (zero_union + 1e-6)).mean().item()
            ),
        }
        add_average(totals, values, batch_size)
        count += batch_size

        pred_np = predictions.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()
        for i in range(batch_size):
            if saved >= image_count:
                break
            target = target_np[i, 0]
            prediction = pred_np[i, 0]
            difference = np.abs(prediction - target)
            stem = f"sample_{saved:03d}"
            save_gray(target, output_dir / f"{stem}_target.png")
            save_gray(prediction, output_dir / f"{stem}_prediction.png")
            save_gray(difference, output_dir / f"{stem}_diff.png")
            comparison = np.concatenate([target, prediction, difference], axis=1)
            save_gray(comparison, output_dir / f"{stem}_comparison.png")
            saved += 1

    if count == 0:
        raise RuntimeError("Validation subset is empty")
    return {key: value / count for key, value in totals.items()}


def main(args):
    cfg = load_config(args.config)
    set_seed(args.seed)
    device = torch.device(
        cfg.train.device
        if torch.cuda.is_available() or cfg.train.device == "cpu"
        else "cpu"
    )

    dataset = BBSMGTrainDataset(
        args.npz_path,
        coordinate_scale=cfg.bbsmg.image_size,
    )
    if dataset.inputs.shape[1] != cfg.bbsmg.input_dim:
        raise ValueError(
            f"NPZ input dimension {dataset.inputs.shape[1]} does not match "
            f"config input_dim={cfg.bbsmg.input_dim}"
        )

    model, checkpoint = load_model(cfg, args.checkpoint, device)
    checkpoint_normalization = (
        checkpoint.get("input_normalization")
        if isinstance(checkpoint, dict)
        else None
    )
    if checkpoint_normalization is None:
        print("[WARN] Legacy checkpoint: normalization reconstructed from NPZ.")
    else:
        expected = np.asarray(dataset.input_normalization["scales"], dtype=np.float64)
        recorded = np.asarray(checkpoint_normalization["scales"], dtype=np.float64)
        if expected.shape != recorded.shape or not np.allclose(expected, recorded):
            raise ValueError(
                f"Checkpoint normalization {recorded.tolist()} does not match "
                f"NPZ normalization {expected.tolist()}"
            )

    indices = fixed_validation_indices(len(dataset), args.val_ratio, args.seed)
    if args.max_samples > 0:
        indices = indices[: args.max_samples]
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_bbsmg_batch,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = evaluate(model, loader, device, output_dir, args.num_images)
    report = {
        "checkpoint": str(args.checkpoint),
        "npz_path": str(args.npz_path),
        "validation_samples": len(indices),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "input_normalization": dataset.input_normalization,
        "metrics": metrics,
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    with open(output_dir / "metrics.csv", "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    print(f"[DONE] Evaluated {len(indices)} validation samples on {device}.")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")
    print(f"[DONE] Reports and comparison images saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--npz_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/evaluate_bbsmg")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_images", type=int, default=20)
    parser.add_argument("--max_samples", type=int, default=0, help="0 evaluates the full validation subset")
    main(parser.parse_args())
