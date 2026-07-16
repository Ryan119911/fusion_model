import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from config import load_config
from models.bbsmg import build_bbsmg
from tools.train_bbsmg import (
    BBSMGTrainDataset,
    _load_model_state,
    collate_bbsmg_batch,
    load_torch_checkpoint,
    set_seed,
)
from utils.feature_schema import checkpoint_schema
from utils.losses import CompositeStrokeLoss
from utils.splits import build_split, load_manifest


def save_gray(array: np.ndarray, path: Path) -> None:
    Image.fromarray(
        np.clip(array * 255.0, 0, 255).astype(np.uint8), mode="L"
    ).save(path)


def _loader(
    dataset: BBSMGTrainDataset,
    indices: Sequence[int],
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        Subset(dataset, list(indices)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_bbsmg_batch,
    )


@torch.no_grad()
def evaluate_subset(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: CompositeStrokeLoss,
    device: torch.device,
    output_dir: Optional[Path] = None,
    image_count: int = 0,
) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    count = 0
    saved = 0
    for batch in loader:
        inputs = batch["inputs"].to(device)
        targets = batch["targets"].to(device).clamp(0.0, 1.0)
        predictions = model(inputs).clamp(0.0, 1.0)
        components = criterion.compute_components(predictions, targets)
        component_values = {name: float(value) for name, value in components.items()}
        pred_binary = predictions >= 0.5
        target_binary = targets >= 0.5
        intersection = (pred_binary & target_binary).sum((1, 2, 3)).float()
        union = (pred_binary | target_binary).sum((1, 2, 3)).float()
        zero = torch.zeros_like(targets)
        values = {
            **component_values,
            "composite_loss": float(criterion.combine_components(components)),
            "plain_mse": float(F.mse_loss(predictions, targets)),
            "mae": float(F.l1_loss(predictions, targets)),
            "ssim_score": 1.0 - component_values["ssim_loss"],
            "dice_score": 1.0 - component_values["dice_loss"],
            "iou_at_0.5": float(
                ((intersection + 1e-6) / (union + 1e-6)).mean()
            ),
            "zero_baseline_mse": float(F.mse_loss(zero, targets)),
            "zero_baseline_mae": float(F.l1_loss(zero, targets)),
        }
        batch_size = inputs.shape[0]
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + value * batch_size
        count += batch_size
        if output_dir is not None and saved < image_count:
            pred_np = predictions.cpu().numpy()
            target_np = targets.cpu().numpy()
            for index in range(batch_size):
                if saved >= image_count:
                    break
                target = target_np[index, 0]
                prediction = pred_np[index, 0]
                comparison = np.concatenate(
                    [target, prediction, np.abs(prediction - target)], axis=1
                )
                save_gray(comparison, output_dir / f"sample_{saved:03d}_comparison.png")
                saved += 1
    if count == 0:
        return {}
    return {name: value / count for name, value in totals.items()}


def _metadata(dataset: BBSMGTrainDataset, index: int) -> Dict[str, Any]:
    if dataset.meta is None:
        return {}
    value = dataset.meta[index]
    if isinstance(value, dict):
        return value
    if hasattr(value, "item"):
        value = value.item()
    return value if isinstance(value, dict) else {}


def main(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    set_seed(args.seed)
    device = torch.device(
        cfg.train.device
        if cfg.train.device == "cpu" or torch.cuda.is_available()
        else "cpu"
    )
    dataset = BBSMGTrainDataset(
        args.npz_path,
        coordinate_scale=cfg.bbsmg.image_size,
        expected_schema=cfg.bbsmg.feature_schema,
        target_cache_dir=cfg.train.target_cache_dir,
    )
    checkpoint = load_torch_checkpoint(args.checkpoint, map_location=device)
    schema = checkpoint_schema(checkpoint, dataset.inputs.shape[1])
    if schema != dataset.feature_schema:
        raise ValueError(
            f"Checkpoint schema {schema} does not match NPZ schema {dataset.feature_schema}"
        )
    model = build_bbsmg(
        input_dim=cfg.bbsmg.input_dim,
        latent_dim=cfg.bbsmg.latent_dim,
        base_channels=cfg.bbsmg.base_channels,
        out_channels=cfg.bbsmg.out_channels,
        image_size=cfg.bbsmg.image_size,
        use_tanh=cfg.bbsmg.use_tanh,
    ).to(device)
    model.load_state_dict(_load_model_state(checkpoint))
    model.eval()

    recorded_norm = checkpoint.get("input_normalization") if isinstance(checkpoint, dict) else None
    if recorded_norm:
        expected = np.asarray(dataset.input_normalization["scales"])
        actual = np.asarray(recorded_norm["scales"])
        if expected.shape != actual.shape or not np.allclose(expected, actual):
            raise ValueError("Checkpoint and NPZ normalization differ")

    if args.split_manifest:
        manifest = load_manifest(args.split_manifest, expected_length=len(dataset))
    elif isinstance(checkpoint, dict) and checkpoint.get("split_manifest"):
        manifest = checkpoint["split_manifest"]
    else:
        _, _, manifest = build_split(
            dataset.meta,
            len(dataset),
            args.val_ratio,
            args.seed,
            strategy=cfg.train.split_strategy,
            group_key=cfg.train.split_group_key,
        )
    indices = list(manifest["val_indices"])
    if args.max_samples > 0:
        indices = indices[: args.max_samples]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    criterion = CompositeStrokeLoss(**asdict(cfg.train.loss)).to(device)
    groups = {
        "all": indices,
        "real": [
            index for index in indices
            if bool(_metadata(dataset, index).get("used_real_image"))
        ],
        "synthetic": [
            index for index in indices
            if not bool(_metadata(dataset, index).get("used_real_image"))
        ],
    }
    metrics: Dict[str, Dict[str, float]] = {}
    for name, group_indices in groups.items():
        if not group_indices:
            metrics[name] = {}
            continue
        metrics[name] = evaluate_subset(
            model,
            _loader(dataset, group_indices, args.batch_size, args.num_workers),
            criterion,
            device,
            output_dir if name == "all" else None,
            args.num_images if name == "all" else 0,
        )

    report = {
        "checkpoint": str(args.checkpoint),
        "npz_path": str(args.npz_path),
        "feature_schema": dataset.feature_schema,
        "validation_samples": len(indices),
        "subgroup_samples": {name: len(values) for name, values in groups.items()},
        "split_manifest": manifest,
        "metrics": metrics,
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    rows = []
    for subgroup, values in metrics.items():
        rows.append({"subgroup": subgroup, "samples": len(groups[subgroup]), **values})
    fieldnames = sorted({key for row in rows for key in row})
    with open(output_dir / "metrics.csv", "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[DONE] Evaluated {len(indices)} validation samples on {device}.")
    for subgroup, values in metrics.items():
        summary = values.get("composite_loss")
        print(f"[{subgroup}] samples={len(groups[subgroup])}, composite_loss={summary}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--npz_path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split_manifest")
    parser.add_argument("--output_dir", default="outputs/evaluate_bbsmg")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_images", type=int, default=20)
    parser.add_argument("--max_samples", type=int, default=0)
    main(parser.parse_args())
