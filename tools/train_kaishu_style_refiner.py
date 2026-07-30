"""Train a source-grouped Kaishu style refiner without Wu leakage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.style_refiner import (
    STYLE_REFINER_CHECKPOINT_FORMAT,
    build_style_refiner,
)


class StyleDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray, indices: np.ndarray):
        self.features, self.targets, self.indices = features, targets, indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        raw = int(self.indices[index])
        return (
            torch.from_numpy(self.features[raw].astype(np.float32) / 255.0),
            torch.from_numpy(self.targets[raw].astype(np.float32) / 255.0),
        )


def grouped_split(
    characters: np.ndarray,
    sources: np.ndarray,
    heldout_character: str,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    generic = np.flatnonzero(characters != heldout_character)
    heldout = np.flatnonzero(characters == heldout_character)
    groups = np.unique(sources[generic])
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    val_count = max(1, int(round(len(groups) * val_ratio)))
    val_groups = set(groups[:val_count].tolist())
    val = np.asarray([i for i in generic if sources[i] in val_groups], dtype=np.int64)
    train = np.asarray([i for i in generic if sources[i] not in val_groups], dtype=np.int64)
    if set(sources[train]).intersection(sources[val]):
        raise RuntimeError("Source-group leakage detected")
    return train, val, heldout


def ranked_adaptation_split(
    heldout_indices: np.ndarray,
    sources: np.ndarray,
    audit_json: str,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    report = json.loads(Path(audit_json).read_text(encoding="utf-8"))
    ranked_sources = [
        Path(record["image_path"]).as_posix()
        for record in report["ranked_candidates"][:top_k]
    ]
    adapt = np.asarray(
        [i for i in heldout_indices if Path(sources[i]).as_posix() in ranked_sources],
        dtype=np.int64,
    )
    test = np.asarray(
        [i for i in heldout_indices if Path(sources[i]).as_posix() not in ranked_sources],
        dtype=np.int64,
    )
    if not len(adapt) or not len(test):
        raise RuntimeError(
            f"Need non-empty Wu adaptation and test sets, got {len(adapt)} and {len(test)}"
        )
    return adapt, test


def loss_components(prediction: torch.Tensor, target: torch.Tensor) -> dict:
    dims = (1, 2, 3)
    mse = F.mse_loss(prediction, target)
    mae = F.l1_loss(prediction, target)
    intersection = (prediction * target).sum(dim=dims)
    dice = 1.0 - (
        (2 * intersection + 1e-6)
        / (prediction.sum(dim=dims) + target.sum(dim=dims) + 1e-6)
    ).mean()
    grad_pred_x = prediction[..., 1:] - prediction[..., :-1]
    grad_true_x = target[..., 1:] - target[..., :-1]
    grad_pred_y = prediction[..., 1:, :] - prediction[..., :-1, :]
    grad_true_y = target[..., 1:, :] - target[..., :-1, :]
    boundary = F.l1_loss(grad_pred_x, grad_true_x) + F.l1_loss(
        grad_pred_y, grad_true_y
    )
    total = mse + 0.35 * mae + 0.35 * dice + 0.25 * boundary
    return {"loss": total, "mse": mse, "mae": mae, "dice_loss": dice, "boundary": boundary}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    totals, count = {}, 0
    for features, targets in loader:
        features, targets = features.to(device), targets.to(device)
        values = loss_components(model(features), targets)
        batch = features.shape[0]
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + float(value) * batch
        count += batch
    return {key: value / max(count, 1) for key, value in totals.items()}


@torch.no_grad()
def save_panels(model, dataset, device, output_dir: Path, prefix: str, limit: int = 8):
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    for panel_index in range(min(limit, len(dataset))):
        features, target = dataset[panel_index]
        prediction = model(features[None].to(device))[0, 0].cpu().numpy()
        geometry = features[0].numpy()
        target_array = target[0].numpy()
        difference = np.abs(prediction - target_array)
        arrays = [geometry, prediction, target_array, difference]
        labels = ["geometry", "refined", "target", "abs diff"]
        canvas = Image.new("L", (128 * 4, 146), 255)
        draw = ImageDraw.Draw(canvas)
        for index, (array, label) in enumerate(zip(arrays, labels)):
            image = Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8))
            canvas.paste(image, (index * 128, 18))
            draw.text((index * 128 + 3, 2), label, fill=0)
        canvas.save(output_dir / f"{prefix}_{panel_index:02d}.png")


def main(args: argparse.Namespace) -> None:
    npz = np.load(args.npz, allow_pickle=False)
    features, targets = npz["features"], npz["targets"]
    characters, sources = npz["characters"], npz["sources"]
    train_idx, val_idx, heldout_idx = grouped_split(
        characters, sources, args.heldout_character, args.val_ratio, args.seed
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(
        StyleDataset(features, targets, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        StyleDataset(features, targets, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    heldout_loader = DataLoader(
        StyleDataset(features, targets, heldout_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    model = build_style_refiner(base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    history = []
    initial_validation = evaluate(model, val_loader, device)
    initial_heldout = evaluate(model, heldout_loader, device)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            components = loss_components(model(batch_features), batch_targets)
            components["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(components["loss"]) * batch_features.shape[0]
            count += batch_features.shape[0]
        metrics = evaluate(model, val_loader, device)
        record = {"epoch": epoch, "train_loss": total / count, **metrics}
        history.append(record)
        checkpoint = {
            "format": STYLE_REFINER_CHECKPOINT_FORMAT,
            "model_state": model.state_dict(),
            "model_config": {"input_channels": 4, "base_channels": args.base_channels},
            "split": {
                "train": len(train_idx),
                "validation": len(val_idx),
                "heldout_wu": len(heldout_idx),
                "grouped_by_source": True,
            },
            "epoch": epoch,
            "metrics": metrics,
        }
        torch.save(checkpoint, output / "style_refiner_last.pt")
        if metrics["loss"] < best:
            best = metrics["loss"]
            torch.save(checkpoint, output / "style_refiner_best.pt")
        print(
            f"[Epoch {epoch:03d}] train={record['train_loss']:.6f}, "
            f"val={metrics['loss']:.6f}, mse={metrics['mse']:.6f}"
        )
    report = {
        "format": STYLE_REFINER_CHECKPOINT_FORMAT,
        "device": str(device),
        "split": checkpoint["split"],
        "heldout_character": args.heldout_character,
        "history": history,
        "best_val_loss": best,
        "initial_validation": initial_validation,
        "initial_heldout_wu": initial_heldout,
    }
    best_checkpoint = torch.load(
        output / "style_refiner_best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(best_checkpoint["model_state"])
    report["generic_heldout_wu"] = evaluate(model, heldout_loader, device)
    selected_checkpoint = best_checkpoint
    selected_reason = "generic_checkpoint_without_wu_adaptation"
    save_panels(
        model,
        StyleDataset(features, targets, heldout_idx),
        device,
        output / "generic_wu_panels",
        "generic",
    )
    if args.variant_audit_json:
        adapt_idx, test_idx = ranked_adaptation_split(
            heldout_idx,
            sources,
            args.variant_audit_json,
            args.adapt_top_k,
        )
        adapt_loader = DataLoader(
            StyleDataset(features, targets, adapt_idx),
            batch_size=min(args.batch_size, len(adapt_idx)),
            shuffle=True,
        )
        test_dataset = StyleDataset(features, targets, test_idx)
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False
        )
        before = evaluate(model, test_loader, device)
        adapter = torch.optim.AdamW(
            model.parameters(), lr=args.adapt_lr, weight_decay=1e-4
        )
        for _ in range(args.adapt_epochs):
            model.train()
            for batch_features, batch_targets in adapt_loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)
                adapter.zero_grad(set_to_none=True)
                loss = loss_components(model(batch_features), batch_targets)["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                adapter.step()
        after = evaluate(model, test_loader, device)
        adaptation_accepted = bool(
            after["loss"] <= before["loss"] and after["mse"] <= before["mse"]
        )
        adapted_checkpoint = dict(best_checkpoint)
        adapted_checkpoint["model_state"] = model.state_dict()
        adapted_checkpoint["adaptation"] = {
            "top_k": args.adapt_top_k,
            "samples": len(adapt_idx),
            "heldout_test_samples": len(test_idx),
            "audit_json": args.variant_audit_json,
        }
        torch.save(adapted_checkpoint, output / "style_refiner_adapted.pt")
        report["wu_adaptation"] = {
            **adapted_checkpoint["adaptation"],
            "test_before": before,
            "test_after": after,
            "accepted": adaptation_accepted,
            "acceptance_rule": (
                "heldout test loss and MSE must both be no worse after adaptation"
            ),
        }
        if adaptation_accepted:
            selected_checkpoint = adapted_checkpoint
            selected_reason = "wu_adaptation_improved_heldout_loss_and_mse"
        else:
            selected_reason = "wu_adaptation_rejected_due_to_heldout_regression"
        save_panels(
            model,
            test_dataset,
            device,
            output / "adapted_wu_test_panels",
            "adapted",
        )
    selected_checkpoint = dict(selected_checkpoint)
    selected_checkpoint["selection_reason"] = selected_reason
    torch.save(selected_checkpoint, output / "style_refiner_selected.pt")
    report["selected_checkpoint"] = "style_refiner_selected.pt"
    report["selection_reason"] = selected_reason
    (output / "training_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--heldout_character", default="武")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--base_channels", type=int, default=24)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--variant_audit_json", default=None)
    parser.add_argument("--adapt_top_k", type=int, default=5)
    parser.add_argument("--adapt_epochs", type=int, default=20)
    parser.add_argument("--adapt_lr", type=float, default=3e-5)
    main(parser.parse_args())
