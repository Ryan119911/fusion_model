# 中文注释：本文件命令行工具：训练 B-BSMG 笔触生成网络并保存检查点。
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from config import load_config, ensure_dirs
from models.bbsmg import build_bbsmg
from utils.types import BBSMGInput


# 中文注释：设置 Python、NumPy 和 PyTorch 随机种子。
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# 中文注释：读取伪配对 NPZ 文件并提供 B-BSMG 训练样本。
class BBSMGTrainDataset(Dataset):
    def __init__(self, npz_path: str, coordinate_scale: float = 128.0):
        path = Path(npz_path)
        if not path.exists():
            raise FileNotFoundError(f"Training npz not found: {npz_path}")

        data = np.load(path, allow_pickle=True)

        self.inputs = data["inputs"].astype(np.float32)
        target_dtype = data["targets"].dtype
        self.targets = data["targets"].astype(np.float32)
        if np.issubdtype(target_dtype, np.integer):
            self.targets /= float(np.iinfo(target_dtype).max)

        # 如果 targets 是 [N,H,W]，转成 [N,1,H,W]
        if self.targets.ndim == 3:
            self.targets = self.targets[:, None, :, :]

        # Prefer explicit NPZ normalization; retain the legacy fallback.
        metadata = {}
        if "metadata_json" in data:
            raw_metadata = data["metadata_json"]
            raw_metadata = (
                raw_metadata.item()
                if getattr(raw_metadata, "ndim", 0) == 0
                else raw_metadata.tolist()
            )
            metadata = json.loads(str(raw_metadata))
        recorded_normalization = metadata.get("input_normalization")
        if recorded_normalization is not None:
            scales = np.asarray(
                recorded_normalization["scales"], dtype=np.float32
            )
            if scales.shape != (self.inputs.shape[1],):
                raise ValueError("NPZ input_normalization dimension is invalid")
            self.input_normalization = dict(recorded_normalization)
        else:
            h_max = max(float(np.nanmax(self.inputs[:, 0])), 1.0)
            scales = np.ones((self.inputs.shape[1],), dtype=np.float32)
            scales[0] = h_max
            # Preserve normalization for existing 10D/legacy 5D datasets.
            if self.inputs.shape[1] > 3:
                scales[3:] = float(coordinate_scale)
            self.input_normalization = {
                "version": 1,
                "input_dim": int(self.inputs.shape[1]),
                "scales": scales.tolist(),
            }
        self.inputs = self.inputs / scales[None, :]

        print("[CHECK] inputs shape:", self.inputs.shape)
        print("[CHECK] targets shape:", self.targets.shape)

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "inputs": torch.from_numpy(self.inputs[index]),
            "targets": torch.from_numpy(self.targets[index]),
        }

# 中文注释：对前景区域赋予更高权重的均方误差损失。
class WeightedMSELoss(nn.Module):
    def __init__(self, pos_weight: float = 8.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = preds.clamp(0.0, 1.0)
        targets = targets.clamp(0.0, 1.0)
        weights = 1.0 + self.pos_weight * targets
        return (weights * (preds - targets) ** 2).mean()


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = preds.clamp(0.0, 1.0)
        targets = targets.clamp(0.0, 1.0)

        dims = (1, 2, 3)
        inter = (preds * targets).sum(dim=dims)
        union = preds.sum(dim=dims) + targets.sum(dim=dims)

        dice = (2.0 * inter + self.eps) / (union + self.eps)
        return 1.0 - dice.mean()


class InkMeanLoss(nn.Module):
    """
    控制单笔图整体墨量，避免过淡或过浓。
    """
    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = preds.clamp(0.0, 1.0)
        targets = targets.clamp(0.0, 1.0)

        pred_mean = preds.mean(dim=(1, 2, 3))
        target_mean = targets.mean(dim=(1, 2, 3))

        return torch.abs(pred_mean - target_mean).mean()


class SSIMLoss(nn.Module):
    """
    结构相似性损失：让生成笔画在局部结构上更接近 target。
    输入要求: preds/targets shape = [B,1,H,W], range=[0,1]
    """
    def __init__(self, window_size: int = 11, sigma: float = 1.5, eps: float = 1e-12):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.eps = eps

        coords = torch.arange(window_size).float() - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()

        window_2d = torch.outer(g, g)
        window_2d = window_2d / window_2d.sum()

        self.register_buffer(
            "window",
            window_2d.view(1, 1, window_size, window_size),
        )

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = preds.clamp(0.0, 1.0)
        targets = targets.clamp(0.0, 1.0)

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        padding = self.window_size // 2
        window = self.window.to(device=preds.device, dtype=preds.dtype)

        mu_x = F.conv2d(preds, window, padding=padding)
        mu_y = F.conv2d(targets, window, padding=padding)

        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x2 = F.conv2d(preds * preds, window, padding=padding) - mu_x2
        sigma_y2 = F.conv2d(targets * targets, window, padding=padding) - mu_y2
        sigma_xy = F.conv2d(preds * targets, window, padding=padding) - mu_xy

        numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
        denominator = (mu_x2 + mu_y2 + c1) * (
            sigma_x2 + sigma_y2 + c2
        )
        # c1/c2 already keep the denominator positive. A previous 1e-6
        # additive epsilon was larger than c1*c2 (9e-8), so even identical
        # black background scored only about 0.08 SSIM.
        ssim_map = numerator / denominator.clamp_min(self.eps)
        ssim = ssim_map.mean().clamp(0.0, 1.0)
        return 1.0 - ssim


class SobelEdgeLoss(nn.Module):
    """
    Sobel 边缘损失：约束笔画轮廓。
    """
    def __init__(self):
        super().__init__()

        kx = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        ky = torch.tensor(
            [[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def edge_map(self, x: torch.Tensor) -> torch.Tensor:
        kx = self.kx.to(device=x.device, dtype=x.dtype)
        ky = self.ky.to(device=x.device, dtype=x.dtype)

        gx = F.conv2d(x, kx, padding=1)
        gy = F.conv2d(x, ky, padding=1)

        return torch.sqrt(gx * gx + gy * gy + 1e-6)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = preds.clamp(0.0, 1.0)
        targets = targets.clamp(0.0, 1.0)

        pred_edge = self.edge_map(preds)
        target_edge = self.edge_map(targets)

        return F.l1_loss(pred_edge, target_edge)

def soft_erode(img: torch.Tensor) -> torch.Tensor:
    p1 = -F.max_pool2d(-img, kernel_size=(3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-img, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)


def soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)


def soft_open(img: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(img))


def soft_skeletonize(img: torch.Tensor, iters: int = 10) -> torch.Tensor:
    img = img.clamp(0.0, 1.0)

    opened = soft_open(img)
    skel = F.relu(img - opened)

    for _ in range(iters):
        img = soft_erode(img)
        opened = soft_open(img)
        delta = F.relu(img - opened)
        skel = skel + F.relu(delta - skel * delta)

    return skel.clamp(0.0, 1.0)


class SoftCLDiceLoss(nn.Module):
    """
    skeleton / centerline loss:
    约束笔画骨架连通性，减少断笔、骨架偏移。
    """
    def __init__(self, iters: int = 10, eps: float = 1e-6):
        super().__init__()
        self.iters = iters
        self.eps = eps

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = preds.clamp(0.0, 1.0)
        targets = targets.clamp(0.0, 1.0)

        skel_pred = soft_skeletonize(preds, self.iters)
        skel_target = soft_skeletonize(targets, self.iters)

        dims = (1, 2, 3)

        tprec = ((skel_pred * targets).sum(dim=dims) + self.eps) / (
            skel_pred.sum(dim=dims) + self.eps
        )

        tsens = ((skel_target * preds).sum(dim=dims) + self.eps) / (
            skel_target.sum(dim=dims) + self.eps
        )

        cldice = (2.0 * tprec * tsens + self.eps) / (
            tprec + tsens + self.eps
        )

        return 1.0 - cldice.mean()
    
class LocalStructureLoss(nn.Module):
    """
    单笔局部结构误差：
    比较每一笔的位置、面积、主轴尺度和方向。
    """
    def __init__(
        self,
        centroid_weight: float = 1.0,
        area_weight: float = 0.5,
        scale_weight: float = 0.5,
        direction_weight: float = 0.5,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.centroid_weight = centroid_weight
        self.area_weight = area_weight
        self.scale_weight = scale_weight
        self.direction_weight = direction_weight
        self.eps = eps

    def _moments(self, x: torch.Tensor):
        x = x.clamp(0.0, 1.0)
        b, c, h, w = x.shape

        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype),
            torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype),
            indexing="ij",
        )

        xx = xx.view(1, 1, h, w)
        yy = yy.view(1, 1, h, w)

        mass = x.sum(dim=(1, 2, 3)) + self.eps

        cx = (x * xx).sum(dim=(1, 2, 3)) / mass
        cy = (x * yy).sum(dim=(1, 2, 3)) / mass
        centroid = torch.stack([cx, cy], dim=-1)

        dx = xx - cx.view(b, 1, 1, 1)
        dy = yy - cy.view(b, 1, 1, 1)

        cov_xx = (x * dx * dx).sum(dim=(1, 2, 3)) / mass
        cov_yy = (x * dy * dy).sum(dim=(1, 2, 3)) / mass
        cov_xy = (x * dx * dy).sum(dim=(1, 2, 3)) / mass

        trace = cov_xx + cov_yy
        det = cov_xx * cov_yy - cov_xy * cov_xy
        tmp = torch.sqrt(torch.clamp(trace * trace / 4.0 - det, min=0.0))

        lambda1 = trace / 2.0 + tmp
        lambda2 = trace / 2.0 - tmp
        eigvals = torch.stack([lambda1, lambda2], dim=-1)

        theta = 0.5 * torch.atan2(2.0 * cov_xy, cov_xx - cov_yy + self.eps)
        main_dir = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)

        area = mass / float(h * w)

        return area, centroid, eigvals, main_dir

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pred_area, pred_centroid, pred_eig, pred_dir = self._moments(preds)
        tgt_area, tgt_centroid, tgt_eig, tgt_dir = self._moments(targets)

        centroid_loss = F.l1_loss(pred_centroid, tgt_centroid)
        area_loss = F.l1_loss(pred_area, tgt_area)
        pred_eig = torch.clamp(pred_eig, min=0.0)
        tgt_eig = torch.clamp(tgt_eig, min=0.0)

        scale_loss = F.l1_loss(
            torch.sqrt(pred_eig + self.eps),
            torch.sqrt(tgt_eig + self.eps),
        )

        # 主方向正负等价，所以用 abs(dot)
        dot = (pred_dir * tgt_dir).sum(dim=-1).abs().clamp(0.0, 1.0)
        direction_loss = (1.0 - dot).mean()

        return (
            self.centroid_weight * centroid_loss
            + self.area_weight * area_loss
            + self.scale_weight * scale_loss
            + self.direction_weight * direction_loss
        )
    
class CompositeStrokeLoss(nn.Module):
    """
    B-BSMG 单笔监督组合损失：
    MSE + SSIM + Dice + Skeleton + Edge + LocalStructure + Ink
    """
    def __init__(
        self,
        mse_weight: float = 1.0,
        ssim_weight: float = 0.5,
        dice_weight: float = 0.5,
        cldice_weight: float = 0.3,
        edge_weight: float = 0.2,
        structure_weight: float = 0.2,
        ink_weight: float = 0.1,
        pos_weight: float = 4.0,
    ):
        super().__init__()

        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
        self.dice_weight = dice_weight
        self.cldice_weight = cldice_weight
        self.edge_weight = edge_weight
        self.structure_weight = structure_weight
        self.ink_weight = ink_weight

        self.mse = WeightedMSELoss(pos_weight=pos_weight)
        self.ssim = SSIMLoss()
        self.dice = DiceLoss()
        self.cldice = SoftCLDiceLoss(iters=10)
        self.edge = SobelEdgeLoss()
        self.structure = LocalStructureLoss()
        self.ink = InkMeanLoss()

    def compute_components(self, preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        preds = preds.clamp(1e-6, 1.0 - 1e-6)
        targets = targets.clamp(0.0, 1.0)

        components = {
            "weighted_mse": self.mse(preds, targets),
            "ssim_loss": self.ssim(preds, targets),
            "dice_loss": self.dice(preds, targets),
            "cldice_loss": self.cldice(preds, targets),
            "edge_loss": self.edge(preds, targets),
            "structure_loss": self.structure(preds, targets),
            "ink_loss": self.ink(preds, targets),
        }
        for name, value in components.items():
            if not torch.isfinite(value):
                print(f"Non-finite validation component: {name}")
        return components

    def combine_components(self, components: Dict[str, torch.Tensor]) -> torch.Tensor:
        return (
            self.mse_weight * components["weighted_mse"]
            + self.ssim_weight * components["ssim_loss"]
            + self.dice_weight * components["dice_loss"]
            + self.cldice_weight * components["cldice_loss"]
            + self.edge_weight * components["edge_loss"]
            + self.structure_weight * components["structure_loss"]
            + self.ink_weight * components["ink_loss"]
        )

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.combine_components(self.compute_components(preds, targets))

# 中文注释：整理 B-BSMG 样本 batch 为模型输入和目标张量。
def collate_bbsmg_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    inputs = torch.stack([item["inputs"] for item in batch], dim=0)
    targets = torch.stack([item["targets"] for item in batch], dim=0)
    return {"inputs": inputs, "targets": targets}


# 中文注释：根据 NPZ 数据构建训练和验证 DataLoader。
def build_dataloaders(npz_path: str, batch_size: int, num_workers: int, val_ratio: float = 0.1, coordinate_scale: float = 128.0):
    dataset = BBSMGTrainDataset(npz_path, coordinate_scale=coordinate_scale)
    val_len = max(1, int(len(dataset) * val_ratio)) if len(dataset) > 1 else 0
    train_len = len(dataset) - val_len
    if val_len > 0:
        train_set, val_set = random_split(dataset, [train_len, val_len])
    else:
        train_set, val_set = dataset, None
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_bbsmg_batch)
    val_loader = None
    if val_set is not None:
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_bbsmg_batch)
    return train_loader, val_loader, dataset.input_normalization


# 中文注释：执行一个 epoch 的训练并返回平均损失。
def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device) -> float:
    model.train()
    running = 0.0
    count = 0
    for batch in loader:
        inputs = batch["inputs"].to(device)
        targets = batch["targets"].to(device)
        optimizer.zero_grad()
        #preds = model(inputs)
        if not torch.isfinite(inputs).all():
            print("[ERROR] inputs has NaN/Inf")
            print("inputs min/max:", inputs.nan_to_num().min().item(), inputs.nan_to_num().max().item())
            raise RuntimeError("Non-finite inputs")

        if not torch.isfinite(targets).all():
            print("[ERROR] targets has NaN/Inf")
            print("targets min/max:", targets.nan_to_num().min().item(), targets.nan_to_num().max().item())
            raise RuntimeError("Non-finite targets")

        preds = model(inputs)

        if not torch.isfinite(preds).all():
            print("[ERROR] preds has NaN/Inf before loss")
            print("preds finite ratio:", torch.isfinite(preds).float().mean().item())
            print("preds nan count:", torch.isnan(preds).sum().item())
            print("preds inf count:", torch.isinf(preds).sum().item())
            raise RuntimeError("Non-finite preds")
        
        loss = criterion(preds, targets)

        if not torch.isfinite(loss):
            print("[WARN] non-finite loss, skip batch")
            optimizer.zero_grad(set_to_none=True)
            continue

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        if not torch.isfinite(grad_norm):
            print("[WARN] non-finite grad_norm, skip optimizer step")
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.step()
        running += float(loss.item()) * inputs.shape[0]
        count += inputs.shape[0]
    return running / max(count, 1)


# 中文注释：在验证集上评估模型平均损失。
@torch.no_grad()
def validate_detailed(model: nn.Module, loader: Optional[DataLoader], criterion: CompositeStrokeLoss, device: torch.device) -> Optional[Dict[str, float]]:
    if loader is None:
        return None
    model.eval()
    running: Dict[str, float] = {}
    count = 0
    for batch in loader:
        inputs = batch["inputs"].to(device)
        targets = batch["targets"].to(device).clamp(0.0, 1.0)
        preds = model(inputs).clamp(0.0, 1.0)
        components = criterion.compute_components(preds, targets)
        total = criterion.combine_components(components)

        pred_binary = preds >= 0.5
        target_binary = targets >= 0.5
        intersection = (pred_binary & target_binary).sum(dim=(1, 2, 3)).float()
        union = (pred_binary | target_binary).sum(dim=(1, 2, 3)).float()
        iou = ((intersection + 1e-6) / (union + 1e-6)).mean()

        values = {name: float(value.item()) for name, value in components.items()}
        values.update({
            "composite_loss": float(total.item()),
            "plain_mse": float(F.mse_loss(preds, targets).item()),
            "mae": float(F.l1_loss(preds, targets).item()),
            "ssim_score": 1.0 - values["ssim_loss"],
            "dice_score": 1.0 - values["dice_loss"],
            "iou_at_0.5": float(iou.item()),
        })
        batch_size = inputs.shape[0]
        for name, value in values.items():
            running[name] = running.get(name, 0.0) + value * batch_size
        count += batch_size
    return {name: value / max(count, 1) for name, value in running.items()}


def append_metrics_csv(path: Path, epoch: int, train_loss: float, lr: float, val_metrics: Optional[Dict[str, float]]) -> None:
    row: Dict[str, Any] = {
        "epoch": epoch,
        "learning_rate": lr,
        "train_loss": train_loss,
    }
    if val_metrics is not None:
        row.update({f"val_{name}": value for name, value in val_metrics.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# 中文注释：保存模型、优化器状态和配置到检查点文件。
def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, scheduler, epoch: int, train_loss: float, val_metrics: Optional[Dict[str, float]], best_val: Optional[float], ckpt_path: str, input_normalization: Optional[Dict[str, Any]] = None) -> None:
    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
    val_loss = val_metrics.get("composite_loss") if val_metrics is not None else None
    encoder_linears = [
        module for module in model.encoder.net if isinstance(module, nn.Linear)
    ]
    model_config = {
        "input_dim": int(encoder_linears[0].in_features),
        "latent_dim": int(encoder_linears[-1].out_features),
        "base_channels": int(model.decoder.fc.out_features // (8 * 8 * 8)),
        "image_size": int(model.decoder.image_size),
    }
    normalization = input_normalization or {}
    checkpoint_format = normalization.get("checkpoint_format")
    if checkpoint_format is None:
        checkpoint_format = (
            "paper_bbsmg_v1"
            if normalization.get("feature_names")
            == ["H_mm", "alpha_rad", "beta_rad", "x0_px", "y0_px"]
            else "bbsmg_legacy"
        )
    torch.save({
        "format": checkpoint_format,
        "epoch": epoch,
        "model_config": model_config,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_metrics": val_metrics,
        "best_val": best_val,
        "input_normalization": input_normalization,
    }, ckpt_path)


def load_resume_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler, device: torch.device, input_normalization: Dict[str, Any], output_dir: Path):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])

    recorded = checkpoint.get("input_normalization")
    if recorded is not None and not np.allclose(
        np.asarray(recorded["scales"], dtype=np.float64),
        np.asarray(input_normalization["scales"], dtype=np.float64),
    ):
        raise ValueError("Resume checkpoint normalization does not match the current NPZ")
    recorded_basis = (recorded or {}).get(
        "regression_angle_basis", "paper_declared_radian"
    )
    current_basis = input_normalization.get(
        "regression_angle_basis", "paper_declared_radian"
    )
    if recorded_basis != current_basis:
        raise ValueError(
            "Resume checkpoint regression angle basis does not match the current NPZ"
        )

    best_val = checkpoint.get("best_val")
    if best_val is None:
        best_path = output_dir / "bbsmg_best.pt"
        if best_path.exists():
            best_checkpoint = torch.load(best_path, map_location="cpu")
            best_val = best_checkpoint.get("best_val", best_checkpoint.get("val_loss"))
        else:
            best_val = checkpoint.get("val_loss")
    return int(checkpoint.get("epoch", 0)) + 1, best_val


# 中文注释：解析命令行参数，准备日志文件并分派到对应子命令。
def main(args):
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.output_dir is not None:
        cfg.train.output_dir = args.output_dir
    elif args.resume is not None:
        cfg.train.output_dir = str(Path(args.resume).parent)
    ensure_dirs(cfg)
    set_seed(cfg.train.seed)

    device = torch.device(cfg.train.device if torch.cuda.is_available() or cfg.train.device == "cpu" else "cpu")
    train_loader, val_loader, input_normalization = build_dataloaders(
        npz_path=args.npz_path,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        val_ratio=args.val_ratio,
        coordinate_scale=cfg.bbsmg.image_size,
    )
    if input_normalization["input_dim"] != cfg.bbsmg.input_dim:
        raise ValueError(
            f"NPZ input dimension {input_normalization['input_dim']} does not "
            f"match config bbsmg.input_dim={cfg.bbsmg.input_dim}. "
            "Rebuild the NPZ or use configs/legacy_5d.yaml for an old model."
        )

    model = build_bbsmg(
        input_dim=cfg.bbsmg.input_dim,
        latent_dim=cfg.bbsmg.latent_dim,
        base_channels=cfg.bbsmg.base_channels,
        out_channels=cfg.bbsmg.out_channels,
        image_size=cfg.bbsmg.image_size,
        use_tanh=cfg.bbsmg.use_tanh,
    ).to(device)

    criterion = CompositeStrokeLoss(
        mse_weight=1.0,
        ssim_weight=0.3,
        dice_weight=0.3,
        cldice_weight=0.05,
        edge_weight=0.1,
        structure_weight=0.05,
        ink_weight=0.1,
        pos_weight=4.0,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )

    best_val = None
    start_epoch = 1
    output_dir = Path(cfg.train.output_dir)
    if args.resume is not None:
        start_epoch, best_val = load_resume_checkpoint(
            args.resume,
            model,
            optimizer,
            scheduler,
            device,
            input_normalization,
            output_dir,
        )
        print(
            f"[RESUME] checkpoint={args.resume}, start_epoch={start_epoch}, "
            f"target_epochs={cfg.train.epochs}, best_val={best_val}"
        )

    if start_epoch > cfg.train.epochs:
        print(
            f"[DONE] Checkpoint already reached epoch {start_epoch - 1}; "
            f"requested total epochs={cfg.train.epochs}."
        )
        return

    metrics_path = output_dir / "training_metrics.csv"
    for epoch in range(start_epoch, cfg.train.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = validate_detailed(model, val_loader, criterion, device)
        val_loss = val_metrics.get("composite_loss") if val_metrics is not None else None
        monitor_loss = val_loss if val_loss is not None else train_loss
        scheduler.step(monitor_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])

        msg = f"[Epoch {epoch:03d}] train_loss={train_loss:.6f}, lr={current_lr:.8g}"
        if val_loss is not None:
            msg += f", val_loss={val_loss:.6f}"
        print(msg)
        if val_metrics is not None:
            print(
                "[VAL COMPONENTS] "
                + ", ".join(
                    f"{name}={value:.6f}"
                    for name, value in val_metrics.items()
                )
            )

        append_metrics_csv(metrics_path, epoch, train_loss, current_lr, val_metrics)

        is_best = val_loss is not None and (best_val is None or val_loss < best_val)
        if is_best:
            best_val = val_loss

        if epoch % cfg.train.save_interval == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_metrics, best_val, str(output_dir / f"bbsmg_epoch_{epoch:03d}.pt"), input_normalization)

        if is_best:
            save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_metrics, best_val, str(output_dir / "bbsmg_best.pt"), input_normalization)

        # Always refresh last checkpoint so interrupted runs can continue.
        save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_metrics, best_val, str(output_dir / "bbsmg_last.pt"), input_normalization)


# 中文注释：作为脚本直接运行时，从这里进入命令行流程或示例测试。
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to yaml config")
    parser.add_argument("--npz_path", type=str, required=True, help="Training dataset npz with keys: inputs, targets")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--epochs", type=int, default=None, help="Override the configured number of epochs")
    parser.add_argument("--output_dir", type=str, default=None, help="Override the configured checkpoint directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume model, optimizer, scheduler and epoch from a checkpoint")
    parser.add_argument("--lr_factor", type=float, default=0.5, help="Learning-rate decay factor")
    parser.add_argument("--lr_patience", type=int, default=3, help="Validation plateaus before reducing learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate")
    args = parser.parse_args()
    main(args)
