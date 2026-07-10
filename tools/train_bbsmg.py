# 中文注释：本文件命令行工具：训练 B-BSMG 笔触生成网络并保存检查点。
import argparse
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
    def __init__(self, npz_path: str):
        path = Path(npz_path)
        if not path.exists():
            raise FileNotFoundError(f"Training npz not found: {npz_path}")

        data = np.load(path, allow_pickle=True)

        self.inputs = data["inputs"].astype(np.float32)
        self.targets = data["targets"].astype(np.float32)

        # 如果 targets 是 [N,H,W]，转成 [N,1,H,W]
        if self.targets.ndim == 3:
            self.targets = self.targets[:, None, :, :]

        # 输入归一化：你现在是 5D 输入
        h_max = max(float(np.nanmax(self.inputs[:, 0])), 1.0)
        self.inputs[:, 0] = self.inputs[:, 0] / h_max

        self.inputs[:, 3] = self.inputs[:, 3] / 128.0
        self.inputs[:, 4] = self.inputs[:, 4] / 128.0

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
    def __init__(self, window_size: int = 11, sigma: float = 1.5, eps: float = 1e-6):
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

        ssim_map = (
            (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
            / ((mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + self.eps)
        )
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

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = preds.clamp(1e-6, 1.0 - 1e-6)
        targets = targets.clamp(0.0, 1.0)

        loss_mse = self.mse(preds, targets)
        loss_ssim = self.ssim(preds, targets)
        loss_dice = self.dice(preds, targets)
        loss_cldice = self.cldice(preds, targets)
        loss_edge = self.edge(preds, targets)
        loss_structure = self.structure(preds, targets)
        loss_ink = self.ink(preds, targets)
        if torch.isnan(loss_mse):
            print("NaN in mse")

        if torch.isnan(loss_ssim):
            print("NaN in ssim")

        if torch.isnan(loss_dice):
            print("NaN in dice")

        if torch.isnan(loss_cldice):
            print("NaN in cldice")

        if torch.isnan(loss_edge):
            print("NaN in edge")

        if torch.isnan(loss_structure):
            print("NaN in structure")

        if torch.isnan(loss_ink):
            print("NaN in ink")


        total = (
            self.mse_weight * loss_mse
            + self.ssim_weight * loss_ssim
            + self.dice_weight * loss_dice
            + self.cldice_weight * loss_cldice
            + self.edge_weight * loss_edge
            + self.structure_weight * loss_structure
            + self.ink_weight * loss_ink
        )

        return total

# 中文注释：整理 B-BSMG 样本 batch 为模型输入和目标张量。
def collate_bbsmg_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    inputs = torch.stack([item["inputs"] for item in batch], dim=0)
    targets = torch.stack([item["targets"] for item in batch], dim=0)
    return {"inputs": inputs, "targets": targets}


# 中文注释：根据 NPZ 数据构建训练和验证 DataLoader。
def build_dataloaders(npz_path: str, batch_size: int, num_workers: int, val_ratio: float = 0.1):
    dataset = BBSMGTrainDataset(npz_path)
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
    return train_loader, val_loader


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
def validate(model: nn.Module, loader: Optional[DataLoader], criterion: nn.Module, device: torch.device) -> Optional[float]:
    if loader is None:
        return None
    model.eval()
    running = 0.0
    count = 0
    for batch in loader:
        inputs = batch["inputs"].to(device)
        targets = batch["targets"].to(device)
        preds = model(inputs)
        loss = criterion(preds, targets)
        running += float(loss.item()) * inputs.shape[0]
        count += inputs.shape[0]
    return running / max(count, 1)


# 中文注释：保存模型、优化器状态和配置到检查点文件。
def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, train_loss: float, val_loss: Optional[float], ckpt_path: str) -> None:
    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
    }, ckpt_path)


# 中文注释：解析命令行参数，准备日志文件并分派到对应子命令。
def main(args):
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    set_seed(cfg.train.seed)

    device = torch.device(cfg.train.device if torch.cuda.is_available() or cfg.train.device == "cpu" else "cpu")
    train_loader, val_loader = build_dataloaders(
        npz_path=args.npz_path,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        val_ratio=args.val_ratio,
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

    best_val = None
    for epoch in range(1, cfg.train.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        msg = f"[Epoch {epoch:03d}] train_loss={train_loss:.6f}"
        if val_loss is not None:
            msg += f", val_loss={val_loss:.6f}"
        print(msg)

        if epoch % cfg.train.save_interval == 0:
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss, str(Path(cfg.train.output_dir) / f"bbsmg_epoch_{epoch:03d}.pt"))

        if val_loss is not None and (best_val is None or val_loss < best_val):
            best_val = val_loss
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss, str(Path(cfg.train.output_dir) / "bbsmg_best.pt"))

    save_checkpoint(model, optimizer, cfg.train.epochs, train_loss, val_loss, str(Path(cfg.train.output_dir) / "bbsmg_last.pt"))


# 中文注释：作为脚本直接运行时，从这里进入命令行流程或示例测试。
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to yaml config")
    parser.add_argument("--npz_path", type=str, required=True, help="Training dataset npz with keys: inputs, targets")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio")
    args = parser.parse_args()
    main(args)
