# 中文注释：本文件命令行工具：训练 B-BSMG 笔触生成网络并保存检查点。
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
import random

import numpy as np
import torch
import torch.nn as nn
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
    # 中文注释：初始化对象并保存后续处理所需的配置和成员变量。
    def __init__(self, npz_path: str):
        path = Path(npz_path)
        if not path.exists():
            raise FileNotFoundError(f"Training npz not found: {npz_path}")
        data = np.load(path, allow_pickle=True)
        self.inputs = data["inputs"].astype(np.float32)
        self.targets = data["targets"].astype(np.float32)
        if self.targets.ndim == 3:
            self.targets = self.targets[:, None, :, :]

    # 中文注释：返回数据集或容器中的样本数量。
    def __len__(self) -> int:
        return self.inputs.shape[0]

    # 中文注释：按索引读取并返回单个样本。
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "inputs": torch.from_numpy(self.inputs[index]),
            "targets": torch.from_numpy(self.targets[index]),
        }

# 中文注释：对前景区域赋予更高权重的均方误差损失。
class WeightedMSELoss(nn.Module):
    # 中文注释：初始化对象并保存后续处理所需的配置和成员变量。
    def __init__(self, pos_weight: float = 8.0):
        super().__init__()
        self.pos_weight = pos_weight

    # 中文注释：定义模型或损失的前向计算逻辑。
    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weights = 1.0 + self.pos_weight * targets
        return (weights * (preds - targets) ** 2).mean()

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
        preds = model(inputs)
        loss = criterion(preds, targets)
        loss.backward()
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

    criterion = WeightedMSELoss(pos_weight=4.0)
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
# 使用说明：该脚本假定训练数据已经被整理成一个 .npz 文件，其中至少包含两个键：inputs 和 targets。inputs 的形状应为 [N,5]，顺序对应 (h, alpha, beta, x0, y0)；
# targets 的形状应为 [N,H,W] 或 [N,1,H,W]，通常为 128×128 的单通道笔触监督图。
# 脚本会自动按 val_ratio 划分训练集和验证集，使用 MSELoss 与 Adam 进行训练，并定期在 outputs 目录下保存 epoch 检查点、best 模型和 last 模型。
# 典型运行方式为：python tools/train_bbsmg.py --config configs/default.yaml --npz_path data/processed/bbsmg_train.npz。
