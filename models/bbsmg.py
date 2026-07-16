# 中文注释：本文件定义 B-BSMG 神经网络，用低维笔画参数生成局部笔触图像。
import torch
import torch.nn as nn
from typing import Any, Dict, Optional


def normalize_bbsmg_inputs(
    params: torch.Tensor,
    normalization: Optional[Dict[str, Any]],
) -> torch.Tensor:
    """Apply the exact feature scaling recorded during training."""
    if normalization is None:
        raise RuntimeError(
            "B-BSMG input normalization is missing. Load a checkpoint that "
            "contains input_normalization, or provide the training NPZ so the "
            "normalization can be reconstructed."
        )

    scales = normalization.get("scales")
    if scales is None:
        raise ValueError("input_normalization does not contain 'scales'")

    scale_tensor = torch.as_tensor(
        scales,
        dtype=params.dtype,
        device=params.device,
    )
    if params.shape[-1] != scale_tensor.numel():
        raise ValueError(
            f"Input dimension {params.shape[-1]} does not match normalization "
            f"dimension {scale_tensor.numel()}"
        )
    return params / scale_tensor


# 中文注释：用多层感知机把笔画参数编码为潜变量。
class MLPEncoder(nn.Module):
    # 中文注释：初始化对象并保存后续处理所需的配置和成员变量。
    def __init__(self, input_dim: int = 10, latent_dim: int = 256, hidden_dims=(128, 256, 512)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            prev = h
        layers.append(nn.Linear(prev, latent_dim))
        layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    # 中文注释：定义模型或损失的前向计算逻辑。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# 中文注释：用转置卷积把潜变量解码为局部笔触图像。
class ConvDecoder(nn.Module):
    # 中文注释：初始化对象并保存后续处理所需的配置和成员变量。
    def __init__(self, latent_dim: int = 256, base_channels: int = 64, out_channels: int = 1, image_size: int = 128, use_tanh: bool = False):
        super().__init__()
        self.image_size = image_size
        self.fc = nn.Linear(latent_dim, base_channels * 8 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels, base_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh() if use_tanh else nn.Sigmoid(),
        )

    # 中文注释：定义模型或损失的前向计算逻辑。
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(x.shape[0], -1, 8, 8)
        x = self.decoder(x)
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = nn.functional.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return x


# 中文注释：组合编码器和解码器的笔触生成网络。
class BBSMG(nn.Module):
    # 中文注释：初始化对象并保存后续处理所需的配置和成员变量。
    def __init__(self, input_dim: int = 10, latent_dim: int = 256, base_channels: int = 64, out_channels: int = 1, image_size: int = 128, use_tanh: bool = False):
        super().__init__()
        self.encoder = MLPEncoder(input_dim=input_dim, latent_dim=latent_dim)
        self.decoder = ConvDecoder(latent_dim=latent_dim, base_channels=base_channels, out_channels=out_channels, image_size=image_size, use_tanh=use_tanh)
        self._init_weights()

    # 中文注释：初始化线性层和卷积层权重，提升训练稳定性。
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # 中文注释：定义模型或损失的前向计算逻辑。
    def forward(self, params: torch.Tensor) -> torch.Tensor:
        z = self.encoder(params)
        img = self.decoder(z)
        return img


# 中文注释：根据配置对象创建 B-BSMG 模型。
def build_bbsmg(input_dim: int = 10, latent_dim: int = 256, base_channels: int = 64, out_channels: int = 1, image_size: int = 128, use_tanh: bool = False) -> BBSMG:
    return BBSMG(input_dim=input_dim, latent_dim=latent_dim, base_channels=base_channels, out_channels=out_channels, image_size=image_size, use_tanh=use_tanh)


# 中文注释：作为脚本直接运行时，从这里进入命令行流程或示例测试。
if __name__ == "__main__":
    model = build_bbsmg()
    x = torch.randn(4, 10)
    y = model(x)
    print("input:", x.shape)
    print("output:", y.shape)
# 使用说明：该模块实现了 B-BSMG 的首版 PyTorch 网络。
# 输入是 5 维参数向量 (h, alpha, beta, x0, y0)，先由 MLPEncoder 编码到潜变量，再由 ConvDecoder 逐步上采样解码为 128×128 的单通道笔触图。
# 默认输出经过 Sigmoid 归一化到 [0,1]，便于直接使用 MSE 或 BCE 类损失进行监督训练；
# 如果后续你希望改成以 tanh 为输出范围，只需在构造时设置 use_tanh=True。
# build_bbsmg() 提供统一的模型构造入口，后续 train_bbsmg.py、fusion_renderer.py 和 trajectory_optimizer.py 都可以直接复用这一接口。
