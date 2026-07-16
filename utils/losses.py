from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedMSELoss(nn.Module):
    def __init__(self, pos_weight: float = 4.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weights = 1.0 + self.pos_weight * targets.clamp(0.0, 1.0)
        return (weights * (preds.clamp(0.0, 1.0) - targets.clamp(0.0, 1.0)) ** 2).mean()


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dims = (1, 2, 3)
        preds = preds.clamp(0.0, 1.0)
        targets = targets.clamp(0.0, 1.0)
        intersection = (preds * targets).sum(dim=dims)
        union = preds.sum(dim=dims) + targets.sum(dim=dims)
        return 1.0 - ((2.0 * intersection + self.eps) / (union + self.eps)).mean()


class InkMeanLoss(nn.Module):
    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(
            preds.clamp(0.0, 1.0).mean(dim=(1, 2, 3)),
            targets.clamp(0.0, 1.0).mean(dim=(1, 2, 3)),
        )


class SSIMLoss(nn.Module):
    def __init__(self, window_size: int = 11, sigma: float = 1.5, eps: float = 1e-6):
        super().__init__()
        coords = torch.arange(window_size).float() - window_size // 2
        gaussian = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        gaussian = gaussian / gaussian.sum()
        window = torch.outer(gaussian, gaussian)
        self.register_buffer("window", (window / window.sum()).view(1, 1, window_size, window_size))
        self.padding = window_size // 2
        self.eps = eps

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = preds.clamp(0.0, 1.0)
        targets = targets.clamp(0.0, 1.0)
        window = self.window.to(device=preds.device, dtype=preds.dtype)
        mu_x = F.conv2d(preds, window, padding=self.padding)
        mu_y = F.conv2d(targets, window, padding=self.padding)
        mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
        sigma_x2 = F.conv2d(preds * preds, window, padding=self.padding) - mu_x2
        sigma_y2 = F.conv2d(targets * targets, window, padding=self.padding) - mu_y2
        sigma_xy = F.conv2d(preds * targets, window, padding=self.padding) - mu_xy
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        score = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
            (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + self.eps
        )
        return 1.0 - score.mean().clamp(0.0, 1.0)


class SobelEdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("kx", torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer("ky", torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))

    def edge_map(self, image: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(image, self.kx.to(image), padding=1)
        gy = F.conv2d(image, self.ky.to(image), padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self.edge_map(preds.clamp(0.0, 1.0)), self.edge_map(targets.clamp(0.0, 1.0)))


def _soft_erode(image: torch.Tensor) -> torch.Tensor:
    return torch.min(
        -F.max_pool2d(-image, kernel_size=(3, 1), stride=1, padding=(1, 0)),
        -F.max_pool2d(-image, kernel_size=(1, 3), stride=1, padding=(0, 1)),
    )


def _soft_skeletonize(image: torch.Tensor, iterations: int) -> torch.Tensor:
    image = image.clamp(0.0, 1.0)
    opened = F.max_pool2d(_soft_erode(image), 3, stride=1, padding=1)
    skeleton = F.relu(image - opened)
    for _ in range(iterations):
        image = _soft_erode(image)
        opened = F.max_pool2d(_soft_erode(image), 3, stride=1, padding=1)
        delta = F.relu(image - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton.clamp(0.0, 1.0)


class SoftCLDiceLoss(nn.Module):
    def __init__(self, iterations: int = 10, eps: float = 1e-6):
        super().__init__()
        self.iterations = iterations
        self.eps = eps

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = preds.clamp(0.0, 1.0)
        targets = targets.clamp(0.0, 1.0)
        skel_pred = _soft_skeletonize(preds, self.iterations)
        skel_target = _soft_skeletonize(targets, self.iterations)
        dims = (1, 2, 3)
        precision = ((skel_pred * targets).sum(dims) + self.eps) / (skel_pred.sum(dims) + self.eps)
        sensitivity = ((skel_target * preds).sum(dims) + self.eps) / (skel_target.sum(dims) + self.eps)
        return 1.0 - ((2 * precision * sensitivity + self.eps) / (precision + sensitivity + self.eps)).mean()


class LocalStructureLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def _moments(self, image: torch.Tensor):
        b, _, h, w = image.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=image.device, dtype=image.dtype),
            torch.linspace(-1.0, 1.0, w, device=image.device, dtype=image.dtype),
            indexing="ij",
        )
        image = image.clamp(0.0, 1.0)
        mass = image.sum((1, 2, 3)) + self.eps
        cx = (image * xx).sum((1, 2, 3)) / mass
        cy = (image * yy).sum((1, 2, 3)) / mass
        dx = xx - cx.view(b, 1, 1, 1)
        dy = yy - cy.view(b, 1, 1, 1)
        cxx = (image * dx * dx).sum((1, 2, 3)) / mass
        cyy = (image * dy * dy).sum((1, 2, 3)) / mass
        cxy = (image * dx * dy).sum((1, 2, 3)) / mass
        trace = cxx + cyy
        radius = torch.sqrt(torch.clamp(trace * trace / 4 - (cxx * cyy - cxy * cxy), min=0.0))
        eig = torch.stack([trace / 2 + radius, trace / 2 - radius], dim=-1).clamp(min=0.0)
        theta = 0.5 * torch.atan2(2 * cxy, cxx - cyy + self.eps)
        direction = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
        return mass / (h * w), torch.stack([cx, cy], dim=-1), eig, direction

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pa, pc, pe, pd = self._moments(preds)
        ta, tc, te, td = self._moments(targets)
        direction = (1.0 - (pd * td).sum(-1).abs().clamp(0.0, 1.0)).mean()
        return (
            F.l1_loss(pc, tc)
            + 0.5 * F.l1_loss(pa, ta)
            + 0.5 * F.l1_loss(torch.sqrt(pe + self.eps), torch.sqrt(te + self.eps))
            + 0.5 * direction
        )


class CompositeStrokeLoss(nn.Module):
    def __init__(
        self,
        weighted_mse: float = 1.0,
        ssim: float = 0.3,
        dice: float = 0.3,
        cldice: float = 0.05,
        edge: float = 0.1,
        structure: float = 0.05,
        ink: float = 0.1,
        positive_weight: float = 4.0,
        cldice_iterations: int = 10,
    ):
        super().__init__()
        self.weights = {
            "weighted_mse": weighted_mse,
            "ssim_loss": ssim,
            "dice_loss": dice,
            "cldice_loss": cldice,
            "edge_loss": edge,
            "structure_loss": structure,
            "ink_loss": ink,
        }
        self.losses = nn.ModuleDict({
            "weighted_mse": WeightedMSELoss(positive_weight),
            "ssim_loss": SSIMLoss(),
            "dice_loss": DiceLoss(),
            "cldice_loss": SoftCLDiceLoss(cldice_iterations),
            "edge_loss": SobelEdgeLoss(),
            "structure_loss": LocalStructureLoss(),
            "ink_loss": InkMeanLoss(),
        })

    def compute_components(self, preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {name: loss(preds, targets) for name, loss in self.losses.items()}

    def combine_components(self, components: Dict[str, torch.Tensor]) -> torch.Tensor:
        return sum(self.weights[name] * value for name, value in components.items())

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.combine_components(self.compute_components(preds, targets))

    def config_dict(self) -> Dict[str, float]:
        return dict(self.weights)
