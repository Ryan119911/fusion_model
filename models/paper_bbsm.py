"""Paper-aligned B-BSM geometry for the simulation prototype.

The regression coefficients are transcribed from the B-BSMG paper. They are
temporary simulation calibration, not real brush or robot calibration data.
External posture angles are always radians.  The explicit angle-basis option
exists because the paper text declares radians while its sampled levels,
plots, and coefficient magnitudes may indicate that degree values were used
when fitting the published regression.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw

try:
    import torch
except ImportError:  # Dataset construction does not require PyTorch.
    torch = None


PAPER_POSTURE_MIN = np.asarray([11.0, 0.0, 0.0], dtype=np.float32)
PAPER_POSTURE_MAX = np.asarray(
    [20.0, np.deg2rad(10.0), np.deg2rad(5.0)], dtype=np.float32
)

# Rows: Lt, Lh, Lr. Columns: H, alpha, beta.
PAPER_REGRESSION_MATRIX = np.asarray(
    [
        [0.0672, 0.0263, 0.0191],
        [0.0196, 0.0039, 0.0073],
        [0.0239, 0.0061, 0.0096],
    ],
    dtype=np.float32,
)
PAPER_REGRESSION_BIAS = np.asarray([0.0267, 0.0372, 0.1137], dtype=np.float32)
PAPER_ANGLE_BASIS_RADIAN = "paper_declared_radian"
PAPER_ANGLE_BASIS_DEGREE_FITTED = "degree_fitted"
PAPER_ANGLE_BASES = (
    PAPER_ANGLE_BASIS_RADIAN,
    PAPER_ANGLE_BASIS_DEGREE_FITTED,
)


@dataclass(frozen=True)
class PaperPrototypeLimits:
    h_min_mm: float = 11.0
    h_max_mm: float = 20.0
    alpha_min_rad: float = 0.0
    alpha_max_rad: float = float(np.deg2rad(10.0))
    beta_min_rad: float = 0.0
    beta_max_rad: float = float(np.deg2rad(5.0))
    gamma_rad: float = 0.0


def regression_matrix_numpy(
    angle_basis: str = PAPER_ANGLE_BASIS_RADIAN,
) -> np.ndarray:
    """Return coefficients acting on external [H_mm, alpha_rad, beta_rad]."""
    if angle_basis not in PAPER_ANGLE_BASES:
        raise ValueError(
            f"Unknown regression angle basis {angle_basis!r}; "
            f"expected one of {PAPER_ANGLE_BASES}"
        )
    matrix = PAPER_REGRESSION_MATRIX.copy()
    if angle_basis == PAPER_ANGLE_BASIS_DEGREE_FITTED:
        matrix[:, 1:] *= np.float32(180.0 / np.pi)
    return matrix


def posture_to_geometry_numpy(
    posture: np.ndarray,
    angle_basis: str = PAPER_ANGLE_BASIS_RADIAN,
) -> np.ndarray:
    posture = np.asarray(posture, dtype=np.float32)
    matrix = regression_matrix_numpy(angle_basis)
    return posture @ matrix.T + PAPER_REGRESSION_BIAS


def posture_to_geometry_torch(
    posture: torch.Tensor,
    angle_basis: str = PAPER_ANGLE_BASIS_RADIAN,
) -> torch.Tensor:
    if torch is None:
        raise RuntimeError("PyTorch is required for differentiable rendering")
    matrix = torch.as_tensor(
        regression_matrix_numpy(angle_basis),
        dtype=posture.dtype,
        device=posture.device,
    )
    bias = torch.as_tensor(
        PAPER_REGRESSION_BIAS, dtype=posture.dtype, device=posture.device
    )
    return posture @ matrix.T + bias


def geometry_to_posture_torch(
    geometry: torch.Tensor,
    reference: torch.Tensor | None = None,
    regularization: float = 1e-4,
    angle_basis: str = PAPER_ANGLE_BASIS_RADIAN,
) -> torch.Tensor:
    """Invert the regression with a small reference-pose regularizer."""
    if torch is None:
        raise RuntimeError("PyTorch is required for differentiable rendering")
    matrix = torch.as_tensor(
        regression_matrix_numpy(angle_basis),
        dtype=geometry.dtype,
        device=geometry.device,
    )
    bias = torch.as_tensor(
        PAPER_REGRESSION_BIAS, dtype=geometry.dtype, device=geometry.device
    )
    if reference is None:
        reference = torch.as_tensor(
            (PAPER_POSTURE_MIN + PAPER_POSTURE_MAX) / 2.0,
            dtype=geometry.dtype,
            device=geometry.device,
        ).expand_as(geometry)
    eye = torch.eye(3, dtype=geometry.dtype, device=geometry.device)
    lhs = matrix.T @ matrix + float(regularization) * eye
    rhs = (geometry - bias) @ matrix + float(regularization) * reference
    return torch.linalg.solve(lhs, rhs.T).T


def clamp_posture_torch(posture: torch.Tensor) -> torch.Tensor:
    if torch is None:
        raise RuntimeError("PyTorch is required for differentiable rendering")
    lower = torch.as_tensor(
        PAPER_POSTURE_MIN, dtype=posture.dtype, device=posture.device
    )
    upper = torch.as_tensor(
        PAPER_POSTURE_MAX, dtype=posture.dtype, device=posture.device
    )
    return torch.maximum(torch.minimum(posture, upper), lower)


def bbsm_boundary(
    lt: float,
    lh: float,
    lr: float,
    samples_per_side: int = 64,
) -> np.ndarray:
    """Return the symmetric cubic Bézier B-BSM outline in model units."""
    p0 = np.asarray([-lt, 0.0], dtype=np.float64)
    p3 = np.asarray([lh, 0.0], dtype=np.float64)
    p1 = np.asarray([(lt - 4.0 * lh) / 3.0, 4.0 * lr / 3.0])
    p2 = np.asarray([lh, 4.0 * lr / 3.0])
    t = np.linspace(0.0, 1.0, samples_per_side, dtype=np.float64)[:, None]
    upper = (
        (1.0 - t) ** 3 * p0
        + 3.0 * (1.0 - t) ** 2 * t * p1
        + 3.0 * (1.0 - t) * t**2 * p2
        + t**3 * p3
    )
    lower = upper[::-1].copy()
    lower[:, 1] *= -1.0
    return np.concatenate([upper, lower], axis=0)


def render_bbsm_mask(
    posture: np.ndarray,
    x0: float,
    y0: float,
    image_size: int = 128,
    pixels_per_model_unit: float = 20.0,
    supersample: int = 4,
    angle_basis: str = PAPER_ANGLE_BASIS_RADIAN,
) -> np.ndarray:
    """Rasterize one analytic B-BSM target with background=0 and ink=1."""
    h, alpha, beta = np.asarray(posture, dtype=np.float64).tolist()
    lt, lh, lr = posture_to_geometry_numpy(
        np.asarray([[h, alpha, beta]], dtype=np.float32),
        angle_basis=angle_basis,
    )[0]
    points = bbsm_boundary(float(lt), float(lh), float(lr))
    c, s = np.cos(beta), np.sin(beta)
    points = points @ np.asarray([[c, -s], [s, c]], dtype=np.float64).T
    points *= float(pixels_per_model_unit)
    points += np.asarray([x0, y0], dtype=np.float64)

    scale = max(int(supersample), 1)
    canvas = Image.new("L", (image_size * scale, image_size * scale), 0)
    ImageDraw.Draw(canvas).polygon(
        [(float(x * scale), float(y * scale)) for x, y in points],
        fill=255,
    )
    if scale > 1:
        canvas = canvas.resize((image_size, image_size), Image.Resampling.LANCZOS)
    return np.asarray(canvas, dtype=np.float32) / 255.0


def safe_anchor_ranges(
    image_size: int = 128,
    pixels_per_model_unit: float = 20.0,
    margin: float = 2.0,
    angle_basis: str = PAPER_ANGLE_BASIS_RADIAN,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    corners = np.stack([PAPER_POSTURE_MIN, PAPER_POSTURE_MAX])
    geometry = posture_to_geometry_numpy(corners, angle_basis=angle_basis)
    radius = float(np.max(geometry) * pixels_per_model_unit + margin)
    return (radius, image_size - 1.0 - radius), (
        radius,
        image_size - 1.0 - radius,
    )
