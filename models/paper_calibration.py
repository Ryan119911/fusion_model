"""Versioned numerical data transcribed or digitized from the two papers.

The B-BSMG regression table is transcribed exactly in ``paper_bbsm.py``.
Wang et al. (2020) state ``K_w = K_d = 0.02`` exactly, but only plot the
Width/Drag/Offset polynomial fits in Figure 4.  The coefficients below are
therefore digitized approximations of the orange fitted curves, not author
supplied calibration coefficients.

The two papers used different brushes and coordinate conventions.  Absolute
centimetres from Wang Figure 4 must not be silently treated as B-BSM model
units.  The renderer transfers only the dimensionless Offset/Drag ratio; its
width and drag dimensions still come from the B-BSMG regression fitted to the
B-BSMG brush.
"""
from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np

try:
    import torch
except ImportError:  # Metadata and NumPy evaluation do not require PyTorch.
    torch = None


WANG2020_PROFILE = "wang2020_figure4_digitized_v1"
LEGACY_OFFSET_PROFILE = "legacy_fraction_v1"
DYNAMIC_PROFILES = (WANG2020_PROFILE, LEGACY_OFFSET_PROFILE)

WANG2020_HEIGHT_MIN_CM = 0.0
WANG2020_HEIGHT_MAX_CM = 1.5
WANG2020_WIDTH_INERTIA = 0.02
WANG2020_DRAG_INERTIA = 0.02

# np.polyval order (highest power first).  Digitized from the orange curves in
# Wang et al. 2020, Figure 4, rendered at 250 dpi.  Axis calibration used
# z=[0,1.5] cm and value=[0,1.75] cm.
WANG2020_WIDTH_COEFFICIENTS = (1.1688916535, -0.0038450050)
WANG2020_DRAG_COEFFICIENTS = (1.0560234265, 0.1625819413)
WANG2020_OFFSET_COEFFICIENTS = (
    -0.9548991597,
    2.0747044422,
    -0.0102216952,
)
WANG2020_DIGITIZATION_RMSE_CM = {
    "width": 0.00119452,
    "drag": 0.00134559,
    "offset": 0.00201777,
}


def _polyval_numpy(coefficients: Sequence[float], value: np.ndarray) -> np.ndarray:
    result = np.zeros_like(value, dtype=np.float32)
    for coefficient in coefficients:
        result = result * value + np.float32(coefficient)
    return result


def _polyval_torch(coefficients: Sequence[float], value: "torch.Tensor"):
    result = torch.zeros_like(value)
    for coefficient in coefficients:
        result = result * value + float(coefficient)
    return result


def bbsm_h_to_wang_height_numpy(h_mm: np.ndarray) -> np.ndarray:
    """Map the B-BSMG descending-height range to Wang's Figure 4 range.

    This is an explicit normalized-range bridge, not a physical robot-frame
    calibration.  Real brush calibration must replace it later.
    """
    h = np.asarray(h_mm, dtype=np.float32)
    normalized = np.clip((h - 11.0) / 9.0, 0.0, 1.0)
    return normalized * WANG2020_HEIGHT_MAX_CM


def bbsm_h_to_wang_height_torch(h_mm: "torch.Tensor"):
    normalized = ((h_mm - 11.0) / 9.0).clamp(0.0, 1.0)
    return normalized * WANG2020_HEIGHT_MAX_CM


def wang2020_curves_numpy(height_cm: np.ndarray) -> Dict[str, np.ndarray]:
    z = np.clip(
        np.asarray(height_cm, dtype=np.float32),
        WANG2020_HEIGHT_MIN_CM,
        WANG2020_HEIGHT_MAX_CM,
    )
    return {
        "width_cm": np.maximum(
            _polyval_numpy(WANG2020_WIDTH_COEFFICIENTS, z), 0.0
        ),
        "drag_cm": np.maximum(
            _polyval_numpy(WANG2020_DRAG_COEFFICIENTS, z), 1e-6
        ),
        "offset_cm": np.maximum(
            _polyval_numpy(WANG2020_OFFSET_COEFFICIENTS, z), 0.0
        ),
    }


def wang2020_curves_torch(height_cm: "torch.Tensor") -> Dict[str, "torch.Tensor"]:
    if torch is None:
        raise RuntimeError("PyTorch is required for differentiable calibration")
    z = height_cm.clamp(WANG2020_HEIGHT_MIN_CM, WANG2020_HEIGHT_MAX_CM)
    return {
        "width_cm": _polyval_torch(
            WANG2020_WIDTH_COEFFICIENTS, z
        ).clamp_min(0.0),
        "drag_cm": _polyval_torch(
            WANG2020_DRAG_COEFFICIENTS, z
        ).clamp_min(1e-6),
        "offset_cm": _polyval_torch(
            WANG2020_OFFSET_COEFFICIENTS, z
        ).clamp_min(0.0),
    }


def wang2020_offset_drag_ratio_torch(h_mm: "torch.Tensor"):
    curves = wang2020_curves_torch(bbsm_h_to_wang_height_torch(h_mm))
    return (curves["offset_cm"] / curves["drag_cm"]).clamp(0.0, 1.5)


def wang2020_dimension_progress_torch(
    h_mm: "torch.Tensor",
) -> Dict[str, "torch.Tensor"]:
    """Return Figure 4 curve progress normalized to each endpoint range."""
    z = bbsm_h_to_wang_height_torch(h_mm)
    curves = wang2020_curves_torch(z)
    lower = wang2020_curves_torch(torch.zeros_like(z))
    upper = wang2020_curves_torch(
        torch.full_like(z, WANG2020_HEIGHT_MAX_CM)
    )
    width_progress = (
        (curves["width_cm"] - lower["width_cm"])
        / (upper["width_cm"] - lower["width_cm"]).clamp_min(1e-6)
    ).clamp(0.0, 1.0)
    drag_progress = (
        (curves["drag_cm"] - lower["drag_cm"])
        / (upper["drag_cm"] - lower["drag_cm"]).clamp_min(1e-6)
    ).clamp(0.0, 1.0)
    return {
        "width_progress": width_progress,
        "drag_progress": drag_progress,
    }


def paper_calibration_metadata(
    active_profile: str = WANG2020_PROFILE,
) -> Dict[str, Any]:
    if active_profile not in DYNAMIC_PROFILES:
        raise ValueError(
            f"Unknown dynamic profile {active_profile!r}; "
            f"expected one of {DYNAMIC_PROFILES}"
        )
    return {
        "profile": active_profile,
        "uses_figure4_curves": active_profile == WANG2020_PROFILE,
        "simulation_only": True,
        "exact_paper_values": {
            "width_inertia_Kw": WANG2020_WIDTH_INERTIA,
            "drag_inertia_Kd": WANG2020_DRAG_INERTIA,
            "psoc_order_range": [3, 8],
            "jacobian": "numerical_difference",
        },
        "figure4_digitized_approximation": {
            "height_range_cm": [
                WANG2020_HEIGHT_MIN_CM,
                WANG2020_HEIGHT_MAX_CM,
            ],
            "coefficient_order": "descending_power",
            "width_coefficients": list(WANG2020_WIDTH_COEFFICIENTS),
            "drag_coefficients": list(WANG2020_DRAG_COEFFICIENTS),
            "offset_coefficients": list(WANG2020_OFFSET_COEFFICIENTS),
            "curve_fit_pixel_rmse_cm": WANG2020_DIGITIZATION_RMSE_CM,
            "source": "Wang et al. 2020, Figure 4 orange fitted curves",
        },
        "cross_paper_bridge": {
            "height": "linear normalized range: B-BSMG H=11..20 mm -> Wang z=0..1.5 cm",
            "width_drag": (
                "Wang Width/Drag curve progress remapped to B-BSMG endpoint "
                "dimensions, retaining B-BSMG angle deltas"
            ),
            "offset": "Wang dimensionless Offset(z)/Drag(z) ratio times dynamic B-BSM drag",
            "status": "simulation assumption; replace with real brush calibration",
        },
        "unreported_values": {
            "wang_polynomial_author_coefficients": None,
            "wang_terminal_height_beta_weights": None,
        },
    }
