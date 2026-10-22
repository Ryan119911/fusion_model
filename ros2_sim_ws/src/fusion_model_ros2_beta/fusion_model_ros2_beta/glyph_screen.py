"""Conservative, quantitative screening of catalog glyph targets.

Script conversion examines labels, not ink shapes.  Passing this screen is
therefore deliberately named ``screened`` rather than ``verified``.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


SCREEN_METHOD = "opencc_bidirectional_and_model_support_ink_ratio_v3"
SCREEN_REPORT_FORMAT = "v16_original_target_glyph_screen_v3"


def screen_glyph(
    character: str, target_ink: np.ndarray, support: np.ndarray,
    *, ratio_max: float, simplified_to_traditional: Callable[[str], str],
    traditional_to_simplified: Callable[[str], str],
) -> dict:
    if len(character) != 1 or not np.isfinite(ratio_max) or ratio_max <= 0:
        raise ValueError("invalid glyph screen character or ratio threshold")
    ink = np.asarray(target_ink, dtype=np.float32)
    area = np.asarray(support, dtype=bool)
    if ink.ndim != 2 or ink.shape != area.shape or not np.isfinite(ink).all():
        raise ValueError("target and trajectory support must share finite 2-D pixels")
    within = float(ink[area].sum())
    outside = float(ink[~area].sum())
    outside_over_inside = outside / max(within, 1.0e-9)
    # Match target_match_score(): area of target pixels at threshold 0.35
    # divided by the 7 px model-support area. This is an area ratio, not the
    # fraction of target ink lying outside the support region.
    target_area = int(np.count_nonzero(ink >= .35))
    support_area = int(np.count_nonzero(area))
    model_support_ink_ratio = target_area / max(support_area, 1)
    s2t = simplified_to_traditional(character)
    t2s = traditional_to_simplified(character)
    script_invariant = s2t == character and t2s == character
    geometry_pass = support_area > 0 and target_area > 0 and model_support_ink_ratio <= ratio_max
    reasons = []
    if not script_invariant:
        reasons.append("character_has_simplified_traditional_variant")
    if not geometry_pass:
        reasons.append("target_to_model_support_ink_area_ratio_exceeds_gate")
    # The ratio is a size/brush-support anomaly, not a glyph classifier: a
    # visually correct thick 一 can exceed 2.0. Keep those for review, while
    # a known simplified/traditional variant is quarantined explicitly.
    status = ("quarantined" if not script_invariant else
              "screened" if geometry_pass else "needs_review")
    return {
        "status": status,
        "character": character,
        "method": SCREEN_METHOD,
        "outside_support_over_inside": outside_over_inside,
        "outside_support_ink_mass": outside,
        "inside_support_ink_mass": within,
        "target_ink_area_at_0_35_px": target_area,
        "model_support_area_px": support_area,
        "target_to_model_support_ink_ratio": model_support_ink_ratio,
        "ratio_max": ratio_max,
        "support_definition": "trajectory_model_support_width_7_px_target_threshold_0_35",
        "script_invariant": script_invariant,
        "opencc_s2t": s2t,
        "opencc_t2s": t2s,
        "reasons": reasons,
        "semantic_glyph_identity_proven": False,
    }
