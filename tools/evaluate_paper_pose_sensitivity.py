"""Evaluate whether a paper B-BSMG checkpoint responds to each pose field."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.bbsmg import normalize_bbsmg_inputs
from models.paper_bbsm import (
    PAPER_POSTURE_MAX,
    PAPER_POSTURE_MIN,
    render_bbsm_mask,
)
from models.paper_fusion_renderer import PaperFusionRenderer


def dice_score(prediction: np.ndarray, target: np.ndarray) -> float:
    pred = prediction >= 0.5
    true = target >= 0.5
    denominator = pred.sum() + true.sum()
    return float(
        (2.0 * np.logical_and(pred, true).sum() + 1e-6)
        / (denominator + 1e-6)
    )


def evaluate_field(
    model: torch.nn.Module,
    normalization: dict,
    field_index: int,
    field_name: str,
    lower: float,
    upper: float,
    samples: int,
    image_size: int,
    pixels_per_model_unit: float,
    supersample: int,
    device: torch.device,
) -> dict:
    feature_names = normalization["feature_names"]
    gamma_conditioned = "gamma_rad" in feature_names
    baseline = np.asarray(
        [
            float((PAPER_POSTURE_MIN[0] + PAPER_POSTURE_MAX[0]) / 2.0),
            float((PAPER_POSTURE_MIN[1] + PAPER_POSTURE_MAX[1]) / 2.0),
            float((PAPER_POSTURE_MIN[2] + PAPER_POSTURE_MAX[2]) / 2.0),
            0.0,
            (image_size - 1.0) / 2.0,
            (image_size - 1.0) / 2.0,
        ],
        dtype=np.float32,
    )
    if not gamma_conditioned:
        baseline = baseline[[0, 1, 2, 4, 5]]
    values = np.linspace(lower, upper, samples, dtype=np.float32)
    raw = np.tile(baseline, (samples, 1))
    raw[:, field_index] = values
    normalized = normalize_bbsmg_inputs(
        torch.as_tensor(raw, device=device),
        normalization,
    )
    with torch.no_grad():
        predictions = model(normalized).clamp(0.0, 1.0)[:, 0].cpu().numpy()

    targets = []
    for row in raw:
        if gamma_conditioned:
            posture = row[:3]
            gamma = float(row[3])
            x0, y0 = float(row[4]), float(row[5])
        else:
            posture = row[:3]
            gamma = 0.0
            x0, y0 = float(row[3]), float(row[4])
        targets.append(
            render_bbsm_mask(
                posture,
                x0,
                y0,
                image_size=image_size,
                pixels_per_model_unit=pixels_per_model_unit,
                supersample=supersample,
                gamma_rad=gamma,
            )
        )
    targets_np = np.stack(targets)
    per_sample_mse = np.mean(
        (predictions - targets_np) ** 2, axis=(1, 2)
    )
    per_sample_dice = [
        dice_score(prediction, target)
        for prediction, target in zip(predictions, targets_np)
    ]
    prediction_response = float(
        np.sqrt(np.mean((predictions[-1] - predictions[0]) ** 2))
    )
    target_response = float(
        np.sqrt(np.mean((targets_np[-1] - targets_np[0]) ** 2))
    )

    center = raw[samples // 2].copy()
    delta = max((upper - lower) * 0.01, 1e-6)
    minus = center.copy()
    plus = center.copy()
    minus[field_index] = max(center[field_index] - delta, lower)
    plus[field_index] = min(center[field_index] + delta, upper)
    local_raw = torch.as_tensor(np.stack([minus, plus]), device=device)
    with torch.no_grad():
        local_prediction = model(
            normalize_bbsmg_inputs(local_raw, normalization)
        )[:, 0]
    local_span = float(plus[field_index] - minus[field_index])
    local_jacobian_l2 = float(
        torch.sqrt(
            torch.mean(
                ((local_prediction[1] - local_prediction[0]) / local_span)
                ** 2
            )
        ).cpu()
    )
    return {
        "field": field_name,
        "range": [float(lower), float(upper)],
        "samples": int(samples),
        "mean_plain_mse": float(per_sample_mse.mean()),
        "max_plain_mse": float(per_sample_mse.max()),
        "mean_dice_at_0.5": float(np.mean(per_sample_dice)),
        "prediction_endpoint_response_rms": prediction_response,
        "target_endpoint_response_rms": target_response,
        "response_ratio": (
            prediction_response / target_response
            if target_response > 1e-9
            else None
        ),
        "local_prediction_jacobian_rms_per_unit": local_jacobian_l2,
    }


def main(args: argparse.Namespace) -> None:
    device = torch.device(
        args.device
        if args.device == "cpu" or torch.cuda.is_available()
        else "cpu"
    )
    renderer = PaperFusionRenderer.from_checkpoint(
        args.checkpoint,
        device=device,
        image_size=args.image_size,
    )
    model = renderer.bbsmg.eval()
    normalization = renderer.input_normalization
    feature_names = normalization.get("feature_names")
    if not feature_names:
        raise ValueError("Checkpoint must declare feature_names")

    limits = {
        "H_mm": (float(PAPER_POSTURE_MIN[0]), float(PAPER_POSTURE_MAX[0])),
        "alpha_rad": (
            float(PAPER_POSTURE_MIN[1]),
            float(PAPER_POSTURE_MAX[1]),
        ),
        "beta_rad": (
            float(PAPER_POSTURE_MIN[2]),
            float(PAPER_POSTURE_MAX[2]),
        ),
    }
    if "gamma_rad" in feature_names:
        gamma_index = feature_names.index("gamma_rad")
        gamma_scale = float(normalization["scales"][gamma_index])
        limits["gamma_rad"] = (-gamma_scale, gamma_scale)

    results = {}
    for field_name, bounds in limits.items():
        field_index = feature_names.index(field_name)
        result = evaluate_field(
            model,
            normalization,
            field_index,
            field_name,
            bounds[0],
            bounds[1],
            args.samples_per_field,
            args.image_size,
            args.pixels_per_model_unit,
            args.supersample,
            device,
        )
        results[field_name] = result
        print(
            f"[POSE] {field_name}: mse={result['mean_plain_mse']:.6f}, "
            f"dice={result['mean_dice_at_0.5']:.6f}, "
            f"response_ratio={result['response_ratio']:.6f}"
        )

    report = {
        "format": "paper_bbsmg_pose_sensitivity_v1",
        "checkpoint": args.checkpoint,
        "checkpoint_format": normalization.get("checkpoint_format"),
        "feature_names": feature_names,
        "gamma_conditioned": "gamma_rad" in feature_names,
        "results": results,
        "interpretation": (
            "Response ratio near one means the network reproduces the analytic "
            "endpoint change. It does not prove real-brush identifiability."
        ),
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[DONE] Pose-sensitivity report: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--output_json",
        default="outputs/paper_pose_sensitivity/metrics.json",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--pixels_per_model_unit", type=float, default=20.0)
    parser.add_argument("--supersample", type=int, default=2)
    parser.add_argument("--samples_per_field", type=int, default=9)
    main(parser.parse_args())
