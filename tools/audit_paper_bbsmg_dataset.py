"""Audit pose coverage and target occupancy of a paper B-BSMG NPZ dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _field_stats(values: np.ndarray, lower: float | None, upper: float | None) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("empty field")
    result = {
        "count": int(values.size),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "quantiles": {
            str(q): float(np.quantile(values, q))
            for q in (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
        },
        "unique_count": int(np.unique(values).size),
    }
    if lower is not None and upper is not None and upper > lower:
        span = upper - lower
        tolerance = 0.02 * span
        result["declared_range"] = [float(lower), float(upper)]
        result["coverage_fraction"] = float(
            np.mean((values >= lower) & (values <= upper))
        )
        result["lower_boundary_fraction_2pct"] = float(
            np.mean(values <= lower + tolerance)
        )
        result["upper_boundary_fraction_2pct"] = float(
            np.mean(values >= upper - tolerance)
        )
        bins = np.linspace(lower, upper, 11)
        counts, _ = np.histogram(values, bins=bins)
        result["decile_counts"] = counts.astype(int).tolist()
        result["empty_deciles"] = int(np.sum(counts == 0))
    return result


def audit_dataset(npz_path: str, output_json: str) -> dict:
    archive = np.load(npz_path, allow_pickle=False)
    if "inputs" not in archive or "targets" not in archive:
        raise ValueError("NPZ must contain inputs and targets")
    inputs = np.asarray(archive["inputs"], dtype=np.float32)
    targets = np.asarray(archive["targets"], dtype=np.float32)
    if inputs.ndim != 2 or inputs.shape[1] < 4:
        raise ValueError(f"expected [N,D] inputs with D>=4, got {inputs.shape}")
    if targets.shape[0] != inputs.shape[0]:
        raise ValueError("inputs and targets have different sample counts")

    metadata = {}
    if "metadata_json" in archive:
        raw = archive["metadata_json"]
        raw_value = raw.item() if raw.ndim == 0 else raw.reshape(-1)[0]
        metadata = json.loads(str(raw_value))
    feature_names = list(
        metadata.get("feature_names", [f"field_{i}" for i in range(inputs.shape[1])])
    )
    declared_limits = metadata.get("limits", {})
    fields = {}
    for index, name in enumerate(feature_names[: inputs.shape[1]]):
        limits = declared_limits.get(name)
        lower = upper = None
        if isinstance(limits, (list, tuple)) and len(limits) == 2:
            lower, upper = float(limits[0]), float(limits[1])
        fields[name] = _field_stats(inputs[:, index], lower, upper)

    target_unit = targets / (255.0 if targets.max() > 1.5 else 1.0)
    ink_ratio = np.mean(target_unit >= 0.5, axis=tuple(range(1, target_unit.ndim)))
    result = {
        "format": "paper_bbsmg_pose_dataset_audit_v1",
        "npz_path": str(npz_path),
        "input_shape": list(inputs.shape),
        "target_shape": list(targets.shape),
        "feature_names": feature_names,
        "fields": fields,
        "target_ink_ratio": {
            "mean": float(ink_ratio.mean()),
            "std": float(ink_ratio.std()),
            "min": float(ink_ratio.min()),
            "max": float(ink_ratio.max()),
        },
        "group_count": (
            int(np.unique(archive["group_ids"]).size)
            if "group_ids" in archive
            else None
        ),
        "metadata": {
            "samples_per_character": metadata.get("samples_per_character"),
            "character_count": metadata.get("character_count"),
            "holdout_characters": metadata.get("holdout_characters"),
            "sampling_mode": metadata.get("sampling_mode"),
            "simulation_only": metadata.get("simulation_only"),
        },
    }
    output = Path(output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] Dataset audit: {output}")
    for name, stats in fields.items():
        print(
            f"[POSE COVERAGE] {name}: range={stats['min']:.6g}..{stats['max']:.6g}, "
            f"empty_deciles={stats.get('empty_deciles', 'n/a')}"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_path", required=True)
    parser.add_argument(
        "--output_json", default="outputs/paper_bbsmg_dataset_audit/metrics.json"
    )
    args = parser.parse_args()
    audit_dataset(args.npz_path, args.output_json)


if __name__ == "__main__":
    main()
