"""Select a physically valid image-optimal pose from a broad multi-start run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.evaluate_paper_multistart import DEFAULT_ERROR_LIMITS
from tools.evaluate_paper_pose_recovery import FIELDS, load_rows


def _load_field(path: str, field: str) -> np.ndarray:
    rows = load_rows(path)
    return np.asarray(
        [float(rows[key][field]) for key in sorted(rows)], dtype=np.float64
    )


def select_candidate(summary: dict, near_optimal_factor: float = 2.0) -> dict:
    if near_optimal_factor < 1.0:
        raise ValueError("near_optimal_factor must be >= 1")
    candidates = []
    for label, quality in summary.get("run_quality", {}).items():
        if quality.get("passed") is not True:
            continue
        mse = quality.get("plain_mse")
        estimate = summary.get("estimates", {}).get(label)
        recovery = summary.get("runs", {}).get(label)
        if mse is None or estimate is None or recovery is None:
            continue
        candidates.append(
            {
                "label": label,
                "plain_mse": float(mse),
                "estimate_csv": estimate,
                "recovery": recovery,
            }
        )
    if not candidates:
        raise ValueError("No candidate passed image/Jacobian/boundary checks")
    candidates.sort(key=lambda item: item["plain_mse"])
    selected = candidates[0]
    best_mse = selected["plain_mse"]
    near_optimal = [
        item
        for item in candidates
        if item["plain_mse"] <= best_mse * near_optimal_factor
    ]
    selected_accuracy = {}
    for field, limit in DEFAULT_ERROR_LIMITS.items():
        value = float(
            selected["recovery"]["metrics"][field]["normalized_rmse"]
        )
        selected_accuracy[field] = {
            "normalized_rmse": value,
            "limit": float(limit),
            "passed": value <= limit,
        }

    ambiguity = {}
    for field, physical_range in FIELDS.items():
        arrays = np.stack(
            [_load_field(item["estimate_csv"], field) for item in near_optimal]
        )
        pointwise_std = arrays.std(axis=0)
        ambiguity[field] = {
            "normalized_cross_candidate_std_rmse": float(
                np.sqrt(np.mean(pointwise_std**2)) / physical_range
            ),
            "candidate_count": len(near_optimal),
        }
    return {
        "format": "paper_pose_candidate_selection_v20",
        "simulation_only": True,
        "source_format": summary.get("format"),
        "near_optimal_factor": float(near_optimal_factor),
        "eligible_candidate_count": len(candidates),
        "selected_label": selected["label"],
        "selected_estimate_csv": selected["estimate_csv"],
        "selected_plain_mse": selected["plain_mse"],
        "selected_accuracy": selected_accuracy,
        "selected_accuracy_passed": all(
            item["passed"] for item in selected_accuracy.values()
        ),
        "near_optimal_labels": [item["label"] for item in near_optimal],
        "near_optimal_ambiguity": ambiguity,
        "warning": (
            "Selection by image fit is valid for optimizer basin discovery, "
            "but near-optimal pose spread remains an uncertainty signal."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--near_optimal_factor", type=float, default=2.0)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()
    summary = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    report = select_candidate(summary, args.near_optimal_factor)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[SELECT] label={report['selected_label']}, "
        f"plain_mse={report['selected_plain_mse']:.8g}, "
        f"accuracy_passed={report['selected_accuracy_passed']}"
    )
    print(
        "[NEAR OPTIMAL] "
        + ",".join(report["near_optimal_labels"])
    )
    print(f"[DONE] Candidate selection: {output}")


if __name__ == "__main__":
    main()
