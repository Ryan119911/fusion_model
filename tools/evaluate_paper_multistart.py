"""Aggregate simulated paper-pose recovery across multiple initial guesses."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.evaluate_paper_pose_recovery import FIELDS, evaluate, load_rows


DEFAULT_ERROR_LIMITS = {
    "z": 0.01,
    "alpha": 0.06,
    "beta": 0.06,
    "gamma": 0.04,
}


def _field_array(path: str, field: str) -> np.ndarray:
    rows = load_rows(path)
    return np.asarray(
        [float(rows[key][field]) for key in sorted(rows)], dtype=np.float64
    )


def evaluate_multistart(
    reference_csv: str,
    estimates: Mapping[str, str],
    error_limits: Mapping[str, float] | None = None,
    stability_limit: float = 0.02,
    iou_limit: float = 0.99,
    boundary_fraction_limit: float = 0.0,
    require_run_reports: bool = False,
) -> dict:
    if len(estimates) < 2:
        raise ValueError("Multi-start evaluation requires at least two runs")
    limits = dict(DEFAULT_ERROR_LIMITS)
    if error_limits is not None:
        limits.update(error_limits)

    run_reports = {
        name: evaluate(reference_csv, path)
        for name, path in estimates.items()
    }
    run_quality = {}
    for name, estimate_path in estimates.items():
        path = Path(estimate_path)
        suffix = "_trajectory.csv"
        report_path = (
            path.with_name(path.name[: -len(suffix)] + "_report.json")
            if path.name.endswith(suffix)
            else path.with_name(path.stem + "_report.json")
        )
        if not report_path.exists():
            run_quality[name] = {
                "available": False,
                "report_json": str(report_path),
                "passed": None,
            }
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        image_metrics = report.get("metrics", {})
        iou = image_metrics.get("iou_at_0.5")
        joint_audit = (
            report.get("identifiability", {})
            .get("joint_jacobian_audit", {})
        )
        jointly_identifiable = joint_audit.get("jointly_identifiable")
        boundary_fractions = {
            field: decision.get("boundary_fraction")
            for field, decision in report.get("field_decisions", {}).items()
            if field in FIELDS
            and decision.get("boundary_fraction") is not None
        }
        maximum_boundary_fraction = max(
            boundary_fractions.values(), default=0.0
        )
        passed = (
            iou is not None
            and float(iou) >= iou_limit
            and jointly_identifiable is True
            and maximum_boundary_fraction <= boundary_fraction_limit
        )
        run_quality[name] = {
            "available": True,
            "report_json": str(report_path),
            "plain_mse": image_metrics.get("plain_mse"),
            "iou_at_0.5": iou,
            "iou_limit": float(iou_limit),
            "jointly_identifiable": jointly_identifiable,
            "maximum_boundary_fraction": float(maximum_boundary_fraction),
            "boundary_fraction_limit": float(boundary_fraction_limit),
            "boundary_fractions": boundary_fractions,
            "passed": passed,
        }
    field_reports = {}
    for field, physical_range in FIELDS.items():
        arrays = np.stack(
            [_field_array(path, field) for path in estimates.values()], axis=0
        )
        pointwise_std = arrays.std(axis=0)
        consensus = arrays.mean(axis=0)
        pairwise_rmse = []
        for first in range(len(arrays)):
            for second in range(first + 1, len(arrays)):
                pairwise_rmse.append(
                    float(
                        np.sqrt(
                            np.mean((arrays[first] - arrays[second]) ** 2)
                        )
                    )
                )
        normalized_errors = {
            name: float(report["metrics"][field]["normalized_rmse"])
            for name, report in run_reports.items()
        }
        normalized_stability = float(
            np.sqrt(np.mean(pointwise_std**2)) / physical_range
        )
        accuracy_passed = all(
            value <= limits[field] for value in normalized_errors.values()
        )
        stability_passed = normalized_stability <= stability_limit
        field_reports[field] = {
            "physical_range": float(physical_range),
            "normalized_rmse_by_run": normalized_errors,
            "median_normalized_rmse": float(
                np.median(list(normalized_errors.values()))
            ),
            "worst_normalized_rmse": float(
                max(normalized_errors.values())
            ),
            "accuracy_limit": float(limits[field]),
            "accuracy_passed": accuracy_passed,
            "consensus_range": [
                float(consensus.min()),
                float(consensus.max()),
            ],
            "cross_start_std_rmse": float(
                np.sqrt(np.mean(pointwise_std**2))
            ),
            "normalized_cross_start_std_rmse": normalized_stability,
            "stability_limit": float(stability_limit),
            "stability_passed": stability_passed,
            "max_pairwise_rmse": float(max(pairwise_rmse)),
            "passed": accuracy_passed and stability_passed,
        }

    observation_checks_passed = all(
        quality["passed"] is True
        for quality in run_quality.values()
        if quality["available"] or require_run_reports
    )
    if require_run_reports and any(
        not quality["available"] for quality in run_quality.values()
    ):
        observation_checks_passed = False
    pose_fields_passed = all(
        report["passed"] for report in field_reports.values()
    )
    return {
        "format": "paper_pose_multistart_v17",
        "simulation_only": True,
        "reference_csv": reference_csv,
        "run_count": len(estimates),
        "estimates": dict(estimates),
        "runs": run_reports,
        "run_quality": run_quality,
        "fields": field_reports,
        "pose_fields_passed": pose_fields_passed,
        "observation_checks_passed": observation_checks_passed,
        "run_reports_required": require_run_reports,
        "overall_passed": (
            pose_fields_passed
            and (observation_checks_passed or not require_run_reports)
        ),
        "export_policy": {
            field: (
                "eligible_simulation_consensus"
                if report["passed"]
                else "withhold_unstable_or_inaccurate"
            )
            for field, report in field_reports.items()
        },
        "interpretation": (
            "Passing this synthetic test validates local optimizer robustness, "
            "not real-brush calibration or physical robot ground truth."
        ),
    }


def _parse_estimate(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--estimate must use LABEL=/path/to/trajectory.csv"
        )
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError(
            "--estimate must contain a non-empty label and path"
        )
    return label, path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_csv", required=True)
    parser.add_argument(
        "--estimate",
        action="append",
        required=True,
        type=_parse_estimate,
        metavar="LABEL=CSV",
    )
    parser.add_argument(
        "--stability_limit",
        type=float,
        default=0.02,
        help="Maximum normalized cross-start RMS standard deviation",
    )
    parser.add_argument("--z_error_limit", type=float, default=0.01)
    parser.add_argument("--alpha_error_limit", type=float, default=0.06)
    parser.add_argument("--beta_error_limit", type=float, default=0.06)
    parser.add_argument("--gamma_error_limit", type=float, default=0.04)
    parser.add_argument("--iou_limit", type=float, default=0.99)
    parser.add_argument(
        "--boundary_fraction_limit",
        type=float,
        default=0.0,
    )
    parser.add_argument("--require_run_reports", action="store_true")
    parser.add_argument(
        "--output_json",
        default="outputs/paper_pose_multistart_v17.json",
    )
    args = parser.parse_args()
    estimates = dict(args.estimate)
    if len(estimates) != len(args.estimate):
        raise ValueError("Every --estimate label must be unique")
    report = evaluate_multistart(
        args.reference_csv,
        estimates,
        error_limits={
            "z": args.z_error_limit,
            "alpha": args.alpha_error_limit,
            "beta": args.beta_error_limit,
            "gamma": args.gamma_error_limit,
        },
        stability_limit=args.stability_limit,
        iou_limit=args.iou_limit,
        boundary_fraction_limit=args.boundary_fraction_limit,
        require_run_reports=args.require_run_reports,
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for field, metrics in report["fields"].items():
        print(
            f"{field}: worst_nrmse={metrics['worst_normalized_rmse']:.6f}, "
            f"cross_start_std={metrics['normalized_cross_start_std_rmse']:.6f}, "
            f"passed={metrics['passed']}"
        )
    print(
        f"[DONE] Multi-start validation: runs={report['run_count']}, "
        f"overall_passed={report['overall_passed']}, output={output}"
    )


if __name__ == "__main__":
    main()
