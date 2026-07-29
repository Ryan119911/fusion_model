"""Run v18 block-coordinate H -> tilt -> gamma multi-start validation."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.build_paper_roundtrip_probe import build_probe
from tools.evaluate_paper_multistart import evaluate_multistart
from tools.evaluate_paper_pose_recovery import evaluate
from tools.run_paper_multistart_validation import run_logged, scale_label


GROUPED_STAGES = (
    ("height", ("H",)),
    ("tilt", ("alpha", "beta")),
    ("gamma", ("gamma",)),
)

SEPARATE_STAGES = (
    ("height", ("H",)),
    ("alpha", ("alpha",)),
    ("beta", ("beta",)),
    ("gamma", ("gamma",)),
)


def build_stage_command(
    args: argparse.Namespace,
    initial_pose_csv: Path,
    stage_dir: Path,
    allowed_fields: tuple[str, ...],
) -> list[str]:
    return [
        sys.executable,
        "-u",
        "tools/invert_paper_trajectory.py",
        "--trajectory_csv",
        args.trajectory_csv,
        "--initial_pose_csv",
        str(initial_pose_csv),
        "--target_image",
        args.target_image,
        "--bbsmg_ckpt",
        args.bbsmg_ckpt,
        "--character",
        args.character,
        "--output_dir",
        str(stage_dir),
        "--output_stem",
        args.output_stem,
        "--device",
        args.device,
        "--order",
        str(args.order),
        "--optimization_size",
        str(args.optimization_size),
        "--max_steps",
        str(args.stage_steps),
        "--damping",
        str(args.damping),
        "--jacobian_mode",
        "finite_difference",
        "--finite_difference_eps",
        str(args.finite_difference_eps),
        "--field_mode",
        "auto",
        "--allowed_pose_fields",
        *allowed_fields,
        "--observability_gate_mode",
        "node_snr",
        "--min_observability_snr",
        str(args.min_observability_snr),
        "--joint_gate_action",
        "prune",
        "--optimize_gamma",
        "--gamma_max_abs_deg",
        str(args.gamma_max_abs_deg),
        "--dynamic_profile",
        "wang2020_figure4_digitized_v1",
        "--offset_transfer_scale",
        str(args.offset_transfer_scale),
        "--pixel_weight",
        str(args.pixel_weight),
        "--h_smoothness_weight",
        str(args.pose_smoothness_weight),
        "--alpha_smoothness_weight",
        str(args.pose_smoothness_weight),
        "--beta_smoothness_weight",
        str(args.pose_smoothness_weight),
        "--gamma_smoothness_weight",
        str(args.pose_smoothness_weight),
        "--h_prior_weight",
        str(args.pose_prior_weight),
        "--alpha_prior_weight",
        str(args.pose_prior_weight),
        "--beta_prior_weight",
        str(args.pose_prior_weight),
        "--gamma_prior_weight",
        str(args.pose_prior_weight),
        "--footprint_longitudinal_scale",
        str(args.footprint_longitudinal_scale),
        "--footprint_transverse_scale",
        str(args.footprint_transverse_scale),
        "--render_max_step_px",
        str(args.render_max_step_px),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory_csv", required=True)
    parser.add_argument("--truth_pose_csv", required=True)
    parser.add_argument("--target_image", required=True)
    parser.add_argument("--bbsmg_ckpt", required=True)
    parser.add_argument("--character", default="武")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_stem", default="wu")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--perturbation_scales",
        type=float,
        nargs="+",
        default=[-2.0, -1.0, -0.5, 0.5, 1.0, 2.0],
    )
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument(
        "--stage_scheme",
        choices=("grouped", "separate"),
        default="grouped",
        help=(
            "grouped runs alpha+beta jointly; separate is the v19 "
            "single-field coordinate-descent schedule"
        ),
    )
    parser.add_argument("--stage_steps", type=int, default=5)
    parser.add_argument("--order", type=int, default=1)
    parser.add_argument("--optimization_size", type=int, default=64)
    parser.add_argument("--damping", type=float, default=0.1)
    parser.add_argument("--finite_difference_eps", type=float, default=0.01)
    parser.add_argument("--min_observability_snr", type=float, default=1.0)
    parser.add_argument("--gamma_max_abs_deg", type=float, default=30.0)
    parser.add_argument("--offset_transfer_scale", type=float, default=1.0)
    parser.add_argument("--pixel_weight", type=float, default=5.0)
    parser.add_argument("--pose_smoothness_weight", type=float, default=0.001)
    parser.add_argument(
        "--pose_prior_weight",
        type=float,
        default=0.0,
        help=(
            "Prior to each start; zero is required for synthetic uniqueness "
            "testing so different guesses are not artificially preserved"
        ),
    )
    parser.add_argument("--footprint_longitudinal_scale", type=float, default=0.22)
    parser.add_argument("--footprint_transverse_scale", type=float, default=0.258)
    parser.add_argument("--render_max_step_px", type=float, default=2.0)
    parser.add_argument("--stability_limit", type=float, default=0.02)
    parser.add_argument("--resume_completed", action="store_true")
    args = parser.parse_args()

    if args.cycles < 1 or args.stage_steps < 1:
        raise ValueError("cycles and stage_steps must be positive")
    if len(set(args.perturbation_scales)) != len(args.perturbation_scales):
        raise ValueError("Every perturbation scale must be unique")
    if len(args.perturbation_scales) < 2:
        raise ValueError("v18 requires at least two perturbation scales")

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    stages = (
        SEPARATE_STAGES
        if args.stage_scheme == "separate"
        else GROUPED_STAGES
    )
    estimates: dict[str, str] = {}
    manifest = {
        "format": (
            "paper_pose_single_field_multistart_runner_v19"
            if args.stage_scheme == "separate"
            else "paper_pose_staged_multistart_runner_v18"
        ),
        "simulation_only": True,
        "truth_pose_csv": args.truth_pose_csv,
        "target_image": args.target_image,
        "perturbation_scales": args.perturbation_scales,
        "cycles": args.cycles,
        "stage_scheme": args.stage_scheme,
        "stages": [
            {"name": name, "allowed_pose_fields": list(fields)}
            for name, fields in stages
        ],
        "pose_prior_weight": args.pose_prior_weight,
        "runs": {},
    }
    for scale in args.perturbation_scales:
        label = scale_label(scale)
        run_dir = root / label
        initial_csv = run_dir / "initial_pose.csv"
        build_probe(
            args.truth_pose_csv,
            str(initial_csv),
            profile="perturbed_initial",
            perturbation_scale=scale,
        )
        current_pose = initial_csv
        run_manifest = {
            "scale": scale,
            "initial_pose_csv": str(initial_csv),
            "stages": [],
        }
        for cycle in range(1, args.cycles + 1):
            for stage_name, allowed_fields in stages:
                stage_dir = run_dir / f"cycle_{cycle}" / stage_name
                estimate_csv = (
                    stage_dir / f"{args.output_stem}_trajectory.csv"
                )
                command = build_stage_command(
                    args, current_pose, stage_dir, allowed_fields
                )
                run_manifest["stages"].append(
                    {
                        "cycle": cycle,
                        "name": stage_name,
                        "allowed_pose_fields": list(allowed_fields),
                        "input_pose_csv": str(current_pose),
                        "estimate_csv": str(estimate_csv),
                        "command": command,
                    }
                )
                manifest["runs"][label] = run_manifest
                (root / "manifest.json").write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                report_path = (
                    stage_dir / f"{args.output_stem}_report.json"
                )
                if not (
                    args.resume_completed
                    and estimate_csv.exists()
                    and report_path.exists()
                ):
                    print(
                        f"[V18] label={label}, cycle={cycle}, "
                        f"stage={stage_name}, fields={','.join(allowed_fields)}"
                    )
                    run_logged(command, stage_dir / "run.log")
                current_pose = estimate_csv
        final_csv = run_dir / f"{args.output_stem}_trajectory.csv"
        final_csv.write_bytes(current_pose.read_bytes())
        final_report = run_dir / f"{args.output_stem}_report.json"
        source_report = current_pose.with_name(
            f"{args.output_stem}_report.json"
        )
        final_report.write_bytes(source_report.read_bytes())
        recovery = evaluate(args.truth_pose_csv, str(final_csv))
        (run_dir / "pose_recovery.json").write_text(
            json.dumps(recovery, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        estimates[label] = str(final_csv)

    summary = evaluate_multistart(
        args.truth_pose_csv,
        estimates,
        stability_limit=args.stability_limit,
        require_run_reports=True,
    )
    summary["format"] = (
        "paper_pose_single_field_multistart_v19"
        if args.stage_scheme == "separate"
        else "paper_pose_staged_multistart_v18"
    )
    summary["stages"] = manifest["stages"]
    summary["cycles"] = args.cycles
    summary["stage_scheme"] = args.stage_scheme
    summary["pose_prior_weight"] = args.pose_prior_weight
    (root / "multistart_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for field, metrics in summary["fields"].items():
        print(
            f"[V18] {field}: "
            f"worst_nrmse={metrics['worst_normalized_rmse']:.6f}, "
            f"cross_start_std={metrics['normalized_cross_start_std_rmse']:.6f}, "
            f"passed={metrics['passed']}"
        )
    print(
        f"[DONE] v18 staged multi-start validation: "
        f"overall_passed={summary['overall_passed']}, output={root}"
    )


if __name__ == "__main__":
    main()
