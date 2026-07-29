"""Run the v17 simulated multi-start PSOC/LM validation sequentially."""
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


def scale_label(scale: float) -> str:
    sign = "p" if scale >= 0 else "m"
    magnitude = f"{abs(scale):g}".replace(".", "p")
    return f"{sign}{magnitude}"


def build_inversion_command(
    args: argparse.Namespace,
    initial_pose_csv: Path,
    run_dir: Path,
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
        str(run_dir),
        "--output_stem",
        args.output_stem,
        "--device",
        args.device,
        "--order",
        str(args.order),
        "--optimization_size",
        str(args.optimization_size),
        "--max_steps",
        str(args.max_steps),
        "--damping",
        str(args.damping),
        "--jacobian_mode",
        "finite_difference",
        "--finite_difference_eps",
        str(args.finite_difference_eps),
        "--field_mode",
        "auto",
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
        "--footprint_longitudinal_scale",
        str(args.footprint_longitudinal_scale),
        "--footprint_transverse_scale",
        str(args.footprint_transverse_scale),
        "--render_max_step_px",
        str(args.render_max_step_px),
    ]


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


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
    parser.add_argument("--order", type=int, default=1)
    parser.add_argument("--optimization_size", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument("--damping", type=float, default=0.1)
    parser.add_argument("--finite_difference_eps", type=float, default=0.01)
    parser.add_argument("--min_observability_snr", type=float, default=1.0)
    parser.add_argument("--gamma_max_abs_deg", type=float, default=30.0)
    parser.add_argument("--offset_transfer_scale", type=float, default=1.0)
    parser.add_argument("--pixel_weight", type=float, default=5.0)
    parser.add_argument("--pose_smoothness_weight", type=float, default=0.001)
    parser.add_argument("--footprint_longitudinal_scale", type=float, default=0.22)
    parser.add_argument("--footprint_transverse_scale", type=float, default=0.258)
    parser.add_argument("--render_max_step_px", type=float, default=2.0)
    parser.add_argument("--stability_limit", type=float, default=0.02)
    parser.add_argument("--resume_completed", action="store_true")
    args = parser.parse_args()

    if len(set(args.perturbation_scales)) != len(args.perturbation_scales):
        raise ValueError("Every perturbation scale must be unique")
    if len(args.perturbation_scales) < 2:
        raise ValueError("v17 requires at least two perturbation scales")

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    estimates: dict[str, str] = {}
    manifest = {
        "format": "paper_pose_multistart_runner_v17",
        "simulation_only": True,
        "truth_pose_csv": args.truth_pose_csv,
        "target_image": args.target_image,
        "perturbation_scales": args.perturbation_scales,
        "runs": {},
    }
    for scale in args.perturbation_scales:
        label = scale_label(scale)
        run_dir = root / label
        initial_csv = run_dir / "initial_pose.csv"
        estimate_csv = run_dir / f"{args.output_stem}_trajectory.csv"
        build_probe(
            args.truth_pose_csv,
            str(initial_csv),
            profile="perturbed_initial",
            perturbation_scale=scale,
        )
        command = build_inversion_command(args, initial_csv, run_dir)
        manifest["runs"][label] = {
            "scale": scale,
            "initial_pose_csv": str(initial_csv),
            "estimate_csv": str(estimate_csv),
            "command": command,
        }
        (root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if not (
            args.resume_completed
            and estimate_csv.exists()
            and (run_dir / f"{args.output_stem}_report.json").exists()
        ):
            print(f"[MULTISTART] label={label}, scale={scale:+g}")
            run_logged(command, run_dir / "run.log")
        recovery = evaluate(args.truth_pose_csv, str(estimate_csv))
        (run_dir / "pose_recovery.json").write_text(
            json.dumps(recovery, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        estimates[label] = str(estimate_csv)

    summary = evaluate_multistart(
        args.truth_pose_csv,
        estimates,
        stability_limit=args.stability_limit,
        require_run_reports=True,
    )
    (root / "multistart_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for field, metrics in summary["fields"].items():
        print(
            f"[V17] {field}: "
            f"worst_nrmse={metrics['worst_normalized_rmse']:.6f}, "
            f"cross_start_std={metrics['normalized_cross_start_std_rmse']:.6f}, "
            f"passed={metrics['passed']}"
        )
    print(
        f"[DONE] v17 multi-start validation: "
        f"overall_passed={summary['overall_passed']}, output={root}"
    )


if __name__ == "__main__":
    main()
