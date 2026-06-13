import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def ensure_exists(path_str: str, desc: str) -> None:
    if path_str is None:
        return
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"{desc} not found: {path_str}")


def get_log_file(log_dir: str, command: str) -> Path:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(log_dir) / f"{command}_{ts}.log"


def run(cmd, log_file: Path = None):
    print("[RUN]", " ".join(cmd))
    if log_file is None:
        subprocess.run(cmd, check=True)
    else:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write("[RUN] " + " ".join(cmd) + "\n")
            subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)


def cmd_fit(args):
    ensure_exists(args.calibration_csv, "Calibration CSV")
    cmd = [sys.executable, "tools/fit_dynamic_model.py", "--config", args.config, "--calibration_csv", args.calibration_csv, "--output_json", args.output_json, "--output_fit_csv", args.output_fit_csv]
    if args.output_plot_dir:
        cmd += ["--output_plot_dir", args.output_plot_dir]
    cmd += ["--aggregate", args.aggregate, "--min_degree", str(args.min_degree), "--max_degree", str(args.max_degree), "--val_ratio", str(args.val_ratio)]
    run(cmd, args.log_file)


def cmd_build(args):
    cmd = [sys.executable, "tools/build_pseudo_pairs.py", "--config", args.config, "--output_npz", args.output_npz]
    run(cmd, args.log_file)


def cmd_train(args):
    ensure_exists(args.npz_path, "Training NPZ")
    cmd = [sys.executable, "tools/train_bbsmg.py", "--config", args.config, "--npz_path", args.npz_path, "--val_ratio", str(args.val_ratio)]
    run(cmd, args.log_file)


def cmd_optimize(args):
    ensure_exists(args.target_image, "Target image")
    if args.bbsmg_ckpt:
        ensure_exists(args.bbsmg_ckpt, "B-BSMG checkpoint")
    if args.trajectory_csv:
        ensure_exists(args.trajectory_csv, "Trajectory CSV")
    cmd = [sys.executable, "tools/optimize_trajectory.py", "--config", args.config, "--target_image", args.target_image, "--output_dir", args.output_dir, "--order", str(args.order)]
    if args.bbsmg_ckpt:
        cmd += ["--bbsmg_ckpt", args.bbsmg_ckpt]
    if args.trajectory_csv:
        cmd += ["--trajectory_csv", args.trajectory_csv]
    if args.sample_id:
        cmd += ["--sample_id", args.sample_id]
    if args.character:
        cmd += ["--character", args.character]
    cmd += ["--index", str(args.index)]
    if getattr(args, "use_6d", False):
        cmd += ["--use_6d"]
    run(cmd, args.log_file)


def cmd_all(args):
    if not args.skip_fit:
        fit_args = argparse.Namespace(config=args.config, calibration_csv=args.calibration_csv, output_json=args.output_json, output_fit_csv=args.output_fit_csv, output_plot_dir=args.output_plot_dir, aggregate=args.aggregate, min_degree=args.min_degree, max_degree=args.max_degree, val_ratio=args.fit_val_ratio, log_file=args.log_file)
        cmd_fit(fit_args)
    if not args.skip_build:
        build_args = argparse.Namespace(config=args.config, output_npz=args.output_npz, log_file=args.log_file)
        cmd_build(build_args)
    if not args.skip_train:
        train_args = argparse.Namespace(config=args.config, npz_path=args.output_npz, val_ratio=args.train_val_ratio, log_file=args.log_file)
        cmd_train(train_args)
    opt_args = argparse.Namespace(config=args.config, target_image=args.target_image, output_dir=args.optimize_output_dir, order=args.order, bbsmg_ckpt=args.bbsmg_ckpt, trajectory_csv=args.trajectory_csv, sample_id=args.sample_id, character=args.character, index=args.index, log_file=args.log_file, use_6d=args.use_6d)
    cmd_optimize(opt_args)


def build_parser():
    parser = argparse.ArgumentParser(description="Fusion Brush unified main program (enhanced)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p1 = subparsers.add_parser("fit")
    p1.add_argument("--config", type=str, default="configs/default.yaml")
    p1.add_argument("--calibration_csv", type=str, required=True)
    p1.add_argument("--output_json", type=str, default="data/processed/dynamic_brush_coeffs.json")
    p1.add_argument("--output_fit_csv", type=str, default="data/processed/dynamic_brush_fit.csv")
    p1.add_argument("--output_plot_dir", type=str, default="data/processed/dynamic_plots")
    p1.add_argument("--aggregate", type=str, default="mean")
    p1.add_argument("--min_degree", type=int, default=1)
    p1.add_argument("--max_degree", type=int, default=4)
    p1.add_argument("--val_ratio", type=float, default=0.2)
    p1.add_argument("--log_dir", type=str, default="outputs/logs")
    p1.set_defaults(func=cmd_fit)

    p2 = subparsers.add_parser("build")
    p2.add_argument("--config", type=str, default="configs/default.yaml")
    p2.add_argument("--output_npz", type=str, default="data/processed/bbsmg_train.npz")
    p2.add_argument("--log_dir", type=str, default="outputs/logs")
    p2.set_defaults(func=cmd_build)

    p3 = subparsers.add_parser("train")
    p3.add_argument("--config", type=str, default="configs/default.yaml")
    p3.add_argument("--npz_path", type=str, required=True)
    p3.add_argument("--val_ratio", type=float, default=0.1)
    p3.add_argument("--log_dir", type=str, default="outputs/logs")
    p3.set_defaults(func=cmd_train)

    p4 = subparsers.add_parser("optimize")
    p4.add_argument("--config", type=str, default="configs/default.yaml")
    p4.add_argument("--target_image", type=str, required=True)
    p4.add_argument("--bbsmg_ckpt", type=str, default=None)
    p4.add_argument("--trajectory_csv", type=str, default=None)
    p4.add_argument("--sample_id", type=str, default=None)
    p4.add_argument("--character", type=str, default=None)
    p4.add_argument("--index", type=int, default=0)
    p4.add_argument("--order", type=int, default=5)
    p4.add_argument("--output_dir", type=str, default="outputs/optimize")
    p4.add_argument("--use_6d", action="store_true", help="Enable 6D optimization if optimize script supports it")
    p4.add_argument("--log_dir", type=str, default="outputs/logs")
    p4.set_defaults(func=cmd_optimize)

    p5 = subparsers.add_parser("all")
    p5.add_argument("--config", type=str, default="configs/default.yaml")
    p5.add_argument("--calibration_csv", type=str, required=True)
    p5.add_argument("--output_json", type=str, default="data/processed/dynamic_brush_coeffs.json")
    p5.add_argument("--output_fit_csv", type=str, default="data/processed/dynamic_brush_fit.csv")
    p5.add_argument("--output_plot_dir", type=str, default="data/processed/dynamic_plots")
    p5.add_argument("--aggregate", type=str, default="mean")
    p5.add_argument("--min_degree", type=int, default=1)
    p5.add_argument("--max_degree", type=int, default=4)
    p5.add_argument("--fit_val_ratio", type=float, default=0.2)
    p5.add_argument("--output_npz", type=str, default="data/processed/bbsmg_train.npz")
    p5.add_argument("--train_val_ratio", type=float, default=0.1)
    p5.add_argument("--target_image", type=str, required=True)
    p5.add_argument("--bbsmg_ckpt", type=str, default="outputs/bbsmg_best.pt")
    p5.add_argument("--trajectory_csv", type=str, default=None)
    p5.add_argument("--sample_id", type=str, default=None)
    p5.add_argument("--character", type=str, default=None)
    p5.add_argument("--index", type=int, default=0)
    p5.add_argument("--order", type=int, default=5)
    p5.add_argument("--optimize_output_dir", type=str, default="outputs/optimize")
    p5.add_argument("--skip_fit", action="store_true")
    p5.add_argument("--skip_build", action="store_true")
    p5.add_argument("--skip_train", action="store_true")
    p5.add_argument("--use_6d", action="store_true", help="Enable 6D optimization if optimize script supports it")
    p5.add_argument("--log_dir", type=str, default="outputs/logs")
    p5.set_defaults(func=cmd_all)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.log_file = get_log_file(args.log_dir, args.command)
    print(f"Log file: {args.log_file}")
    args.func(args)


if __name__ == "__main__":
    main()
