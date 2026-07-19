# 中文注释：本文件提供命令行入口，串联动态模型拟合、伪样本构建、B-BSMG 训练和轨迹优化流程。
import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime


# 中文注释：在启动子命令前检查输入路径是否存在，尽早暴露配置错误。
def ensure_exists(path_str: str, desc: str) -> None:
    if path_str is None:
        return
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"{desc} not found: {path_str}")


# 中文注释：按命令名和时间戳生成日志文件路径。
def get_log_file(log_dir: str, command: str) -> Path:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(log_dir) / f"{command}_{ts}.log"


# 中文注释：执行子命令，并可选择将标准输出和错误写入日志。
def run(cmd, log_file: Path = None):
    print("[RUN]", " ".join(cmd))
    if log_file is None:
        subprocess.run(cmd, check=True)
    else:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write("[RUN] " + " ".join(cmd) + "\n")
            subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)


# 中文注释：组织并执行动态笔刷模型拟合命令。
def cmd_fit(args):
    ensure_exists(args.calibration_csv, "Calibration CSV")
    cmd = [sys.executable, "tools/fit_dynamic_model.py", "--config", args.config, "--calibration_csv", args.calibration_csv, "--output_json", args.output_json, "--output_fit_csv", args.output_fit_csv]
    if args.output_plot_dir:
        cmd += ["--output_plot_dir", args.output_plot_dir]
    cmd += ["--aggregate", args.aggregate, "--min_degree", str(args.min_degree), "--max_degree", str(args.max_degree), "--val_ratio", str(args.val_ratio)]
    run(cmd, args.log_file)


# 中文注释：组织并执行伪配对样本构建命令。
def cmd_build(args):
    cmd = [sys.executable, "tools/build_pseudo_pairs.py", "--config", args.config, "--output_npz", args.output_npz]
    run(cmd, args.log_file)


# 中文注释：组织并执行 B-BSMG 训练命令。
def cmd_train(args):
    ensure_exists(args.npz_path, "Training NPZ")
    cmd = [sys.executable, "tools/train_bbsmg.py", "--config", args.config, "--npz_path", args.npz_path, "--val_ratio", str(args.val_ratio)]
    if getattr(args, "resume", None):
        ensure_exists(args.resume, "Resume checkpoint")
        cmd += ["--resume", args.resume]
    if getattr(args, "epochs", None) is not None:
        cmd += ["--epochs", str(args.epochs)]
    if getattr(args, "output_dir", None):
        cmd += ["--output_dir", args.output_dir]
    cmd += [
        "--lr_factor", str(getattr(args, "lr_factor", 0.5)),
        "--lr_patience", str(getattr(args, "lr_patience", 3)),
        "--min_lr", str(getattr(args, "min_lr", 1e-6)),
    ]
    run(cmd, args.log_file)


def cmd_build_character(args):
    cmd = [
        sys.executable,
        "tools/build_character_pairs.py",
        "--config", args.config,
        "--output_npz", args.output_npz,
    ]
    if args.trajectory_csv:
        cmd += ["--trajectory_csv", args.trajectory_csv]
    if args.character:
        cmd += ["--character", args.character]
    if args.target_character:
        cmd += ["--target_character", args.target_character]
    if args.target_image:
        ensure_exists(args.target_image, "Whole-character target image")
        cmd += ["--target_image", args.target_image]
    if args.chirography:
        cmd += ["--chirography", args.chirography]
    if args.require_real_target:
        cmd += ["--require_real_target"]
    cmd += ["--trajectory_width", str(args.trajectory_width)]
    run(cmd, args.log_file)


def cmd_train_character(args):
    ensure_exists(args.npz_path, "Whole-character training NPZ")
    cmd = [
        sys.executable,
        "tools/train_character.py",
        "--config", args.config,
        "--npz_path", args.npz_path,
        "--val_ratio", str(args.val_ratio),
        "--lr_factor", str(args.lr_factor),
        "--lr_patience", str(args.lr_patience),
        "--min_lr", str(args.min_lr),
    ]
    if args.output_dir:
        cmd += ["--output_dir", args.output_dir]
    if args.epochs is not None:
        cmd += ["--epochs", str(args.epochs)]
    if args.batch_size is not None:
        cmd += ["--batch_size", str(args.batch_size)]
    if args.resume:
        ensure_exists(args.resume, "Whole-character resume checkpoint")
        cmd += ["--resume", args.resume]
    if args.init_character_checkpoint:
        ensure_exists(args.init_character_checkpoint, "Character initialization checkpoint")
        cmd += ["--init_character_checkpoint", args.init_character_checkpoint]
    run(cmd, args.log_file)


def cmd_evaluate_character(args):
    ensure_exists(args.npz_path, "Whole-character evaluation NPZ")
    ensure_exists(args.checkpoint, "Whole-character checkpoint")
    cmd = [
        sys.executable,
        "tools/evaluate_character.py",
        "--config", args.config,
        "--npz_path", args.npz_path,
        "--checkpoint", args.checkpoint,
        "--output_dir", args.output_dir,
        "--split", args.split,
        "--num_images", str(args.num_images),
    ]
    if args.character:
        cmd += ["--character", args.character]
    run(cmd, args.log_file)


def cmd_predict_character(args):
    ensure_exists(args.checkpoint, "Whole-character checkpoint")
    if args.trajectory_csv:
        ensure_exists(args.trajectory_csv, "Trajectory CSV")
    if args.target_image:
        ensure_exists(args.target_image, "Whole-character target image")
    cmd = [
        sys.executable,
        "tools/predict_character.py",
        "--config", args.config,
        "--checkpoint", args.checkpoint,
        "--index", str(args.index),
        "--output_dir", args.output_dir,
        "--trajectory_width", str(args.trajectory_width),
    ]
    for name in ("trajectory_csv", "character", "sample_id", "target_image", "output_stem"):
        value = getattr(args, name)
        if value is not None:
            cmd += ["--" + name, str(value)]
    run(cmd, args.log_file)


# 中文注释：组织并执行轨迹优化命令。
def cmd_optimize(args):
    ensure_exists(args.target_image, "Target image")
    if args.bbsmg_ckpt:
        ensure_exists(args.bbsmg_ckpt, "B-BSMG checkpoint")
    if args.trajectory_csv:
        ensure_exists(args.trajectory_csv, "Trajectory CSV")
    cmd = [sys.executable, "tools/optimize_trajectory.py", "--config", args.config, "--target_image", args.target_image, "--output_dir", args.output_dir, "--order", str(args.order)]
    if args.bbsmg_ckpt:
        cmd += ["--bbsmg_ckpt", args.bbsmg_ckpt]
    if getattr(args, "normalization_npz", None):
        cmd += ["--normalization_npz", args.normalization_npz]
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


# 中文注释：按顺序执行完整流水线中未被跳过的步骤。
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
    opt_args = argparse.Namespace(config=args.config, target_image=args.target_image, output_dir=args.optimize_output_dir, order=args.order, bbsmg_ckpt=args.bbsmg_ckpt, normalization_npz=args.normalization_npz, trajectory_csv=args.trajectory_csv, sample_id=args.sample_id, character=args.character, index=args.index, log_file=args.log_file, use_6d=args.use_6d)
    cmd_optimize(opt_args)


# 中文注释：构建命令行参数解析器和所有子命令参数。
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
    p3.add_argument("--resume", type=str, default=None)
    p3.add_argument("--epochs", type=int, default=None)
    p3.add_argument("--output_dir", type=str, default=None)
    p3.add_argument("--lr_factor", type=float, default=0.5)
    p3.add_argument("--lr_patience", type=int, default=3)
    p3.add_argument("--min_lr", type=float, default=1e-6)
    p3.add_argument("--log_dir", type=str, default="outputs/logs")
    p3.set_defaults(func=cmd_train)

    p4 = subparsers.add_parser("optimize")
    p4.add_argument("--config", type=str, default="configs/default.yaml")
    p4.add_argument("--target_image", type=str, required=True)
    p4.add_argument("--bbsmg_ckpt", type=str, default=None)
    p4.add_argument("--normalization_npz", type=str, default=None)
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
    p5.add_argument("--normalization_npz", type=str, default=None)
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

    p6 = subparsers.add_parser("build-character", help="Build direct whole-character pairs")
    p6.add_argument("--config", default="configs/default.yaml")
    p6.add_argument("--trajectory_csv", default=None)
    p6.add_argument("--output_npz", default="data/processed/character_train.npz")
    p6.add_argument("--character", default=None)
    p6.add_argument("--target_character", default=None)
    p6.add_argument("--target_image", default=None)
    p6.add_argument("--chirography", default=None)
    p6.add_argument("--require_real_target", action="store_true")
    p6.add_argument("--trajectory_width", type=int, default=3)
    p6.add_argument("--log_dir", default="outputs/logs")
    p6.set_defaults(func=cmd_build_character)

    p7 = subparsers.add_parser("train-character", help="Train the spatial whole-character U-Net")
    p7.add_argument("--config", default="configs/default.yaml")
    p7.add_argument("--npz_path", required=True)
    p7.add_argument("--output_dir", default=None)
    p7.add_argument("--epochs", type=int, default=None)
    p7.add_argument("--batch_size", type=int, default=None)
    p7.add_argument("--val_ratio", type=float, default=0.1)
    p7.add_argument("--resume", default=None)
    p7.add_argument("--init_character_checkpoint", default=None)
    p7.add_argument("--lr_factor", type=float, default=0.5)
    p7.add_argument("--lr_patience", type=int, default=3)
    p7.add_argument("--min_lr", type=float, default=1e-6)
    p7.add_argument("--log_dir", default="outputs/logs")
    p7.set_defaults(func=cmd_train_character)

    p8 = subparsers.add_parser("evaluate-character", help="Evaluate complete-character images")
    p8.add_argument("--config", default="configs/default.yaml")
    p8.add_argument("--npz_path", required=True)
    p8.add_argument("--checkpoint", required=True)
    p8.add_argument("--output_dir", default="outputs/eval_character")
    p8.add_argument("--split", choices=("val", "train", "all"), default="val")
    p8.add_argument("--character", default=None)
    p8.add_argument("--num_images", type=int, default=20)
    p8.add_argument("--log_dir", default="outputs/logs")
    p8.set_defaults(func=cmd_evaluate_character)

    p9 = subparsers.add_parser("predict-character", help="Predict one complete character with U-Net")
    p9.add_argument("--config", default="configs/default.yaml")
    p9.add_argument("--trajectory_csv", default=None)
    p9.add_argument("--checkpoint", required=True)
    p9.add_argument("--character", default=None)
    p9.add_argument("--sample_id", default=None)
    p9.add_argument("--index", type=int, default=0)
    p9.add_argument("--target_image", default=None)
    p9.add_argument("--output_dir", default="outputs/predict_character")
    p9.add_argument("--output_stem", default=None)
    p9.add_argument("--trajectory_width", type=int, default=3)
    p9.add_argument("--log_dir", default="outputs/logs")
    p9.set_defaults(func=cmd_predict_character)

    return parser


# 中文注释：解析命令行参数，准备日志文件并分派到对应子命令。
def main():
    parser = build_parser()
    args = parser.parse_args()
    args.log_file = get_log_file(args.log_dir, args.command)
    print(f"Log file: {args.log_file}")
    args.func(args)


# 中文注释：作为脚本直接运行时，从这里进入命令行流程或示例测试。
if __name__ == "__main__":
    main()
