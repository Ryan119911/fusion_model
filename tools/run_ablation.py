import argparse
import subprocess
import sys
from pathlib import Path


def run(command, log_path: Path):
    print("[RUN]", " ".join(command))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)


def main(args):
    configs = sorted(Path().glob(args.config_glob))
    if not configs:
        raise FileNotFoundError(f"No configs matched {args.config_glob}")
    root = Path(args.output_root)
    for config in configs:
        name = config.stem
        output = root / name
        checkpoint = output / "bbsmg_last.pt"
        train = [
            sys.executable, "-u", "tools/train_bbsmg.py",
            "--config", str(config), "--npz_path", args.npz_path,
            "--output_dir", str(output), "--epochs", str(args.epochs),
        ]
        if args.resume and checkpoint.exists():
            train += ["--resume", str(checkpoint)]
        run(train, output / "train.log")
        evaluate = [
            sys.executable, "-u", "tools/evaluate_bbsmg.py",
            "--config", str(config), "--npz_path", args.npz_path,
            "--checkpoint", str(output / "bbsmg_best.pt"),
            "--output_dir", str(output / "evaluation"),
            "--batch_size", str(args.batch_size),
        ]
        run(evaluate, output / "evaluate.log")
        print(f"[DONE] {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_path", required=True)
    parser.add_argument(
        "--config_glob", default="configs/ablations/stroke10_v1_*.yaml"
    )
    parser.add_argument("--output_root", default="outputs/ablations")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    main(parser.parse_args())
