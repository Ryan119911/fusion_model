import argparse
import subprocess
import sys
from pathlib import Path


def run_tool(tool: str, arguments) -> None:
    command = [sys.executable, "-u", f"tools/{tool}.py", *arguments]
    print("[RUN]", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fusion model command dispatcher")
    subparsers = parser.add_subparsers(dest="command", required=True)

    passthrough = {
        "build": "build_pseudo_pairs",
        "train": "train_bbsmg",
        "evaluate": "evaluate_bbsmg",
        "render-character": "render_character_comparison",
        "optimize": "optimize_trajectory",
        "fit-dynamic": "fit_dynamic_model",
        "audit": "audit_npz",
        "ablate": "run_ablation",
        "summarize": "summarize_experiments",
    }
    for command in passthrough:
        subparsers.add_parser(command, add_help=False)
    args, remaining = parser.parse_known_args()
    run_tool(passthrough[args.command], remaining)


if __name__ == "__main__":
    main()
