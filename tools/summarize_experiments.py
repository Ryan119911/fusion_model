import argparse
import csv
import json
from pathlib import Path


def main(args):
    rows = []
    for metrics_path in sorted(Path(args.output_root).glob("*/evaluation/metrics.json")):
        report = json.loads(metrics_path.read_text(encoding="utf-8"))
        all_metrics = report.get("metrics", {}).get("all", {})
        rows.append({
            "experiment": metrics_path.parent.parent.name,
            "samples": report.get("validation_samples"),
            "composite_loss": all_metrics.get("composite_loss"),
            "plain_mse": all_metrics.get("plain_mse"),
            "mae": all_metrics.get("mae"),
            "ssim_score": all_metrics.get("ssim_score"),
            "dice_score": all_metrics.get("dice_score"),
            "iou_at_0.5": all_metrics.get("iou_at_0.5"),
        })
    if not rows:
        raise FileNotFoundError("No evaluation metrics were found")
    rows.sort(key=lambda row: float(row["composite_loss"]))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print(f"[DONE] {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default="outputs/ablations")
    parser.add_argument("--output", default="outputs/ablations/summary.csv")
    main(parser.parse_args())
