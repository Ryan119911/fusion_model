import argparse
import csv
import json
from pathlib import Path


METRIC_FIELDS = (
    "composite_loss",
    "plain_mse",
    "mae",
    "ssim_score",
    "dice_score",
    "iou_at_0.5",
    "zero_baseline_mse",
    "zero_baseline_mae",
)


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _percent_reduction(before, after):
    return 100.0 * (float(before) - float(after)) / float(before)


def _find(rows, experiment, subgroup="all"):
    return next(
        row
        for row in rows
        if row["experiment"] == experiment and row["subgroup"] == subgroup
    )


def write_analysis(path: Path, rows):
    experiments = {row["experiment"] for row in rows}
    names = {
        "a": "stroke10_v1_a_random_legacy_loss",
        "b": "stroke10_v1_b_grouped_legacy_loss",
        "c": "stroke10_v1_c_grouped_full",
        "d": "stroke10_v1_d_no_topology",
    }
    if not set(names.values()).issubset(experiments):
        return
    a = _find(rows, names["a"])
    b = _find(rows, names["b"])
    c = _find(rows, names["c"])
    d = _find(rows, names["d"])
    b_real = _find(rows, names["b"], "real")
    b_synthetic = _find(rows, names["b"], "synthetic")
    real_ratio_a = _find(rows, names["a"], "real")["samples"] / a["samples"]
    real_ratio_b = b_real["samples"] / b["samples"]
    lines = [
        "# 消融实验结果分析",
        "",
        "## 主要结论",
        "",
        f"- C 相对 B：MSE 下降 {_percent_reduction(b['plain_mse'], c['plain_mse']):.2f}%，MAE 下降 {_percent_reduction(b['mae'], c['mae']):.2f}%，Dice 变化 {float(c['dice_score']) - float(b['dice_score']):+.5f}，IoU 变化 {float(c['iou_at_0.5']) - float(b['iou_at_0.5']):+.5f}。",
        f"- D 相对 C：MSE 变化 {-_percent_reduction(c['plain_mse'], d['plain_mse']):+.2f}%，MAE 变化 {-_percent_reduction(c['mae'], d['mae']):+.2f}%，Dice 变化 {float(d['dice_score']) - float(c['dice_score']):+.5f}，IoU 变化 {float(d['iou_at_0.5']) - float(c['iou_at_0.5']):+.5f}。",
        f"- B 的真实子集 MSE={float(b_real['plain_mse']):.6f}、IoU={float(b_real['iou_at_0.5']):.4f}；合成子集 MSE={float(b_synthetic['plain_mse']):.6f}、IoU={float(b_synthetic['iou_at_0.5']):.4f}。",
        f"- A/B 真实样本占比分别为 {real_ratio_a:.1%} 和 {real_ratio_b:.1%}，因此 A/B 总指标不是纯划分策略对照。",
        "",
        "## 原因判断",
        "",
        "- 当前主问题是合成到真实的域差异，可能来自纹理、抗锯齿、笔画宽度、风格和图像配准。",
        "- C/D 差异很小，单种子不足以判定拓扑、边缘和结构损失是否必要；应补多种子均值和标准差。",
        "- 不同损失权重会改变 composite loss 的尺度，跨配置时应比较 MSE、MAE、Dice 和 IoU，而不是直接比较 composite loss。",
        "- 真实子集 MSE 高于全零基线时，说明错误位置的出墨代价很大；优先测试真实样本过采样、加权或真实图微调。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(args):
    subgroup_rows = []
    for metrics_path in sorted(Path(args.output_root).glob("*/evaluation/metrics.json")):
        report = json.loads(metrics_path.read_text(encoding="utf-8"))
        subgroup_samples = report.get("subgroup_samples", {})
        for subgroup, metrics in report.get("metrics", {}).items():
            subgroup_rows.append({
                "experiment": metrics_path.parent.parent.name,
                "subgroup": subgroup,
                "samples": subgroup_samples.get(subgroup),
                **{field: metrics.get(field) for field in METRIC_FIELDS},
            })
    if not subgroup_rows:
        raise FileNotFoundError("No evaluation metrics were found")
    rows = [row for row in subgroup_rows if row["subgroup"] == "all"]
    rows.sort(key=lambda row: float(row["plain_mse"]))
    output = Path(args.output)
    _write_csv(output, rows)
    subgroup_rows.sort(key=lambda row: (row["experiment"], row["subgroup"]))
    subgroup_output = Path(args.subgroup_output)
    _write_csv(subgroup_output, subgroup_rows)
    write_analysis(Path(args.analysis_output), subgroup_rows)
    for row in rows:
        print(row)
    print(f"[DONE] {output}")
    print(f"[DONE] {subgroup_output}")
    print(f"[DONE] {args.analysis_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default="outputs/ablations")
    parser.add_argument("--output", default="outputs/ablations/summary.csv")
    parser.add_argument(
        "--subgroup_output", default="outputs/ablations/subgroup_summary.csv"
    )
    parser.add_argument(
        "--analysis_output", default="outputs/ablations/analysis.md"
    )
    main(parser.parse_args())
