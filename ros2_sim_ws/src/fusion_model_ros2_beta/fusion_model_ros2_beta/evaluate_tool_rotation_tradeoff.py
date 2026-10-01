"""Paired original-target inversion and read-only actual-UR10 planning.

Run ``--mode generate`` with the model Python; ``--mode plan`` with the ROS
Python. ``--mode all`` (ROS Python) delegates generation to --python. No
controller publication or executor spin is performed by this tool.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def generate(args):
    from .offline_fontsize_inversion import OfflineInversionConfig
    from .original_target_inversion import OriginalTargetFontSizeGenerator
    from .trajectory_catalog import TrajectoryEntry
    references = json.loads(Path(args.manifest).read_text(encoding="utf-8"))["references"]
    references = [r for r in references if r["character"] == args.character]
    if len(references) != 1:
        raise ValueError("comparison requires one explicitly pinned character/sample")
    root, output = Path(args.model_root).resolve(), Path(args.output).resolve()
    record = references[0]
    source = TrajectoryEntry(character=record["character"], sample_id=record["sample_id"],
                             status="ready", trajectory_csv=Path(record["source_trajectory"]))
    experiment = {"format": "v16_original_target_absolute_rotation_comparison_v1",
                  "manifest": str(Path(args.manifest).resolve()), "manifest_sha256": digest(args.manifest),
                  "character": args.character, "size_m": args.size, "steps": args.steps, "order": args.order,
                  "motion_published": False, "production_promoted": False, "cases": []}
    request = output / "request.json"
    request_data = vars(args).copy(); request_data.pop("mode")
    if request.is_file() and json.loads(request.read_text()) != request_data:
        raise RuntimeError("output belongs to a different experiment; choose a new output directory")
    save_json(request, request_data)
    for weight in args.rotation_weights:
        case = {"rotation_weight": weight, "generation_passed": False, "planning": {"feasible": False, "status": "not_run"}}
        experiment["cases"].append(case)
        try:
            config = OfflineInversionConfig(model_root=root, python_executable=Path(args.python),
                bbsmg_checkpoint=root / "outputs/paper_bbsmg_general_v16_pose_dense/bbsmg_best.pt",
                cache_root=output / "cache", timeout_s=args.timeout_s, device=args.device,
                order=args.order, max_steps=args.steps, joint_foreground_weight=args.foreground_weight,
                joint_hard_neural_domain=args.hard_domain, joint_xy_target_skeleton_weight=args.skeleton_weight,
                joint_tool_absolute_rotation_weight=weight)
            case["config"] = {k: str(v) if isinstance(v, Path) else v for k,v in asdict(config).items()}
            generator = OriginalTargetFontSizeGenerator(config, references)
            candidate = generator.generate(source, args.size, progress=lambda msg: print(msg, flush=True))
            folder = Path(candidate.output_dir)
            report = json.loads((folder / "inversion_report.json").read_text())
            meta = json.loads((folder / "offline_inversion.json").read_text())
            audit = json.loads((folder / "neural_ink_audit.json").read_text())
            case.update(candidate=str(folder), generation_passed=True,
                physical_csv_sha256=digest(folder / "physical_trajectory.csv"),
                original_target_sha256=meta["target_provenance"]["original_target_sha256"],
                quality=report["metrics"], skeleton_distance=report.get("optimized_trajectory_target_skeleton_distance"),
                absolute_tool_rotation=report["lm"]["diagnostics"]["absolute_tool_rotation"],
                checkpoint_selection=report["lm"]["diagnostics"]["checkpoint_selection"],
                training_domain=audit["training_domain"], neural_stream_parity_max_error=audit["stream_vs_forward_max_abs_error"])
            print(json.dumps({"weight": weight, "quality": case["quality"], "tool_rotation": case["absolute_tool_rotation"]["total_rad"]}), flush=True)
        except Exception as exc:
            case.update(error_type=type(exc).__name__, error=str(exc)); traceback.print_exc()
        save_json(output / "comparison.json", experiment)
    return experiment


def plan(args):
    import numpy as np
    sys.path.insert(0, args.model_root)
    from utils.joint_rotation_metrics import joint_rotation_metrics
    import rclpy
    from .brush_trajectory_driver import (BrushTrajectoryDriver, _joint_singularity_margin,
        _brush_rotation, JOINT_LOWER_LIMITS, JOINT_UPPER_LIMITS)
    from .joint_candidate import load_joint_candidate
    from . import ur10_actual_kinematics as kinematics
    import inspect
    output = Path(args.output).resolve()
    report_path = output / "comparison.json"
    experiment = json.loads(report_path.read_text(encoding="utf-8"))
    if experiment["manifest_sha256"] != digest(experiment["manifest"]):
        raise ValueError("original target manifest changed after generation")
    for case in experiment["cases"]:
        if not case["generation_passed"]:
            continue
        folder = Path(case["candidate"])
        if case["physical_csv_sha256"] != digest(folder / "physical_trajectory.csv"):
            raise ValueError("physical controls changed after generation")
        planned = {"feasible": False, "status": "running", "motion_published": False,
                   "physical_csv_sha256": case["physical_csv_sha256"], "planner": "BrushTrajectoryDriver._build_targets",
                   "pen_up_orientation_owner": "existing ROS UR10 planner",
                   "tool_rotation_convention": case["absolute_tool_rotation"]["convention"]}
        case["planning"] = planned
        candidate_report = json.loads((folder / "inversion_report.json").read_text())
        diagnostics = candidate_report["lm"]["diagnostics"]
        case["trajectory_continuity"] = diagnostics.get("trajectory_continuity")
        case["bound_fraction_within_1pct"] = diagnostics.get("bound_fraction_within_1pct")
        case["field_decisions"] = diagnostics.get("field_decisions")
        planned["robot_profile"] = {"robot_model":kinematics.ROBOT_MODEL, "robot_serial":kinematics.ROBOT_SERIAL,
            "calibration_hash":kinematics.CALIBRATION_HASH,
            "planner_source_sha256":digest(inspect.getfile(BrushTrajectoryDriver)),
            "kinematics_source_sha256":digest(kinematics.__file__),
            "gamma_converter_sha256":digest(Path(__file__).with_name("gamma_semantics.py")) if Path(__file__).with_name("gamma_semantics.py").is_file() else None,
            "tool_cost_sha256":digest(Path(args.model_root)/"optim/tool_orientation.py")}
        save_json(report_path, experiment)
        node = None
        rclpy.init(args=["--ros-args", "-p", "enable_input_page:=false", "-p", "strict_ik:=true",
                         "-p", "publish_on_start:=false"])
        try:
            candidate = load_joint_candidate(folder, require_training_domain=True)
            if not math.isclose(candidate.metadata["font_size_m"], experiment["size_m"], abs_tol=1e-12):
                raise ValueError("candidate size differs from experiment size")
            node = BrushTrajectoryDriver()
            node._build_targets(entries=[candidate], layout_mode="horizontal", font_size_m=experiment["size_m"])
            targets = node._targets
            q = np.asarray([t.joints for t in targets])
            if not targets or not np.all((q >= JOINT_LOWER_LIMITS) & (q <= JOINT_UPPER_LIMITS)):
                raise ValueError("empty path or joint limits violated")
            metrics = joint_rotation_metrics(q, [t.point.state for t in targets],
                [t.point.stroke_id for t in targets], [t.duration_s for t in targets])
            metrics["minimum_singularity_margin"] = min(_joint_singularity_margin(t.joints) for t in targets)
            metrics["joint_limit_min_margin_rad"] = float(np.minimum(q - JOINT_LOWER_LIMITS, JOINT_UPPER_LIMITS - q).min())
            # Independent parity of optimizer audit vs the exact driver function
            # on the final physical CSV. Lift states are not exported here.
            rows = list(csv.DictReader((folder / "physical_trajectory.csv").open(encoding="utf-8-sig")))
            matrices = np.asarray([_brush_rotation(float(r["alpha"]),float(r["beta"]),float(r["gamma"])) for r in rows])
            same = np.array([rows[i]["stroke_id"] == rows[i-1]["stroke_id"] for i in range(1,len(rows))])
            rel = matrices[:-1].transpose(0,2,1) @ matrices[1:]
            total = float(np.arccos(np.clip((np.trace(rel,axis1=1,axis2=2)-1)/2,-1,1))[same].sum())
            error = abs(total - case["absolute_tool_rotation"]["total_rad"])
            metrics["optimizer_vs_ros_tool_rotation_error_rad"] = error
            if error > .01:
                raise ValueError(f"optimizer/ROS rotation convention parity failed: {error} rad")
            plan_csv = folder / "ur10_planned_joints.csv"
            with plan_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["target", "state", "stroke_id", "duration_s", "x_m", "y_m", "z_m", "alpha", "beta", "gamma"] + [f"joint_{i+1}_rad" for i in range(6)])
                for index, target in enumerate(targets):
                    p = target.point
                    writer.writerow([index,p.state,p.stroke_id,target.duration_s,p.x,p.y,p.z,p.alpha,p.beta,p.gamma,*target.joints])
            planned.update(status="completed", feasible=True, metrics=metrics,
                           planned_csv=str(plan_csv), planned_csv_sha256=digest(plan_csv),
                           checks=["strict IK", "calibrated UR10 kinematics", "paper/table/attachment collision envelopes",
                                   "joint limits", "singularity", "joint continuity", "time parameterization"])
        except Exception as exc:
            planned.update(status="rejected", error=str(exc), error_type=type(exc).__name__,
                           partial_target_count=len(node._targets) if node is not None else 0)
            # Never score a failed partial plan as lower wrist travel.
            traceback.print_exc()
        finally:
            if node is not None:
                node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        save_json(folder / "ur10_rotation_planning.json", planned)
        save_json(report_path, experiment)
        print(json.dumps({"weight":case["rotation_weight"], "planning":planned},ensure_ascii=False),flush=True)
    summarize(output, experiment, args.max_iou_drop)
    return experiment


def summarize(output, experiment, max_iou_drop):
    baseline = next((c for c in experiment["cases"] if c["rotation_weight"] == 0 and c.get("generation_passed")), None)
    rows, eligible = [], []
    for case in experiment["cases"]:
        quality, planned = case.get("quality", {}), case.get("planning", {})
        metrics = planned.get("metrics", {})
        row = {"rotation_weight":case["rotation_weight"], "generation_passed":case["generation_passed"],
               "iou":quality.get("iou_at_0.5"), "dice":quality.get("dice_at_0.5"), "mse":quality.get("plain_mse"),
               "tool_contact_rotation_deg":case.get("absolute_tool_rotation",{}).get("total_deg"),
               "ur10_feasible":planned.get("feasible",False), "wrist_total_deg":metrics.get("wrist_total_deg"),
               "wrist3_total_deg":metrics.get("wrist3_total_deg"), "planned_duration_s":metrics.get("planned_duration_s")}
        if baseline and row["iou"] is not None:
            row["iou_delta_vs_baseline"] = row["iou"] - baseline["quality"]["iou_at_0.5"]
            b = baseline.get("planning",{}).get("metrics",{}).get("wrist_total_deg")
            if b is not None and row["wrist_total_deg"] is not None:
                row["wrist_delta_deg_vs_baseline"] = row["wrist_total_deg"] - b
                if row["ur10_feasible"] and row["iou_delta_vs_baseline"] >= -max_iou_drop:
                    eligible.append(case)
        rows.append(row)
    experiment["paired_results"] = rows
    experiment["selection_rule"] = {"max_iou_drop":max_iou_drop,
        "rule":"feasible complete UR10 plan, same original target, IoU drop within bound, minimize actual three-wrist accumulated travel",
        "not_a_real_robot_safety_certification":True}
    chosen = min(eligible, key=lambda c:c["planning"]["metrics"]["wrist_total_deg"]) if eligible else None
    experiment["recommended_rotation_weight"] = chosen["rotation_weight"] if chosen else None
    save_json(output / "comparison.json", experiment)
    names = list(dict.fromkeys(k for row in rows for k in row))
    with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle,fieldnames=names); writer.writeheader(); writer.writerows(rows)
    from PIL import Image, ImageDraw
    panels = Image.new("RGB", (720, max(1,len(rows))*230), "white")
    draw = ImageDraw.Draw(panels)
    for index,(case,row) in enumerate(zip(experiment["cases"],rows)):
        top = index*230
        draw.text((10,top+5), f"weight={row['rotation_weight']}  IoU={row['iou']}  wrists(deg)={row['wrist_total_deg']}  feasible={row['ur10_feasible']}", fill="black")
        if case.get("generation_passed") and case.get("candidate"):
            folder = Path(case["candidate"])
            for column,name in enumerate(("inversion_target.png","inversion_rendered.png","inversion_diff.png")):
                with Image.open(folder/name) as im:
                    ink = im.convert("L"); bw = ink.point(lambda p:255-p)
                    panels.paste(bw.resize((192,192)),(10+column*235,top+28))
                draw.text((10+column*235,top+220), ("Original target","Forward render","Absolute diff")[column], fill="black")
    panels.save(output / "comparison.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode",choices=("all","generate","plan"),default="all")
    parser.add_argument("--model-root",default="/home/robot/coppeliasim/machine_learning/model")
    parser.add_argument("--python",default="/home/robot/miniconda3/envs/ddpm/bin/python")
    parser.add_argument("--character",default="武")
    parser.add_argument("--size",type=float,default=.29)
    parser.add_argument("--steps",type=int,default=16)
    parser.add_argument("--order",type=int,default=11)
    parser.add_argument("--rotation-weights",type=float,nargs="+",default=[0.,10.,100.])
    parser.add_argument("--foreground-weight",type=float,default=1.)
    parser.add_argument("--skeleton-weight",type=float,default=0.)
    parser.add_argument("--hard-domain",action="store_true")
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--timeout-s",type=int,default=14400)
    parser.add_argument("--max-iou-drop",type=float,default=.01)
    args = parser.parse_args()
    if (len(set(args.rotation_weights)) != len(args.rotation_weights) or 0 not in args.rotation_weights
            or any(not math.isfinite(w) or w < 0 for w in args.rotation_weights)):
        parser.error("use unique finite nonnegative weights including zero for the paired baseline")
    Path(args.output).resolve().mkdir(parents=True,exist_ok=True)
    if args.mode == "all":
        command_args = sys.argv[1:].copy()
        if "--mode" in command_args:
            position = command_args.index("--mode"); del command_args[position:position+2]
        subprocess.run([args.python,"-u","-m","fusion_model_ros2_beta.evaluate_tool_rotation_tradeoff",*command_args,"--mode","generate"],check=True)
    elif args.mode == "generate":
        generate(args); return 0
    result = plan(args)
    return 0 if all(c["generation_passed"] and c["planning"]["feasible"] for c in result["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
