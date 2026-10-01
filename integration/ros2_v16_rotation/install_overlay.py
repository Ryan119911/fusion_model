"""Apply this narrow overlay only to the exact audited existing ROS source.

No package installs, builds, ROS publication or changes to kinematics. Changed
sources are backed up before patching; unknown/concurrent edits fail closed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--apply", action="store_true", help="default is read-only hash check")
    args = parser.parse_args()
    bundle = Path(__file__).resolve().parent
    target, model = Path(args.package_root).resolve(), Path(args.model_root).resolve()
    manifest = json.loads((bundle/"manifest.json").read_text())
    for relative in ("optim/tool_orientation.py", "utils/joint_rotation_metrics.py"):
        if not (model/relative).is_file():
            raise RuntimeError(f"missing model support: {relative}; update the model code first")
    states = []
    for item in manifest["files"]:
        path = (target/item["path"]).resolve()
        if not path.is_relative_to(target):
            raise ValueError("patch path escapes package root")
        actual = sha(path)
        states.append("before" if actual == item["before_sha256"] else "after" if actual == item["after_sha256"] else "unknown")
    if "unknown" in states or len(set(states)) != 1:
        raise RuntimeError("ROS source differs from audited before/after snapshot; preserve and review it before applying")
    assets = {"evaluate_tool_rotation_tradeoff.py":"fusion_model_ros2_beta/evaluate_tool_rotation_tradeoff.py",
              "test_tool_rotation_tradeoff.py":"test/test_tool_rotation_tradeoff.py"}
    for source,destination in assets.items():
        path = target/destination
        if path.exists() and sha(path) != sha(bundle/source):
            raise RuntimeError(f"new asset has independent edits: {path}")
    report = {"base_state":states[0], "applied":False, "package_root":str(target)}
    if args.apply:
        if states[0] == "before":
            patch = ["patch", "-p1", "--forward", "--directory", str(target), "--input", str(bundle/"ros_package.patch")]
            subprocess.run(patch + ["--dry-run"], check=True)
            backup = target.parent / "rotation_overlay_backups"
            backup.mkdir(exist_ok=True)
            filename = backup / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+".tgz")
            with tarfile.open(filename,"w:gz") as archive:
                for item in manifest["files"]:
                    archive.add(target/item["path"],arcname=item["path"])
            # Recheck immediately before patch: do not clobber a concurrent edit.
            if any(sha(target/i["path"]) != i["before_sha256"] for i in manifest["files"]):
                raise RuntimeError("source changed during preparation; no patch was applied")
            subprocess.run(patch,check=True)
            report["backup"] = str(filename)
        for source,destination in assets.items():
            path = target/destination
            path.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(bundle/source,path)
        if any(sha(target/i["path"]) != i["after_sha256"] for i in manifest["files"]):
            raise RuntimeError("after-patch hash mismatch; inspect backup and package before running")
        report["applied"] = True
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
