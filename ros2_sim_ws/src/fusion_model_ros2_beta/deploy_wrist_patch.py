"""Opt-in source deployment with baseline guards; never change formal YAML."""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

BASELINE={
    'fusion_model_ros2_beta/brush_trajectory_driver.py':'70fd16d6778aa6df0d3b30287b7813483eb88d6fcbe9451cd4b250f524959421',
    'fusion_model_ros2_beta/joint_optimizer.py':'1d0849da12e4ab0c5a8afff8c3b9f88ebf59e24e16d9edac83570f274a1513fe',
    'fusion_model_ros2_beta/offline_fontsize_inversion.py':'ea0c3a0eeb102c8e93ad8d4ba08e2c0528e85099dd52170ef017c8021ace7293',
    'fusion_model_ros2_beta/original_target_inversion.py':'80e63c7c50b0eb9c49ddf8a37bb825698093ed56f97269da02232513b857945f',
    'fusion_model_ros2_beta/run_joint_inversion.py':'7b165ccdcb60272f7b3a1ea41c859f2262261c7891527ec878bcd9bbbcafd74d',
    'fusion_model_ros2_beta/joint_candidate.py':'dd63bb5d4465bfb1263863a73807633be781e726877f6f6b72894330388f5555',
    'launch/ur10_brush_ros2_beta.launch.py':'2813aad008cc4084dfd51dea53ad912c4e36bcc1353f25189a957d953f1b0165',
}
NEW_NAMES=('ur10_wrist_cost.py','wrist_aware_transitions.py','precise_ur10_validation.py',
    'original_image_metrics.py','evaluate_actual_wrist_tradeoff.py','joint_path_audit.py','robot_context_guard.py')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def deploy(source,target,backup,expected=None):
    source,target,backup=(Path(p).resolve() for p in (source,target,backup))
    expected=dict(BASELINE if expected is None else expected)
    files=list(expected)+['fusion_model_ros2_beta/'+name for name in NEW_NAMES]
    plan=[]
    # Validate EVERY file before writing any; refuse unrelated/new user changes.
    for rel in files:
        src,dst=source/rel,target/rel
        if not dst.resolve().is_relative_to(target): raise ValueError('deployment escaped package root')
        incoming=sha(src)
        current=sha(dst) if dst.exists() else None
        if current==incoming: continue
        if current!=expected.get(rel): raise ValueError('preserving changed file; baseline mismatch: '+rel)
        plan.append((rel,src,dst,current,incoming))
    backup.mkdir(parents=True,exist_ok=False)
    report=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),source=str(source),target=str(target),
        formal_configuration_changed=False,coppeliasim_modified=False,motion_published=False,files={})
    for rel,src,dst,current,incoming in plan:
        if current is not None:
            saved=backup/rel;saved.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(dst,saved)
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        report['files'][rel]={'before_sha256':current,'after_sha256':incoming}
    (backup/'deployment.json').write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--target',required=True);p.add_argument('--backup',required=True)
    args=p.parse_args();print(json.dumps(deploy(args.source,args.target,args.backup),indent=2))
