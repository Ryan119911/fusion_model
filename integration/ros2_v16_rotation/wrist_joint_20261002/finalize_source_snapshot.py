"""Freeze the opt-in ROS source for final offline re-planning; no launch."""
from __future__ import annotations
import hashlib
import json
import shutil
from pathlib import Path

ROOT=Path('/home/robot/ros2_ws/evaluation/wrist_joint_20261002')
ACTIVE=Path('/home/robot/ros2_ws/src/fusion_model_ros2_beta')
REL='fusion_model_ros2_beta/evaluate_actual_wrist_tradeoff.py'
PREVIOUS_EVALUATOR='1650a91f889d0ba111e1f1f5a5c325a37cacf5b51095db8cb819fe0afc8c43ee'
GEOMETRY='342d2d47ce4faaf823faad71ebb0234bd04c14bbd42cac778ad468d57507adfc'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config_hashes():
    return {str(p.relative_to(ACTIVE)):sha(p) for p in (ACTIVE/'config').rglob('*') if p.is_file()}


def freeze():
    snapshot=ROOT/'deployed_source'
    if snapshot.exists():raise ValueError('preserve existing final source snapshot')
    config_before=config_hashes()
    geometry=ACTIVE/'fusion_model_ros2_beta/ur10_actual_kinematics.py'
    if sha(geometry)!=GEOMETRY:raise ValueError('current calibrated tool/camera geometry changed')
    incoming=ROOT/'source'/REL;destination=ACTIVE/REL
    before=sha(destination);after=sha(incoming)
    if before not in (PREVIOUS_EVALUATOR,after):raise ValueError('preserving unexpected active evaluator change')
    if before!=after:
        backup=ROOT/'deployment_backup_20261002_final'/REL
        backup.parent.mkdir(parents=True,exist_ok=False)
        shutil.copy2(destination,backup)
        shutil.copy2(incoming,destination)
    shutil.copytree(ACTIVE,snapshot,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache','.git'))
    if config_hashes()!=config_before:raise ValueError('formal configuration unexpectedly changed')
    report=dict(active_source=str(ACTIVE),final_snapshot=str(snapshot),
        updated_evaluator=dict(before_sha256=before,after_sha256=after),
        current_factory_tool_camera_geometry_sha256=sha(geometry),config_sha256=config_before,
        formal_configuration_changed=False,motion_published=False,coppeliasim_modified=False)
    (ROOT/'final_source_snapshot.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':freeze()
