"""Guarded evidence-only FK export update; no joint path or config change."""
from __future__ import annotations
import hashlib
import json
import shutil
from pathlib import Path

ROOT=Path('/home/robot/ros2_ws/evaluation/wrist_joint_20261002')
ACTIVE=Path('/home/robot/ros2_ws/src/fusion_model_ros2_beta')
BEFORE='943dc236a45e1f199bf6aeb37b5e145674239d20078feddf4fd0b50dc3ab1c80'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def refresh():
    prior=json.loads((ROOT/'final_source_snapshot.json').read_text())
    cfg={str(p.relative_to(ACTIVE)):sha(p) for p in (ACTIVE/'config').rglob('*') if p.is_file()}
    if cfg!=prior['config_sha256']:raise ValueError('formal configuration changed')
    if sha(ACTIVE/'fusion_model_ros2_beta/ur10_actual_kinematics.py')!=prior['current_factory_tool_camera_geometry_sha256']:
        raise ValueError('calibrated geometry changed')
    plan=[]
    for name,target in (('active',ACTIVE),('deployed',ROOT/'deployed_source')):
        for rel,expected in (('fusion_model_ros2_beta/evaluate_actual_wrist_tradeoff.py',BEFORE),
                             ('fusion_model_ros2_beta/export_robot_path.py',None)):
            src,dst=ROOT/'source'/rel,target/rel
            current=sha(dst) if dst.exists() else None
            if current==sha(src):continue
            if current!=expected:raise ValueError('preserving changed file: '+str(dst))
            plan.append((name,rel,src,dst,current,sha(src)))
    backup=ROOT/'exporter_backup_20261002';backup.mkdir(exist_ok=False)
    report={'joint_path_modified':False,'formal_configuration_changed':False,'motion_published':False,'files':[]}
    for name,rel,src,dst,current,incoming in plan:
        if current is not None:
            p=backup/name/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(dst,p)
        shutil.copy2(src,dst)
        report['files'].append(dict(snapshot=name,path=rel,before_sha256=current,after_sha256=incoming))
    (ROOT/'exporter_source_update.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':refresh()
