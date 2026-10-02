"""Deploy only the report-planning lock; preserve all optimization sources."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT=Path('/home/robot/ros2_ws/evaluation/wrist_joint_20261002')
ACTIVE=Path('/home/robot/ros2_ws/src/fusion_model_ros2_beta')
REL='fusion_model_ros2_beta/evaluate_actual_wrist_tradeoff.py'
BEFORE='903f2b4ca9400d6deb278188e311cfa974117ca68701c2038c9109ad72fb3090'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def refresh():
    prior=json.loads((ROOT/'final_source_snapshot.json').read_text())
    cfg={str(p.relative_to(ACTIVE)):sha(p) for p in (ACTIVE/'config').rglob('*') if p.is_file()}
    if cfg!=prior['config_sha256']:raise ValueError('formal configuration changed')
    if sha(ACTIVE/'fusion_model_ros2_beta/ur10_actual_kinematics.py')!=prior['current_factory_tool_camera_geometry_sha256']:
        raise ValueError('calibrated geometry changed')
    incoming=ROOT/'source'/REL;after=sha(incoming)
    for target in (ACTIVE,ROOT/'deployed_source'):
        if sha(target/REL) not in (BEFORE,after):raise ValueError('preserving changed report code')
    backup=ROOT/'report_lock_backup_20261002';backup.mkdir(exist_ok=False)
    for name,target in (('active',ACTIVE),('deployed',ROOT/'deployed_source')):
        saved=backup/name/REL;saved.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(target/REL,saved);shutil.copy2(incoming,target/REL)
    (ROOT/'report_lock_update.json').write_text(json.dumps(dict(before_sha256=BEFORE,after_sha256=after,
        optimization_sources_changed=False,joint_path_modified=False,formal_configuration_changed=False,
        motion_published=False),indent=2))


if __name__=='__main__':refresh()
