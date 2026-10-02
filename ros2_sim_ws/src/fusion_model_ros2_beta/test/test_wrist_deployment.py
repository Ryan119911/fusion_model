import importlib.util
from pathlib import Path
import hashlib
import pytest


def module():
    path=Path(__file__).parents[1]/'deploy_wrist_patch.py'
    spec=importlib.util.spec_from_file_location('deploy_wrist_patch',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def fixture(tmp_path):
    m=module();src=tmp_path/'src';target=tmp_path/'target'
    rel='fusion_model_ros2_beta/brush_trajectory_driver.py'
    for root in (src,target):(root/rel).parent.mkdir(parents=True)
    old=b'default=False\n';(target/rel).write_bytes(old);(src/rel).write_text('default=False\nfeature=True\n')
    for name in m.NEW_NAMES:(src/'fusion_model_ros2_beta'/name).write_text('# new opt-in module\n')
    return m,src,target,{rel:hashlib.sha256(old).hexdigest()}


def test_preserves_dirty_file_and_never_partially_deploys(tmp_path):
    m,src,target,expected=fixture(tmp_path)
    p=target/'fusion_model_ros2_beta/brush_trajectory_driver.py';p.write_text('user change\n')
    with pytest.raises(ValueError,match='preserving changed file'):
        m.deploy(src,target,tmp_path/'backup',expected)
    assert p.read_text()=='user change\n'
    assert not (target/'fusion_model_ros2_beta/ur10_wrist_cost.py').exists()


def test_scoped_source_deployment_keeps_config_and_original_backup(tmp_path):
    m,src,target,expected=fixture(tmp_path)
    yaml=target/'formal.yaml';yaml.write_text('production: unchanged\n')
    backup=tmp_path/'backup';report=m.deploy(src,target,backup,expected)
    assert yaml.read_text()=='production: unchanged\n'
    assert (backup/'fusion_model_ros2_beta/brush_trajectory_driver.py').read_text()=='default=False\n'
    assert not report['formal_configuration_changed'] and not report['motion_published']
