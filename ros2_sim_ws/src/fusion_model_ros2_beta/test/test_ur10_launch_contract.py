"""No ROS graph or robot commands: validate the fake-hardware launch contract."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

PACKAGE = Path(__file__).resolve().parents[1]
LAUNCH = PACKAGE / 'launch' / 'ur10_brush_ros2_beta.launch.py'


def test_topics_belong_to_controllers_not_manager():
    source = LAUNCH.read_text()
    assert 'name=controller_manager_name' not in source
    assert '("controller_manager:__node", controller_manager_name)' in source
    tree = ast.parse(LAUNCH.read_text())
    namespace = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in {'controller_manager_name', 'state_broadcaster_name',
                        'trajectory_controller_name', 'controller_topic'}:
                namespace[name] = eval(compile(ast.Expression(node.value), str(LAUNCH), 'eval'), {}, namespace)
    assert namespace['controller_topic'] == '/fusion_beta_joint_trajectory_controller/joint_trajectory'
    state_topics = [eval(compile(ast.Expression(node), str(LAUNCH), 'eval'), {}, namespace)
                    for node in ast.walk(tree) if isinstance(node, ast.JoinedStr)
                    and any(isinstance(part, ast.Constant) and part.value == '/joint_states'
                            for part in node.values)]
    assert state_topics == ['/fusion_beta_joint_state_broadcaster/joint_states']


@pytest.mark.parametrize('returncode', [0, 1, -15])
def test_downstream_starts_only_after_success(returncode):
    spec = importlib.util.spec_from_file_location('actual_ur10_launch', LAUNCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    next_action = object()
    result = module._after_success(next_action)(SimpleNamespace(returncode=returncode), None)
    if returncode == 0:
        assert result == [next_action]
    else:
        assert next_action not in result
        assert any(type(action).__name__ == 'EmitEvent' for action in result)


def test_launch_model_cannot_connect_to_hardware():
    root = ET.parse(PACKAGE / 'urdf' / 'ur10_official.urdf').getroot()
    plugins = [item.text.strip() for item in root.findall('./ros2_control/hardware/plugin')]
    assert plugins == ['mock_components/GenericSystem']
