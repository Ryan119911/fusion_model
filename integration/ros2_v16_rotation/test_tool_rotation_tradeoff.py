"""Source-package tests; model tests exercise torch cost/gradient separately."""
import importlib
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_ros_local_gamma_and_matrix_match_model_cost():
    torch = pytest.importorskip("torch")
    from optim.tool_orientation import absolute_tool_rotations
    from fusion_model_ros2_beta.gamma_semantics import local_to_absolute
    # Driver rotation is Rx(pi) after Rz/Ry/Rx; import in the ROS environment
    # when rclpy is available, otherwise use the exact published formula.
    try:
        from fusion_model_ros2_beta.brush_trajectory_driver import _brush_rotation
    except ModuleNotFoundError as exc:
        if exc.name not in {"rclpy", "ament_index_python", "geometry_msgs", "sensor_msgs", "trajectory_msgs", "visualization_msgs"}:
            raise
        def _brush_rotation(a,b,g):
            ca,sa,cb,sb,cg,sg = np.cos(a),np.sin(a),np.cos(b),np.sin(b),np.cos(g),np.sin(g)
            return np.array([[cg,-sg,0],[sg,cg,0],[0,0,1]]) @ np.array([[cb,0,sb],[0,1,0],[-sb,0,cb]]) @ np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]]) @ np.diag([1,-1,-1])
    xy = np.array([[1,2],[2,4],[3,5],[9,9],[8,10]],dtype=float)
    ids = np.array([0,0,0,1,1]); groups = [np.where(ids==s)[0] for s in (0,1)]
    pose = np.array([[15,.1,.04]]*len(xy)); gamma = np.array([.1,.2,.3,-.1,-.2])
    absolute = local_to_absolute(gamma,xy*np.array([1,-1]),ids)
    expected = np.array([_brush_rotation(.1,.04,g) for g in absolute])
    actual = absolute_tool_rotations(torch.tensor(xy),torch.tensor(pose),torch.tensor(gamma),groups)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_config_rotation_weight_validation_and_command_forwarding(tmp_path):
    from dataclasses import replace
    from fusion_model_ros2_beta.offline_fontsize_inversion import OfflineInversionConfig
    from fusion_model_ros2_beta.joint_fontsize_inversion import build_joint_command
    executable, checkpoint = tmp_path/"python", tmp_path/"bbsmg.pt"
    executable.touch(); checkpoint.touch()
    cfg = OfflineInversionConfig(tmp_path, executable, checkpoint, tmp_path/"cache", joint_tool_absolute_rotation_weight=12.)
    cfg.validate()
    with pytest.raises(ValueError):
        replace(cfg,joint_tool_absolute_rotation_weight=-1).validate()
    # Six-field contract remains independently optimized, not a fused-from-H proxy.
    cmd = build_joint_command(cfg, tmp_path/"source",tmp_path/"initial",tmp_path/"target",tmp_path/"out","武","wu")
    assert "--optimize_xy" in cmd and "--optimize_gamma" in cmd
    assert cmd[cmd.index("--field_mode")+1] == "all"
    original = importlib.import_module("fusion_model_ros2_beta.original_target_inversion")
    source = Path(original.__file__).read_text()
    assert "--beta_tool_absolute_rotation_weight" in source
    assert "gamma_semantics.py" in source and "optim/tool_orientation.py" in source


def test_comparison_does_not_select_partial_plan(tmp_path):
    from fusion_model_ros2_beta.evaluate_tool_rotation_tradeoff import summarize
    baseline = {"rotation_weight":0,"generation_passed":True,"quality":{"iou_at_0.5":.5},
                "planning":{"feasible":False,"status":"rejected","partial_target_count":30}}
    experiment = {"cases":[baseline]}
    summarize(tmp_path,experiment,.01)
    assert experiment["recommended_rotation_weight"] is None
    assert experiment["paired_results"][0]["wrist_total_deg"] is None


def test_comparison_keeps_image_quality_bound(tmp_path):
    from fusion_model_ros2_beta.evaluate_tool_rotation_tradeoff import summarize
    def case(weight,iou,travel):
        return {"rotation_weight":weight,"generation_passed":True,"quality":{"iou_at_0.5":iou},
                "planning":{"feasible":True,"metrics":{"wrist_total_deg":travel}}}
    # Do not fake candidate file images in this unit test; only summary data.
    cases = [case(0,.9,100),case(10,.895,80),case(100,.8,10)]
    for c in cases:
        c["generation_passed"] = False
    # A real generated result always contains images; test the policy via a
    # temporary minimal fixture so the same summary implementation executes.
    from PIL import Image
    for i,c in enumerate(cases):
        folder=tmp_path/str(i); folder.mkdir()
        for name in ("inversion_target.png","inversion_rendered.png","inversion_diff.png"):
            Image.new("L",(2,2)).save(folder/name)
        c.update(generation_passed=True,candidate=str(folder))
    experiment={"cases":cases}
    summarize(tmp_path,experiment,.01)
    assert experiment["recommended_rotation_weight"] == 10
