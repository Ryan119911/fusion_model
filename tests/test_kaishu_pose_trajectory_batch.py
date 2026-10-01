from types import SimpleNamespace

import numpy as np
from PIL import Image

from tools.invert_kaishu_trajectory_batch import (
    _child_command,
    character_stem,
    choose_target,
    group_target_items,
    filter_target_items,
    parse_target_overrides,
    save_target_image,
    select_trajectory_samples,
    target_match_score,
    target_compatibility_failures,
)


class FakeSample:
    def __init__(self, character, sample_id, point_count):
        self.character = character
        self.meta = {"sample_id": sample_id}
        self._point_count = point_count

    def all_points(self):
        return [object()] * self._point_count


def test_character_stem_is_portable_and_collision_free():
    assert character_stem("武") == "char_u6b66"
    assert character_stem("A") != character_stem("Ａ")


def test_selects_longest_sample_and_records_duplicates():
    selected, counts = select_trajectory_samples(
        [
            FakeSample("武", "short", 2),
            FakeSample("武", "long", 4),
            FakeSample("文", "only", 3),
        ],
        selection="longest",
    )
    assert selected["武"].meta["sample_id"] == "long"
    assert counts == {"武": 2, "文": 1}


def test_target_match_score_prefers_supported_points(tmp_path):
    target = np.zeros((16, 16), dtype=np.float32)
    target[5:11, 5:11] = 1.0
    score = target_match_score(
        target,
        np.asarray([[7.0, 7.0], [8.0, 8.0]], dtype=np.float32),
        tolerance_px=1,
    )
    assert score["coverage"] == 1.0
    assert score["mean_distance_px"] == 0.0
    assert score["score"] == 1.0


def test_target_match_score_penalizes_unrenderably_thick_target():
    support = np.zeros((16, 16), dtype=bool)
    support[7:9, 2:14] = True
    thin = support.astype(np.float32)
    thick = np.zeros((16, 16), dtype=np.float32)
    thick[4:12, 2:14] = 1.0
    points = np.asarray([[3.0, 8.0], [12.0, 8.0]], dtype=np.float32)
    thin_score = target_match_score(thin, points, model_support=support)
    thick_score = target_match_score(thick, points, model_support=support)
    assert thin_score["coverage"] == thick_score["coverage"] == 1.0
    assert thin_score["model_compatibility_score"] > thick_score["model_compatibility_score"]


def test_target_compatibility_rejects_wrong_structure():
    args = SimpleNamespace(
        min_target_coverage=0.75,
        min_model_support_dice=0.30,
        min_target_support_ink_ratio=0.45,
        max_target_support_ink_ratio=1.65,
    )
    failures = target_compatibility_failures(
        {
            "selection": "best_model_support",
            "trajectory_match": {
                "coverage": 0.30,
                "model_support_dice": 0.08,
                "target_to_model_support_ink_ratio": 0.23,
            },
        },
        args,
    )
    assert "trajectory_coverage_below_threshold" in failures
    assert "model_support_dice_below_threshold" in failures
    assert "target_support_ink_ratio_too_small" in failures


def test_target_image_is_black_ink_on_white_background(tmp_path):
    path = tmp_path / "target.png"
    ink = np.zeros((4, 4), dtype=np.float32)
    ink[1:3, 1:3] = 1.0
    save_target_image(ink, path)
    array = np.asarray(Image.open(path).convert("L"))
    assert array[0, 0] == 255
    assert array[1, 1] == 0


def test_target_override_parser_is_repeatable_and_rejects_malformed_values():
    assert parse_target_overrides(["武=target.png", "文=wen.png"]) == {
        "武": "target.png",
        "文": "wen.png",
    }
    try:
        parse_target_overrides(["武"])
    except ValueError as exc:
        assert "CHARACTER=IMAGE_PATH" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("malformed override should be rejected")


def test_group_target_items_is_deterministic():
    dataset = SimpleNamespace(
        index=[
            {"character": "武", "image_path": "b.jpg", "shape_index": 1},
            {"character": "武", "image_path": "a.jpg", "shape_index": 2},
            {"character": "文", "image_path": "c.jpg", "shape_index": 0},
        ]
    )
    grouped = group_target_items(dataset)
    assert [item["image_path"] for item in grouped["武"]] == ["a.jpg", "b.jpg"]


def test_target_exclusion_matches_image_and_bbox():
    grouped = {
        "武": [
            {"image_path": "data/0022.jpg", "bbox": [1, 2, 3, 4]},
            {"image_path": "data/good.jpg", "bbox": [1, 2, 3, 4]},
        ]
    }
    filtered, removed = filter_target_items(
        grouped,
        [{"character": "武", "image_path": "0022.jpg", "bbox": [1, 2, 3, 4], "reason": "corrupt"}],
    )
    assert len(filtered["武"]) == 1
    assert filtered["武"][0]["image_path"].endswith("good.jpg")
    assert removed[0]["reason"] == "corrupt"


def test_child_command_contains_selected_character_and_flags(tmp_path):
    args = SimpleNamespace(
        python="python",
        trajectory_csv="traj.csv",
        target_image="unused",
        bbsmg_ckpt="model.pt",
        device="cpu",
        image_size=128,
        padding=16,
        order=5,
        max_steps=3,
        damping=0.05,
        optimization_size=64,
        point_batch_size=8,
        pixel_weight=3.0,
        h_smoothness_weight=0.2,
        h_point_velocity_weight=5.0,
        h_point_acceleration_weight=10.0,
        initial_h_mm=15.5,
        initial_alpha_deg=0.0,
        initial_beta_deg=0.0,
        initial_gamma_deg=0.0,
        xy_max_offset_px=4.0,
        xy_smoothness_weight=1.0,
        xy_prior_weight=0.15,
        dynamic_profile="wang2020_figure4_digitized_v1",
        pixels_per_model_unit=20.0,
        patch_floor=0.05,
        footprint_longitudinal_scale=0.22,
        footprint_transverse_scale=0.262,
        render_max_step_px=2.0,
        optimize_xy=True,
        fused_pose_from_height=True,
        field_mode=None,
        optimize_gamma=False,
        gamma_max_abs_deg=180.0,
        search_orders=False,
        order_min=3,
        order_max=8,
        cap_order_to_points=True,
    )
    command = _child_command(args, "武", "sample", tmp_path / "target.png", tmp_path)
    assert "武" in command
    assert "--optimize_xy" in command
    assert "--fused_pose_from_height" in command
    assert "--field_mode" in command
    assert command[command.index("--field_mode") + 1] == "h_only"
    assert command[command.index("--h_point_velocity_weight") + 1] == "5.0"
