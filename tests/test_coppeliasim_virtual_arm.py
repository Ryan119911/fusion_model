from pathlib import Path

from tools.coppeliasim_virtual_arm import (
    CoordinateMapper,
    PoseRow,
    build_report,
    interpolate_stroke,
    mapped_strokes,
)


def _rows():
    return [
        PoseRow("武", "s", 0, 0, 0.0, 0.0, 12.0, 0.1, 0.2, 0.3, 0),
        PoseRow("武", "s", 0, 1, 1.0, 1.0, 15.0, 0.1, 0.2, 0.4, 1),
        PoseRow("武", "s", 1, 0, 0.0, 1.0, 13.0, 0.0, 0.0, 0.0, 2),
        PoseRow("武", "s", 1, 1, 1.0, 0.0, 14.0, 0.0, 0.0, 0.0, 0),
    ]


def test_mapper_lifts_up_states_and_preserves_pose_orientation():
    mapper = CoordinateMapper(0.0, 1.0, 0.0, 1.0, 12.0, 15.0)
    strokes = mapped_strokes(_rows(), mapper)
    down = strokes[0][0]
    up = strokes[1][0]
    assert up.position[2] > down.position[2]
    assert down.orientation == (0.1, 0.2, 0.3)


def test_default_mapping_is_for_standard_top_view_and_legacy_flip_is_opt_in():
    rows = _rows()
    top_mapper = CoordinateMapper(0.0, 1.0, 0.0, 1.0, 12.0, 15.0)
    legacy_mapper = CoordinateMapper(0.0, 1.0, 0.0, 1.0, 12.0, 15.0, flip_y=True)
    top = top_mapper.map_row(rows[0])
    legacy = legacy_mapper.map_row(rows[0])
    assert top_mapper.flip_y is False
    assert legacy_mapper.flip_y is True
    assert top.position[1] == -legacy.position[1]


def test_interpolation_stays_inside_one_stroke():
    mapper = CoordinateMapper(0.0, 1.0, 0.0, 1.0, 12.0, 15.0)
    strokes = mapped_strokes(_rows(), mapper)
    dense = interpolate_stroke(strokes[0], 0.01)
    assert dense
    assert {point.stroke_id for point in dense} == {0}
    assert all(left.stroke_id == right.stroke_id for left, right in zip(dense, dense[1:]))


def test_report_declares_no_cross_stroke_segments():
    mapper = CoordinateMapper(0.0, 1.0, 0.0, 1.0, 12.0, 15.0)
    strokes = mapped_strokes(_rows(), mapper)
    report = build_report(_rows(), strokes, mapper)
    assert report["strokes"] == 2
    assert report["cross_stroke_segments"] == 0
    assert report["safety"]["actual_robot_ik"] is False
    assert report["coordinate_mapping"]["view_reference"] == "top_view"
