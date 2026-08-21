from pathlib import Path

from tools.coppeliasim_virtual_arm import (
    CoordinateMapper,
    PoseRow,
    _euler_xyz_to_quaternion,
    _interpolate_euler_shortest,
    _rotate_vector_by_quaternion,
    _world_bbox_min_z,
    build_report,
    interpolate_stroke,
    mapped_strokes,
    save_pressure_trajectory,
    trajectory_provenance,
    validate_trajectory_identity,
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


def test_bbsmg_h_maps_to_negative_signed_brush_compression():
    mapper = CoordinateMapper(0.0, 1.0, 0.0, 1.0, 11.0, 20.0)
    maximum_pressure = mapper.map_row(
        PoseRow("武", "s", 0, 0, 0.0, 0.0, 11.0, 0.0, 0.0, 0.0, 1)
    )
    neutral = mapper.map_row(
        PoseRow("武", "s", 0, 1, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 1)
    )
    assert abs(maximum_pressure.signed_pressure_z + 0.015) < 1e-12
    assert abs(neutral.signed_pressure_z) < 1e-12
    assert maximum_pressure.position[2] < mapper.paper_z
    assert neutral.position[2] == mapper.paper_z


def test_pressure_export_keeps_original_h_and_writes_negative_z(tmp_path: Path):
    mapper = CoordinateMapper(0.0, 1.0, 0.0, 1.0, 11.0, 20.0)
    path = save_pressure_trajectory(_rows(), mapper, tmp_path)
    text = path.read_text(encoding="utf-8-sig")
    assert "negative_brush_compression" in text
    assert "original_h_mm" in text
    assert "-0.013333" in text


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


def test_report_includes_trajectory_provenance():
    mapper = CoordinateMapper(0.0, 1.0, 0.0, 1.0, 12.0, 15.0)
    strokes = mapped_strokes(_rows(), mapper)
    provenance = {"resolved_path": "/tmp/pose.csv", "sha256": "a" * 64}
    report = build_report(_rows(), strokes, mapper, provenance=provenance)
    assert report["trajectory_source"] == provenance


def test_trajectory_provenance_and_identity_guard(tmp_path: Path):
    csv_path = tmp_path / "pose.csv"
    csv_path.write_text(
        "character,sample_id,stroke_id,point_id,x,y,z,alpha,beta,gamma,state,prototype,pose_frame,z_unit,angle_unit,gamma_semantics\n"
        "武,s,0,0,0,0,12,0.1,0.2,0.3,0,paper_target_local_footprint_v44,paper_model,mm,rad,absolute_forward_xy_heading\n"
        "武,s,0,1,1,1,15,0.0,0.2,-0.3,1,paper_target_local_footprint_v44,paper_model,mm,rad,absolute_forward_xy_heading\n",
        encoding="utf-8",
    )
    provenance = trajectory_provenance(csv_path, character="武", sample_id="s")
    assert provenance["resolved_path"] == str(csv_path.resolve())
    assert len(provenance["sha256"]) == 64
    assert provenance["prototype"] == ["paper_target_local_footprint_v44"]
    assert provenance["pose_ranges"]["z"] == [12.0, 15.0]
    assert provenance["nonzero_pose_counts"]["alpha"] == 1
    validate_trajectory_identity(
        provenance,
        required_prototype="paper_target_local_footprint_v44",
        required_sha256=provenance["sha256"].upper(),
    )


def test_trajectory_identity_guard_rejects_wrong_prototype(tmp_path: Path):
    provenance = {
        "resolved_path": str(tmp_path / "pose.csv"),
        "prototype": ["paper_psoc_lm_v11_staged_pose"],
        "sha256": "a" * 64,
    }
    try:
        validate_trajectory_identity(
            provenance,
            required_prototype="paper_target_local_footprint_v44",
        )
    except ValueError as exc:
        assert "prototype mismatch" in str(exc)
    else:
        raise AssertionError("wrong trajectory prototype should be rejected")


def test_csv_euler_quaternion_rotates_about_z():
    quaternion = _euler_xyz_to_quaternion((0.0, 0.0, 3.141592653589793 / 2.0))
    rotated = _rotate_vector_by_quaternion(quaternion, (1.0, 0.0, 0.0))
    assert abs(rotated[0]) < 1e-7
    assert abs(rotated[1] - 1.0) < 1e-7
    assert abs(rotated[2]) < 1e-7


def test_euler_interpolation_uses_short_wrapped_delta():
    value = _interpolate_euler_shortest((0.0, 0.0, 3.0), (0.0, 0.0, -3.0), 0.5)
    assert abs(abs(value[2]) - 3.141592653589793) < 1e-7


def test_world_bbox_min_z_uses_oriented_geometry_not_object_center():
    # Local Z is rotated into world X; local X controls the world-space Z
    # extent.  The object center is at world Z=0.10, but geometry reaches 0.06.
    matrix = [
        0.0, 0.0, 1.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        -1.0, 0.0, 0.0, 0.10,
    ]
    bounds = [-0.04, -0.02, -0.20, 0.04, 0.02, 0.20]
    assert abs(_world_bbox_min_z(matrix, bounds) - 0.06) < 1e-9
