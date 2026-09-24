"""Reconstruct A-style TARGETS only; never return executable robot trajectories.

The legacy polygon renderer is deliberately confined to reference preparation.
Generated robot controls must still pass through the V16 six-field inverse and
the unchanged neural forward renderer. This is not a physical brush calibration.
"""
import json
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
from .offline_fontsize_inversion import _sha256, _read_pose_rows


def build_A_reference(entry, destination, *, font_size_m, expected_source_sha256):
    size = float(font_size_m)
    if not math.isfinite(size) or not 0 < size <= .29:
        raise ValueError('reference size must be positive and at most 290 mm')
    source = Path(entry.trajectory_csv).resolve()
    if _sha256(source) != expected_source_sha256:
        raise ValueError('reference source SHA256 mismatch')
    _, rows = _read_pose_rows(source)
    if any(r['character'] != entry.character or r['sample_id'] != entry.sample_id or int(r['state'])==3
           for r in rows):
        raise ValueError('A reference requires a single contact-only source sample')
    # Lazy import: target preparation needs ROS Python dependencies but constructs
    # no ROS node, publishes nothing, and never edits the formal renderer.
    from fusion_model_ros2 import brush_trajectory_driver as legacy
    from fusion_model_ros2 import paper_brush_model as brush
    points, _ = legacy.load_points(source, entry.character, entry.sample_id)
    mapped = legacy.map_source_points(points, paper_width=size+.03,paper_height=size+.03,
        paper_z=0.,paper_offset_x=0.,paper_offset_y=0.,margin=.015,
        max_compression=.02,lift_height=.025,reference_trajectory_extent=.29)
    params=dict(max_step_m=.002,lift_height_m=.025,brush_bezier_samples=14)
    fake=SimpleNamespace(_param=lambda key:params[key],
        _brush_model=brush.FlexibleBrushModel(brush.PaperBrushParameters()),
        get_logger=lambda:SimpleNamespace(info=lambda message:None))
    dense=legacy.BrushTrajectoryDriver._densify_mapped(fake,mapped)
    fake._targets=[SimpleNamespace(point=p) for p in dense]
    legacy.BrushTrajectoryDriver._build_flexible_footprints(fake)
    polygons=[p for p in fake._footprints if p is not None]
    if not polygons:
        raise ValueError('reference renderer produced no contact ink')
    canvas_m=.29*128/96
    target=brush.render_world_footprints(polygons,paper_width_m=canvas_m,paper_height_m=canvas_m,
        paper_offset_x_m=0.,paper_offset_y_m=0.,image_size=128,supersampling=4)
    if not np.isfinite(target).all() or float(target.max()) <= 0:
        raise ValueError('invalid A reference image')
    destination=Path(destination).resolve()
    destination.mkdir(parents=True,exist_ok=False)
    target_path=destination/'A_reference.png'
    Image.fromarray(np.uint8((1-target)*255)).save(target_path)
    record=dict(character=entry.character,sample_id=entry.sample_id,
        source_sha256=expected_source_sha256,target_image=str(target_path),target_sha256=_sha256(target_path),
        font_size_m=size,reference_only=True,mode='reconstructed_A_legacy_polygon_target',
        reference_frame=dict(image_size=128,padding=16,reference_font_size_m=.29),
        implementation_sha256=dict(builder=_sha256(Path(__file__)),
            driver=_sha256(Path(legacy.__file__)),brush=_sha256(Path(brush.__file__))),
        parameters=params,physical_calibration=False)
    (destination/'A_reference.json').write_text(json.dumps(record,indent=2,ensure_ascii=False))
    return record
