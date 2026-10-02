"""Do not reuse wrist-optimized glyphs at a different physical robot layout."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from . import ur10_actual_kinematics as robot


def load_pinned_context(folder, metadata):
    if not metadata.get('actual_wrist_weight',0.): return None
    request=json.loads((Path(folder)/'original_target_generation_request.json').read_text())
    path=Path(request['config']['joint_robot_context'])
    if hashlib.sha256(path.read_bytes()).hexdigest()!=request['robot_context_sha256']:
        raise ValueError('wrist-optimized candidate robot context changed')
    return json.loads(path.read_text())


def validate_placement(context, *, paper_xy, paper_z, brush_length):
    if context is None: return
    if (context['calibration_hash']!=robot.CALIBRATION_HASH
            or context['kinematics_sha256']!=hashlib.sha256(Path(robot.__file__).read_bytes()).hexdigest()):
        raise ValueError('wrist-optimized candidate factory/tool/camera geometry changed')
    actual=np.r_[paper_xy,paper_z,brush_length]
    expected=np.r_[context['paper_offset_xy_m'],context['paper_z_m'],context['brush_length_m']]
    if not np.allclose(actual,expected,rtol=0,atol=1e-9):
        raise ValueError('wrist-optimized candidate belongs to a different paper placement or TCP; reinvert')
