"""Geometry boundary for operator-authorized millimetre-scale air alignment.

This does not authorize gross positioning, contact motion or reorientation.
Fresh wrench, model audit and the unchanged buffered guard are still required.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def local_air_target(start, planned, zero_pose, tip_offset, normal):
    start, planned, zero_pose = [np.asarray(p, float) for p in (start, planned, zero_pose)]
    tip_offset, normal = np.asarray(tip_offset, float), np.asarray(normal, float)
    if any(p.shape != (6,) for p in (start, planned, zero_pose)) or tip_offset.shape != (3,) or normal.shape != (3,):
        raise ValueError('invalid air alignment geometry')
    if not np.isfinite(np.r_[start,planned,zero_pose,tip_offset,normal]).all() or abs(np.linalg.norm(normal)-1) > 1e-6:
        raise ValueError('nonfinite geometry or invalid paper normal')
    if np.linalg.norm(planned[:3]-start[:3]) > .005+1e-12:
        raise ValueError('coarse positioning required: fine alignment is limited to 5 mm')
    if (Rotation.from_rotvec(start[3:]).inv()*Rotation.from_rotvec(planned[3:])).magnitude() > np.deg2rad(.05):
        raise ValueError('fine alignment does not rotate the tool')
    target = planned.copy(); target[3:] = start[3:]
    zero_tip = zero_pose[:3]+Rotation.from_rotvec(zero_pose[3:]).apply(tip_offset)
    heights = [(p[:3]+Rotation.from_rotvec(p[3:]).apply(tip_offset)-zero_tip)@normal for p in (start,target)]
    if heights[1] < .003-1e-9:
        raise ValueError('target does not retain 3 mm nominal clearance above observed local zero')
    if heights[0] < .003 and heights[1] < heights[0]-1e-9:
        raise ValueError('no downward fine alignment from below 3 mm nominal clearance')
    return target.tolist()
