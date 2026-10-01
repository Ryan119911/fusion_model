"""Read-only joint-path rotation accounting (no IK and no motion commands)."""
from __future__ import annotations
import numpy as np


JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")


def joint_rotation_metrics(joints, states, stroke_ids, durations=None):
    q = np.asarray(joints, dtype=float)
    states, ids = np.asarray(states), np.asarray(stroke_ids)
    if q.ndim != 2 or q.shape[1] != 6 or len(q) < 2 or not np.isfinite(q).all():
        raise ValueError("need a finite complete [N,6] planned joint path")
    if states.shape != (len(q),) or ids.shape != (len(q),):
        raise ValueError("state and stroke arrays must match the planned path")
    # Do NOT wrap deltas. These are unwrapped controller targets, not Euler
    # angles. Wrapping would hide a full wrist turn that the robot must execute.
    change = np.abs(np.diff(q, axis=0))
    writing = (states[1:] != 3) & (states[:-1] != 3) & (ids[1:] == ids[:-1])
    total, contact, transfers = change.sum(0), change[writing].sum(0), change[~writing].sum(0)
    result = {"delta_convention": "absolute differences of actual unwrapped planned joints; no modulo 2pi",
              "target_count": len(q), "within_stroke_edge_count": int(writing.sum()),
              "pen_up_or_approach_edge_count": int((~writing).sum()),
              "joint_total_rad": dict(zip(JOINT_NAMES, total.tolist())),
              "joint_total_deg": dict(zip(JOINT_NAMES, np.rad2deg(total).tolist())),
              "wrist_total_rad": float(total[3:].sum()), "wrist_total_deg": float(np.rad2deg(total[3:].sum())),
              "wrist_writing_rad": float(contact[3:].sum()), "wrist_transfer_rad": float(transfers[3:].sum()),
              "wrist3_total_rad": float(total[5]), "wrist3_total_deg": float(np.rad2deg(total[5])),
              "max_joint_step_rad": float(change.max()), "max_wrist_step_rad": float(change[:, 3:].max()),
              "max_internal_wrist_step_rad": float(change[1:, 3:].max()) if len(change) > 1 else 0.,
              "joint_range_rad": {name: [float(q[:, i].min()), float(q[:, i].max())] for i, name in enumerate(JOINT_NAMES)}}
    if durations is not None:
        times = np.asarray(durations, dtype=float)
        if times.shape != (len(q),) or not np.isfinite(times).all() or np.any(times <= 0):
            raise ValueError("durations must be positive finite planned intervals")
        result["planned_duration_s"] = float(times.sum())
        result["max_joint_speed_rad_s"] = float((change / times[1:, None]).max())
    return result
