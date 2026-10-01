"""ROS-compatible absolute brush rotations for local-gamma inversion.

Canvas y points down; the exported source/paper y points up. Uniform scale
and translation do not affect heading. Pen-up interpolation is deliberately
absent: the ROS planner, not this image objective, owns stroke transitions.
"""
from __future__ import annotations

import math
import numpy as np
import torch


CONVENTION = "Rz(wrap(local_gamma+paper_forward_heading)) @ Ry(beta) @ Rx(alpha) @ Rx(pi)"


def canvas_paper_headings(xy: torch.Tensor, point_indices) -> torch.Tensor:
    """Match gamma_semantics.forward_headings after CanvasTransform.unmap.

    The last point reuses its preceding forward segment. A zero-length
    segment has heading zero, just as numpy atan2(0, 0) in the ROS exporter;
    use a constant safe argument there to keep autograd finite.
    """
    heading = xy.new_zeros(len(xy))
    for indices in point_indices:
        index = torch.as_tensor(indices, dtype=torch.long, device=xy.device)
        if len(index) < 2:
            continue
        delta = xy[index[1:]] - xy[index[:-1]]
        moving = delta.square().sum(dim=1) > 0
        dx = torch.where(moving, delta[:, 0], torch.ones_like(delta[:, 0]))
        dy = torch.where(moving, -delta[:, 1], torch.zeros_like(delta[:, 1]))
        segment = torch.atan2(dy, dx)
        heading = heading.index_copy(0, index, torch.cat((segment, segment[-1:])))
    return heading


def brush_rotations(alpha, beta, gamma_absolute):
    """Exactly the driver rotation, including the fixed downward Rx(pi)."""
    sa, ca = torch.sin(alpha), torch.cos(alpha)
    sb, cb = torch.sin(beta), torch.cos(beta)
    sg, cg = torch.sin(gamma_absolute), torch.cos(gamma_absolute)
    zero, one = torch.zeros_like(alpha), torch.ones_like(alpha)
    rx = torch.stack((one, zero, zero, zero, ca, -sa, zero, sa, ca), dim=-1).reshape(-1, 3, 3)
    ry = torch.stack((cb, zero, sb, zero, one, zero, -sb, zero, cb), dim=-1).reshape(-1, 3, 3)
    rz = torch.stack((cg, -sg, zero, sg, cg, zero, zero, zero, one), dim=-1).reshape(-1, 3, 3)
    down = alpha.new_tensor([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
    return rz @ ry @ rx @ down


def absolute_tool_rotations(xy_canvas, posture, gamma_local, point_indices):
    gamma_absolute = gamma_local + canvas_paper_headings(xy_canvas, point_indices)
    # Sin/cos implement wrapping without a discontinuous optimization variable.
    return brush_rotations(posture[:, 1], posture[:, 2], gamma_absolute)


def absolute_tool_rotation_residuals(xy_canvas, posture, gamma_local, point_indices, weight):
    """SO(3) chord changes, within each stroke only; not an angular speed.

    ||R_next-R_prev||_F/sqrt(2) = 2 sin(theta/2). This periodic residual
    avoids acos gradients at zero and +pi/-pi Euler discontinuities.
    """
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("absolute tool rotation weight must be finite and nonnegative")
    if weight == 0:
        return []
    rotations = absolute_tool_rotations(xy_canvas, posture, gamma_local, point_indices)
    result = []
    for indices in point_indices:
        index = torch.as_tensor(indices, dtype=torch.long, device=xy_canvas.device)
        if len(index) >= 2:
            result.append(math.sqrt(weight / 2.) * (rotations[index[1:]] - rotations[index[:-1]]).flatten())
    return result


def absolute_tool_rotation_report(xy_canvas, posture, gamma_local, point_indices):
    with torch.no_grad():
        # Audit in float64: acos(trace) can accumulate spurious motion for
        # identical float32 matrices on a long stroke.
        rotations = absolute_tool_rotations(xy_canvas.double(), posture.double(), gamma_local.double(), point_indices).cpu().numpy()
        heading = canvas_paper_headings(xy_canvas.double(), point_indices).cpu().numpy()
        gamma = gamma_local.detach().cpu().numpy() + heading
        gamma = np.arctan2(np.sin(gamma), np.cos(gamma))
    angles, per_stroke, zero_segments = [], [], 0
    xy = xy_canvas.detach().cpu().numpy()
    for number, indices in enumerate(point_indices):
        index = np.asarray(indices, dtype=int)
        values = np.empty(0)
        if len(index) >= 2:
            relative = rotations[index[:-1]].transpose(0, 2, 1) @ rotations[index[1:]]
            values = np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2) - 1.) / 2., -1., 1.))
            zero_segments += int(np.count_nonzero(np.sum(np.diff(xy[index], axis=0)**2, axis=1) == 0))
            angles.extend(values.tolist())
        per_stroke.append({"stroke_index": number, "total_rad": float(values.sum()), "max_step_rad": float(values.max()) if len(values) else 0.})
    return {"convention": CONVENTION, "xy_frame": "canvas x right/y down reflected to paper x right/y up",
            "scope": "within-stroke contact points only; pen-up handled by ROS planning",
            "total_rad": float(sum(angles)), "total_deg": float(np.rad2deg(sum(angles))),
            "max_step_rad": float(max(angles, default=0.)), "segment_count": len(angles),
            "zero_length_segments": zero_segments, "per_stroke": per_stroke,
            "gamma_absolute_range_rad": [float(gamma.min()), float(gamma.max())] if len(gamma) else []}
