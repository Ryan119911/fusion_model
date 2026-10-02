"""Actual factory-calibrated, continuous IK inside the image LM residual.

No rclpy dependency, publication, nominal DH, gamma surrogate or modulo of
joint differences. The existing planner remains the final collision gate.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from . import ur10_actual_kinematics as robot
from .gamma_semantics import local_to_absolute
from .joint_continuity import nearest_equivalent, continuous_joint_path

LOWER = np.array([-2*np.pi, -2*np.pi, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi])
UPPER = -LOWER
INITIAL = np.array([0., -1.2, 1.2, -1.5, -np.pi/2, 0.])
ORIGINS = [robot.transform(xyz, rpy) for xyz, rpy in robot.FACTORY_JOINT_ORIGINS]
END = robot.transform(rpy=robot.WRIST3_TO_FLANGE_RPY) @ robot.transform(rpy=robot.FLANGE_TO_TOOL0_RPY)


def fk_jacobian(q):
    current = np.eye(4)
    axes, centres = [], []
    for origin, angle in zip(ORIGINS, q):
        current = current @ origin
        axes.append(current[:3, 2].copy()); centres.append(current[:3, 3].copy())
        c, s = np.cos(angle), np.sin(angle)
        rz = np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1.]])
        current = current @ rz
    current = current @ END
    axes = np.asarray(axes).T
    linear = np.cross(axes.T, current[:3,3] - np.asarray(centres)).T
    return current, np.concatenate((linear, axes), axis=0)


def ik(target, seed, max_iterations=70):
    q = np.asarray(seed, float).copy()
    for _ in range(max_iterations):
        pose, jac = fk_jacobian(q)
        error = np.r_[target[:3,3]-pose[:3,3],
                      Rotation.from_matrix(target[:3,:3] @ pose[:3,:3].T).as_rotvec()]
        if np.linalg.norm(error[:3]) < 2e-7 and np.linalg.norm(error[3:]) < 2e-6:
            q = nearest_equivalent(q, np.asarray(seed), LOWER, UPPER)
            return q, True
        delta = np.linalg.solve(jac.T@jac + 1e-7*np.eye(6), jac.T@error)
        q += np.clip(delta, -.18, .18)
    return q, False


def tool_rotation(alpha, beta, gamma):
    return robot._rot_z(gamma) @ robot._rot_y(beta) @ robot._rot_x(alpha) @ robot._rot_x(np.pi)


class ActualWristCost:
    def __init__(self, context, weight):
        self.context = json.loads(Path(context).read_text()) if isinstance(context,(str,Path)) else dict(context)
        self.weight = float(weight)
        if not np.isfinite(self.weight) or self.weight < 0:
            raise ValueError('invalid wrist weight')
        expected = self.context.get('kinematics_sha256')
        actual = hashlib.sha256(Path(robot.__file__).read_bytes()).hexdigest()
        if expected and expected != actual:
            raise ValueError('factory kinematics source changed')
        if self.context['calibration_hash'] != robot.CALIBRATION_HASH:
            raise ValueError('factory calibration mismatch')
        self.reference = None
        self.calls = 0
        self.last_report = {}

    def targets(self, xy, posture, gamma, indices):
        xy, posture, gamma = np.asarray(xy,float), np.asarray(posture,float), np.asarray(gamma,float)
        ids = np.zeros(len(xy),int)
        for stroke, members in enumerate(indices): ids[members] = stroke
        absolute = local_to_absolute(gamma, xy * [1.,-1.], ids)
        c = self.context
        positions = np.empty((len(xy),3))
        # Match CanvasTransform and export_physical_trajectory exactly; this
        # is fixed once from original source, never refit to a candidate bbox.
        positions[:,:2] = (xy - c['canvas_center']) * [c['metres_per_pixel'], -c['metres_per_pixel']]
        positions[:,:2] += np.asarray(c['paper_offset_xy_m'])
        positions[:,2] = c['paper_z_m'] - posture[:,0]/1000.
        targets = np.repeat(np.eye(4)[None], len(xy), axis=0)
        for i,(a,b,g) in enumerate(zip(posture[:,1],posture[:,2],absolute)):
            r = tool_rotation(a,b,g)
            targets[i,:3,:3] = r
            targets[i,:3,3] = positions[i] - r[:,2]*c['brush_length_m']
        return targets

    def solve(self, xy, posture, gamma, indices):
        targets = self.targets(xy,posture,gamma,indices)
        qs, failures = [], []
        previous = np.asarray(self.context.get('initial_joints',INITIAL),float)
        for i,target in enumerate(targets):
            seed = self.reference[i] if self.reference is not None else previous
            q, ok = ik(target,seed)
            if not ok: failures.append(i)
            qs.append(q); previous=q
        qs = np.asarray(qs)
        try:
            qs = continuous_joint_path(qs, np.asarray(self.context.get('initial_joints',INITIAL)), LOWER,UPPER)
        except ValueError:
            failures.append(-1)
        if self.reference is None and not failures: self.reference=qs.copy()
        margins = np.minimum(qs-LOWER, UPPER-qs)
        singularity = min(abs(np.sin(q[2])),abs(np.sin(q[4]))) if len(qs)==1 else min(min(abs(np.sin(q[2])),abs(np.sin(q[4]))) for q in qs)
        valid = not failures and margins.min() >= 0 and singularity >= self.context.get('minimum_singularity_margin',.08)
        self.last_report = {'valid':bool(valid),'ik_failures':failures,
            'minimum_joint_limit_margin_rad':float(margins.min()), 'minimum_singularity_margin':float(singularity),
            'calls':self.calls,'joint_delta_convention':'actual unwrapped differences; NO modulo 2pi',
            'factory_calibration_hash':robot.CALIBRATION_HASH,'optimized_fields':['x','y','H','alpha','beta','gamma'],
            'pen_up_owner':'separate ROS wrist-aware transition planner'}
        return qs,valid

    def residual(self, xy, posture, gamma, indices):
        import torch
        self.calls += 1
        q,valid=self.solve(xy.detach().cpu().numpy(),posture.detach().cpu().numpy(),gamma.detach().cpu().numpy(),indices)
        delta=np.diff(q[:,3:6],axis=0)
        # Smooth L1 accumulated ACTUAL q4/q5/q6, including endpoint differences
        # across strokes. Lift-waypoint excursion is optimized by ROS separately.
        epsilon=1e-4
        result=np.sign(delta)*np.sqrt(2*self.weight*(np.sqrt(delta**2+epsilon**2)-epsilon))
        self.last_report.update(wrist_sum_rad=float(abs(delta).sum()),
            per_wrist_rad=abs(delta).sum(axis=0).tolist(), valid=bool(valid))
        return torch.as_tensor(result.reshape(-1),device=xy.device,dtype=xy.dtype)
