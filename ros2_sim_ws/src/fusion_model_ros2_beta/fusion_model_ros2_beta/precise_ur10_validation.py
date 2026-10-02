"""Exact calibrated IK for the offline paired validator, not a ROS default."""
import numpy as np
from scipy.spatial.transform import Rotation
from .ur10_wrist_cost import fk_jacobian,ik


def solve_exact(target,seed,**kwargs):
    q,ok=ik(target,seed,max_iterations=kwargs.get('max_iterations',180))
    pose,_=fk_jacobian(q)
    position=float(np.linalg.norm(target[:3,3]-pose[:3,3]))
    angle=float(np.linalg.norm(Rotation.from_matrix(target[:3,:3]@pose[:3,:3].T).as_rotvec()))
    return q,position,angle,bool(ok)
