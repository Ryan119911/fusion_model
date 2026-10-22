"""Independent, unwrapped final-path checks using calibrated full-chain FK/J."""
from __future__ import annotations
import numpy as np
from .ur10_wrist_cost import fk_jacobian, LOWER, UPPER, INITIAL


def audit_path(targets, speed_limit=.6, singular_value_min=.01):
    qs=np.asarray([t.joints for t in targets],float)
    durations=np.asarray([t.duration_s for t in targets],float)
    if len(qs)<2 or not np.isfinite(qs).all() or not (durations>0).all():
        raise ValueError('invalid complete robot path')
    previous=np.vstack([INITIAL,qs[:-1]])
    delta=qs-previous # Deliberately not wrapped modulo 2 pi.
    speed=float(np.max(abs(delta)/durations[:,None]))
    singular=[]
    # Normalize translation by physical arm length; dimensionless full-chain
    # singular values include the shoulder as well as elbow/wrist singularities.
    scale=np.asarray([1/1.1838]*3+[1.]*3)[:,None]
    for left,right in zip(previous,qs):
        n=max(1,int(np.ceil(np.max(abs(right-left))/.04)))
        for q in np.linspace(left,right,n+1)[1:]:
            _,jac=fk_jacobian(q)
            singular.append(float(np.linalg.svd(jac*scale,compute_uv=False)[-1]))
    margin=float(np.minimum(qs-LOWER,UPPER-qs).min())
    result=dict(delta_convention='actual unwrapped q difference, no modulo',
        max_joint_speed_rad_s=speed,speed_limit_rad_s=float(speed_limit),
        minimum_joint_limit_margin_rad=margin,
        minimum_full_chain_normalized_singular_value=min(singular),
        minimum_allowed_full_chain_normalized_singular_value=float(singular_value_min),
        interpolation_sample_count=len(singular),
        acceleration_and_torque_validation='not performed; no hardware execution authorized')
    if margin < -1e-9 or speed>speed_limit+1e-8 or min(singular)<singular_value_min:
        raise ValueError(f'independent robot path gate failed: {result}')
    result['passed']=True
    return result
