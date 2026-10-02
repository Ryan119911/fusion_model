"""Pen-up-only joint path optimization. Contact targets are immutable.

Optimizes actual q4/q5/q6 travel with constrained shortcutting and a lifted
continuous-IK alternative. It never returns each stroke to a home pose.
"""
from __future__ import annotations
from dataclasses import replace
import numpy as np
from scipy.spatial.transform import Rotation
from .ur10_wrist_cost import fk_jacobian, ik


def travel(path):
    return float(abs(np.diff(np.asarray(path),axis=0)[:,3:6]).sum())


def optimize_penup(node):
    targets=list(node._targets)
    audits=[]; changed=[]; i=0
    from .brush_trajectory_driver import JointTarget, _joint_singularity_margin, JOINT_LOWER_LIMITS, JOINT_UPPER_LIMITS
    brush=float(node._param('brush_length_m')); paper=float(node._param('paper_z_m'))
    lift=float(node._param('lift_height_m')); margin=float(node._param('minimum_singularity_margin'))
    def safe(left,right,above=None):
        if node._motion_obstacle_collision(left,right) is not None: return False
        count=max(2,int(np.ceil(np.max(abs(right-left))/.04))+1)
        for q in np.linspace(left,right,count):
            if np.any(q<JOINT_LOWER_LIMITS) or np.any(q>JOINT_UPPER_LIMITS) or _joint_singularity_margin(q)<margin: return False
            pose,_=fk_jacobian(q)
            if pose[2,2]>-np.cos(np.deg2rad(30.)): return False
            if above is not None:
                if (pose[:3,3]+pose[:3,2]*brush)[2] < above-1e-6: return False
        return True
    while i<len(targets):
        if targets[i].point.state != 3:
            changed.append(targets[i]); i+=1; continue
        start=i
        while i<len(targets) and targets[i].point.state==3: i+=1
        # Preserve initial approach, contact transitions within a stroke, and
        # any final lift. Only ordinary between-stroke intervals are optimized.
        if start==0 or i==len(targets) or targets[start-1].point.stroke_id==targets[i].point.stroke_id:
            changed.extend(targets[start:i]); continue
        left,right=targets[start-1],targets[i]
        original=np.asarray([t.joints for t in targets[start-1:i+1]])
        best=original; method='retained_safe_original'
        # Shortcut the joint path, but only while lifted. Contact-adjacent
        # lift/descent boundaries stay fixed; no accidental paper-level bridge.
        height=paper+max(.004,lift*.5)
        high=[]
        for k,q in enumerate(original):
            pose,_=fk_jacobian(q)
            if (pose[:3,3]+pose[:3,2]*brush)[2]>=height: high.append(k)
        if len(high)>1:
            a,b=high[0],high[-1]; kept=[a]; cursor=a
            while cursor<b:
                next_k=cursor+1
                for k in range(b,cursor,-1):
                    if safe(original[cursor],original[k],height): next_k=k; break
                kept.append(next_k); cursor=next_k
            shortened=np.r_[original[:a],original[kept],original[b+1:]]
            if travel(shortened) < travel(best)-1e-8: best=shortened; method='wrist_cost_constrained_shortcuts'
        # Alternative: lift at each endpoint with its CONTACT orientation,
        # then a minimum-travel joint bridge above the paper. Both contact q
        # endpoints are held EXACTLY, not re-solved into another posture.
        try:
            lifted=[]
            for endpoint in (left,right):
                base=np.asarray(endpoint.joints); pose,_=fk_jacobian(base)
                pose[2,3] += lift+float(endpoint.point.press_depth_mm)/1000.
                q,ok=ik(pose,base)
                if not ok: raise ValueError('endpoint lift IK failed')
                lifted.append(q)
            candidate=np.asarray([left.joints,*lifted,right.joints])
            if (safe(candidate[0],candidate[1]) and safe(candidate[1],candidate[2],height)
                    and safe(candidate[2],candidate[3]) and travel(candidate)<travel(best)-1e-8):
                best=candidate; method='lifted_continuous_ik_minimum_wrist_bridge'
        except ValueError:
            pass
        dense=[]
        for q0,q1 in zip(best,best[1:]):
            count=max(1,int(np.ceil(np.max(abs(q1-q0))/.12)))
            dense.extend(np.linspace(q0,q1,count+1)[1:])
        # Last target is the right CONTACT and is emitted by the next loop.
        for q in dense[:-1]:
            pose,_=fk_jacobian(q); tip=pose[:3,3]+pose[:3,2]*brush
            alpha,beta,gamma=Rotation.from_matrix(pose[:3,:3] @ np.diag([1.,-1.,-1.])).as_euler('xyz')
            airborne=replace(left.point,x=float(tip[0]),y=float(tip[1]),z=float(tip[2]),
                alpha=float(alpha),beta=float(beta),gamma=float(gamma),state=3,press_depth_mm=0.)
            changed.append(JointTarget(airborne,tuple(map(float,q)),float(node._param('air_interval_s'))))
        audits.append({'from_stroke':left.point.stroke_id,'to_stroke':right.point.stroke_id,
            'method':method,'before_wrist_rad':travel(original),'after_wrist_rad':travel(best),
            'contact_endpoints_changed':False,'returned_home':bool(any(
                np.max(abs(q-np.array([0.,-1.2,1.2,-1.5,-np.pi/2,0.])))<1e-6 for q in best[1:-1]))})
    node._targets=changed
    node._penup_rotation_audit={'method':'actual_wrist_cost_constrained_joint_path_optimization',
        'transitions':audits,'contact_targets_preserved':True,
        'per_stroke_home_reset':any(a['returned_home'] for a in audits)}
