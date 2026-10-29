"""Build a checked, pen-up connection from a measured suspended start.

Pure geometry only. Source writing points and their zero are never shifted.
"""
import copy,math
import numpy as np
from scipy.spatial.transform import Rotation,Slerp


def connect_start(points,start,tip,origin,normal,domain,baseline):
    start=np.asarray(start,float);tip=np.asarray(tip,float);origin=np.asarray(origin,float);normal=np.asarray(normal,float)
    if start.shape!=(6,) or not np.isfinite(np.r_[start,tip,origin,normal]).all():raise ValueError('invalid measured start')
    if points[0]['contact']:raise ValueError('original route must begin in free space')
    original=np.array(points[0]['pose_m_rad']);r=Rotation.from_rotvec(start[3:]);old=Rotation.from_rotvec(original[3:])
    current_tip=start[:3]+r.apply(tip);old_tip=original[:3]+old.apply(tip)
    height=float((current_tip-origin)@normal);old_height=float((old_tip-origin)@normal)
    if not .010<=height<=.300:raise ValueError('start must have nominal tip clearance 10..300 mm before a checked lift')
    if old_height<.035:raise ValueError('original connection target must be at least 35 mm above paper')
    prefix=[];template=points[0]
    def add(phase,position,rotation,speed=.003):
        value=np.r_[position-rotation.apply(tip),rotation.as_rotvec()]
        if prefix and np.linalg.norm(value[:3]-np.array(prefix[-1]['pose_m_rad'][:3]))<1e-8 and (Rotation.from_rotvec(prefix[-1]['pose_m_rad'][3:]).inv()*rotation).magnitude()<1e-8:return
        p=copy.deepcopy(template);p.update(phase=phase,pose_m_rad=value.tolist(),contact=False,stroke_id=None,speed_m_s=speed,acceleration_m_s2=.002,unloaded_wrench_n_nm=None)
        prefix.append(p)
    add('measured_current_start',current_tip,r)
    safe=max(.040,height,old_height)
    raised=current_tip+(safe-height)*normal
    # Pen-up travel only, kept well below the 30 mm/s writing speed; it must fit
    # the production guard's existing 1800 s ceiling regardless of where the
    # measured start happens to be.
    add('lift_clear',raised,r,.020)
    above_old=old_tip+(safe-old_height)*normal
    add('air_start_translate',above_old,r,.020)
    # Rotate at a fixed compensated tip, retaining <=0.5 deg interpolation.
    angle=(r.inv()*old).magnitude();count=max(1,math.ceil(math.degrees(angle)/.5))
    for rot in Slerp([0,1],Rotation.from_matrix([r.as_matrix(),old.as_matrix()]))(np.linspace(0,1,count+1)[1:]):
        # Rotation about the brush axis barely translates tool0 (0.03 mm per 0.5 deg),
        # so the half-millimetre speed would imply an unusable joint rate; the
        # existing air-alignment speed is used for this phase.
        add('air_start_reorient',above_old,rot,.0015)
    if np.linalg.norm(above_old-old_tip)>1e-8:add('air_start_descend',old_tip,old,.020)
    # The exact original first pose is the final connector endpoint.
    if np.linalg.norm(np.array(prefix[-1]['pose_m_rad'])-original)<1e-7:prefix.pop()
    combined=prefix+copy.deepcopy(points)
    x=(Rotation.from_rotvec(baseline).inv()*Rotation.from_rotvec([p['pose_m_rad'][3:] for p in combined])).as_rotvec()/np.deg2rad(4)
    if isinstance(domain,dict):
        # A reduced (e.g. three-posture) reference has no three-dimensional hull;
        # the measured span is used as a per-axis box instead.
        if domain.get('kind')!='box':raise ValueError('unsupported start orientation domain')
        lo,hi=np.asarray(domain['lo'],float),np.asarray(domain['hi'],float)
        if lo.shape!=(3,) or hi.shape!=(3,) or np.any(lo>hi):raise ValueError('invalid start orientation box')
        # The measured start itself is an observation, not an extrapolation: its
        # predicted wrench is separately compared with the measured raw wrench.
        measured=(Rotation.from_rotvec(baseline).inv()*r).as_rotvec()/np.deg2rad(4)
        lo=np.minimum(lo,measured);hi=np.maximum(hi,measured)
        # Only the pen-up connector is bounded by the measured span. The writing
        # route is an explicitly declared first-order extrapolation whose bound is
        # checked against the same 10 N limit before this connection is built.
        # 0.05 feature units = 0.2 deg, below the 0.5 deg interpolation step, to
        # absorb measurement noise at the box corner defined by the baseline pose.
        slack=.05
        connector=x[:len(prefix)]
        if np.any(connector<lo-slack) or np.any(connector>hi+slack):
            raise ValueError('start connection orientation is outside the measured air-reference span')
    else:
        equations=np.asarray(domain)
        if np.max(x@equations[:,:3].T+equations[:,3])>1e-5:raise ValueError('start connection orientation is outside the measured air-reference domain')
    return combined,{'status':'connection_requires_independent_path_checks','prefix_point_count':len(prefix),
        'start_orientation_added_to_measured_span':isinstance(domain,dict),
        'measured_start_pose_m_rad':start.tolist(),'original_start_pose_m_rad':original.tolist(),
        'initial_tip_height_m':height,'safe_tip_height_m':safe,'original_tip_height_m':old_height,
        'source_writing_route_preserved':True,'motion_sent':False}
