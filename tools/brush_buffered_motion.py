#!/usr/bin/env python3
"""Compile one finite, heartbeat-guarded URScript; no per-waypoint PC stops.

The input is a physical controller-base six-dimensional route, not neural H.
No depth is clipped or silently substituted here. CLI only exports a program;
the caller must independently validate the mapped/blended physical path.
"""
import argparse
import hashlib
import ipaddress
import json
import math
from pathlib import Path

import numpy as np


def route_identity(points):
    return hashlib.sha256(json.dumps(points,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def validate_points(points):
    if not isinstance(points,list) or not 2<=len(points)<=10000:
        raise ValueError('finite route of 2..10000 physical waypoints required')
    for point in points:
        pose=np.asarray(point['pose_m_rad'],float)
        if pose.shape!=(6,) or not np.isfinite(pose).all():
            raise ValueError('finite six-dimensional physical pose required')
        if type(point['contact']) is not bool:
            raise ValueError('contact must be boolean')
        if point['contact'] and type(point.get('stroke_id')) is not int:
            raise ValueError('contact stroke_id required')
        for name,upper in [('speed_m_s',.030),('acceleration_m_s2',.100)]:
            value=point[name]
            if type(value) not in (int,float) or not math.isfinite(value) or not 0<value<=upper:
                raise ValueError('invalid physical motion limit: '+name)
    if points[0]['contact'] or points[-1]['contact']:
        raise ValueError('complete route must begin/end out of contact')


def blend_radii(points,maximum=.0005):
    """Bound rounding to <=0.5 mm, only at interior same-stroke waypoints.

    Source poses remain verbatim. Never blend across contact/lift transitions.
    Adjacent balls are disjoint; near reversals must be reviewed explicitly.
    """
    validate_points(points)
    if not math.isfinite(maximum) or not 0<=maximum<=.0005:
        raise ValueError('blend cap must be within 0..0.5 mm')
    radius=[0.]*len(points)
    xyz=np.array([p['pose_m_rad'][:3] for p in points],float)
    for i in range(1,len(points)-1):
        left,current,right=points[i-1:i+2]
        if not all(p['contact'] for p in (left,current,right)):
            continue
        if len({p['stroke_id'] for p in (left,current,right)})!=1:
            continue
        incoming=xyz[i]-xyz[i-1];outgoing=xyz[i+1]-xyz[i]
        a,b=float(np.linalg.norm(incoming)),float(np.linalg.norm(outgoing))
        if min(a,b)<1e-6:
            raise ValueError('duplicate/near duplicate contact points; review instead of skipping')
        if incoming@outgoing/(a*b)<-.95:
            raise ValueError('near reversal cannot be blended without explicit path review')
        radius[i]=min(maximum,.15*a,.15*b)
    for i in range(len(points)-1):
        if radius[i]+radius[i+1]>=np.linalg.norm(xyz[i+1]-xyz[i]) and radius[i]+radius[i+1]>0:
            raise ValueError('overlapping blend regions')
    return radius


def compile_program(points,guard_host,guard_port,maximum_blend=.0005):
    ip=ipaddress.ip_address(guard_host)
    if ip.version!=4 or ip.is_unspecified or ip.is_loopback or ip.is_multicast:
        raise ValueError('explicit robot-reachable IPv4 guard host required')
    if type(guard_port) is not int or not 1024<=guard_port<=65535:
        raise ValueError('invalid guard port')
    radii=blend_radii(points,maximum_blend)
    # The main thread owns guard reads. It kills the motion thread before
    # stopl, matching UR's own external-control thread cancellation pattern.
    lines=['def brush_buffered():',
           '  if not socket_open("%s", %d, "brush_guard"):'%(str(ip),guard_port),
           '    halt','  end',
           '  hello = socket_read_binary_integer(1, "brush_guard", 0.15)',
           '  if hello[0] != 1:','    socket_close("brush_guard")','    halt','  end',
           '  if hello[1] != 1:','    socket_close("brush_guard")','    halt','  end',
           '  global brush_done = False','  thread brush_motion():']
    for i,(point,radius) in enumerate(zip(points,radii)):
        pose=','.join('%.9f'%float(x) for x in point['pose_m_rad'])
        lines.append('    movel(p[%s], a=%.6f, v=%.6f, r=%.9f)'%
                     (pose,point['acceleration_m_s2'],point['speed_m_s'],radius))
    lines+=['    global brush_done = True','  end',
            '  motion_handle = run brush_motion()',
            '  sequence = 1','  while not brush_done:',
            '    heartbeat = socket_read_binary_integer(1, "brush_guard", 0.15)',
            '    valid = False','    if heartbeat[0] == 1:',
            '      if heartbeat[1] == sequence + 1:','        valid = True','      end','    end',
            '    if not valid:','      kill motion_handle','      stopl(0.5)',
            '      socket_close("brush_guard")','      halt','    end',
            '    sequence = heartbeat[1]','  end','  join motion_handle',
            '  socket_close("brush_guard")','end','']
    script='\n'.join(lines)
    manifest={'route_sha256':route_identity(points),'program_sha256':hashlib.sha256(script.encode()).hexdigest(),
              'waypoint_count':len(points),'native_movel_count':len(points),
              'interior_blended_count':sum(r>0 for r in radii),'blend_radii_m':radii,
              'maximum_blend_m':maximum_blend,'host_stop_waits_between_waypoints':0,
              'guard_timeout_s':.15,'guard_host':str(ip),'guard_port':guard_port,
              'motion_thread_killed_before_protective_stop':True,
              'fixed_depth_clamp_present':False,'source_six_dimensional_poses_retained':True,
              'motion_sent':False,'hardware_validated':False}
    return script,manifest


def execution_blockers(bundle,manifest,folder):
    """Export is allowed without certificates; real execution is not."""
    errors=[]
    if bundle.get('physical_mapping_verified') is not True:
        errors.append('model H/alpha/beta/gamma to physical brush pose mapping not verified')
    if bundle.get('force_reference_verified') is not True:
        errors.append('force reference over the complete posture path not verified')
    for name in ('physical_mapping','blended_path_audit','force_reference'):
        item=bundle.get('evidence',{}).get(name,{})
        path=folder/item.get('path','__missing__')
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=item.get('sha256'):
            errors.append('missing/changed evidence: '+name)
            continue
        report=json.loads(path.read_text())
        if report.get('passed') is not True or report.get('route_sha256')!=manifest['route_sha256']:
            errors.append('evidence not bound to exact complete route: '+name)
        if name=='blended_path_audit' and report.get('program_sha256')!=manifest['program_sha256']:
            errors.append('blended path audit does not bind this program')
    if bundle.get('controller_watchdog_hardware_validated') is not True:
        errors.append('controller heartbeat timeout/stop not yet hardware validated')
    return errors


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--guard-host',required=True)
    p.add_argument('--guard-port',type=int,default=50071)
    p.add_argument('--output-prefix',type=Path,required=True)
    args=p.parse_args();bundle=json.loads(args.bundle.read_text())
    script,manifest=compile_program(bundle['points'],args.guard_host,args.guard_port)
    manifest['execution_blockers']=execution_blockers(bundle,manifest,args.bundle.resolve().parent)
    manifest['scope']=bundle.get('scope','unspecified')
    args.output_prefix.parent.mkdir(parents=True,exist_ok=True)
    with Path(str(args.output_prefix)+'.script.preview').open('x') as f:f.write(script)
    with Path(str(args.output_prefix)+'.json').open('x') as f:json.dump(manifest,f,indent=2)
    print(json.dumps(manifest))


if __name__=='__main__':main()
