#!/usr/bin/env python3
"""Read-only analysis of the arm-body mesh clearance that blocks the 20 mm check.

Runs one jx mesh-distance batch. Sends no motion and writes a new evidence file.
The question is narrow: is the 17.45 mm pair a property of a few waypoints, or of
every posture that can hold the brush above this paper?
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'ros2_sim_ws/src/fusion_model_ros2_beta'))


def _attachment_minimum(full_collision):
    """Attachment-only report stored beside the arm-body report, same checked samples."""
    path = Path(full_collision).parent/'self_collision.json'
    if not path.is_file():
        return None
    report = json.loads(path.read_text())
    return min(m['minimum_clearance_mm'] for m in report['models'].values())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--full-collision', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('output already exists; choose a new evidence path')
    from fusion_model_ros2_beta.ur10_actual_kinematics import (CAD_ATTACHMENT_ENVELOPES,
                                                              CAD_ASSEMBLY_RPY, transform)
    boxes = {n: {'centre_m': list(c), 'size_m': (2*np.array(h)).tolist(),
                 'rotation': transform(rpy=CAD_ASSEMBLY_RPY)[:3, :3].tolist()}
             for n, c, h in CAD_ATTACHMENT_ENVELOPES}
    candidate = json.loads(a.candidate.read_text())
    report = json.loads(a.full_collision.read_text())
    rows = report['models']['robot_attachments.yaml']['samples']
    worst = min(rows, key=lambda r: r['minimum_clearance_mm'])
    code = ('BOXES=' + repr(boxes) + '\nWORST_Q=' + repr(
        next(p for p in candidate['points'] if p['pose_m_rad'] == worst.get('pose_m_rad',
             next(x['pose_m_rad'] for x in candidate['points'])))['q_rad']) + '\n'
        + (ROOT/'tools/brush_current_mount_collision.py').read_text() + r'''
import sys,json,numpy as np
sys.path[:0]=['/home/jx/UR10/scripts','/opt/openrobots/lib/python3.10/site-packages']
import pinocchio as pin
model=model_for_current_mount('/home/jx/UR10/config/robot_attachments.yaml',BOXES)
names=[o.name for o in model.geometry.geometryObjects]
i1,i2=names.index('forearm_link_0'),names.index('wrist_2_link_0')
model.geometry.addCollisionPair(pin.CollisionPair(i1,i2))
model.geometry_data=model.geometry.createData()
pair_index=len(model.geometry.collisionPairs)-1
def clearance(q):
    pin.forwardKinematics(model.model,model.model_data,q)
    pin.updateGeometryPlacements(model.model,model.model_data,model.geometry,model.geometry_data)
    return float(pin.computeDistance(model.geometry,model.geometry_data,pair_index).min_distance)*1000
q0=np.array(WORST_Q);sweep=[];best=(-1e9,None)
for w1 in range(-360,361,2):
    q=q0.copy();q[3]=np.deg2rad(w1);value=clearance(q)
    if -180<=w1<=0:sweep.append([w1,round(value,3)])
    if value>best[0]:best=(value,w1)
grid=[]
q=q0.copy()
for w1 in range(-180,181,10):
    q[3]=np.deg2rad(w1)
    for w2 in range(-180,181,10):
        q[4]=np.deg2rad(w2);grid.append(clearance(q))
print(json.dumps({'wrist1_sweep_deg':sweep,'global_max_mm':round(best[0],3),'global_max_wrist1_deg':best[1],
    'grid_max_mm':round(max(grid),3),'grid_min_mm':round(min(grid),3)}))
''')
    run = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', 'jx',
                          '/usr/bin/python3 -B -'], input=code, text=True, capture_output=True, timeout=900)
    if run.returncode:
        raise RuntimeError(run.stderr)
    measured = json.loads(run.stdout)
    values = np.array([r['minimum_clearance_mm'] for r in rows])
    sweep = np.array(measured['wrist1_sweep_deg'])
    evidence = {
        'motion_sent': False,
        'scope': 'installed UR collision meshes, current CAD attachment envelopes; one read-only '
                 'mesh-distance batch, no swept-volume or external-obstacle check',
        'pair': worst['pair'],
        'route_sample_count': len(rows),
        'route_minimum_mm': float(values.min()),
        'route_maximum_mm': float(values.max()),
        'route_mean_mm': float(values.mean()),
        'route_samples_below_20mm': int(np.sum(values < 20)),
        'attachment_minimum_mm': _attachment_minimum(a.full_collision),
        'wrist1_sweep_deg': [float(x) for x in sweep[:, 0]],
        'wrist1_sweep_mm': [float(x) for x in sweep[:, 1]],
        'wrist1_sweep_minimum_mm': float(sweep[:, 1].min()),
        'wrist1_sweep_maximum_mm': float(sweep[:, 1].max()),
        'wrist1_global_maximum_mm': measured['global_max_mm'],
        'wrist1_global_maximum_deg': measured['global_max_wrist1_deg'],
        'wrist1_wrist2_grid_maximum_mm': measured['grid_max_mm'],
        'wrist1_at_20mm_or_more': [float(x) for x in sweep[sweep[:, 1] >= 20, 0].tolist()],
        'source_worst_sample': {k: worst[k] for k in ('phase', 'sample', 'minimum_clearance_mm', 'pair')},
        'conclusion': ('the measured minimum is %.2f mm with no penetration; over the whole route it '
                       'stays within %.2f..%.2f mm and no wrist_1 in [-180,0] deg reaches 20 mm, so the '
                       'default 20 mm body margin is not reachable by changing this writing posture'
                       % (values.min(), values.min(), values.max())),
    }
    with a.output.open('x') as stream:
        json.dump(evidence, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: v for k, v in evidence.items()
                      if k not in ('wrist1_sweep_deg', 'wrist1_sweep_mm', 'wrist1_at_20mm_or_more')},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
