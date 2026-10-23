#!/usr/bin/env python3
"""Run existing remote geometry distance checks on supplied offline q only.

No state reader, camera service, robot socket or URScript is used. Existing
Pinocchio/hppfcl runtime is reused through SSH; nothing is installed/deployed.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

REMOTE = '''import sys,json
sys.path[:0]=['/home/jx/UR10/scripts','/opt/openrobots/lib/python3.10/site-packages']
from self_collision import SelfCollisionModel
from robot_attachments import load
request=json.load(sys.stdin)
result={'scope':'offline supplied q; existing robot collision meshes plus configured attachment envelopes','motion_sent':False,'models':{}}
for name in ['robot_attachments.yaml','robot_attachments.preliminary-cad-20260921.yaml']:
 geometry=load('/home/jx/UR10/config/'+name,calibration_path='/home/jx/UR10/config/handeye_eye_in_hand_20260921.yaml',root='/home/jx/UR10')
 model=SelfCollisionModel(geometry=geometry)
 rows=[]
 for item in request:
  gap,pair=model.min_clearance(item['q_rad'])
  rows.append({'phase':item['phase'],'sample':item['sample'],'clearance_mm':gap*1000,'pair':pair})
 result['models'][name]={'minimum_clearance_mm':min(r['clearance_mm'] for r in rows),'passed_20mm':all(r['clearance_mm']>=20 for r in rows),'samples':rows}
print(json.dumps(result,allow_nan=False))
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();d=json.loads(a.candidate.read_text())
    command='PYTHONNOUSERSITE=1 /usr/bin/python3 -B -c '+shlex.quote(REMOTE)
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','jx',command],
        input=json.dumps(d['audit']['samples']),text=True,capture_output=True,timeout=90)
    if r.returncode:raise RuntimeError(r.stderr[-2000:])
    result=json.loads(r.stdout);result['candidate_path']=str(a.candidate.resolve())
    result['supplied_samples']=d['audit']['samples']
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:{n:v for n,v in m.items() if n!='samples'} for k,m in result['models'].items()}))


if __name__=='__main__':main()
