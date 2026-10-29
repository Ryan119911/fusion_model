#!/usr/bin/env python3
"""Default local preview; --check runs one read-only jx mesh-distance batch."""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ros2_sim_ws/src/fusion_model_ros2_beta'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--query',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--check',action='store_true')
    p.add_argument('--body-pairs',choices=('on','off'),default='on',
                   help='on adds the installed UR SRDF arm-body pairs; off reports attachment-only clearance')
    a=p.parse_args()
    request=json.loads(a.query.read_text())
    if a.check and a.output.exists():p.error('output already exists; choose a new evidence path')
    if not a.check:
        print(json.dumps({'motion_sent':False,'mode':'offline_preview','samples':len(request['audit']['samples'])}));return
    from fusion_model_ros2_beta.ur10_actual_kinematics import CAD_ATTACHMENT_ENVELOPES,CAD_ASSEMBLY_RPY,transform
    boxes={n:{'centre_m':list(c),'size_m':(2*np.array(h)).tolist(),'rotation':transform(rpy=CAD_ASSEMBLY_RPY)[:3,:3].tolist()} for n,c,h in CAD_ATTACHMENT_ENVELOPES}
    code=('REQUEST='+repr(request)+'\nBOXES='+repr(boxes)+'\nBODY_PAIRS=('+repr(a.body_pairs=='on')+')\n'
          +(ROOT/'tools/brush_current_mount_collision.py').read_text()+r'''
import sys,json,itertools,xml.etree.ElementTree as ET
from pathlib import Path
sys.path[:0]=['/home/jx/UR10/scripts','/opt/openrobots/lib/python3.10/site-packages']
import pinocchio as pin
srdf=Path('/opt/ros/humble/share/ur_moveit_config/srdf/ur_macro.srdf.xacro').read_text()
disabled={frozenset((e.attrib['link1'].replace('${prefix}',''),e.attrib['link2'].replace('${prefix}',''))) for e in ET.fromstring(srdf).iter('disable_collisions')}
result={'motion_sent':False,'models':{},'supplied_samples':REQUEST['audit']['samples'],'srdf_source':srdf,
 'scope':'all non-disabled robot body mesh pairs using installed UR SRDF, plus existing attachment-to-arm pairs with current CAD envelopes; discrete route and native-blend-neighborhood samples, not continuous swept volume or external obstacle verification'}
for name in ['robot_attachments.yaml','robot_attachments.preliminary-cad-20260921.yaml']:
 model=model_for_current_mount('/home/jx/UR10/config/'+name,BOXES)
 robot=[i for i,obj in enumerate(model.geometry.geometryObjects) if obj.parentFrame<len(model.model.frames)]
 added=[]
 for i,j in itertools.combinations(robot,2):
  if not BODY_PAIRS:break
  x,y=model.geometry.geometryObjects[i],model.geometry.geometryObjects[j]
  lx,ly=model.model.frames[x.parentFrame].name,model.model.frames[y.parentFrame].name
  if lx==ly or frozenset((lx,ly)) in disabled:continue
  model.geometry.addCollisionPair(pin.CollisionPair(i,j));added.append([lx,ly])
 model.geometry_data=model.geometry.createData()
 rows=[]
 for row in REQUEST['audit']['samples']:
  gap,pair=model.min_clearance(row['q_rad'])
  rows.append({'phase':row['phase'],'sample':row.get('sample',row.get('source_waypoint_index')),'minimum_clearance_mm':gap*1000,'pair':pair})
 minimum=min(x['minimum_clearance_mm'] for x in rows)
 result['models'][name]={'minimum_clearance_mm':minimum,'passed_20mm':minimum>=20,'robot_body_pairs':added,'samples':rows}
result['passed']=all(m['passed_20mm'] for m in result['models'].values())
print(json.dumps(result))
''')
    run=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','jx','/usr/bin/python3 -B -'],input=code,text=True,capture_output=True,timeout=240)
    if run.returncode:raise RuntimeError(run.stderr)
    result=json.loads(run.stdout)
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'passed':result['passed'],'motion_sent':False,'models':{n:{k:v for k,v in m.items() if k!='samples'} for n,m in result['models'].items()}}))
    if not result['passed']:raise SystemExit(1)

if __name__=='__main__':main()
