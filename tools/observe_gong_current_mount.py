#!/usr/bin/env python3
"""MU entry for current-mount unloaded observation. Default preview; no motion mode."""
import argparse
import ast
import datetime as dt
import json
from pathlib import Path
import shlex
import subprocess

ROOT=Path(__file__).resolve().parents[1]
REMOTE=r'''
import sys,json,time,types,struct,traceback
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.dont_write_bytecode=True
sys.path[:0]=['/home/jx/UR10/scripts','/home/jx/.dsh-analysis','/opt/openrobots/lib/python3.10/site-packages']
import probe_paper_height as support
from calibrate_pen_tcp import capture
from brush_live_state import ContinuousState
out=Path(OUT);out.mkdir(exist_ok=False)
result={'status':'collecting','motion_sent':False,'force_tare_performed':False,'camera_triggered':False,
        'operator_unloaded_confirmed':True,'sample_id':SAMPLE,'mount_model':MOUNT,
        'force_data_max_age_s':1.5,'scope':'one static unloaded observation, not a varying-posture force map'}
reader=force=None
try:
    result['controller']=support.controller_ready('192.168.1.3')
    result['tcp_capture']=capture('192.168.1.3',30011,1.)
    with (out/'controller_feedback.jsonl').open('x') as feedback,(out/'raw_force.jsonl').open('x') as raw:
        original_fresh=support.fresh
        def logged_fresh(r,*args,**kwargs):
            v=original_fresh(r,*args,**kwargs)
            feedback.write(json.dumps({'sample_id':SAMPLE,'host_unix_ns':time.time_ns(),
                'controller_time_s':struct.unpack_from('>d',r.last_frame,4)[0],
                'actual_q_rad':v[0].tolist(),'actual_tool0_pose_m_rad':v[1].tolist(),'tcp_velocity':v[2].tolist()})+'\n')
            return v
        support.fresh=logged_fresh
        reader=ContinuousState(support,'192.168.1.3')
        q,start,velocity=reader.stationary()
        result.update(initial_q_rad=q.tolist(),initial_pose_m_rad=start.tolist(),initial_velocity=velocity.tolist())
        ns={};exec(FORCE_SOURCE,ns)
        logs={};exec(RECORDING_SOURCE,logs)
        force=support.RawForce();force.recent=types.MethodType(ns['recent_force'],force)
        with force.lock:force.rows=logs['RecordingRows'](force.rows,raw,SAMPLE)
        try:
            baseline,statistics=force.baseline()
            _,end,_=reader.stationary()
            pe=float(np.linalg.norm(end[:3]-start[:3]));ae=float((Rotation.from_rotvec(start[3:]).inv()*Rotation.from_rotvec(end[3:])).magnitude())
            if pe>.0001 or ae>np.deg2rad(.05):raise RuntimeError('pose changed during unloaded observation')
            active=np.asarray(result['tcp_capture']['active_tcp_offset_m_rad'])
            if not np.allclose(active,0,atol=1e-8,rtol=0):raise RuntimeError('active TCP is not the recorded zero tool0 convention')
            tip=start[:3]+Rotation.from_rotvec(start[3:]).apply(CANDIDATE['tcp']['translation_m'])
            height=float((tip-np.array(CANDIDATE['paper']['reconstructed_center_tip_base_m']))@np.array(CANDIDATE['paper']['paper_normal_base']))
            result.update(status='static_unloaded_observation_collected',median_wrench_n_nm=baseline.tolist(),statistics=statistics,
                final_pose_m_rad=end.tolist(),pose_drift_m=pe,orientation_drift_rad=ae,
                nominal_tip_base_m=tip.tolist(),nominal_tip_paper_height_m=height,
                varying_posture_reference_ready=False,robot_execution_ready=False)
        finally:
            force.close();force=None;reader.close();reader=None
    result['source_records']={}
    for name in ['config/robot_attachments.yaml','config/handeye_eye_in_hand_20261027.yaml','scripts/robot_attachments.py','scripts/self_collision.py']:
        p=Path('/home/jx/UR10')/name
        if p.is_file():result['source_records'][name]=p.read_text()
except BaseException as exc:
    result.update(status='stopped_on_error',error=type(exc).__name__+': '+str(exc),error_traceback=traceback.format_exc())
finally:
    if force is not None:force.close()
    if reader is not None:reader.close()
    result['finished_at']=time.time()
    (out/'RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_records','error_traceback','tcp_capture')},ensure_ascii=False),flush=True)
if result['status']=='stopped_on_error':raise SystemExit(1)
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collect',action='store_true')
    p.add_argument('--unloaded-confirmed',action='store_true')
    p.add_argument('--candidate',type=Path,required=True)
    a=p.parse_args()
    if a.collect and not a.unloaded_confirmed:p.error('--collect requires --unloaded-confirmed')
    c=json.loads(a.candidate.read_text())
    if not a.collect:
        print(json.dumps({'status':'observation_preview_no_connection','motion_sent':False,'candidate':str(a.candidate),'requires_unloaded_confirmation':True}));return
    sid='gong-current-mount-observe-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=ROOT/'outputs/brush_mapping_20261027'/sid;out.mkdir(parents=True,exist_ok=False)
    remote='/home/jx/.dsh-analysis/'+sid
    source=(ROOT/'tools/brush_write_capture.py').read_text()
    cls=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='RecordingRows')
    code='OUT='+repr(remote)+'\nSAMPLE='+repr(sid)+'\nMOUNT='+repr(c['mount_model'])+'\nCANDIDATE='+repr(c)+'\n'
    code+='FORCE_SOURCE='+repr((ROOT/'tools/brush_force_snapshot.py').read_text())+'\n'
    code+='RECORDING_SOURCE='+repr('import json\nfrom collections import deque\n'+ast.unparse(cls))+'\n'+REMOTE
    (out/'request.py').write_text(code)
    command='source /opt/ros/humble/setup.bash; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1; exec /usr/bin/python3 -B -'
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','jx','bash -c '+shlex.quote(command)],input=code,text=True,capture_output=True,timeout=60)
    (out/'stdout.txt').write_text(r.stdout);(out/'stderr.txt').write_text(r.stderr)
    cp=subprocess.run(['scp','-q','-r','jx:'+remote,str(out/'remote_evidence')],capture_output=True,text=True,timeout=30)
    (out/'download_stderr.txt').write_text(cp.stderr)
    print(json.dumps({'local_evidence':str(out),'ssh_exit_code':r.returncode,'raw_evidence_downloaded':cp.returncode==0,'remote_output':r.stdout.strip()},ensure_ascii=False))
    if r.returncode or cp.returncode:raise SystemExit(1)


if __name__=='__main__':main()
