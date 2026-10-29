import copy
import io
import json
import sys
import tempfile
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]/'tools'))
import brush_unloaded_calibration as air


def fixture():
    poses = [[.6, 0, -.434, 0, 0, 0], [.6001, 0, -.434, 0, 0, 0]]
    points = [{'phase':'unloaded_reorientation','pose_m_rad':p,'q_rad':[0]*6,
        'contact':False,'stroke_id':None,'speed_m_s':.001,'acceleration_m_s2':.002,
        'unloaded_wrench_n_nm':[0]*6} for p in poses]
    rows = [{'sample':i,'pose_m_rad':p,'cad_paper_clearance_mm':50,
             'shaft_capsule_clearance_mm':30} for i,p in enumerate(poses)]
    audit = {'passed_numeric':True,'samples':rows}
    collision = {'supplied_samples':rows,'models':{str(i):{'passed_20mm':True,
                 'minimum_clearance_mm':30} for i in range(2)}}
    return dict(points=points,audit=audit,collision=collision,origin=[0,0,0],normal=[0,0,1],
                tip=[0,0,.474],anchor=[.6,0,.04],baseline_rotation=[0,0,0])


class AirPolicyTests(unittest.TestCase):
    def check(self, d):
        air.validate_air_route(**d)

    def test_air_route_accepted_without_claiming_force_map(self):
        self.check(fixture())

    def test_timed_moves_preserve_watchdog_and_bind_actual_source(self):
        points=fixture()['points']
        source,original=air.compile_program(points,'192.168.1.10',50071)
        timed,manifest=air.compile_air_program(points)
        import re
        self.assertEqual(re.sub(r', t=[0-9.]+, r=',', r=',timed),source)
        self.assertEqual(manifest['source_compiler_manifest'],original)
        self.assertNotIn('program_sha256',manifest)
        self.assertTrue(manifest['watchdog_scaffold_unchanged'])

    def test_near_pure_rotation_has_bounded_explicit_duration(self):
        points=fixture()['points']
        points[1]['pose_m_rad']=[.600001,0,-.434,0,0,np.deg2rad(2)]
        points[1]['speed_m_s']=1e-7
        _,manifest=air.compile_air_program(points)
        self.assertGreater(manifest['move_duration_s'][1],10)
        self.assertLess(manifest['move_duration_s'][1],12)

    def test_contact_rejected(self):
        d=fixture();d['points'][1]['contact']=True
        with self.assertRaisesRegex(ValueError,'contact'):self.check(d)

    def test_low_clearance_rejected(self):
        d=fixture();d['origin']=[0,0,.01]
        with self.assertRaisesRegex(ValueError,'air cell'):self.check(d)

    def test_changed_route_and_missing_reference_rejected(self):
        d=fixture();d['points']=copy.deepcopy(d['points']);d['points'][1]['pose_m_rad'][0]+=.001
        with self.assertRaisesRegex(ValueError,'audit end'):self.check(d)
        d=fixture();d['points'][1]['unloaded_wrench_n_nm']=None
        with self.assertRaises(ValueError):self.check(d)

    def test_changed_collision_samples_rejected(self):
        d=fixture();d['collision']=copy.deepcopy(d['collision']);d['collision']['supplied_samples'][0]['sample']=99
        with self.assertRaisesRegex(ValueError,'collision report'):self.check(d)

    def test_live_height_and_orientation_envelope_rejected(self):
        reader=air.AirReader(None,[0,0,0],[0,0,1],[0,0,.474],[.6,0,.04],io.StringIO())
        with self.assertRaisesRegex(RuntimeError,'air cell'):
            reader.check((np.zeros(6),np.array([.6,0,-.444,0,0,0]),np.zeros(6)))
        from scipy.spatial.transform import Rotation
        rotation=Rotation.from_euler('z',11,degrees=True)
        pose=np.r_[np.array([.6,0,.04])-rotation.apply([0,0,.474]),rotation.as_rotvec()]
        with self.assertRaisesRegex(RuntimeError,'orientation'):
            reader.check((np.zeros(6),pose,np.zeros(6)))

    def test_force_fault_after_send_closes_callback_and_stops(self):
        import threading
        d=fixture();d.update(maximum_runtime_s=20,reference_id='test-air')
        sample=(np.deg2rad([0,-90,-90,-90,90,0]),np.array(d['points'][0]['pose_m_rad']),np.zeros(6))
        class Reader:
            def __init__(self,*a):pass
            def stationary(self):return sample
            def fresh(self,**kw):return sample
            def close(self):pass
        class Force:
            def __init__(self):self.rows=deque(maxlen=50000);self.lock=threading.Lock();self.calls=0
            def recent(self):
                self.calls+=1
                wrench=[101,0,0,0,0,0] if self.calls>=3 else [0]*6
                return [(0,0,0,wrench)]
            def close(self):pass
        class Channel:
            closed=False
            def setblocking(self,*a):pass
            def recv(self,*a):raise BlockingIOError
            def sendall(self,*a):pass
            def close(self):self.closed=True
        channel=Channel()
        class Server:
            def setsockopt(self,*a):pass
            def bind(self,*a):pass
            def listen(self,*a):pass
            def settimeout(self,*a):pass
            def accept(self):return channel,('192.168.1.3',1234)
            def close(self):pass
        commands=[]
        support=SimpleNamespace(controller_ready=lambda *a:None,model_gate=lambda *a:None,
            RawForce=Force,command=lambda host,script:commands.append(script),close_commands=lambda:None)
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);watch=folder/'watch.json';watch.write_text(json.dumps({'passed':True,'compiler_sha256':'original-compiler-fixture'}));d['watchdog_path']=str(watch)
            with patch.object(air,'ContinuousState',Reader),patch.object(air.socket,'socket',Server),\
                 patch('write_gong_1mm.sha',return_value='original-compiler-fixture'),\
                 patch('pathlib.Path.home',return_value=folder):
                (folder/'UR10/outputs/paper_writing').mkdir(parents=True)
                with self.assertRaisesRegex(RuntimeError,'force increment'):
                    air.execute_air_segment(d,folder/'run',support,lambda *a:{'active_tcp_offset_m_rad':[0]*6})
            report=json.loads((folder/'run/execution.json').read_text())
            self.assertEqual(report['status'],'stopped_on_error')
            self.assertTrue(report['motion_sent'])
            self.assertFalse(report['contact_force_map_verified'])
            self.assertTrue(channel.closed)
            self.assertEqual(len(commands),2)
            self.assertEqual(commands[-1],'stopl(0.5)\n')

    def test_original_joint_speed_stop_retained(self):
        import time
        sample=(np.deg2rad([0,-90,-90,-90,90,0]),np.array([.6,0,-.434,0,0,0]),np.zeros(6))
        reader=SimpleNamespace(fresh=lambda **kw:sample)
        force=SimpleNamespace(recent=lambda:[(0,0,0,[0]*6)])
        with self.assertRaisesRegex(RuntimeError,'joint speed'):
            air.health_tick(reader,force,fixture()['points'],(time.monotonic()-.01,sample[0]-.02))


if __name__=='__main__':unittest.main()
