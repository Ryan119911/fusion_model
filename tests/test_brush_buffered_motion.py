import copy
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
from brush_buffered_motion import blend_radii,compile_program,execution_blockers
from brush_buffered_guard import health_tick,reference_at_pose


def point(x,y,z,contact,stroke=0):
    return {'pose_m_rad':[x,y,z,np.pi,0,0],'contact':contact,'stroke_id':stroke,
            'speed_m_s':.03,'acceleration_m_s2':.1,'unloaded_wrench_n_nm':[0]*6}


class BufferedTests(unittest.TestCase):
    def setUp(self):
        self.points=[point(.5,0,.52,False),point(.5,0,.5,True),
                     point(.51,0,.49,True),point(.52,.002,.48,True),
                     point(.52,.002,.52,False)]

    def test_one_program_without_host_waits(self):
        script,report=compile_program(self.points,'192.0.2.10',50071)
        self.assertEqual(script.count('def brush_buffered()'),1)
        self.assertEqual(script.count('movel('),5)
        self.assertNotIn('sleep(',script)
        self.assertNotIn('stationary',script)
        self.assertEqual(report['host_stop_waits_between_waypoints'],0)
        self.assertGreater(report['interior_blended_count'],0)

    def test_full_physical_z_and_orientation_not_clamped(self):
        self.points[2]['pose_m_rad'][3:]=[3.0,.1,.2]
        source=copy.deepcopy(self.points)
        script,_=compile_program(self.points,'192.0.2.10',50071)
        self.assertIn('0.490000000,3.000000000,0.100000000,0.200000000',script)
        self.assertEqual(self.points,source)

    def test_no_contact_or_stroke_boundary_blends(self):
        r=blend_radii(self.points)
        self.assertEqual([r[0],r[1],r[-2],r[-1]],[0]*4)
        self.points[2]['stroke_id']=1
        self.assertEqual(blend_radii(self.points),[0]*5)

    def test_tiny_segments_do_not_overlap(self):
        self.points[2]['pose_m_rad'][:3]=[.5001,0,.4999]
        r=blend_radii(self.points)
        xyz=np.array([p['pose_m_rad'][:3] for p in self.points])
        self.assertTrue(all(r[i]+r[i+1]<np.linalg.norm(xyz[i+1]-xyz[i]) for i in range(4)))

    def test_watchdog_precedes_motion_and_kill_precedes_stop(self):
        script,_=compile_program(self.points,'192.0.2.10',50071)
        self.assertLess(script.index('hello[1] != 1'),script.index('run brush_motion()'))
        self.assertLess(script.index('kill motion_handle'),script.index('stopl(0.5)'))
        self.assertIn('heartbeat[1] == sequence + 1',script)
        self.assertIn('"brush_guard", 0.15',script)

    def test_unverified_complete_model_cannot_run(self):
        _,manifest=compile_program(self.points,'192.0.2.10',50071)
        with tempfile.TemporaryDirectory() as folder:
            errors=execution_blockers({},manifest,Path(folder))
        self.assertTrue(any('physical' in e for e in errors))
        self.assertTrue(any('heartbeat' in e for e in errors))

    def test_force_reference_interpolates(self):
        points=[point(.5,0,.5,False),point(.6,0,.5,False)]
        points[1]['unloaded_wrench_n_nm']=[2,4,6,0,0,0]
        actual=np.array([.55,0,.5,np.pi,0,0])
        np.testing.assert_allclose(reference_at_pose(points,actual),[1,2,3,0,0,0])

    def test_stale_state_or_force_prevents_heartbeat_permission(self):
        class StaleReader:
            def fresh(self,**kwargs):raise TimeoutError('stale state')
        with self.assertRaises(TimeoutError):health_tick(StaleReader(),None,self.points)

    def test_route_escape_prevents_heartbeat_permission(self):
        class Escaped:
            def fresh(self,**kwargs):return np.zeros(6),np.array([.7,0,.5,np.pi,0,0]),np.zeros(6)
        with self.assertRaisesRegex(RuntimeError,'escaped'):health_tick(Escaped(),None,self.points)


if __name__=='__main__':unittest.main()
