import copy,sys,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import brush_unloaded_calibration as air

class SessionReuseTests(unittest.TestCase):
    def test_reuse_is_bound_to_full_audit_and_exact_poses_and_joints(self):
        points=[{'pose_m_rad':[0]*6,'q_rad':[0]*6},{'pose_m_rad':[.01,0,0,0,0,0],'q_rad':[.1]*6}]
        rows=[{'sample':0}];session={'points':copy.deepcopy(points),'audit':{'passed_numeric':True,'samples':rows},
          'collision':{'supplied_samples':rows,'models':{str(i):{'passed_20mm':True,'minimum_clearance_mm':30} for i in range(2)}}}
        air.validate_prechecked_session(points,session)
        bad=copy.deepcopy(points);bad[1]['pose_m_rad'][0]+=.001
        with self.assertRaises(ValueError):air.validate_prechecked_session(bad,session)
        session['collision']['supplied_samples']=[]
        with self.assertRaises(ValueError):air.validate_prechecked_session(points,session)
    def test_final_settle_keeps_original_point_one_mm_limit(self):
        target=np.zeros(6);poses=[np.array([.00011,0,0,0,0,0]),np.array([.00005,0,0,0,0,0])]
        class Reader:
            def stationary(self):return None,poses.pop(0),None
        with patch.object(air,'health_tick') as tick:
            end=air.wait_final_air_pose(Reader(),object(),[{'pose_m_rad':target}],timeout=1.)
            self.assertAlmostEqual(end[0],.00005);self.assertEqual(tick.call_count,2)
        poses.append(np.array([.00011,0,0,0,0,0]))
        with patch.object(air,'health_tick'):
            with self.assertRaises(RuntimeError):air.wait_final_air_pose(Reader(),object(),[{'pose_m_rad':target}],timeout=0.)

    def test_static_window_restarts_and_rejects_persistent_movement(self):
        class Force:
            def recent(self):return [(0,1,2,[0]*6)]
        clock=[0.]
        def now():clock[0]+=.1;return clock[0]
        def tick(*args):
            pose=np.array([.00008 if clock[0]<1. else -.00008,0,0,0,0,0])
            return None,pose,None
        with patch.object(air.time,'monotonic',side_effect=now),patch.object(air,'health_tick',side_effect=tick):
            end,rows=air.collect_static_air_window(None,Force(),[{'pose_m_rad':[0]*6}],np.array([.00008,0,0,0,0,0]))
            self.assertGreater(clock[0],4.);self.assertTrue(rows);self.assertAlmostEqual(end[0],-.00008)
        clock[0]=0.
        with patch.object(air.time,'monotonic',side_effect=now),patch.object(air,'health_tick',side_effect=lambda *args:(None,np.array([clock[0]*.002,0,0,0,0,0]),None)):
            with self.assertRaisesRegex(RuntimeError,'bounded settling'):
                air.collect_static_air_window(None,Force(),[{'pose_m_rad':[0]*6}],np.zeros(6))

if __name__=='__main__':unittest.main()
