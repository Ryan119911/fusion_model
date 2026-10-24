import json
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
from write_gong_1mm import build_points


class PressLimitTests(unittest.TestCase):
    def setUp(self):
        self.data={'center_zero_controller_pose_m_rad':[.54,-.19,.502,np.pi,0,0],
                   'paper_u_base':[0,1,0],'paper_normal_base':[0,0,1],
                   'canonical_controller_corners_m':[[.65,-.05,.502],[.44,-.05,.502],[.44,-.34,.502]],
                   'glyph_xy_m':[[-.03,.02],[-.02,.02],[-.01,.02],[0,.02],[.02,.02],[.03,.02],
                                 [0,.02],[.001,.01],[0,-.01],[0,-.02],
                                 [-.04,-.02],[-.03,-.02],[-.02,-.02],[0,-.02],[.03,-.02],[.04,-.02]],
                   'glyph_stroke_ids':[0]*6+[1]*4+[2]*6,'force_reference_baseline_n_nm':[0]*6}
        self.start=np.array([.54,-.18,.512,np.pi,0,0])

    def test_one_mm_exact_command_cap_and_three_strokes(self):
        points,plane=build_points(self.data,self.start,1.)
        contact=[p for p in points if p['contact']]
        self.assertEqual(len(contact),16)
        self.assertEqual({p['stroke_id'] for p in contact},{0,1,2})
        for p in contact:self.assertAlmostEqual(p['pose_m_rad'][2],.501)
        self.assertFalse(points[-1]['contact'])
        self.assertAlmostEqual(points[-1]['pose_m_rad'][2],.512)
        self.assertEqual(plane['commanded_press_limit_mm'],1.)

    def test_larger_or_invalid_values_rejected(self):
        for value in (1.001,2,0,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):build_points(self.data,self.start,value)

    def test_depth_change_does_not_rescale_xy_or_change_orientation(self):
        before,_=build_points(self.data,self.start,.3)
        after,_=build_points(self.data,self.start,1.)
        for a,b in zip(before,after):
            np.testing.assert_allclose(a['pose_m_rad'][:2],b['pose_m_rad'][:2])
            np.testing.assert_allclose(a['pose_m_rad'][3:],b['pose_m_rad'][3:])
            if a['contact']:self.assertAlmostEqual(a['pose_m_rad'][2]-b['pose_m_rad'][2],.0007)

    def test_tilt_change_refused(self):
        self.start[3]+=.01
        with self.assertRaises(ValueError):build_points(self.data,self.start,1.)


if __name__=='__main__':unittest.main()
