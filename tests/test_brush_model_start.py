import copy,sys,unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from brush_model_start import connect_start


class StartConnectionTests(unittest.TestCase):
    def setUp(self):
        self.tip=np.array([0.,0.,.474]);self.origin=np.zeros(3);self.normal=np.array([0.,0.,1.])
        self.domain=np.array([[1,0,0,-1],[-1,0,0,-1],[0,1,0,-1],[0,-1,0,-1],[0,0,1,-1],[0,0,-1,-1]])
        self.points=[{'pose_m_rad':[.2,0,.040-.474,0,0,0],'contact':False,'stroke_id':None,'phase':'old_start'},
            {'pose_m_rad':[.3,0,-.474-.0009,0,0,0],'contact':True,'stroke_id':0,'phase':'write_line','model_fields':{'model_H_mm':20}}]
    def connect(self,start):return connect_start(self.points,start,self.tip,self.origin,self.normal,self.domain,[0,0,0])
    def test_low_suspended_start_lifts_before_translating_and_keeps_source(self):
        saved=copy.deepcopy(self.points);start=[0,0,.0148-.474,0,0,0]
        pts,d=self.connect(start);self.assertEqual(self.points,saved)
        self.assertEqual(pts[d['prefix_point_count']:],saved)
        self.assertEqual(pts[1]['phase'],'lift_clear');self.assertEqual(pts[1]['pose_m_rad'][:2],start[:2])
        self.assertAlmostEqual(pts[1]['pose_m_rad'][2]+self.tip[2],.04)
        self.assertTrue(all(not p['contact'] for p in pts[:d['prefix_point_count']]))
    def test_high_start_translates_high_before_lowering(self):
        pts,d=self.connect([0,0,.1-.474,0,0,0])
        p=pts[1];self.assertAlmostEqual(p['pose_m_rad'][2]+self.tip[2],.1)
        self.assertAlmostEqual(d['safe_tip_height_m'],.1)
    def test_contact_or_reference_extrapolation_is_rejected(self):
        with self.assertRaises(ValueError):self.connect([0,0,-.474,0,0,0])
        with self.assertRaises(ValueError):self.connect([0,0,.04-.474,0,0,np.deg2rad(12)])
    def test_matching_start_does_not_duplicate_or_shift_route(self):
        pts,d=self.connect(self.points[0]['pose_m_rad']);self.assertEqual(pts,self.points)
        self.assertEqual(d['prefix_point_count'],0)

if __name__=='__main__':unittest.main()
