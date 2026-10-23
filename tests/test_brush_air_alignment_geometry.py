from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from brush_air_alignment_geometry import local_air_target


class AirAlignmentTests(unittest.TestCase):
    zero=[.6,0,.45,np.pi,0,0]
    tip=[0,0,.474]
    normal=[0,0,1]

    def test_millimetre_translation_retains_actual_rotation(self):
        start=[.6,0,.456,np.pi,0,.0001]
        goal=[.6001,0,.455,np.pi,0,0]
        target=local_air_target(start,goal,self.zero,self.tip,self.normal)
        self.assertEqual(target[3:],start[3:])
        self.assertEqual(target[:3],goal[:3])

    def test_gross_35mm_move_is_not_silently_treated_as_fine_alignment(self):
        with self.assertRaisesRegex(ValueError,'coarse positioning'):
            local_air_target([.6,0,.490,np.pi,0,0],[.6,0,.455,np.pi,0,0],self.zero,self.tip,self.normal)

    def test_target_close_to_observed_zero_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'3 mm nominal clearance'):
            local_air_target([.6,0,.455,np.pi,0,0],[.6,0,.4525,np.pi,0,0],self.zero,self.tip,self.normal)

    def test_short_unloaded_retreat_from_lower_position_is_allowed(self):
        target=local_air_target([.6,0,.452,np.pi,0,0],[.6,0,.455,np.pi,0,0],self.zero,self.tip,self.normal)
        self.assertAlmostEqual(target[2],.455)

    def test_pose_change_cannot_be_hidden_in_small_translation(self):
        with self.assertRaisesRegex(ValueError,'does not rotate'):
            local_air_target([.6,0,.456,np.pi,0,0],[.6,0,.455,np.pi-.01,0,0],self.zero,self.tip,self.normal)


if __name__=='__main__':unittest.main()
