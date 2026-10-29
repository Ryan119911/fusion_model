import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
from brush_trial_force import trial_force_level,TrialForceStop


class Force:
    def __init__(self,values):self.values=values
    def recent(self):return [(0,0,0,list(v)+[0,0,0]) for v in self.values]


class GuardTests(unittest.TestCase):
    def test_10n_boundary(self):
        self.assertAlmostEqual(trial_force_level(Force([[9.99,0,0]]*10),np.zeros(6),10),9.99)
        with self.assertRaises(TrialForceStop):trial_force_level(Force([[10,0,0]]*10),np.zeros(6),10)

    def test_operator_selected_100n_boundary(self):
        self.assertAlmostEqual(trial_force_level(Force([[99.99,0,0]]*10),np.zeros(6),100),99.99)
        with self.assertRaises(TrialForceStop):trial_force_level(Force([[100,0,0]]*10),np.zeros(6),100)

    def test_one_raw_spike_stops(self):
        with self.assertRaises(TrialForceStop):
            trial_force_level(Force([[0,0,0]]*9+[[0,10.01,0]]),np.zeros(6),10)

    def test_vector_norm_and_original_reference(self):
        baseline=np.array([4,12,64,0,0,0])
        with self.assertRaises(TrialForceStop):trial_force_level(Force([[10,20,64]]),baseline,10)
        self.assertAlmostEqual(trial_force_level(Force([[5,12,64]]*10),baseline,10),1.)

    def test_invalid_thresholds_and_data(self):
        for threshold in (0,100.01,float('nan'),float('inf')):
            with self.assertRaises(ValueError):trial_force_level(Force([[0,0,0]]),np.zeros(6),threshold)
        with self.assertRaises(RuntimeError):trial_force_level(Force([[float('nan'),0,0]]),np.zeros(6),10)


if __name__=='__main__':unittest.main()
