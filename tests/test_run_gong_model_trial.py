import contextlib,copy,io,json,sys,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import run_gong_model_trial as trial


class ModelTrialTests(unittest.TestCase):
    def test_execute_requires_explicit_initial_trial_acceptance(self):
        with patch.object(sys,'argv',['run_gong_model_trial.py','--execute']),patch.object(trial,'load_checked_trial') as load,contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):trial.main()
            load.assert_not_called()

    def test_default_preview_never_starts_ssh_or_motion(self):
        payload={'candidate':{'points':[{}],'conversion':{},'zero':{},'control_counts':[6,4,6]},
                 'reference_acceptance':{'moving_max_force_error_n':4.3,'model_kind':'linear','static_reference_count':3},
                 'reference_extrapolation':{'complete':False,'max_distance_deg':3.77,'extrapolation_bound_n':1.97},
                 'collision':{'models':{'fixture':{'minimum_clearance_mm':47.18}}},
                 'body_collision':{'minimum_clearance_mm':17.44,'below_default_20mm':True}}
        with patch.object(sys,'argv',['run_gong_model_trial.py']),patch.object(trial,'load_checked_trial',return_value=payload),patch.object(trial.subprocess,'Popen') as process,contextlib.redirect_stdout(io.StringIO()) as output:
            trial.main();process.assert_not_called()
            self.assertFalse(json.loads(output.getvalue())['motion_sent'])

    @unittest.skipUnless((trial.DEFAULT/'candidate.PREVIEW_ONLY.json').exists(),'recorded site proposal not installed')
    def test_recorded_trial_preserves_model_and_detects_coordinate_corruption(self):
        d=json.loads((trial.DEFAULT/'candidate.PREVIEW_ONLY.json').read_text());trial.check_conversion(d)
        bad=copy.deepcopy(d);point=next(p for p in bad['points'] if p['contact']);point['pose_m_rad'][2]+=.001
        with self.assertRaises(ValueError):trial.check_conversion(bad)
        bad=copy.deepcopy(d)
        for p in bad['points']:
            if p['contact']:p['model_fields']['gamma_absolute_paper_rad']=0.
        with self.assertRaises(ValueError):trial.check_conversion(bad)

if __name__=='__main__':unittest.main()
