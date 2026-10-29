import copy
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import prepare_gong_model_trial as preparation
import run_gong_model_trial as old_runner


def example():
    rotation=Rotation.from_euler('xyz',[.08,-.06,1.55]).as_matrix()
    u,v,n=rotation.T
    data={'paper':{'paper_u_base':u.tolist(),'paper_v_base':v.tolist(),
                  'paper_normal_base':n.tolist(),'reconstructed_center_tip_base_m':[.5,0,.1]},
          'tcp':{'translation_m':[0,0,.315]},'original_controls':[],
          'conversion':{'paper_translation_mm':[0.,-65.],'xy_scale':1.,
            'H_command_mm_per_model_mm':.045,'alpha_gain':.2,'beta_gain':.2,'gamma_gain':.02,
            'baseline_tool0_rotation_rad':[0,0,0],'tool_brush_axis':[0,0,1]},
          'points':[],'force_reference_verified':True,'audit':{'passed_numeric':True}}
    def point(pose):return {'phase':'air','pose_m_rad':pose,'contact':False,'stroke_id':None,
                           'speed_m_s':.003,'acceleration_m_s2':.002,'unloaded_wrench_n_nm':[1]*6}
    data['points'].append(point([.5,0,.5,0,0,0]))
    for sid,count in enumerate([6,4,6]):
        for i in range(count):
            fields={'paper_x_m':i*.001,'paper_y_m':sid*.001,'model_H_mm':11.,
                    'alpha_rad':0.,'beta_rad':0.,'gamma_absolute_paper_rad':0.}
            data['original_controls'].append({'stroke_id':str(sid),**fields})
            xyz=np.array([.5,0,.1])+i*.001*u+(sid*.001-.065)*v-.000495*n-[0,0,.315]
            p=point([*xyz,0,0,0]);p.update(phase='stroke',contact=True,stroke_id=sid,
                model_fields=fields,commanded_depth_mm=.495)
            data['points'].append(p)
        data['points'].append(point([*(xyz+.04*n),0,0,0]))
    return data


class PreparationTests(unittest.TestCase):
    def test_whole_route_shift_follows_paper_and_preserves_all_fields(self):
        old=example();unchanged=copy.deepcopy(old);new=preparation.relocate(old,50)
        self.assertEqual(old,unchanged)
        delta=np.array(new['relocation']['actual_base_translation_mm'])
        np.testing.assert_allclose(delta[:2],[-50,0],atol=1e-9)
        self.assertAlmostEqual(delta@np.array(old['paper']['paper_normal_base']),0,places=9)
        for a,b in zip(old['points'],new['points']):
            np.testing.assert_allclose(np.array(b['pose_m_rad'][:3])-a['pose_m_rad'][:3],delta/1000,atol=1e-12)
            self.assertEqual(a.get('model_fields'),b.get('model_fields'))
        self.assertFalse(new['force_reference_verified']);self.assertFalse(new['robot_execution_ready'])
        self.assertNotIn('audit',new)
        self.assertTrue(all(p['unloaded_wrench_n_nm'] is None for p in new['points']))

    def test_nonfinite_or_wrong_direction_rejected(self):
        for value in [-1,0,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):preparation.relocate(example(),value)

    def test_prepare_entry_has_no_execute_mode(self):
        with patch.object(sys,'argv',['prepare_gong_model_trial.py','--execute']),patch.object(preparation,'prepare') as prepare,contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):preparation.main()
            prepare.assert_not_called()

    def test_old_remote_entry_rejects_old_mount_before_ssh(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder,'candidate.PREVIEW_ONLY.json').write_text(json.dumps({'mount_model':{}}))
            with patch.object(sys,'argv',['run_gong_model_trial.py','--folder',folder,'--preflight']),patch.object(old_runner,'load_checked_trial') as load,patch.object(old_runner.subprocess,'Popen') as process,contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):old_runner.main()
                load.assert_not_called();process.assert_not_called()

if __name__=='__main__':unittest.main()
