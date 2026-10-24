import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from prepare_brush_capture_runtime import prepare, collect, pose_matches


class RuntimePreparationTests(unittest.TestCase):
    def fixture(self, folder):
        def point(x,y,z,contact=False):
            return {'pose_m_rad':[x,y,z,3.141592653589793,0.,0.],
                    'contact':contact,'stroke_id':0 if contact else None,
                    'speed_m_s':.003,'acceleration_m_s2':.002}
        write=[point(.5,0,.50),point(.5,0,.46,True),point(.5,.01,.46,True),point(.5,.01,.50)]
        transfer=[write[-1],point(.7,.1,.55)]
        sample={'sample_id':'trial','planned_write':write,'planned_transfer':transfer,
                'planned_initial_lift':[point(.5,0,.46),write[0]],
                'paper_frame':{'paper_normal_base':[0,0,1]},'source_files':[],
                'commanded_depth_mm':.3}
        rows=[{'pose_m_rad':p['pose_m_rad'],'q_rad':[0.]*6} for p in write+transfer]
        candidate={'planned_write':write,'planned_transfer':transfer,
                   'audit':{'passed_numeric':True,'samples':rows}}
        collision={'supplied_samples':rows,'models':{n:{'passed_20mm':True}
                   for n in ('robot_attachments.yaml','robot_attachments.preliminary-cad-20260921.yaml')}}
        paths=[]
        for name,value in [('sample',sample),('candidate',candidate),('collision',collision)]:
            path=folder/(name+'.json');path.write_text(json.dumps(value));paths.append(path)
        return paths,sample

    def test_offline_export_keeps_motion_and_uncollected_references_blocked(self):
        with tempfile.TemporaryDirectory() as t:
            folder=Path(t);paths,sample=self.fixture(folder);out=folder/'prepared'
            with patch('prepare_brush_capture_runtime.subprocess.run',side_effect=AssertionError('network used')):
                result=prepare(*paths,out,'192.168.1.10')
            self.assertFalse(result['robot_execution_ready'])
            for stage in ('write','transfer'):
                bundle=json.loads((out/stage/'bundle.TEMPLATE.json').read_text())
                manifest=json.loads((out/stage/'manifest.PREVIEW.json').read_text())
                self.assertFalse(bundle['force_reference_verified'])
                self.assertFalse(bundle['controller_watchdog_hardware_validated'])
                self.assertTrue(manifest['execution_blockers'])
                for actual,original in zip(bundle['points'],sample['planned_'+stage]):
                    self.assertIsNone(actual['unloaded_wrench_n_nm'])
                    for key in ('pose_m_rad','speed_m_s','acceleration_m_s2','contact'):
                        self.assertEqual(actual[key],original[key])
            plan=json.loads((out/'reference_plan.json').read_text())
            for job in plan['jobs']:
                if 'contact points' in job['purpose']:
                    self.assertAlmostEqual(job['target_tool0_pose_m_rad'][2],.4653)
            self.assertFalse(plan['automatic_pose_advance'])
            self.assertFalse((out/'initial_lift/program.script.preview').exists())

    def test_foreign_collision_report_is_rejected_before_export(self):
        with tempfile.TemporaryDirectory() as t:
            folder=Path(t);paths,_=self.fixture(folder)
            d=json.loads(paths[2].read_text());d['supplied_samples'][0]['q_rad'][0]=1.
            paths[2].write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError,'different complete path'):
                prepare(*paths,folder/'bad','192.168.1.10')
            self.assertFalse((folder/'bad').exists())

    def test_reference_needs_unloaded_confirmation_and_preserves_remote_failure(self):
        with tempfile.TemporaryDirectory() as t:
            folder=Path(t);plan=folder/'plan.json'
            plan.write_text(json.dumps({'sample_id':'trial','source_files':[],
                'jobs':[{'reference_id':'r1','role':'train','target_tool0_pose_m_rad':[0.]*6}]}))
            runner=SimpleNamespace(called=False)
            def remote(*args,**kwargs):
                runner.called=True;compile(kwargs['input'],'collector','exec')
                return SimpleNamespace(stdout=json.dumps({'status':'failed','error':'pose mismatch','motion_sent':False}),stderr='')
            with self.assertRaisesRegex(ValueError,'confirm'):
                collect(plan,'r1',folder/'no',False,remote)
            self.assertFalse(runner.called)
            result=collect(plan,'r1',folder/'failed',True,remote)
            self.assertEqual(result['status'],'failed')
            self.assertEqual(json.loads((folder/'failed/reference.json').read_text())['error'],'pose mismatch')

    def test_rotation_wrap_does_not_reject_same_orientation(self):
        self.assertTrue(pose_matches([0,0,0,0,0,3.141592653589793],
                                     [0,0,0,0,0,-3.141592653589793]))


if __name__=='__main__':unittest.main()
