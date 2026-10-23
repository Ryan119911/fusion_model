import itertools,json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
from validate_brush_unloaded_sweep import validate


class ReferenceValidationTests(unittest.TestCase):
    def fixture(self,folder):
        angles=[(0,0,0)]
        for i in range(3):
            for sign in (-1,1):
                x=[0,0,0];x[i]=sign;angles.append(x)
        angles+=list(itertools.product((-1,1),repeat=3))
        angles +=[(.5,.5,.5),(-.5,.5,-.5),(.5,-.5,-.5),(-.5,-.5,.5),(0,0,0)]
        refs=[]
        for i,x in enumerate(angles):
            rotation=(np.array(x)*np.deg2rad(4)).tolist()
            wrench=[.4*x[0],.5*x[1],60+.2*x[2],0,0,0]
            sid=f'pose-{i:02d}';dest=folder/sid;dest.mkdir()
            role='train' if i<15 else 'holdout' if i<19 else 'drift_check'
            refs.append({'reference_id':sid,'role':role,'reference':{'actual_pose_m_rad':[0,0,0,*rotation],
                         'median_n_nm':wrench}})
            rows=[{'host_unix_ns':int(t*1e9),'actual_tool0_pose_m_rad':[0,0,0,*rotation],
                   'tcp_velocity':[.001,0,0,0,0,0]} for t in (1,1.01)]
            (dest/'controller_feedback.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            raw=[{'received_unix_s':t,'wrench_n_nm':wrench} for t in (1,1.01)]
            (dest/'raw_force.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in raw))
        (folder/'request.json').write_text(json.dumps({'jobs':[{'baseline_rotation':[0,0,0]}]}))
        result={'status':'air_references_collected_pending_independent_validation','completed_references':refs}
        (folder/'result.json').write_text(json.dumps(result))
        return result

    def test_reference_fit_does_not_accept_contact_or_brush_mapping(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);self.fixture(folder);report=validate(folder)
            self.assertTrue(report['passed'])
            self.assertFalse(report['write_force_reference_verified'])
            self.assertFalse(report['brush_H_mapping_accepted'])

    def test_heldout_is_not_used_for_fitting_and_can_reject(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);result=self.fixture(folder);before=validate(folder)
            result['completed_references'][15]['reference']['median_n_nm'][0]+=2
            (folder/'result.json').write_text(json.dumps(result));after=validate(folder)
            self.assertFalse(after['passed'])
            np.testing.assert_allclose(before['local_fit']['coefficients_10x6'],after['local_fit']['coefficients_10x6'])

    def test_return_drift_rejects_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);result=self.fixture(folder)
            result['completed_references'][-1]['reference']['median_n_nm'][2]+=1
            (folder/'result.json').write_text(json.dumps(result))
            self.assertFalse(validate(folder)['passed'])

    def test_resumed_observations_keep_original_raw_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);result=self.fixture(folder);parent=folder/'previous-sweep';parent.mkdir()
            for ref in result['completed_references'][:6]:
                source=parent/ref['reference_id'];(folder/ref['reference_id']).rename(source)
                ref['reference_source_directory']=str(source)
            (folder/'result.json').write_text(json.dumps(result))
            self.assertTrue(validate(folder)['passed'])

    def test_explicit_10n_limit_rechecks_raw_dynamic_error(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);self.fixture(folder)
            path=folder/'pose-00/raw_force.jsonl'
            rows=[json.loads(s) for s in path.read_text().splitlines()]
            for row in rows:row['wrench_n_nm'][0]+=4.3
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            self.assertFalse(validate(folder)['passed'])
            report=validate(folder,10.)
            self.assertTrue(report['passed'])
            self.assertEqual(report['reference_error_limit_n'],10.)
            self.assertEqual(report['force_stop_threshold_n_unchanged'],10.)
            self.assertFalse(report['write_force_reference_verified'])
            with self.assertRaises(ValueError):validate(folder,11.)


if __name__=='__main__':unittest.main()
