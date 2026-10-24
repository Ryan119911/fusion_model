import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
from brush_pose_batch import run_pose_batch


class Backend:
    def __init__(self,fail=None):self.fail=fail;self.calls=[];self.motion_sent=False
    def preflight_all(self,jobs):
        self.calls.append('preflight')
        if self.fail=='preflight':raise RuntimeError('preflight rejected')
    def event(self,event):pass
    def held_orientation(self,job,stage,index):
        self.motion_sent=True;self.calls.append((job['sample_id'],stage['kind']))
        if self.fail==stage['kind']:raise RuntimeError('stopped')
    def air_reference(self,job,stage,index):
        self.motion_sent=True;self.calls.append((job['sample_id'],'air_rotation'))
        if self.fail=='air_rotation':raise RuntimeError('no valid reference')


class BatchTests(unittest.TestCase):
    def jobs(self):return [{'sample_id':str(i),'stages':[{'kind':k} for k in ('transfer','air_rotation','write')]} for i in range(2)]
    def test_missing_reference_stage_rejected_before_motion(self):
        jobs=self.jobs();jobs[0]['stages'].pop(1);backend=Backend()
        report=run_pose_batch(jobs,backend);self.assertFalse(report['motion_sent']);self.assertEqual(backend.calls,[])
    def test_preflight_failure_sends_nothing(self):
        backend=Backend('preflight');report=run_pose_batch(self.jobs(),backend)
        self.assertFalse(report['motion_sent']);self.assertEqual(backend.calls,['preflight'])
    def test_failed_air_reference_does_not_write_or_start_next_sample(self):
        backend=Backend('air_rotation');report=run_pose_batch(self.jobs(),backend)
        self.assertEqual(report['completed_samples'],[])
        self.assertEqual(backend.calls,['preflight',('0','transfer'),('0','air_rotation')])
    def test_failed_write_stops_without_retry(self):
        backend=Backend('write');report=run_pose_batch(self.jobs(),backend)
        self.assertEqual(report['completed_samples'],[]);self.assertEqual(len(backend.calls),4)
    def test_all_samples_run_in_required_order(self):
        backend=Backend();report=run_pose_batch(self.jobs(),backend)
        self.assertEqual(report['completed_samples'],['0','1'])
        self.assertEqual(report['status'],'selected_pose_depth_samples_written_waiting_shared_capture')


if __name__=='__main__':unittest.main()
