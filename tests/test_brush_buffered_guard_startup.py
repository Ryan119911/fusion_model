"""Preflight lifetime tests with fake robot/network; never opens robot sockets."""
import json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock,patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import brush_buffered_guard as guard


class GuardStartupTests(unittest.TestCase):
    def run_fake(self,change_pose=False,fail_during_motion=False):
        readers=[];gates=[];commands=[];q=np.zeros(6);pose=np.zeros(6)
        class Reader:
            def __init__(self,*args):self.closed=False;self.index=len(readers);readers.append(self)
            def stationary(self):
                current=pose.copy()
                if change_pose and self.index==1:current[3]=.01
                return q.copy(),current,np.zeros(6)
            def close(self):self.closed=True
        class Force:
            def __init__(self):
                if len(gates)!=4 or any(not r.closed for r in readers):raise AssertionError('force startup preceded completed geometry checks')
                self.rclpy=SimpleNamespace(ok=lambda:True)
            def recent(self):return []
            def close(self):pass
        def model_gate(*args):
            if any(not r.closed for r in readers):raise AssertionError('live reader stayed armed during expensive preflight')
            gates.append(args)
        support=SimpleNamespace(controller_ready=lambda host:{},model_gate=model_gate,RawForce=Force,
            force_snapshot=lambda f:{},command=lambda *x:commands.append(x),close_commands=lambda:None)
        channel=MagicMock();channel.recv.return_value=b''
        server=MagicMock();server.accept.return_value=(channel,('192.168.1.3',1234))
        p={'pose_m_rad':pose.tolist(),'contact':False,'q_rad':q.tolist(),'unloaded_wrench_n_nm':[0.]*6}
        bundle={'points':[p.copy(),p.copy()],'start_pose_m_rad':pose.tolist(),'active_tcp_offset_m_rad':[0.]*6,'maximum_runtime_s':1.}
        ticks=[]
        def tick(*args,**kwargs):
            ticks.append(1)
            if fail_during_motion and len(ticks)>1:raise RuntimeError('continuous state stream failed: stale >150ms')
            return (0,q.copy()),pose.copy(),0.
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);report=folder/'execution.json'
            with patch.object(guard,'ContinuousState',Reader),patch.object(guard,'compile_program',return_value=('script',{})),patch.object(guard,'execution_blockers',return_value=[]),patch.object(guard,'health_tick',side_effect=tick),patch.object(guard.socket,'socket',return_value=server),patch.object(Path,'home',return_value=folder):
                try:guard.execute_buffered(bundle,folder,'192.168.1.10',50071,report,support,lambda *a:{'active_tcp_offset_m_rad':[0.]*6})
                except RuntimeError:
                    if not(change_pose or fail_during_motion):raise
                result=json.loads(report.read_text())
        return result,readers,commands

    def test_new_live_stream_starts_only_after_heavy_preflight(self):
        result,readers,commands=self.run_fake()
        self.assertEqual(result['status'],'completed_buffered_program');self.assertEqual(len(readers),2)
        self.assertTrue(all(r.closed for r in readers));self.assertEqual(len(commands),1)

    def test_changed_start_stops_before_any_motion(self):
        result,_,commands=self.run_fake(change_pose=True)
        self.assertFalse(result['motion_sent']);self.assertEqual(commands,[])
        self.assertIn('robot moved during preflight',result['error'])

    def test_live_stream_failure_still_stops_and_does_not_reconnect(self):
        result,readers,commands=self.run_fake(fail_during_motion=True)
        self.assertEqual(result['status'],'stopped_on_error');self.assertEqual(len(readers),2)
        self.assertIn('150ms',result['error']);self.assertEqual(commands[-1][1],'stopl(0.5)\n')

if __name__=='__main__':unittest.main()
