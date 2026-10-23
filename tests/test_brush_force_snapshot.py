import sys,threading,unittest
from pathlib import Path
from collections import deque
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from brush_force_snapshot import recent_force
from brush_buffered_guard import PreparedRoute,reference_at_pose
import numpy as np

class SnapshotTests(unittest.TestCase):
    def force(self):return SimpleNamespace(lock=threading.Lock(),rows=deque((i/1000,100+i/1000,100+i/1000,[0.]*6) for i in range(50000)))
    def test_reads_only_newest_ten_and_preserves_source_check(self):
        force=self.force()
        with patch('brush_force_snapshot.time.monotonic',return_value=50.):
            rows=recent_force(force);self.assertEqual(len(rows),10);self.assertEqual(rows[-1][0],49.999)
            force.rows[-1]=(49.999,150.,148.499,[0.]*6)
            with self.assertRaisesRegex(RuntimeError,'source timestamp'):recent_force(force)
    def test_real_receipt_staleness_still_rejects(self):
        with patch('brush_force_snapshot.time.monotonic',return_value=51.5):
            with self.assertRaisesRegex(RuntimeError,'stale >1.5 s') as caught:recent_force(self.force())
            diagnostic=caught.exception.diagnostics
            self.assertAlmostEqual(diagnostic['receipt_age_s'],1.501)
            self.assertEqual(diagnostic['receipt_limit_s'],1.5)
            self.assertEqual(diagnostic['buffer_rows'],50000)
    def test_updated_receipt_age_window_accepts_up_to_1_5_seconds(self):
        for age in (.2,1.,1.499,1.5):
            with self.subTest(age=age),patch('brush_force_snapshot.time.monotonic',return_value=49.999+age):
                self.assertEqual(len(recent_force(self.force())),10)
    def test_source_age_window_and_future_timestamps(self):
        force=self.force()
        with patch('brush_force_snapshot.time.monotonic',return_value=50.):
            for delay in (.2,1.5):
                force.rows[-1]=(49.999,150.,150.-delay,[0.]*6)
                self.assertEqual(len(recent_force(force)),10)
            force.rows[-1]=(49.999,150.,150.1,[0.]*6)
            with self.assertRaisesRegex(RuntimeError,'source timestamp'):recent_force(force)
    def test_prepared_reference_keeps_same_interpolation(self):
        points=[{'pose_m_rad':[x,0,0,0,0,0],'unloaded_wrench_n_nm':[10*x]*6} for x in (0.,1.)]
        np.testing.assert_allclose(reference_at_pose(PreparedRoute(points),[.4,0,0,0,0,0]),[4.]*6)

if __name__=='__main__':unittest.main()
