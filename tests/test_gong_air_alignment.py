import ast,contextlib,io,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import run_gong_air_alignment as alignment

class AirAlignmentTests(unittest.TestCase):
    def test_cell_translation_preserves_existing_air_checks(self):
        source=(alignment.ROOT/'tools/brush_unloaded_calibration.py').read_text()
        new=alignment.air_source_at_height(source,.256)
        prefix,body=new.split('\n',1)
        self.assertEqual(body.replace('AIR_HEIGHT_RANGE_M[0] <= height <= AIR_HEIGHT_RANGE_M[1]','.035 <= height <= .045'),source)
        limits=ast.literal_eval(prefix.split('=',1)[1]);self.assertAlmostEqual(limits[1]-limits[0],.010)
        self.assertGreaterEqual(limits[0],.035);self.assertLessEqual(limits[1],.300)
    def test_fast_profile_changes_timing_only_and_keeps_margin(self):
        source=(alignment.ROOT/'tools/brush_unloaded_calibration.py').read_text()
        fast=alignment.air_source_at_height(source,.256,1.5)
        first,second,body=fast.split('\n',2)
        self.assertEqual(second,'AIR_TIME_MARGIN=1.5')
        restored=body.replace('AIR_TIME_MARGIN*','2.5*').replace('AIR_HEIGHT_RANGE_M[0] <= height <= AIR_HEIGHT_RANGE_M[1]','.035 <= height <= .045')
        self.assertEqual(restored,source)
        for margin in [1.0,0.,float('nan'),3.]:
            with self.assertRaises(ValueError):alignment.air_source_at_height(source,.256,margin)

    def test_air_cell_cannot_extend_outside_checked_height_range(self):
        for h in [.01,.035,.301,float('nan')]:
            with self.assertRaises(ValueError):alignment.air_source_at_height('',h)
    def test_preview_never_opens_ssh_and_execute_needs_unloaded_confirmation(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'candidate.json';p.write_text(json.dumps({'points':[{},{}],'tip_height_m':.25}))
            with patch.object(sys,'argv',['run_gong_air_alignment.py','--candidate',str(p)]),patch.object(alignment.subprocess,'Popen') as process,contextlib.redirect_stdout(io.StringIO()) as output:
                alignment.main();process.assert_not_called();self.assertFalse(json.loads(output.getvalue())['motion_sent'])
            with patch.object(sys,'argv',['run_gong_air_alignment.py','--candidate',str(p),'--execute']),patch.object(alignment.subprocess,'Popen') as process,contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):alignment.main()
                process.assert_not_called()

    def test_reference_plan_preview_is_offline_and_remote_source_compiles(self):
        compile(alignment.REMOTE,'remote','exec')
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'references.json'
            p.write_text(json.dumps({'jobs':[{'points':[{},{}]}],'tip_height_m':.256}))
            with patch.object(sys,'argv',['run_gong_air_alignment.py','--candidate',str(p)]),patch.object(alignment.subprocess,'Popen') as process,contextlib.redirect_stdout(io.StringIO()) as output:
                alignment.main()
                process.assert_not_called()
                self.assertFalse(json.loads(output.getvalue())['motion_sent'])

if __name__=='__main__':unittest.main()
