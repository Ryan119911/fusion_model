import json
from pathlib import Path
import sys
import tempfile
import io
from collections import deque
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from brush_write_capture import run_workflow, interpolate, matrix, segment_hits_mesh, optical_check, RecordingRows, build_trial, LiveBackend
from brush_capture_measure import image_array, cloud_xyz, persist_message, metric_rectification, contour_metrics, build_dataset, measure


class ManualDatasetTests(unittest.TestCase):
    def fixture(self, root):
        run=root/'run';run.mkdir();write=root/'write';write.mkdir()
        names=['camera/image.cdr','camera/image.bin','camera/image_mono.png','camera/camera_info.json',
               'camera/cloud.cdr','camera/cloud_xyz_m.npy','camera/cloud_depth_z_m.npy',
               'camera/textured_cloud.cdr','photo_stationary_1.json','photo_stationary_2.json',
               'capture_controller_feedback.jsonl']
        for name in names:
            p=run/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('test fixture')
        execution=write/'execution.json'
        execution.write_text(json.dumps({'status':'completed_buffered_program','motion_sent':True,
                                        'force_trace':{'rows':[{'wrench_n_nm':[0]*6}]}}))
        (write/'bundle.json').write_text(json.dumps({'sample_id':'s','commanded_press_mm':.3}))
        (write/'controller_feedback.jsonl').write_text('{"actual_tool0_pose_m_rad":[0,0,0,0,0,0]}\n')
        (run/'sample.json').write_text(json.dumps({'sample_id':'s','acquisition_kind':'manual_photo_of_executed_fixed_pose_line',
            'write_execution_source':str(execution),'write_speed_mm_s':1.,'photo_tool0_pose_m_rad':[0]*6}))
        (run/'result.json').write_text(json.dumps({'status':'captured','sample_id':'s'}))
        measurement=root/'measurement.json';measurement.write_text(json.dumps({'sample_id':'s','split':'train',
            'repeat_group':'g','raw_run_directory':str(run),'commanded_depth_mm':.3,
            'measured_depth_paper_normal_mm':None,'metrics':{'width_mm':2.,'length_mm':10.},
            'metric_status':'estimate','eligible_for_accepted_mapping_fit':False}))
        return run,execution,measurement

    def test_manual_photo_indexes_actual_write_without_inventing_transfer(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);run,execution,m=self.fixture(root)
            result=build_dataset([m],root/'dataset')
            self.assertEqual(result['sample_count'],1)
            row=json.loads((root/'dataset/dataset.json').read_text())['samples'][0]
            self.assertEqual(row['transfer_mode'],'manual')
            self.assertFalse(row['manual_transfer_telemetry_recorded'])
            self.assertEqual(row['write_execution_source'],str(execution))
            self.assertFalse((run/'transfer_execution.json').exists())

    def test_manual_photo_rejects_missing_raw_force_or_wrong_write_identity(self):
        for problem in ('force','identity'):
            with self.subTest(problem=problem),tempfile.TemporaryDirectory() as t:
                root=Path(t);run,execution,m=self.fixture(root)
                if problem=='force':
                    d=json.loads(execution.read_text());d['force_trace']['rows']=[];execution.write_text(json.dumps(d))
                else:
                    (execution.parent/'bundle.json').write_text(json.dumps({'sample_id':'other','commanded_press_mm':.3}))
                with self.assertRaisesRegex(ValueError,'raw writing force|identity/status/depth'):
                    build_dataset([m],root/'dataset')

    def test_automatic_capture_still_requires_transfer_records(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);run,execution,m=self.fixture(root)
            p=run/'sample.json';d=json.loads(p.read_text());d['acquisition_kind']='automatic';p.write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError,'transfer_execution.json'):
                build_dataset([m],root/'dataset')


class WorkflowTests(unittest.TestCase):
    def test_camera_not_triggered_if_initial_state_window_fails(self):
        calls=[]
        class Reader:
            def __init__(self,*a):calls.append('connect')
            def stationary(self):
                calls.append('initial_window')
                raise RuntimeError('initial state unavailable')
            def close(self):calls.append('close')
        backend=LiveBackend();backend.support=NS()
        with tempfile.TemporaryDirectory() as t, \
             patch('brush_live_state.ContinuousState',Reader), \
             patch('brush_capture_measure.capture_raw') as trigger:
            with self.assertRaisesRegex(RuntimeError,'initial state unavailable'):
                backend.capture({'photo_tool0_pose_m_rad':[0]*6},Path(t))
            trigger.assert_not_called()
        self.assertEqual(calls,['connect','initial_window','close'])

    def test_dataset_cannot_count_unbacked_measurement_as_real_sample(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'measurement.json'
            path.write_text(json.dumps({'sample_id':'s','raw_run_directory':t,'split':'train','repeat_group':'g'}))
            with self.assertRaisesRegex(ValueError,'raw sample files missing'):
                build_dataset([path],Path(t)/'dataset')
    def test_preview_never_uses_hardware_backend(self):
        class Backend:
            def preflight(self, *a): raise AssertionError('hardware reached')
        with tempfile.TemporaryDirectory() as t:
            result=run_workflow({'sample_id':'s1'},Path(t)/'s1',False,Backend())
            self.assertEqual(result['status'],'preview_only')
            self.assertFalse(result['camera_triggered'])

    def test_order_and_sample_identity(self):
        calls=[]
        class Backend:
            def preflight(self,s,o): calls.append(('preflight',s['sample_id']))
            def move(self,n,s,o): calls.append((n,s['sample_id']));return n
            def stationary(self,s,o):calls.append(('stationary',s['sample_id']))
            def capture(self,s,o):calls.append(('capture',s['sample_id']))
        with tempfile.TemporaryDirectory() as t:
            run_workflow({'sample_id':'same-id'},Path(t)/'run',True,Backend())
        self.assertEqual([c[0] for c in calls],['preflight','write','transfer','stationary','capture','stationary'])
        self.assertEqual({c[1] for c in calls},{'same-id'})

    def test_transfer_failure_does_not_trigger_camera_or_retry(self):
        calls=[]
        class Backend:
            def preflight(self,*a):pass
            def move(self,name,*a):
                calls.append(name)
                if name=='transfer':raise RuntimeError('guard stop')
            def capture(self,*a):raise AssertionError('camera called')
        with tempfile.TemporaryDirectory() as t:
            out=Path(t)/'run'
            with self.assertRaisesRegex(RuntimeError,'guard stop'):run_workflow({'sample_id':'s'},out,True,Backend())
            self.assertEqual(json.loads((out/'result.json').read_text())['status'],'failed')
        self.assertEqual(calls,['write','transfer'])


class RawAndGeometryTests(unittest.TestCase):
    def test_actual_cloud_arrays_and_depth_are_saved(self):
        xyz=np.array([[[1.,2.,3.],[4.,5.,6.]]],np.float32)
        msg=NS(width=2,height=1,point_step=12,row_step=24,is_bigendian=False,is_dense=True,data=xyz.tobytes(),
               fields=[NS(name=k,datatype=7,offset=i*4,count=1) for i,k in enumerate('xyz')],
               header=NS(frame_id='cloud',stamp=NS(sec=7,nanosec=8)))
        with tempfile.TemporaryDirectory() as t:
            persist_message(t,'cloud',msg,lambda m:b'raw-cloud-message')
            np.testing.assert_array_equal(np.load(Path(t)/'cloud_xyz_m.npy'),xyz)
            np.testing.assert_array_equal(np.load(Path(t)/'cloud_depth_z_m.npy'),xyz[:,:,2])
            self.assertEqual((Path(t)/'cloud.cdr').read_bytes(),b'raw-cloud-message')

    def test_measurement_from_saved_capture_keeps_command_and_real_depth_separate(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);run=root/'trial';camera=run/'camera';camera.mkdir(parents=True)
            sample={'sample_id':'s1','split':'holdout','repeat_group':'new','commanded_depth_mm':.6,
                    'handeye':{'rotation_matrix':np.eye(3).tolist(),'translation_m':[0,0,0],
                               'child_frame':'color','validation_scope':'synthetic test'},
                    'paper_frame':{'paper_u_base':[1,0,0],'paper_v_base':[0,1,0],'paper_normal_base':[0,0,1],
                        'reconstructed_center_tip_base_m':[0,0,0],'absolute_plane_independently_validated':False},
                    'tcp':{'rotation_status':'nominal'}}
            sample_path=root/'sample.json';sample_path.write_text(json.dumps(sample));(run/'sample.json').write_text(json.dumps(sample))
            (run/'result.json').write_text(json.dumps({'status':'captured','sample_id':'s1'}))
            info={'width':500,'height':500,'k':[500,0,250,0,500,250,0,0,1],'d':[],
                  'distortion_model':'plumb_bob','frame_id':'color'}
            (camera/'camera_info.json').write_text(json.dumps(info))
            (camera/'capture.json').write_text(json.dumps({'status':'raw_saved','sample_id':'s1',
                'stationary_reference':{'sample':{'flange_pose_m_rad':[0,0,.1,float(np.pi),0,0]}}}))
            image=np.full((500,500),255,np.uint8);cv2.rectangle(image,(225,245),(275,255),0,-1)
            cv2.imwrite(str(camera/'image_mono.png'),image)
            result=measure(sample_path,run,root/'measurement',[-10,10,-10,10])
            self.assertAlmostEqual(result['metrics']['width_mm'],2,delta=.4)
            self.assertEqual(result['commanded_depth_mm'],.6)
            self.assertIsNone(result['measured_depth_paper_normal_mm'])
            self.assertIsNone(result['model_H_mm'])
    def test_changed_depth_and_tilt_preserve_requested_tip_location(self):
        root=Path(__file__).resolve().parents[1]/'outputs/brush_mapping_20261022'
        paper=json.loads((root/'paper-frame-reuse-20261022.json').read_text());tcp=json.loads((root/'brush-pose-nominal-20261022.json').read_text())
        spec={'sample_id':'holdout-tilt','split':'holdout','repeat_group':'new-trial',
              'commanded_depth_mm':.6,'paper_start_uv_mm':[20.,-30.],
              'commanded_tilt_delta_uv_rad':[.08,-.04],'commanded_axis_delta_rad':.1}
        trial,points,target=build_trial(paper,tcp,spec)
        pressed=matrix(points[1]['pose_m_rad']);t=np.array(tcp['translation_m'])
        u,v,n=[np.array(paper[k]) for k in ('paper_u_base','paper_v_base','paper_normal_base')]
        expected=np.array(paper['reconstructed_center_tip_base_m'])+.020*u-.030*v-.0006*n
        np.testing.assert_allclose(pressed[:3,3]+pressed[:3,:3]@t,expected,atol=1e-12)
        self.assertEqual(trial['split'],'holdout')
        self.assertNotEqual(points[1]['pose_m_rad'][3:],paper['historical_center_zero_controller_pose_m_rad'][3:])

    def test_default_trial_retains_previously_checked_geometry(self):
        root=Path(__file__).resolve().parents[1]/'outputs/brush_mapping_20261022'
        paper=json.loads((root/'paper-frame-reuse-20261022.json').read_text());tcp=json.loads((root/'brush-pose-nominal-20261022.json').read_text())
        old=json.loads((root/'write_capture_offline/sample_checked_final.json').read_text())
        _,points,_=build_trial(paper,tcp)
        for a,b in zip(points,old['planned_write']):
            np.testing.assert_allclose(a['pose_m_rad'],b['pose_m_rad'],atol=1e-12)
            self.assertEqual(a['speed_m_s'],b['speed_m_s'])
    def test_force_log_does_not_lose_rows_when_guard_window_rolls(self):
        output=io.StringIO();rows=RecordingRows(deque(maxlen=2),output)
        for i in range(7):rows.append((i,i+.1,i+.01,[0.,0.,float(i),0.,0.,0.]))
        self.assertEqual(len(rows),2)
        self.assertEqual(len(output.getvalue().splitlines()),7)
    def test_padded_mono_image_and_cloud_rows_preserved(self):
        image=NS(encoding='mono8',width=2,height=2,step=4,is_bigendian=False,data=bytes([1,2,0,0,3,4,0,0]))
        np.testing.assert_array_equal(image_array(image),[[1,2],[3,4]])
        points=np.zeros((2,32),np.uint8)
        for row in range(2):points[row,:24]=np.array([[row,2,3],[row+1,4,5]],dtype='>f4').view(np.uint8).reshape(-1)
        cloud=NS(width=2,height=2,point_step=12,row_step=32,is_bigendian=True,data=points.tobytes(),
                 fields=[NS(name=k,datatype=7,offset=i*4) for i,k in enumerate('xyz')])
        xyz=cloud_xyz(cloud);np.testing.assert_array_equal(xyz[1,1],[2,4,5])

    def test_raw_pixel_and_serialized_payload_saved_not_only_stats(self):
        msg=NS(encoding='mono8',width=2,height=2,step=2,is_bigendian=False,data=bytes([1,2,3,4]),
               header=NS(frame_id='camera',stamp=NS(sec=7,nanosec=8)))
        with tempfile.TemporaryDirectory() as t:
            persist_message(t,'image',msg,lambda m:b'original-serialized-message')
            self.assertEqual((Path(t)/'image.bin').read_bytes(),msg.data)
            self.assertEqual((Path(t)/'image.cdr').read_bytes(),b'original-serialized-message')
            np.testing.assert_array_equal(cv2.imread(str(Path(t)/'image_mono.png'),0),[[1,2],[3,4]])

    def test_interpolation_linear_not_screw_and_rays_stop_at_paper(self):
        a=[0,0,0,0,0,0];b=[.02,0,0,0,0,1.]
        rows=interpolate(a,b);np.testing.assert_allclose(rows[:,1:3],0)
        triangle=np.array([[[-1,-1,.5],[1,-1,.5],[0,1,.5]]])
        self.assertEqual(segment_hits_mesh(np.zeros(3),np.array([[0,0,1.],[0,0,.2]]),triangle),[True,False])

    def test_metric_projection_recovers_known_size_and_rejects_clipped_roi(self):
        image=np.full((500,500),255,np.uint8);cv2.rectangle(image,(225,245),(275,255),0,-1)
        info={'width':500,'height':500,'k':[500,0,250,0,500,250,0,0,1],'d':[], 'distortion_model':'plumb_bob'}
        cam=np.diag([1.,-1.,-1.,1.]);cam[2,3]=.1
        rect,xs,ys=metric_rectification(image,info,cam,np.zeros(3),np.eye(3),[-10,10,-10,10],.1)
        metrics,_=contour_metrics(rect,xs,ys)
        self.assertAlmostEqual(metrics['length_mm'],10,delta=.4)
        self.assertAlmostEqual(metrics['width_mm'],2,delta=.4)
        with self.assertRaisesRegex(ValueError,'outside'):metric_rectification(image,info,cam,np.zeros(3),np.eye(3),[-100,100,-100,100],1)
        with self.assertRaisesRegex(ValueError,'ROI edge'):contour_metrics(np.zeros((20,20),np.uint8),np.arange(20),np.arange(20))


if __name__=='__main__':unittest.main()
