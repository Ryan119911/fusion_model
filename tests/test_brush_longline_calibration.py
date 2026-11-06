"""Software-only tests: synthetic geometry/images and injected runtime adapters."""
import copy
import datetime as dt
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import brush_longline_calibration as core
import brush_longline_measure as measure
import brush_longline_runtime as runtime


def config():
    return core.read(ROOT / 'examples/brush_longline_experiment.json')


class GeometryTests(unittest.TestCase):
    def test_all_scan_kinds_compile_without_hardware(self):
        for kind in ('zero','h','alpha','beta','joint'):
            p=core.build_plan(config(),kind,'test_'+kind)
            self.assertFalse(p['motion_sent'])
            self.assertFalse(p['robot_execution_ready'])
            self.assertFalse(p['points'][0]['contact'])
            self.assertFalse(p['points'][-1]['contact'])
            self.assertGreater(len(p['samples']),100)

    def test_tilted_paper_and_two_axis_tip_compensation(self):
        c=config();basis=Rotation.from_euler('xyz',[7,-10,20],degrees=True).as_matrix()
        for name,value in zip(('paper_u_base','paper_v_base','paper_normal_base'),basis.T):c['geometry'][name]=value.tolist()
        p=core.build_plan(c,'joint','geometry')
        tip=np.array(c['geometry']['tip_tool_m']);origin=np.array(c['geometry']['paper_origin_base_m'])
        for row in p['samples']:
            pose=np.array(p['points'][row['waypoint_index']]['pose_m_rad'])
            rot=Rotation.from_rotvec(pose[3:])
            actual=pose[:3]+rot.apply(tip)
            expected=origin+basis[:,0]*row['paper_u_mm']/1000+basis[:,1]*row['paper_v_mm']/1000-basis[:,2]*row['H_cmd_mm']/1000
            np.testing.assert_allclose(actual,expected,atol=1e-12)

    def test_zero_is_linear_and_constant_posture(self):
        c=config();c['inputs']['alpha_deg']['baseline']=1.;c['inputs']['beta_deg']['baseline']=-1.
        _,y=core.input_curve(c,'zero')
        np.testing.assert_allclose(np.diff(y[:,0]),np.diff(y[:,0])[0])
        np.testing.assert_allclose(np.ptp(y[:,1:],axis=0),0)

    def test_single_variable_scans_do_not_change_other_inputs(self):
        for kind,index in [('h',0),('alpha',1),('beta',2)]:
            _,y=core.input_curve(config(),kind)
            self.assertGreater(np.ptp(y[:,index]),0)
            for other in set(range(3))-{index}:self.assertEqual(np.ptp(y[:,other]),0)

    def test_invalid_basis_bounds_and_numeric_inputs_are_rejected(self):
        for mutate in [lambda c:c['geometry'].update(paper_v_base=[1,0,0]),
                       lambda c:c['motion'].update(start_uv_mm=[90,0]),
                       lambda c:c['inputs']['H_mm'].update(max=float('nan')),
                       lambda c:c['inputs']['H_mm'].update(max=26),
                       lambda c:c['motion'].update(speed_mm_s=0)]:
            c=config();mutate(c)
            with self.assertRaises(ValueError):core.build_plan(c,'h','bad')

    def test_unsafe_id_knots_and_angular_speed_are_rejected(self):
        with self.assertRaises(ValueError):core.build_plan(config(),'h','../escape')
        c=config();c['knots']=[{'fraction':0,'H_mm':.5,'alpha_deg':0,'beta_deg':0},{'fraction':1,'H_mm':2,'alpha_deg':0,'beta_deg':0}]
        with self.assertRaisesRegex(ValueError,'envelope'):core.build_plan(c,'joint','bad')
        c=config();c['motion']['max_angular_speed_deg_s']=.001
        with self.assertRaisesRegex(ValueError,'angular'):core.build_plan(c,'joint','bad')

    def test_feedback_reconstructs_physical_inputs(self):
        c=config();p=core.build_plan(c,'joint','feedback')
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'feedback.jsonl'
            rows=[]
            for i,row in enumerate(p['samples']):
                point=p['points'][row['waypoint_index']]
                rows.append(json.dumps({'host_monotonic_s':10+i*.01,'actual_tool0_pose_m_rad':point['pose_m_rad'],'tcp_velocity':[.005,0,0,0,0,0]}))
            path.write_text('\n'.join(rows))
            actual=measure.feedback_samples(path,p)
            for row,original in zip(actual,p['samples']):
                np.testing.assert_allclose([row['H_exec_mm'],row['alpha_exec_rad'],row['beta_exec_rad']],
                                           [original['H_cmd_mm'],original['alpha_cmd_rad'],original['beta_cmd_rad']],atol=1e-10)


class MeasurementTests(unittest.TestCase):
    def profile_fixture(self):
        p=core.build_plan(config(),'h','image')
        before=np.full((300,1400),255,np.uint8);after=before.copy();after[130:160,100:1300]=0
        calibration={'image_to_paper_uv_mm':[[.1,0,-70],[0,-.1,15],[0,0,1]],
                     'roi_uv_mm':[-65,65,-14,14],'mm_per_pixel':.1,'bin_mm':1,'difference_threshold':30}
        return before,after,calibration,p

    def test_metric_width_is_measured_without_inventing_input_labels(self):
        rows,_,_=measure.ink_profile(*self.profile_fixture())
        self.assertAlmostEqual(np.median([r['width_mm'] for r in rows]),3.,places=5)
        self.assertTrue(all(r['H_exec_mm']=='' and r['input_correspondence']=='unverified' for r in rows))
        with self.assertRaises(ValueError):measure.fit_response(rows)

    def test_multiple_components_and_touching_edges_fail(self):
        before,after,c,p=self.profile_fixture();after[30:50,300:400]=0
        with self.assertRaisesRegex(ValueError,'isolated'):measure.ink_profile(before,after,c,p)
        before,after,c,p=self.profile_fixture();after[0:20,100:1300]=0
        with self.assertRaises(ValueError):measure.ink_profile(before,after,c,p)

    def test_small_feedback_jitter_is_reported_but_large_reversal_is_rejected(self):
        before,after,c,p=self.profile_fixture()
        c['input_alignment']={'verified':True,'evidence':'synthetic known correspondence test'}
        feedback=[{'s_mm':i*.25,'H_exec_mm':.5,'alpha_exec_rad':0.,'beta_exec_rad':0.,'host_monotonic_s':i*.01,'forward_speed_mm_s':3.} for i in range(481)]
        small=dict(feedback[20],s_mm=feedback[20]['s_mm']-.02)
        feedback.insert(21,small)
        rows,_,_=measure.ink_profile(before,after,c,p,feedback)
        self.assertEqual(rows[10]['feedback_nonadvancing_samples_skipped'],1)
        feedback[21]['s_mm']=feedback[20]['s_mm']-.5
        with self.assertRaisesRegex(ValueError,'reversed'):measure.ink_profile(before,after,c,p,feedback)

    def test_visual_zero_has_height_bracket_and_does_not_update_config(self):
        p=core.build_plan(config(),'zero','zero');old=copy.deepcopy(p)
        feedback=[{'host_monotonic_s':i*.01,'H_exec_mm':-.5+i*.01} for i in range(101)]
        side=[{'host_monotonic_s':.4+i*.01,'tip_gap_mm':v,'status':'visible'} for i,v in enumerate([.3,.2,.08,.04,0])]
        z=measure.estimate_zero(side,feedback,p,.1,3)
        self.assertEqual(z['zero_kind'],'visual_contact_candidate')
        self.assertAlmostEqual(z['first_contact_monotonic_s'],.42)
        self.assertAlmostEqual(z['H_at_contact_mm'],-.08)
        self.assertEqual(p,old)

    def test_side_video_preserves_measured_frame_times_and_gap(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);video=d/'side.avi';log=d/'frames.jsonl'
            writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),20,(160,120))
            self.assertTrue(writer.isOpened())
            records=[]
            for i,y in enumerate([50,60,70,80]):
                image=np.full((120,160,3),255,np.uint8);image[20:y,70:90]=0;writer.write(image)
                records.append(json.dumps({'video_frame_index':i,'host_monotonic_s':100+i*.07}))
            writer.release();log.write_text('\n'.join(records))
            cal={'brush_roi_px':[50,10,110,105],'paper_line_abc':[0,-1,100],'mm_per_pixel':.1,'dark_threshold':100,'minimum_brush_area_px':20}
            rows=measure.side_features(video,log,cal)
            self.assertEqual([r['host_monotonic_s'] for r in rows],[100,100.07,100.14,100.21])
            np.testing.assert_allclose([r['tip_gap_mm'] for r in rows],[5.1,4.1,3.1,2.1],atol=.11)

    def test_initial_contact_and_unobserved_event_do_not_define_zero(self):
        p=core.build_plan(config(),'zero','zero')
        f=[{'host_monotonic_s':i*.01,'H_exec_mm':0.} for i in range(101)]
        for gaps in ([0,0,0],[1,1,1]):
            side=[{'host_monotonic_s':.4+i*.01,'tip_gap_mm':v,'status':'visible'} for i,v in enumerate(gaps)]
            with self.assertRaises(ValueError):measure.estimate_zero(side,f,p)

    def test_hand_camera_uses_separate_measured_photo_poses(self):
        p=core.build_plan(config(),'h','handeye');image=np.full((600,800,3),255,np.uint8)
        c={'mode':'handeye','roi_uv_mm':[-65,65,-15,15],'camera_info':{'k':[500,0,400,0,500,300,0,0,1],'d':[0]*5,'distortion_model':'plumb_bob','height':600,'width':800},
           'tool0_from_camera':np.eye(4).tolist(),'image_to_paper_uv_mm':np.eye(3).tolist()}
        with tempfile.TemporaryDirectory() as d:
            before=Path(d)/'before.json';after=Path(d)/'after.json'
            core.save(before,{'flange_pose_m_rad':[.5,0,1,np.pi,0,0]})
            core.save(after,{'flange_pose_m_rad':[.55,0,1,np.pi,0,0]})
            images,result=measure.handeye_images([image,image],c,p,before,after)
            self.assertNotIn('image_to_paper_uv_mm',result)
            self.assertFalse(np.allclose(result['before_image_to_paper_uv_mm'],result['after_image_to_paper_uv_mm']))
            for key,x in [('before_image_to_paper_uv_mm',400),('after_image_to_paper_uv_mm',368.75)]:
                uv=np.array(result[key])@np.array([x,300,1.])
                np.testing.assert_allclose(uv[:2]/uv[2],[0,0],atol=1e-10)

    def test_response_recovers_joint_effects_and_flags_domain(self):
        rng=np.random.default_rng(9);rows=[]
        for i in range(100):
            h,a,b=rng.uniform(.2,.8),rng.uniform(-.03,.03),rng.uniform(-.03,.03)
            rows.append({'trial_id':'t'+str(i//20),'repeat_group':'r'+str(i//20),'split':'train','status':'valid','input_correspondence':'verified',
                         'data_kind':'synthetic_test','H_exec_mm':h,'alpha_exec_rad':a,'beta_exec_rad':b,'width_mm':1+3*h+2*a-4*b+5*h*a,'center_offset_mm':a-b})
        model=measure.fit_response(rows,ridge=0)
        predicted,metrics=measure.predict_rows(model,rows)
        self.assertLess(max(metrics['rmse_mm']),1e-10)
        outside=dict(rows[0],H_exec_mm=20)
        self.assertFalse(measure.predict_rows(model,[outside])[0][0]['in_training_hull'])
        rows[0]['split']='holdout'
        with self.assertRaisesRegex(ValueError,'holdout'):measure.fit_response(rows)

    def test_collinear_inputs_are_not_joint_calibration(self):
        rows=[{'trial_id':'x','repeat_group':'x','split':'train','status':'valid','input_correspondence':'verified','data_kind':'synthetic_test',
               'H_exec_mm':i*.01,'alpha_exec_rad':i*.02,'beta_exec_rad':i*.03,'width_mm':i,'center_offset_mm':0} for i in range(30)]
        with self.assertRaisesRegex(ValueError,'identifiable'):measure.fit_response(rows)


class RuntimeTests(unittest.TestCase):
    def test_preparation_refuses_example_geometry_before_any_hardware_import(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'plan.json';core.save(path,core.build_plan(config(),'h','preview'))
            with patch.object(runtime.importlib,'import_module',side_effect=AssertionError('hardware imported')):
                with self.assertRaisesRegex(ValueError,'geometry'):runtime.prepare_live(path,'missing','missing','192.0.2.10','missing',Path(d)/'run')

    def test_changed_plan_is_detected(self):
        with tempfile.TemporaryDirectory() as d:
            p=core.build_plan(config(),'h','changed');p['points'][2]['pose_m_rad'][2]+=.001
            path=Path(d)/'plan.json';core.save(path,p)
            with self.assertRaisesRegex(ValueError,'differs'):runtime.checked_plan(path)

    def prepared_fixture(self,d):
        p=core.build_plan(config(),'h','preview');core.save(d/'plan.json',p)
        core.save(d/'bundle.json',{'points':p['points']})
        prepared={'schema':'brush_longline_prepared_v1','plan_sha256':runtime.sha(d/'plan.json'),'bundle_sha256':runtime.sha(d/'bundle.json')}
        core.save(d/'prepared.json',prepared)
        return d/'prepared.json'

    def test_run_default_has_no_camera_or_robot_access(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);prepared=self.prepared_fixture(d)
            def forbidden(*a,**k):raise AssertionError('hardware accessed')
            result=runtime.run_trial(prepared,d/'preview',recording_factory=forbidden,executor=forbidden)
            self.assertFalse(result['motion_sent']);self.assertFalse(result['camera_triggered'])

    def test_execute_without_camera_configuration_fails_before_motion(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);prepared=self.prepared_fixture(d)
            with self.assertRaisesRegex(ValueError,'cameras'):runtime.run_trial(prepared,d/'bad',True)
            self.assertFalse(core.read(d/'bad/RESULT.json')['motion_sent'])

    def test_force_reference_requires_fresh_bound_validation(self):
        points=core.build_plan(config(),'h','force')['points']
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'validated'):runtime.reference_wrenches({},points,d)
            record={'schema':'brush_longline_force_reference_v1','passed':True,'recorded_at':(dt.datetime.now(dt.timezone.utc)-dt.timedelta(hours=1)).isoformat()}
            with self.assertRaisesRegex(ValueError,'30 minutes'):runtime.reference_wrenches(record,points,d)

    def live_fixture(self,d):
        prepared_path=self.prepared_fixture(d)
        plan=core.read(d/'plan.json');points=copy.deepcopy(plan['points'])
        for point in points:point.update(q_rad=[0,-1,-1,-1,1,0],unloaded_wrench_n_nm=[0.]*6)
        script,manifest=core.compile_program(points,'192.0.2.10',50071)
        evidence={}
        for name in ('physical_mapping','blended_path_audit','force_reference'):
            core.save(d/(name+'.json'),{'passed':True,'route_sha256':manifest['route_sha256'],'program_sha256':manifest['program_sha256']})
            evidence[name]={'path':name+'.json','sha256':runtime.sha(d/(name+'.json'))}
        bundle={'points':points,'physical_mapping_verified':True,'force_reference_verified':True,'controller_watchdog_hardware_validated':True,'evidence':evidence}
        (d/'bundle.json').unlink();core.save(d/'bundle.json',bundle)
        import brush_buffered_motion as compiler
        core.save(d/'watchdog.json',{'passed':True,'compiler_sha256':runtime.sha(compiler.__file__)})
        prepared=core.read(prepared_path)
        prepared.update(bundle_sha256=runtime.sha(d/'bundle.json'),compiler_sha256=runtime.sha(compiler.__file__),
                        prepared_at=dt.datetime.now(dt.timezone.utc).isoformat(),force_reference_recorded_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                        watchdog={'path':str(d/'watchdog.json'),'sha256':runtime.sha(d/'watchdog.json')},guard_host='192.0.2.10',guard_port=50071)
        prepared_path.unlink();core.save(prepared_path,prepared)
        core.save(d/'cameras.json',{})
        return prepared_path,d/'cameras.json'

    def test_camera_start_failure_never_calls_executor_and_is_archived(self):
        class BrokenRecording:
            closed=False
            def __init__(self,*a):pass
            def start(self):raise RuntimeError('camera unavailable')
            def close(self):BrokenRecording.closed=True
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);prepared,cameras=self.live_fixture(d)
            def forbidden(*a,**k):raise AssertionError('motion attempted')
            def capture(*a):return {'max_motion_mm':0,'max_motion_deg':0,'flange_pose_m_rad':[0,0,0,0,0,0]}
            with self.assertRaisesRegex(RuntimeError,'camera unavailable'):
                runtime.run_trial(prepared,d/'run',True,cameras,d,executor=forbidden,recording_factory=BrokenRecording,support=SimpleNamespace(),capture=capture)
            self.assertTrue(BrokenRecording.closed)
            self.assertFalse(core.read(d/'run/RESULT.json')['motion_sent'])

    def test_execution_failure_retains_motion_status_and_skips_after_capture(self):
        class Recording:
            finished=False;closed=False
            def __init__(self,*a):pass
            def start(self):pass
            def health(self):pass
            def finish(self):Recording.finished=True
            def close(self):Recording.closed=True
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);prepared,cameras=self.live_fixture(d)
            def executor(bundle,folder,host,port,output,support,capture):
                core.save(output,{'status':'stopped_on_error','motion_sent':True})
                raise RuntimeError('injected guard stop')
            with self.assertRaisesRegex(RuntimeError,'guard stop'):
                runtime.run_trial(prepared,d/'run',True,cameras,d,executor=executor,recording_factory=Recording,
                                  support=SimpleNamespace(fresh=lambda *a:None,RawForce=type('Force',(),{})),
                                  capture=lambda *a:{'max_motion_mm':0,'max_motion_deg':0,'flange_pose_m_rad':[0,0,0,0,0,0]})
            self.assertFalse(Recording.finished);self.assertTrue(Recording.closed)
            self.assertTrue(core.read(d/'run/RESULT.json')['motion_sent'])

    def test_validated_force_fit_is_bound_and_has_expected_wrenches(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);samples=[];holdout=[]
            rotations=[[-.05,0,0],[.05,0,0],[0,-.05,0],[0,.05,0],[0,0,-.05],[0,0,.05],[.005,.004,.003],[-.005,-.004,-.003]]
            for i,angles in enumerate(rotations):
                folder=d/str(i);folder.mkdir();core.save(folder/'raw_force.json',{'synthetic_test':True})
                x=np.array(angles)/np.deg2rad(4)
                wrench=[1+x[0],2+x[1],3+x[2],0,0,0]
                core.save(folder/'sample.json',{'schema':'brush_longline_air_sample_v1','status':'recorded',
                    'operator_unloaded_confirmed':True,'recorded_at':dt.datetime.now(dt.timezone.utc).isoformat(),
                    'pose_m_rad':[.5,0,.5]+angles,'wrench_n_nm':wrench,'raw_force_sha256':runtime.sha(folder/'raw_force.json')})
                (samples if i<6 else holdout).append(folder/'sample.json')
            result=runtime.fit_force(samples,holdout,.1,.02,d/'fit')
            self.assertTrue(result['passed'])
            record=core.read(d/'fit/force_reference.json')
            points=[{'pose_m_rad':[.5,0,.5,0,0,0]}]
            np.testing.assert_allclose(runtime.reference_wrenches(record,points,d/'fit'),[[1,2,3,0,0,0]],atol=1e-8)
            record['local_fit']['offset'][0]=100
            with self.assertRaisesRegex(ValueError,'bind'):runtime.reference_wrenches(record,points,d/'fit')


if __name__=='__main__':unittest.main()
