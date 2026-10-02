"""Fail-closed acceptance tests; no ROS node or controller is instantiated."""
import json
from pathlib import Path
from PIL import Image
from fusion_model_ros2_beta.evaluate_actual_wrist_tradeoff import summarize


def test_runtime_source_inventory_names_existing_modules():
    from fusion_model_ros2_beta.evaluate_actual_wrist_tradeoff import runtime_source_hashes
    inventory=runtime_source_hashes()
    assert 'ur10_actual_kinematics.py' in inventory
    assert all(len(v)==64 for v in inventory.values())


def test_case_lock_prevents_concurrent_same_case_generation(tmp_path):
    from threading import Thread,Event
    from fusion_model_ros2_beta.evaluate_actual_wrist_tradeoff import exclusive_case
    first,second,release=Event(),Event(),Event()
    def worker_one():
        with exclusive_case(tmp_path,1,'B'):
            first.set();assert release.wait(5.)
    def worker_two():
        with exclusive_case(tmp_path,1,'B'):second.set()
    a=Thread(target=worker_one);b=Thread(target=worker_two)
    a.start();assert first.wait(5.);b.start()
    assert not second.wait(.05)
    release.set();a.join(5.);b.join(5.)
    assert second.is_set() and not a.is_alive() and not b.is_alive()


def fixture(root: Path, iou=.96, feasible=True):
    for slots in (1, 3):
        directory=root/f'canvas_{slots}_slots'; directory.mkdir()
        (directory/'experiment.json').write_text(json.dumps({
            'canvas': {'font_size_m': .29 if slots==1 else .1397024400097029},
            'slots': slots, 'steps':16, 'order':11}))
        for case,wrist in zip('ABC',(300.,250.,200.)):
            candidate=directory/case; candidate.mkdir()
            for name in ('inversion_target.png','inversion_rendered.png','inversion_diff.png'):
                Image.new('L',(8,8)).save(candidate/name)
            data=dict(case=case,candidate=str(candidate),initial_pose_sha256='initial',
                target_sha256='target',checkpoint_sha256='frozen',steps_completed=16,
                metrics={'iou_at_0.5':iou,'ssim_original':.96,'plain_mse':.002},
                planning={'feasible':feasible,'metrics':{'wrist_total_deg':wrist}})
            (directory/f'{case}_result.json').write_text(json.dumps(data))
    # Successful test evidence is mandatory, not just a visual fixture.
    tests=root/'tests';tests.mkdir()
    for name in ('ros','model','ssim'):
        (tests/f'{name}.xml').write_text('<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0"/></testsuites>')


def test_planning_same_case_is_serialized_before_residual_file_write(tmp_path,monkeypatch):
    from threading import Thread,Event
    from types import SimpleNamespace
    from fusion_model_ros2_beta import evaluate_actual_wrist_tradeoff as m
    entered,second,release=Event(),Event(),Event();calls=[]
    def stub(args):
        calls.append(args.worker)
        if args.worker==1:entered.set();assert release.wait(5.)
        else:second.set()
    monkeypatch.setattr(m,'_plan',stub)
    def args(worker):return SimpleNamespace(output=str(tmp_path),slots=1,case='B',worker=worker)
    a=Thread(target=m.plan,args=(args(1),));b=Thread(target=m.plan,args=(args(2),))
    a.start();assert entered.wait(5.);b.start();assert not second.wait(.05)
    release.set();a.join(5.);b.join(5.)
    assert calls==[1,2] and second.is_set() and not a.is_alive() and not b.is_alive()


def test_bad_glyph_fails_even_if_rotation_and_relative_iou_pass(tmp_path):
    fixture(tmp_path,iou=.5)
    summarize(tmp_path)
    result=json.loads((tmp_path/'verification.json').read_text())
    assert result['complete'] and not result['accepted']
    assert all(r['C_relative_quality_passed'] for r in result['rotation_acceptance'])
    assert not result['all_visual_checks_passed']


def test_partial_or_rejected_robot_path_is_not_accepted(tmp_path):
    fixture(tmp_path,feasible=False)
    summarize(tmp_path)
    result=json.loads((tmp_path/'verification.json').read_text())
    assert not result['accepted'] and not result['all_robot_checks_passed']


def test_different_initialization_fails_paired_acceptance(tmp_path):
    fixture(tmp_path)
    p=tmp_path/'canvas_3_slots/C_result.json';r=json.loads(p.read_text())
    r['initial_pose_sha256']='not_the_same';p.write_text(json.dumps(r))
    summarize(tmp_path)
    result=json.loads((tmp_path/'verification.json').read_text())
    assert not result['accepted']


def test_missing_test_evidence_fails_acceptance(tmp_path):
    fixture(tmp_path)
    (tmp_path/'tests/model.xml').unlink()
    summarize(tmp_path)
    result=json.loads((tmp_path/'verification.json').read_text())
    assert not result['accepted'] and not result['tests_passed']
