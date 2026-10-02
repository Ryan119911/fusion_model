import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('verify_frozen_requests',Path(__file__).parents[1]/'verify_frozen_requests.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def fixture(root):
    inputs=root/'inputs';inputs.mkdir()
    for name in ('initial.csv','target.png','checkpoint.pt','renderer.py'):(inputs/name).write_text(name)
    manifest=inputs/'references_wu_verified.json'
    manifest.write_text(json.dumps({'references':[dict(character='武',source_trajectory=str(inputs/'initial.csv'),
        original_target_image=str(inputs/'target.png'))]}))
    for slots in (1,3):
        directory=root/f'canvas_{slots}_slots';directory.mkdir()
        identity=dict(manifest_sha256=module.sha(manifest),source_sha256=module.sha(inputs/'initial.csv'),
            original_target_sha256=module.sha(inputs/'target.png'))
        (directory/'experiment.json').write_text(json.dumps(identity))
        context=directory/'robot_context.json';context.write_text('{}')
        for case in 'ABC':
            candidate=directory/case;candidate.mkdir()
            for filename in ('physical_trajectory.csv','scaled_initial_pose.csv','scaled_target.png'):
                (candidate/filename).write_text(filename)
            request=dict(config=dict(bbsmg_checkpoint=str(inputs/'checkpoint.pt'),max_steps=16,order=11,
                joint_robot_context=str(context)),checkpoint_sha256=module.sha(inputs/'checkpoint.pt'),
                original_target_sha256=identity['original_target_sha256'],seed_sha256=None,
                robot_context_sha256=module.sha(context),implementation={str(inputs/'renderer.py'):module.sha(inputs/'renderer.py')})
            (candidate/'original_target_generation_request.json').write_text(json.dumps(request))
            data=dict(candidate=str(candidate),checkpoint_sha256=request['checkpoint_sha256'],steps_completed=16,
                physical_csv_sha256=module.sha(candidate/'physical_trajectory.csv'),
                initial_pose_sha256=module.sha(candidate/'scaled_initial_pose.csv'),target_sha256=module.sha(candidate/'scaled_target.png'))
            (directory/f'{case}_generated.json').write_text(json.dumps(data))


def test_all_six_requests_pin_original_inputs_and_model(tmp_path):
    fixture(tmp_path);result=module.verify(tmp_path)
    assert result['verified'] and result['checked_request_count']==6


def test_changed_frozen_renderer_is_rejected(tmp_path):
    fixture(tmp_path);(tmp_path/'inputs/renderer.py').write_text('changed')
    with pytest.raises(ValueError,match='frozen input/source changed'):module.verify(tmp_path)


def test_reduced_budget_is_rejected(tmp_path):
    fixture(tmp_path);p=tmp_path/'canvas_1_slots/A_generated.json';r=json.loads(p.read_text());r['steps_completed']=4;p.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='full fixed'):module.verify(tmp_path)


def test_historical_image_seed_is_rejected(tmp_path):
    fixture(tmp_path);p=tmp_path/'canvas_1_slots/A/original_target_generation_request.json';r=json.loads(p.read_text());r['seed_sha256']='forbidden';p.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='historical target seed'):module.verify(tmp_path)
