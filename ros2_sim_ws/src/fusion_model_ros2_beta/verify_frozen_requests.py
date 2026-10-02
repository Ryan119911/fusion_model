"""Check canonical inputs, frozen V16 and inversion source against saved requests."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(root):
    root=Path(root).resolve()
    manifest=root/'inputs/references_wu_verified.json'
    references=json.loads(manifest.read_text())['references']
    record=next(r for r in references if r['character']=='武')
    checked={};cases=[]
    def check(path,expected):
        path=Path(path);actual=sha(path)
        if actual!=expected:raise ValueError('frozen input/source changed: '+str(path))
        checked[str(path.resolve())]=actual
    for slots in (1,3):
        directory=root/f'canvas_{slots}_slots'
        identity=json.loads((directory/'experiment.json').read_text())
        check(manifest,identity['manifest_sha256'])
        check(record['source_trajectory'],identity['source_sha256'])
        check(record['original_target_image'],identity['original_target_sha256'])
        for case in 'ABC':
            data=json.loads((directory/f'{case}_generated.json').read_text())
            candidate=Path(data['candidate'])
            request=json.loads((candidate/'original_target_generation_request.json').read_text())
            check(request['config']['bbsmg_checkpoint'],request['checkpoint_sha256'])
            if data['checkpoint_sha256']!=request['checkpoint_sha256']:
                raise ValueError('candidate checkpoint identity differs')
            if request['original_target_sha256']!=identity['original_target_sha256']:
                raise ValueError('candidate does not use canonical target')
            for path,expected in request['implementation'].items():check(path,expected)
            for filename,key in (('physical_trajectory.csv','physical_csv_sha256'),
                                 ('scaled_initial_pose.csv','initial_pose_sha256'),
                                 ('scaled_target.png','target_sha256')):
                check(candidate/filename,data[key])
            check(request['config']['joint_robot_context'],request['robot_context_sha256'])
            if request['config']['max_steps']!=16 or request['config']['order']!=11 or data['steps_completed']!=16:
                raise ValueError('not the full fixed 16-step / order-11 comparison')
            cases.append({'slots':slots,'case':case,'source_initialization_only':True,
                'historical_seed_used':request.get('seed_sha256') is not None})
            if request.get('seed_sha256') is not None:raise ValueError('historical target seed forbidden')
    return {'verified':True,'checked_request_count':len(cases),'cases':cases,'checked_files_sha256':checked}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=verify(args.output)
    (Path(args.output)/'frozen_input_verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'verified':True,'requests':result['checked_request_count'],
        'fixed_files':len(result['checked_files_sha256'])},indent=2))
