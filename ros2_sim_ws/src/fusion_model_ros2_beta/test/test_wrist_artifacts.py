"""Integrity checks may pass while the engineering result remains rejected."""
import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('verify_wrist_artifacts',
    Path(__file__).resolve().parents[1]/'verify_wrist_artifacts.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(root):
    p=root/'trajectory.csv';p.write_text('q4,q5,q6\n0,0,6.28\n')
    report={'complete':True,'accepted':False,'sha256':{'trajectory.csv':module.digest(p)}}
    save(root,report)
    return report


def save(root,report):
    p=root/'verification.json';p.write_text(json.dumps(report))
    (root/'verification.sha256').write_text(module.digest(p)+'  verification.json\n')


def test_intact_rejected_report_is_not_promoted(tmp_path):
    fixture(tmp_path)
    result=module.verify(tmp_path)
    assert result['artifact_sha256_verified'] and not result['accepted']


def test_changed_trajectory_is_rejected(tmp_path):
    fixture(tmp_path);(tmp_path/'trajectory.csv').write_text('different')
    with pytest.raises(ValueError,match='artifact SHA256'):module.verify(tmp_path)


def test_incomplete_report_is_rejected(tmp_path):
    r=fixture(tmp_path);r['complete']=False;save(tmp_path,r)
    with pytest.raises(ValueError,match='incomplete'):module.verify(tmp_path)


def test_path_escape_is_rejected(tmp_path):
    r=fixture(tmp_path);r['sha256']['../outside.csv']='0'*64;save(tmp_path,r)
    with pytest.raises(ValueError,match='escapes'):module.verify(tmp_path)


def test_changed_verification_report_is_rejected(tmp_path):
    fixture(tmp_path);(tmp_path/'verification.json').write_text('{}')
    with pytest.raises(ValueError,match='verification.json SHA256'):module.verify(tmp_path)
