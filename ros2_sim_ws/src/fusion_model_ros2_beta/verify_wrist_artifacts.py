"""Verify every hashed artifact, independently of the acceptance decision.

Run on the frozen Ubuntu evaluation directory. A valid hash inventory does
not make a failed visual/robot acceptance pass.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(root):
    root=Path(root).resolve()
    report_path=root/'verification.json'
    declared=(root/'verification.sha256').read_text().split()
    if len(declared)!=2 or declared[1]!='verification.json':
        raise ValueError('invalid verification.sha256')
    if digest(report_path)!=declared[0]:
        raise ValueError('verification.json SHA256 mismatch')
    report=json.loads(report_path.read_text())
    if not report.get('complete'):
        raise ValueError('full paired experiment is incomplete')
    inventory=report['sha256']
    if not inventory:
        raise ValueError('empty artifact inventory')
    for relative,expected in inventory.items():
        path=(root/relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError('artifact path escapes evaluation root: '+relative)
        if not path.is_file() or digest(path)!=expected:
            raise ValueError('artifact SHA256 mismatch: '+relative)
    return {'artifact_sha256_verified':True,'artifact_count':len(inventory),
        'complete':True,'accepted':bool(report['accepted']),
        'note':'hash integrity is not visual or hardware acceptance'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    print(json.dumps(verify(parser.parse_args().output),indent=2))
