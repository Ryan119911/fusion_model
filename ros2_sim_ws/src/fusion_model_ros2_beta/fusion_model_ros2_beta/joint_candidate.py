"""Explicit experimental joint-pose contract for isolated ROS planning tests.

This loader does not add candidates to the web catalog or promote a cache.
"""
import hashlib
import json
import math
from pathlib import Path
from .trajectory_catalog import TrajectoryEntry

CONTRACT = 'v16_joint_pose_candidate_v1'
FIELDS = {'x', 'y', 'H', 'alpha', 'beta', 'gamma'}
V16_CHECKPOINT_SHA256 = '30d5e0c37dc7b7913c02fea26930babafa46a4a07b5f698520df277a04d5b717'


def load_joint_candidate(folder, *, require_training_domain=False):
    folder = Path(folder)
    meta = json.loads((folder/'offline_inversion.json').read_text())
    audit = json.loads((folder/'neural_ink_audit.json').read_text())
    controls = json.loads((folder/'control_change_audit.json').read_text())
    report = json.loads((folder/'inversion_report.json').read_text())
    if audit.get('checkpoint_sha256') != V16_CHECKPOINT_SHA256:
        raise ValueError('candidate did not use the pinned V16 checkpoint')
    if require_training_domain:
        from .neural_domain import CONTRACT as DOMAIN_CONTRACT, LIMITS
        domain = audit.get('training_domain', {})
        features = domain.get('features', {})
        if (domain.get('contract') != DOMAIN_CONTRACT or domain.get('passed') is not True
                or not isinstance(domain.get('dense_count'), int) or domain['dense_count'] <= 0
                or any(features.get(name, {}).get('outside_count') != 0
                       or features.get(name, {}).get('training_limits') != list(bounds)
                       for name, bounds in LIMITS.items())):
            raise ValueError('candidate missing or failed actual neural training-domain audit')
    if set(report['optimized_fields']) != FIELDS:
        raise ValueError('candidate is not a six-field joint inversion')
    if not controls['candidate_checks']['eligible_for_further_validation']:
        raise ValueError('candidate failed preliminary geometry checks')
    selection = report['lm']['diagnostics']['checkpoint_selection']
    if selection['metric'] != 'regularized_cost_with_trajectory_constraints':
        raise ValueError('candidate selection omitted trajectory constraints')
    selected, terminal = (float(selection[k]) for k in
                          ('selected_regularized_cost', 'terminal_regularized_cost'))
    if not all(math.isfinite(v) for v in (selected, terminal)) or selected > terminal+1e-5*max(1, abs(terminal)):
        raise ValueError('candidate has an inconsistent selected objective')
    for filename, key in [('physical_trajectory.csv', 'physical_csv_sha256'),
                          ('neural_ink.npz', 'stream_sha256')]:
        if hashlib.sha256((folder/filename).read_bytes()).hexdigest() != audit[key]:
            raise ValueError('candidate content hash mismatch: '+filename)
    error = float(audit['stream_vs_forward_max_abs_error'])
    if not math.isfinite(error) or error > 1e-5:
        raise ValueError('candidate model-forward parity failed')
    size = float(meta['font_size_m'])
    if not math.isfinite(size) or size <= 0:
        raise ValueError('invalid candidate physical size')
    return TrajectoryEntry(character=meta['character'], status='ready',
        trajectory_csv=folder/'physical_trajectory.csv', target_image=folder/'scaled_target.png',
        sample_id=meta['sample_id'], output_dir=folder,
        quality='experimental_pending_robot_and_visual_validation', metadata=dict(
            physical_pose_contract=CONTRACT, optimized_fields=sorted(FIELDS),
            checkpoint_sha256=V16_CHECKPOINT_SHA256,
            coordinate_frame='glyph_center_m', xy_unit='m', font_size_m=size,
            offline_full_pose_inversion=True, gamma_relative_to_path=False,
            neural_ink_path=str(folder/'neural_ink.npz'),
            neural_ink_audit=str(folder/'neural_ink_audit.json')))


def is_joint_contract(metadata):
    return (metadata.get('physical_pose_contract') == CONTRACT
            and set(metadata.get('optimized_fields', [])) == FIELDS)


class JointCandidateCache:
    """Read-only, opt-in interface for isolated testing of prepared inversions.

    Records explicitly pin the original source CSV hash and candidate folder.
    This is NOT an online generator or a visual acceptance decision. A miss
    fails closed: no legacy fallback, nearest-size selection or pose scaling.
    """

    def __init__(self, records):
        self._records = []
        keys = set()
        for record in records:
            folder = Path(record['folder']).resolve()
            source_hash = record['source_sha256']
            if (not isinstance(source_hash, str) or len(source_hash) != 64
                    or any(c not in '0123456789abcdef' for c in source_hash)):
                raise ValueError('invalid pinned source SHA256')
            entry = load_joint_candidate(folder)
            key = (entry.character, entry.sample_id, entry.metadata['font_size_m'], source_hash)
            if key in keys:
                raise ValueError('ambiguous candidate selection; explicitly choose one result')
            keys.add(key)
            self._records.append((key, folder))

    def identity(self):
        return dict(model_weight_version='V16', checkpoint_sha256=V16_CHECKPOINT_SHA256,
                    inversion_pipeline=CONTRACT, optimized_fields=sorted(FIELDS),
                    mode='experimental_precomputed_only', candidate_count=len(self._records),
                    runtime_scaling=False, legacy_fallback=False,
                    visual_accepted=False, online_generation_available=False)

    def generate(self, source, font_size_m, progress=None):
        size = float(font_size_m)
        if not math.isfinite(size) or size <= 0:
            raise ValueError('invalid requested physical size')
        if source.trajectory_csv is None:
            raise ValueError('source trajectory is required')
        digest = hashlib.sha256(Path(source.trajectory_csv).read_bytes()).hexdigest()
        matches = [folder for (char, sample, candidate_size, source_hash), folder in self._records
                   if char == source.character and sample == source.sample_id
                   and abs(candidate_size-size) <= 1e-9 and source_hash == digest]
        if len(matches) != 1:
            raise ValueError('no unique validated six-field inversion for this source and physical size; '
                             'a new model inversion is required (no scaling or legacy fallback)')
        # Recheck file hashes on EVERY request, not just when the interface starts.
        result = load_joint_candidate(matches[0])
        if (result.character != source.character or result.sample_id != source.sample_id
                or abs(result.metadata['font_size_m']-size) > 1e-9):
            raise ValueError('candidate identity changed after indexing')
        if progress:
            progress(f'“{source.character}” {size*1000:.3f} mm 六维反演候选已校验（实验）')
        return result
