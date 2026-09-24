"""Explicit opt-in backend selection. Never silently downgrade joint requests."""
from dataclasses import replace
import json
from pathlib import Path
from .offline_fontsize_inversion import OfflineFontSizeGenerator
from .joint_fontsize_inversion import JointFontSizeGenerator
from .original_target_inversion import (
    MANIFEST_FORMAT as ORIGINAL_TARGET_MANIFEST_FORMAT,
    OriginalTargetFontSizeGenerator,
)


def _manifest_records(path, expected_format, *, path_fields):
    manifest = Path(path).expanduser().resolve()
    data = json.loads(manifest.read_text(encoding='utf-8'))
    if data.get('format') != expected_format or not isinstance(data.get('references'), list):
        raise ValueError('invalid '+expected_format+' manifest format')
    if not data['references']:
        raise ValueError(expected_format+' manifest is empty')
    references = []
    for item in data['references']:
        record = dict(item)
        for key in path_fields:
            if record.get(key):
                value = Path(record[key]).expanduser()
                record[key] = str((manifest.parent/value).resolve() if not value.is_absolute() else value.resolve())
        references.append(record)
    return references


def make_inversion_backend(config, backend='legacy_fused', reference_manifest='',
                           joint_steps=16, original_target_manifest=''):
    if backend == 'legacy_fused':
        return OfflineFontSizeGenerator(config)
    if backend not in ('joint_A', 'joint_target'):
        raise ValueError('unknown inversion backend: '+str(backend))
    if not isinstance(joint_steps, int) or isinstance(joint_steps, bool) or joint_steps <= 0:
        raise ValueError('joint optimization steps must be a positive integer')
    configured = replace(config, max_steps=joint_steps, optimization_size=128)
    if backend == 'joint_A':
        if not reference_manifest:
            raise ValueError('joint_A requires an explicit A reference manifest')
        references = _manifest_records(
            reference_manifest, 'v16_A_reference_manifest_v1',
            path_fields=('target_image', 'seed_folder'))
        return JointFontSizeGenerator(configured, references)
    if not original_target_manifest:
        raise ValueError('joint_target requires an explicit original-target manifest')
    references = _manifest_records(
        original_target_manifest, ORIGINAL_TARGET_MANIFEST_FORMAT,
        path_fields=('source_trajectory', 'original_target_image',
                     'normalized_target_image', 'target_source_json',
                     'source_annotation_json', 'seed_folder'))
    # The candidate loader requires every dense neural input to be in-domain.
    # Enforce the same constraint during optimization, including ROS callers.
    configured = replace(configured, joint_hard_neural_domain=True)
    return OriginalTargetFontSizeGenerator(configured, references)
