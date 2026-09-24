"""Opt-in six-field generation from an explicitly pinned reconstructed A target.

Not enabled in the production node. All sizes are generated in a fixed model
coordinate frame; ROS receives physical controls and unmodified forward ink.
References are explicit because an unverified catalog image is not an A target.
"""
from dataclasses import asdict
import csv
import json
import math
import os
from pathlib import Path
import subprocess
from PIL import Image, ImageChops

from .offline_fontsize_inversion import (
    OfflineFontSizeGenerator, _sha256, _read_pose_rows, _xy_reference,
    scale_initial_pose_csv, export_physical_trajectory)
from .joint_candidate import load_joint_candidate, FIELDS, V16_CHECKPOINT_SHA256
from .inversion_candidate_checks import check_candidate

INTERFACE = 'v16_A_joint_fontsize_v1'


def scale_reference_canvas(source, destination, ratio, image_size=128):
    """Resize the entire A canvas, not its crop; reject clipped foreground."""
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError('invalid target scale')
    with Image.open(source) as opened:
        image = opened.convert('L')
    if image.size != (image_size, image_size):
        raise ValueError('reference must use the declared fixed model canvas')
    corners = [image.getpixel(p) for p in ((0,0),(0,image_size-1),(image_size-1,0),(image_size-1,image_size-1))]
    background = sorted(corners)[2]
    extent = max(1, round(image_size*ratio))
    if extent > 4096:
        raise ValueError('unsupported reference enlargement')
    resampling = getattr(Image, 'Resampling', Image).LANCZOS
    scaled = image.resize((extent, extent), resampling)
    canvas = Image.new('L', image.size, background)
    if extent <= image_size:
        canvas.paste(scaled, ((image_size-extent)//2, (image_size-extent)//2))
    else:
        offset = (extent-image_size)//2
        foreground = ImageChops.difference(scaled, Image.new('L', scaled.size, background))
        bounds = foreground.point(lambda v: 255 if v > 8 else 0).getbbox()
        if bounds and (bounds[0] < offset or bounds[1] < offset or
                       bounds[2] > offset+image_size or bounds[3] > offset+image_size):
            raise ValueError('scaled target would clip ink at the model canvas boundary')
        canvas = scaled.crop((offset, offset, offset+image_size, offset+image_size))
    canvas.save(destination)


def build_joint_command(config, source, initial, target, output, character, sample_id):
    pixel_size = config.reference_font_size_m/(config.image_size-2*config.padding)
    # Preserve the documented legacy simulation unit bridge; not size-dependent.
    axis_scale = .01/(20*pixel_size)
    pairs = {
        '--trajectory_csv':source, '--initial_pose_csv':initial, '--initial_pose_xy_source':'csv',
        '--target_image':target, '--bbsmg_ckpt':config.bbsmg_checkpoint,
        '--character':character, '--sample_id':sample_id, '--output_dir':output,
        '--output_stem':'inversion', '--device':config.device,
        '--image_size':config.image_size, '--padding':config.padding, '--order':config.order,
        '--max_steps':config.max_steps, '--damping':.05, '--optimization_size':config.image_size,
        '--point_batch_size':config.point_batch_size, '--pixel_weight':config.joint_foreground_weight,
        '--field_mode':'all', '--xy_max_offset_px':2.0,
        '--xy_segment_length_weight':3.0, '--xy_segment_direction_weight':10.0,
        '--xy_target_skeleton_weight':config.joint_xy_target_skeleton_weight,
        '--xy_target_skeleton_max_distance_px':config.joint_xy_target_skeleton_max_distance_px,
        '--xy_target_skeleton_threshold':config.joint_xy_target_skeleton_threshold,
        '--h_smoothness_weight':.2, '--h_point_velocity_weight':10.0,
        '--h_point_acceleration_weight':10.0, '--h_prior_weight':.001,
        '--alpha_smoothness_weight':1.0, '--beta_smoothness_weight':1.0, '--gamma_smoothness_weight':1.0,
        '--beta_angle_point_weight':10.0, '--beta_depth_spatial_weight':10.0,
        '--beta_depth_mm_per_pixel':1000*pixel_size, '--beta_difference_scheme':'central',
        '--beta_neural_domain_weight':100.0,
        '--dynamic_profile':'wang2020_figure4_digitized_v1', '--pixels_per_model_unit':20.0,
        '--footprint_longitudinal_scale':axis_scale, '--footprint_transverse_scale':axis_scale,
        '--render_max_step_px':2.0,
    }
    command = [str(config.python_executable), '-u', str(Path(__file__).with_name('run_joint_inversion.py'))]
    for key, value in pairs.items():
        command.extend((key, str(value)))
    return command + ['--optimize_xy', '--optimize_gamma', '--cap_order_to_points', '--beta_project_posture_logits'] + (
        ['--beta_hard_neural_domain'] if config.joint_hard_neural_domain else [])


class JointFontSizeGenerator:
    """API-compatible generator, intentionally opt-in until visual/ROS acceptance.

    Each reference record pins character, sample_id, source_sha256, target_image,
    target_sha256 and font_size_m. Optional seed_folder is a validated joint
    candidate at the reference size, never a runtime-scaled replacement output.
    """
    def __init__(self, config, references):
        self._exporter = OfflineFontSizeGenerator(config)  # validation/export only
        self.config = config
        if self._exporter._checkpoint_sha256 != V16_CHECKPOINT_SHA256:
            raise ValueError('joint generator requires the pinned V16 checkpoint')
        if (config.image_size, config.padding, config.reference_font_size_m) != (128,16,.29):
            raise ValueError('A references currently require the validated 128/16/0.29m model frame')
        if config.max_steps < 1:
            raise ValueError('max_steps must be positive')
        if not math.isfinite(config.joint_foreground_weight) or not 0 <= config.joint_foreground_weight <= 12:
            raise ValueError('joint foreground weight must be finite and between 0 and 12')
        self.references = {}
        for item in references:
            record = dict(item)
            key = (record['character'], record['sample_id'], record['source_sha256'])
            if key in self.references:
                raise ValueError('ambiguous A reference')
            size = float(record['font_size_m'])
            if not math.isfinite(size) or not 0 < size <= config.reference_font_size_m:
                raise ValueError('invalid A reference physical size')
            if _sha256(Path(record['target_image'])) != record['target_sha256']:
                raise ValueError('A reference target hash mismatch')
            self.references[key] = record

    def identity(self):
        return dict(interface_version=INTERFACE, model_weight_version='V16',
                    foreground_loss_weight=self.config.joint_foreground_weight,
                    inversion_pipeline='V16 six-field A-reference inversion',
                    checkpoint_sha256=V16_CHECKPOINT_SHA256, optimized_fields=sorted(FIELDS),
                    fixed_fields=[], derived_fields=[], runtime_scaling=False,
                    mode='experimental_joint_generation', legacy_fallback=False,
                    visual_accepted=False, reference_count=len(self.references))

    def generate(self, entry, font_size_m, progress=None):
        size = float(font_size_m)
        if not math.isfinite(size) or not 0 < size <= self.config.reference_font_size_m:
            raise ValueError('physical size must be positive and at most 290 mm')
        if entry.trajectory_csv is None or not entry.sample_id:
            raise ValueError('missing source trajectory or sample id')
        source = Path(entry.trajectory_csv).resolve()
        source_digest = _sha256(source)
        record = self.references.get((entry.character, entry.sample_id, source_digest))
        if record is None:
            raise ValueError('no pinned A reference for this source; no catalog/legacy fallback')
        target = Path(record['target_image']).resolve()
        if _sha256(target) != record['target_sha256']:
            raise ValueError('A reference target changed')
        fields, original = _read_pose_rows(source)
        if any(r['character'] != entry.character or r['sample_id'] != entry.sample_id or int(r['state']) == 3
               for r in original):
            raise ValueError('source must contain only this sample and contact trajectories')
        seed = None
        seed_size = None
        if record.get('seed_folder'):
            seed_entry = load_joint_candidate(record['seed_folder'])
            if (seed_entry.character, seed_entry.sample_id) != (entry.character, entry.sample_id):
                raise ValueError('seed identity mismatch')
            seed_size = seed_entry.metadata['font_size_m']
            seed = Path(record['seed_folder'])/'inversion_trajectory.csv'
            seed_meta = json.loads((Path(record['seed_folder'])/'offline_inversion.json').read_text())
            if Path(seed_meta['source_trajectory']).resolve() != source:
                raise ValueError('seed source coordinate frame mismatch')
        implementation_files = [Path(__file__), Path(__file__).with_name('joint_optimizer.py'),
            Path(__file__).with_name('domain_seed.py'),
            Path(__file__).with_name('constrained_step.py'),
            Path(__file__).with_name('neural_domain.py'), Path(__file__).with_name('joint_candidate.py'),
            Path(__file__).with_name('run_joint_inversion.py'), Path(__file__).with_name('export_neural_ink.py'),
            self.config.model_root/'models/paper_fusion_renderer.py',
            self.config.model_root/'tools/invert_paper_trajectory.py']
        identity = dict(interface=INTERFACE, source_sha256=source_digest,
            target_sha256=record['target_sha256'], reference_size_m=record['font_size_m'],
            requested_size_m=size, checkpoint_sha256=V16_CHECKPOINT_SHA256,
            seed_sha256=_sha256(seed) if seed else None, seed_size_m=seed_size,
            config={k:str(v) if isinstance(v, Path) else v for k,v in asdict(self.config).items()},
            implementation={str(p):_sha256(p) for p in implementation_files})
        import hashlib
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        output = self.config.cache_root/f'char_u{ord(entry.character):04x}'/f'joint_{key[:24]}'
        marker = output/'joint_generation_complete.json'
        if output.exists():
            if not marker.is_file() or json.loads(marker.read_text()) != identity:
                raise RuntimeError(f'incomplete or conflicting generation; inspect {output}')
            return load_joint_candidate(output, require_training_domain=True)
        output.mkdir(parents=True, exist_ok=False)  # exclusive ownership; never overwrite a running request
        (output/'joint_generation_request.json').write_text(json.dumps(identity, indent=2))
        initial, scaled_target = output/'scaled_initial_pose.csv', output/'scaled_target.png'
        cx, cy, span = scale_initial_pose_csv(source, initial, size/self.config.reference_font_size_m)
        if seed:
            seed_fields, rows = _read_pose_rows(seed)
            if [(r['stroke_id'],r['point_id']) for r in rows] != [(r['stroke_id'],r['point_id']) for r in original]:
                raise ValueError('seed sample sequence mismatch')
            # Scale ONLY the initial guess about the original frame centre.
            # H and all rotations remain seed values, then all six fields optimize.
            for row in rows:
                row['x'] = repr(cx+(float(row['x'])-cx)*size/seed_size)
                row['y'] = repr(cy+(float(row['y'])-cy)*size/seed_size)
            with initial.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=seed_fields)
                writer.writeheader(); writer.writerows(rows)
        scale_reference_canvas(target, scaled_target, size/float(record['font_size_m']))
        command = build_joint_command(self.config, source, initial, scaled_target, output, entry.character, entry.sample_id)
        if progress:
            progress(f'正在对“{entry.character}” {size*1000:.3f} mm 进行 V16 六维模型反演')
        environment = dict(os.environ, BETA_INVERSION_MODEL_ROOT=str(self.config.model_root), PYTHONUNBUFFERED='1')
        with (output/'joint_process.log').open('w') as log:
            subprocess.run(command, cwd=self.config.model_root, env=environment, stdout=log,
                           stderr=subprocess.STDOUT, timeout=self.config.timeout_s, check=True)
        report = json.loads((output/'inversion_report.json').read_text())
        if set(report['optimized_fields']) != FIELDS or '--fused_pose_from_height' in command:
            raise ValueError('model did not return six optimized fields')
        _, before = _read_pose_rows(initial)
        _, after = _read_pose_rows(output/'inversion_trajectory.csv')
        checks = check_candidate(before, after)
        if not checks['eligible_for_further_validation'] or [r['state'] for r in before] != [r['state'] for r in after]:
            raise ValueError('generated trajectory failed geometry/state checks')
        export_physical_trajectory(output/'inversion_trajectory.csv', output/'physical_trajectory.csv',
            source_center_x=cx, source_center_y=cy, source_span=span,
            reference_font_size_m=self.config.reference_font_size_m, font_size_m=size,
            inversion_label='v16_joint_pose_metric_units')
        metadata = dict(format=INTERFACE, character=entry.character, sample_id=entry.sample_id,
            font_size_m=size, reference_font_size_m=self.config.reference_font_size_m,
            source_trajectory=str(source), source_sha256=source_digest, checkpoint=str(self.config.bbsmg_checkpoint),
            checkpoint_sha256=V16_CHECKPOINT_SHA256, command=command, optimized_fields=sorted(FIELDS),
            fixed_fields=[], derived_fields=[], quality_metrics=report['metrics'],
            A_reference=record, visual_accepted=False)
        (output/'offline_inversion.json').write_text(json.dumps(metadata, indent=2))
        (output/'control_change_audit.json').write_text(json.dumps(dict(candidate_checks=checks,
            optimized_fields=sorted(FIELDS), metrics=report['metrics']), indent=2))
        self._exporter._ensure_neural_ink(output)
        result = load_joint_candidate(output, require_training_domain=True)
        # Only validated CSV/forward parity constitutes completion, not subprocess exit.
        with marker.open('x') as handle:
            json.dump(identity, handle, indent=2)
        return result
