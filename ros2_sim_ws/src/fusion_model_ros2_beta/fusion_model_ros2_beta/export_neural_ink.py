"""Run under the model Python environment; export the actual forward ink stream.

The upstream renderer is called unchanged. A subclass observes its transformed
patches, so sampling, dynamics, gamma, network and transmittance are shared.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-root', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    root, output = Path(args.model_root), Path(args.output_dir)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    from fusion_model_ros2_beta.neural_domain import capture_inputs, domain_report
    from models.paper_fusion_renderer import PaperDynamicConfig, PaperFusionRenderer
    from models.geometry import CanvasTransform

    metadata = json.loads((output / 'offline_inversion.json').read_text())
    report = json.loads((output / 'inversion_report.json').read_text())
    calibration = report['forward_calibration']
    command = metadata['command']
    size = int(command[command.index('--image_size') + 1])
    padding = int(command[command.index('--padding') + 1])
    device = command[command.index('--device') + 1]
    def rows(path):
        with Path(path).open(encoding='utf-8-sig', newline='') as f:
            return sorted(csv.DictReader(f), key=lambda r: (int(r['stroke_id']), int(r['point_id'])))
    source = rows(metadata['source_trajectory'])
    poses = rows(output / 'inversion_trajectory.csv')
    physical = rows(output / 'physical_trajectory.csv')
    if any(int(r['state']) == 3 for r in poses):
        raise ValueError('Model export requires contact-only source trajectories')
    bounds = np.array([[float(r['x']), float(r['y'])] for r in source])
    low, high = bounds.min(axis=0), bounds.max(axis=0)
    span = max(high-low)
    centre = (low+high)/2
    metres = float(metadata['reference_font_size_m']) / span
    transform = CanvasTransform(low[0], high[0], low[1], high[1], size, padding)
    xy = np.array([transform.map_point(float(r['x']), float(r['y'])) for r in poses], dtype=np.float32)
    posture = np.array([[float(r[k]) for k in ('z','alpha','beta')] for r in poses], dtype=np.float32)
    ids = np.array([int(r['stroke_id']) for r in poses], dtype=np.int64)
    gamma = np.array([float(r['gamma']) for r in poses], dtype=np.float32)
    dynamic = PaperDynamicConfig(
        width_inertia=calibration['width_inertia_Kw'], drag_inertia=calibration['drag_inertia_Kd'],
        calibration_profile=calibration['dynamic_profile'],
        offset_transfer_scale=calibration['offset_transfer_scale'],
        pixels_per_model_unit=calibration['pixels_per_model_unit'],
        inverse_regularization=calibration['pose_inverse_regularization'],
        patch_floor=calibration['patch_floor'], footprint_scale=calibration['footprint_scale'],
        footprint_longitudinal_scale=calibration['footprint_longitudinal_scale'],
        footprint_transverse_scale=calibration['footprint_transverse_scale'],
        render_max_step_px=calibration['render_max_step_px'],
        fused_pose_from_height='--fused_pose_from_height' in command,
    )
    class ObservedRenderer(PaperFusionRenderer):
        def _rotate_about(self, *a, **kw):
            patches = super()._rotate_about(*a, **kw)
            self.observed.append(patches.detach().cpu().numpy()[:, 0])
            return patches
    renderer = ObservedRenderer.from_checkpoint(metadata['checkpoint'], device=device, image_size=size, dynamic=dynamic)
    renderer.observed = []
    tensors = [torch.as_tensor(v, device=device) for v in (xy, posture, ids, gamma)]
    with torch.no_grad(), capture_inputs(renderer) as actual_inputs:
        reference = renderer(*tensors)[0, 0].cpu().numpy()
        dense_xy, _, dense_ids = renderer.densify_for_rendering(*tensors[:3])
    training_domain = domain_report(torch.cat(actual_inputs).cpu().numpy(), renderer.input_normalization)
    patches = np.concatenate(renderer.observed)
    frames = 1.0 - np.cumprod(np.maximum(1.0-patches, 1e-6), axis=0)
    error = float(np.max(np.abs(frames[-1]-reference)))
    if error > 2e-6:
        raise RuntimeError(f'forward ink stream mismatch: {error}')
    pixel_origin = (np.array(transform.unmap_point(0, 0))-centre)*metres
    pixel_size = float(metadata['reference_font_size_m'])/(size-2*padding)
    mapped = (np.array([transform.unmap_point(*p) for p in xy])-centre)*metres
    physical_xy = np.array([[float(r['x']),float(r['y'])] for r in physical])
    roundtrip = float(np.max(np.abs(mapped-physical_xy)))
    if roundtrip > 1e-6:
        raise RuntimeError(f'CSV/world coordinate mismatch: {roundtrip}')
    dense_world = (np.array([transform.unmap_point(*p) for p in dense_xy.cpu().numpy()])-centre)*metres
    np.savez_compressed(output/'neural_ink.npz', frames=frames, xy_m=dense_world,
                        stroke_ids=dense_ids.cpu().numpy(), origin_m=pixel_origin, pixel_size_m=pixel_size)
    Image.fromarray(np.uint8(np.clip(1-reference,0,1)*255)).save(output/'executed_forward.png')
    target = np.array(Image.open(output/'scaled_target.png').convert('L'),dtype=np.float32)/255
    if np.mean([target[0,0],target[-1,0],target[0,-1],target[-1,-1]]) > 0.5:
        target = 1-target
    pred_mask, target_mask = reference>=0.5, target>=0.5
    audit = {
        'renderer': 'PaperFusionRenderer (V16 neural patches)',
        'physical_csv_sha256': sha(output/'physical_trajectory.csv'),
        'checkpoint_sha256': sha(metadata['checkpoint']),
        'renderer_source_sha256': sha(root/'models/paper_fusion_renderer.py'),
        'exporter_sha256': sha(__file__), 'stream_sha256': sha(output/'neural_ink.npz'),
        'frame_count': len(frames), 'image_size': size,
        'metres_per_pixel': pixel_size, 'coordinate_roundtrip_max_error_m': roundtrip,
        'stream_vs_forward_max_abs_error': error,
        'stream_vs_forward_mse': float(np.mean((frames[-1]-reference)**2)),
        'target_metrics': {'iou': float((pred_mask & target_mask).sum()/max((pred_mask | target_mask).sum(),1)),
                           'mse': float(np.mean((reference-target)**2)),
                           'ink_area_ratio': float(pred_mask.sum()/max(target_mask.sum(),1))},
        'scope': 'commanded CSV forward prediction; not measured brush contact or physical calibration',
        'calibration': calibration,
        'training_domain': training_domain,
    }
    (output/'neural_ink_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
