"""Read-only legacy renderer reconstruction; writes only a new experiment folder.

No node is constructed and no joint commands are sent. This is a candidate
reference, not proof of the historical screenshot's exact code revision.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import math
from dataclasses import replace
import numpy as np
from PIL import Image

WS = Path('/home/robot/ros2_ws/src')
sys.path[:0] = [str(WS/'fusion_model_ros2'), str(WS/'fusion_model_ros2_beta')]
from fusion_model_ros2 import brush_trajectory_driver as old
from fusion_model_ros2.paper_brush_model import FlexibleBrushModel, PaperBrushParameters, render_world_footprints
from fusion_model_ros2_beta.offline_fontsize_inversion import scale_initial_pose_csv, export_physical_trajectory
from fusion_model_ros2_beta import paper_brush_model as beta_model


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--historical-beta', action='store_true')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    model = Path('/home/robot/coppeliasim/machine_learning/model')
    params = dict(max_step_m=.002, lift_height_m=.025, brush_bezier_samples=14)
    fake = SimpleNamespace(_param=lambda k: params[k],
                           _brush_model=FlexibleBrushModel(PaperBrushParameters()),
                           get_logger=lambda: SimpleNamespace(info=print))
    if args.historical_beta:
        fake._brush_model = beta_model.FlexibleBrushModel(beta_model.PaperBrushParameters())
    manifest = dict(status='candidate_reference_pending_visual_confirmation',
                    historical_log='/home/robot/.ros/log/python3_2888538_1789719626034.log',
                    mode='reconstructed_pre_offline_beta' if args.historical_beta else 'older_formal_polygon_scaling',
                    caveat='Pre-offline mapping recovered from this task history at ordinal 13778; screenshot/job identity still requires visual confirmation.',
                    driver_sha256=sha(old.__file__), records=[])
    for height in [.32, .40]:
        width = .52 if height == .32 else .8
        cell_height = (height-.03-.045)/2+.03
        paper_polygons = []
        for index, char in enumerate('武汉'):
            directory = output/f'area_{round(height*1000)}'/f'char_u{ord(char):04x}'
            directory.mkdir(parents=True)
            source_csv = model/f'outputs/kaishu_pose_trajectory_batch_v1/char_u{ord(char):04x}/pose_refined.csv'
            with source_csv.open() as f:
                sample_id = next(csv.DictReader(f))['sample_id']
            points, digest = old.load_points(source_csv, char, sample_id)
            mapped = old.map_source_points(points, paper_width=width, paper_height=cell_height,
                paper_z=0., paper_offset_x=0., paper_offset_y=0., margin=.015,
                max_compression=.020, lift_height=.025, reference_trajectory_extent=.29)
            layout_scale = mapped[0].footprint_scale
            if args.historical_beta:
                recovered = []
                bp = fake._brush_model.parameters
                for point in mapped:
                    if point.state == 3:
                        recovered.append(replace(point, footprint_scale=1.))
                        continue
                    dims = bp.dimensions(point.press_depth_mm, point.alpha, point.beta)
                    desired = beta_model.BrushDimensions(dims.tip_length_m*layout_scale,
                        dims.heel_length_m*layout_scale, dims.half_width_m*layout_scale)
                    depth, alpha, beta = bp.inverse_virtual_pose(desired,
                        reference_pose=(point.press_depth_mm, point.alpha, point.beta), max_depth_mm=20.)
                    recovered.append(replace(point, press_depth_mm=depth, z=-depth,
                        alpha=math.copysign(alpha, point.alpha), beta=math.copysign(beta, point.beta),
                        gamma=math.atan2(math.sin(point.gamma),math.cos(point.gamma)), footprint_scale=1.))
                mapped = [replace(p,z=-p.press_depth_mm/1000) if p.state!=3 else p for p in recovered]
            dense = old.BrushTrajectoryDriver._densify_mapped(fake, mapped)
            fake._targets = [SimpleNamespace(point=p) for p in dense]
            old.BrushTrajectoryDriver._build_flexible_footprints(fake)
            polygons = [p for p in fake._footprints if p is not None]
            span = max(max(p.x for p in mapped)-min(p.x for p in mapped),
                       max(p.y for p in mapped)-min(p.y for p in mapped))
            canvas_m = .29*128/96
            target = render_world_footprints(polygons, paper_width_m=canvas_m, paper_height_m=canvas_m,
                       paper_offset_x_m=0., paper_offset_y_m=0., image_size=128, supersampling=4)
            Image.fromarray(np.uint8((1-target)*255)).save(directory/'scaled_target.png')
            high = render_world_footprints(polygons, paper_width_m=canvas_m, paper_height_m=canvas_m,
                       paper_offset_x_m=0., paper_offset_y_m=0., image_size=512, supersampling=2)
            Image.fromarray(np.uint8((1-high)*255)).save(directory/'legacy_reference.png')
            # A same-size, unchanged-posture neural forward control.
            initial = directory/'inversion_trajectory.csv'
            scale_initial_pose_csv(source_csv, initial, span/.29)
            if args.historical_beta:
                with initial.open() as f:
                    reader = csv.DictReader(f)
                    fields, rows = reader.fieldnames, list(reader)
                pose_by_id = {(p.stroke_id,p.point_id): p for p in points}
                recovered_by_id = {(p.stroke_id,p.point_id): m for p,m in zip(points,mapped)}
                for row in rows:
                    p = recovered_by_id[(int(row['stroke_id']),int(row['point_id']))]
                    row.update(z=p.press_depth_mm, alpha=p.alpha, beta=p.beta, gamma=p.gamma)
                with initial.open('w',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
            low = np.array([[p.x,p.y] for p in points]).min(axis=0)
            high_xy = np.array([[p.x,p.y] for p in points]).max(axis=0)
            export_physical_trajectory(initial, directory/'physical_trajectory.csv',
                source_center_x=float((low[0]+high_xy[0])/2), source_center_y=float((low[1]+high_xy[1])/2),
                source_span=float(max(high_xy-low)), reference_font_size_m=.29, font_size_m=span)
            cache = model/'outputs/ros2_v16_fontsize_cache/char_u6b66/font_0120mm_0133057e7973'
            metadata = json.loads((cache/'offline_inversion.json').read_text())
            metadata.update(source_trajectory=str(source_csv), font_size_m=span,
                            character=char, sample_id=sample_id, reference_font_size_m=.29)
            (directory/'offline_inversion.json').write_text(json.dumps(metadata, indent=2))
            report = json.loads((cache/'inversion_report.json').read_text())
            (directory/'inversion_report.json').write_text(json.dumps({'forward_calibration':report['forward_calibration']}))
            shift = np.array([0., height/2-cell_height/2-index*(cell_height+.015)])
            paper_polygons.extend([p+shift for p in polygons])
            manifest['records'].append(dict(character=char, area_height_m=height,
                source=str(source_csv), source_sha256=digest, sample_id=sample_id,
                dominant_extent_m=span, layout_scale=layout_scale, footprint_scale=mapped[0].footprint_scale,
                contact_count=sum(p.state!=3 for p in dense), directory=str(directory)))
        paper = render_world_footprints(paper_polygons, paper_width_m=.8, paper_height_m=.4,
                    paper_offset_x_m=0., paper_offset_y_m=0., image_size=1024, supersampling=1)
        Image.fromarray(np.uint8((1-paper)*255)).resize((1024,512)).save(output/f'legacy_area_{round(height*1000)}.png')
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == '__main__':
    main()
