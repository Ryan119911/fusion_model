#!/usr/bin/env python3
"""Save real Mech-Eye pixels/points and measure a paper ROI in metric units.

Hardware imports are inside capture_raw; offline measurement never imports ROS.
"""
import argparse
import csv
import datetime as dt
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def stamp(header):
    return int(header.stamp.sec)*1000000000+int(header.stamp.nanosec)


def image_array(msg):
    enc = msg.encoding.lower(); dtype = None; channels = 1
    if enc in ('mono8', '8uc1'): dtype = np.dtype('u1')
    elif enc in ('bgr8', 'rgb8'): dtype = np.dtype('u1'); channels = 3
    elif enc in ('mono16', '16uc1'): dtype = np.dtype('>u2' if msg.is_bigendian else '<u2')
    else: raise ValueError('unsupported raw image encoding: '+enc)
    if msg.step<msg.width*channels*dtype.itemsize or len(msg.data)!=msg.step*msg.height: raise ValueError('image buffer/stride mismatch')
    shape = (msg.height, msg.width) if channels==1 else (msg.height, msg.width, channels)
    strides = (msg.step, dtype.itemsize) if channels==1 else (msg.step, channels, 1)
    return np.ndarray(shape, dtype=dtype, buffer=bytes(msg.data), strides=strides).copy()


def cloud_xyz(msg):
    fields = {f.name: f for f in msg.fields}
    if not all(k in fields and fields[k].datatype==7 for k in 'xyz'): raise ValueError('FLOAT32 xyz required')
    if msg.row_step<msg.width*msg.point_step or len(msg.data)!=msg.row_step*msg.height: raise ValueError('invalid cloud row stride')
    dtype = np.dtype('>f4' if msg.is_bigendian else '<f4')
    return np.stack([np.ndarray((msg.height, msg.width), dtype=dtype, buffer=bytes(msg.data),
                     offset=fields[k].offset, strides=(msg.row_step, msg.point_step)).astype(np.float32) for k in 'xyz'], axis=-1)


def info_record(info):
    return {'frame_id': info.header.frame_id, 'stamp_ns': stamp(info.header),
            'width': info.width, 'height': info.height, 'k': list(info.k), 'd': list(info.d),
            'r': list(info.r), 'p': list(info.p), 'distortion_model': info.distortion_model,
            'binning_x': info.binning_x, 'binning_y': info.binning_y,
            'roi': {k: getattr(info.roi, k) for k in ('x_offset', 'y_offset', 'height', 'width', 'do_rectify')}}


def persist_message(out, name, msg, serialize):
    """CDR + original bytes and metadata, and usable lossless arrays."""
    out = Path(out)
    with (out/(name+'.cdr')).open('xb') as f: f.write(serialize(msg))
    with (out/(name+'.bin')).open('xb') as f: f.write(bytes(msg.data))
    common = {'frame_id': msg.header.frame_id, 'stamp_ns': stamp(msg.header),
              'width': msg.width, 'height': msg.height, 'is_bigendian': bool(msg.is_bigendian),
              'bytes': len(msg.data), 'received_host_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
    if hasattr(msg, 'encoding'):
        common.update(encoding=msg.encoding, step=msg.step)
        pixels = image_array(msg)
        if msg.encoding.lower()=='rgb8': pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(out/(name+'.png')), pixels): raise OSError('image export failed')
        if pixels.ndim==3:
            if np.max(np.ptp(pixels.astype(int), axis=2))>1: raise ValueError('PRO XS output is not monochrome as expected')
            mono = pixels[:, :, 0]
        else: mono = pixels
        if not cv2.imwrite(str(out/(name+'_mono.png')), mono): raise OSError('mono export failed')
    else:
        common.update(point_step=msg.point_step, row_step=msg.row_step, is_dense=bool(msg.is_dense),
                      fields=[{'name': f.name, 'offset': f.offset, 'datatype': f.datatype, 'count': f.count} for f in msg.fields])
        xyz = cloud_xyz(msg); np.save(out/(name+'_xyz_m.npy'), xyz)
        # Camera-frame Z depth derived from the original organized XYZ, in metres.
        np.save(out/(name+'_depth_z_m.npy'), xyz[:, :, 2])
        common['depth_kind'] = 'camera-frame Z derived from original organized XYZ; no image alignment assumption'
    save(out/(name+'.json'), common)
    return common


def capture_raw(sample, output, stationary_record, timeout=20.):
    # Reuse the existing ROS clients and service definitions. Never reinstall SDK.
    import importlib.util
    import rclpy
    from rclpy.serialization import serialize_message
    from sensor_msgs.msg import CameraInfo
    module_path = sample['camera_capture_reuse_script']
    spec = importlib.util.spec_from_file_location('existing_mecheye_validator', module_path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    rclpy.init(args=[]); node = module.CaptureValidator(); info = []
    node.create_subscription(CameraInfo, '/mechmind/camera_info', lambda m: info.append(m), 10)
    out = Path(output)/'camera'; out.mkdir(exist_ok=False)
    records = {'sample_id': sample['sample_id'], 'stationary_reference': stationary_record,
               'captures_are_sequential': True, 'image_cloud_exact_synchronization_claimed': False,
               'source_calibrations': sample['source_files'], 'captures': {}}
    try:
        for key, client, service in [('image', node.color_client, module.CaptureColorImage),
                                     ('cloud', node.cloud_client, module.CapturePointCloud),
                                     ('textured_cloud', node.textured_client, module.CaptureTexturedPointCloud)]:
            request_ns = node.get_clock().now().nanoseconds
            msg, _ = node.capture(key, client, service, timeout)
            if stamp(msg.header)<request_ns-50000000: raise RuntimeError('stale captured '+key)
            records['captures'][key] = persist_message(out, key, msg, serialize_message)
            records['captures'][key]['request_ros_ns'] = request_ns
        deadline = time.monotonic()+timeout
        while not info and time.monotonic()<deadline: rclpy.spin_once(node, timeout_sec=.1)
        if not info: raise RuntimeError('no CameraInfo; raw data saved, metric measurement unavailable')
        image = records['captures']['image']
        matching = [m for m in info if m.width==image['width'] and m.height==image['height']
                    and m.header.frame_id==image['frame_id']]
        if not matching: raise RuntimeError('CameraInfo frame/dimensions do not match image')
        camera_info = info_record(matching[-1]); save(out/'camera_info.json', camera_info)
        with (out/'camera_info.cdr').open('xb') as f: f.write(serialize_message(matching[-1]))
        # Do not assume point_cloud and color_map frames coincide. Retain TF if present.
        import tf2_ros
        from rclpy.time import Time
        tf_buffer = tf2_ros.Buffer(); listener = tf2_ros.TransformListener(tf_buffer, node)
        transforms = {}
        for key in ('cloud', 'textured_cloud'):
            source = records['captures'][key]['frame_id']; target = image['frame_id']
            if source==target: transforms[key] = np.eye(4).tolist(); continue
            deadline = time.monotonic()+2.
            while time.monotonic()<deadline:
                rclpy.spin_once(node, timeout_sec=.05)
                try:
                    tf = tf_buffer.lookup_transform(target, source, Time(nanoseconds=records['captures'][key]['stamp_ns']))
                    q = tf.transform.rotation; t = tf.transform.translation
                    m = np.eye(4); m[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix(); m[:3, 3] = [t.x, t.y, t.z]
                    transforms[key] = m.tolist(); break
                except tf2_ros.TransformException: continue
            if key not in transforms: transforms[key] = None
        records['image_from_cloud_frames'] = transforms
        records['status'] = 'raw_saved'
    except BaseException as exc:
        records.update(status='capture_failed', error=type(exc).__name__+': '+str(exc)); raise
    finally:
        save(out/'capture.json', records); node.destroy_node(); rclpy.shutdown()
    return records


def transform_from_pose(p):
    m = np.eye(4); m[:3, :3] = Rotation.from_rotvec(p[3:]).as_matrix(); m[:3, 3] = p[:3]
    return m


def metric_rectification(image, info, base_cam, origin, axes, roi_mm, mm_per_pixel):
    """Project a metric paper grid through calibrated distortion into raw pixels."""
    u, v, _ = np.asarray(axes); origin = np.asarray(origin)
    if info['distortion_model'] not in ('plumb_bob', 'rational_polynomial', ''): raise ValueError('unsupported camera distortion model')
    if info.get('binning_x', 0)>1 or info.get('binning_y', 0)>1: raise ValueError('binned image needs corresponding adjusted K')
    if info.get('roi', {}).get('x_offset', 0) or info.get('roi', {}).get('y_offset', 0): raise ValueError('cropped CameraInfo ROI requires adjusted K')
    if image.shape[:2]!=(info['height'], info['width']): raise ValueError('image/CameraInfo dimensions differ')
    k = np.array(info['k'], dtype=float).reshape(3, 3); distortion = np.array(info['d'], dtype=float)
    if not np.isfinite(k).all() or min(k[0, 0], k[1, 1])<=0: raise ValueError('invalid camera calibration')
    xmin, xmax, ymin, ymax = roi_mm
    if mm_per_pixel<=0 or xmin>=xmax or ymin>=ymax: raise ValueError('invalid metric ROI')
    xs = np.arange(xmin, xmax+mm_per_pixel/2, mm_per_pixel)
    ys = np.arange(ymax, ymin-mm_per_pixel/2, -mm_per_pixel)
    xx, yy = np.meshgrid(xs, ys)
    base_points = origin+xx[:, :, None]/1000*u+yy[:, :, None]/1000*v
    camera_points = (base_points-base_cam[:3, 3])@base_cam[:3, :3]
    if np.any(camera_points[:, :, 2]<=0): raise ValueError('ROI is behind camera')
    projected, _ = cv2.projectPoints(camera_points.reshape(-1, 3), np.zeros(3), np.zeros(3), k, distortion)
    uv = projected.reshape(*xx.shape, 2).astype(np.float32)
    inside = (uv[:, :, 0]>=1)&(uv[:, :, 0]<info['width']-2)&(uv[:, :, 1]>=1)&(uv[:, :, 1]<info['height']-2)
    if not np.all(inside): raise ValueError('paper ROI outside calibrated image field of view')
    rectified = cv2.remap(image, uv[:, :, 0], uv[:, :, 1], cv2.INTER_LINEAR)
    return rectified, xs, ys


def contour_metrics(gray, xs, ys, threshold=None):
    if gray.ndim!=2: raise ValueError('monochrome image required')
    if threshold is None: level, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    else: level, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    components = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA]>=8]
    if len(components)!=1: raise ValueError('expected one isolated ink component; found '+str(len(components)))
    chosen = (labels==components[0]).astype(np.uint8)*255
    if np.any(chosen[[0, -1]]) or np.any(chosen[:, [0, -1]]): raise ValueError('ink touches ROI edge')
    contours, _ = cv2.findContours(chosen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pixels = contours[0][:, 0, :]; xy = np.column_stack((xs[pixels[:, 0]], ys[pixels[:, 1]]))
    (cx, cy), (w, h), angle = cv2.minAreaRect(xy.astype(np.float32))
    # Principal rectangle; not a skeleton-length claim for curved strokes.
    length, width = max(w, h), min(w, h)
    if h>w: angle += 90.
    return {'threshold': float(level), 'width_mm': float(width), 'length_mm': float(length),
            'length_definition': 'major extent of minimum-area bounding rectangle',
            'contour_area_mm2': float(cv2.contourArea(xy.astype(np.float32))),
            'center_paper_mm': [cx, cy], 'angle_paper_deg': float(angle%180), 'contour_paper_mm': xy.tolist()}, chosen


def measure(sample_file, run_dir, output, roi_mm=None, mm_per_pixel=.1):
    run_dir, output = Path(run_dir), Path(output); sample = json.loads(Path(sample_file).read_text())
    if sample!=json.loads((run_dir/'sample.json').read_text()):raise ValueError('measurement sample differs from acquisition configuration')
    capture = json.loads((run_dir/'camera/capture.json').read_text())
    if capture['status']!='raw_saved': raise ValueError('capture incomplete')
    result=json.loads((run_dir/'result.json').read_text())
    if result['status']!='captured': raise ValueError('post-capture stationary check missing/failed')
    if result['sample_id']!=sample['sample_id'] or capture['sample_id']!=sample['sample_id']:raise ValueError('capture/measurement sample id mismatch')
    start=np.array(sample.get('paper_start_uv_mm',[0.,0.]));length=sample.get('line_length_mm',10.)
    if roi_mm is None:roi_mm=[start[0]-15,start[0]+length+15,start[1]-20,start[1]+20]
    output.mkdir(parents=True, exist_ok=False)
    info = json.loads((run_dir/'camera/camera_info.json').read_text())
    image = cv2.imread(str(run_dir/'camera/image_mono.png'), cv2.IMREAD_UNCHANGED)
    if image is None: raise ValueError('raw monochrome image missing')
    stationary = capture['stationary_reference']['sample']['flange_pose_m_rad']
    handeye = sample['handeye']; h = np.eye(4); h[:3, :3] = handeye['rotation_matrix']; h[:3, 3] = handeye['translation_m']
    if info['frame_id']!=handeye['child_frame']:raise ValueError('image frame differs from calibrated handeye child frame')
    base_cam = transform_from_pose(stationary)@h
    paper = sample['paper_frame']; axes = [paper[k] for k in ('paper_u_base', 'paper_v_base', 'paper_normal_base')]
    origin = paper['reconstructed_center_tip_base_m']
    rectified, xs, ys = metric_rectification(image, info, base_cam, origin, axes, roi_mm, mm_per_pixel)
    if rectified.dtype!=np.uint8:
        rectified = (rectified.astype(float)/np.iinfo(rectified.dtype).max*255).astype(np.uint8)
    measurements, mask = contour_metrics(rectified, xs, ys)
    cv2.imwrite(str(output/'rectified_mono.png'), rectified); cv2.imwrite(str(output/'ink_mask.png'), mask)
    save(output/'contour_paper_mm.json', measurements.pop('contour_paper_mm'))
    # Plane residuals require a measured cloud->image transform, never frame-name guessing.
    cloud_transform = capture.get('image_from_cloud_frames', {}).get('cloud')
    if cloud_transform is not None:
        xyz = np.load(run_dir/'camera/cloud_xyz_m.npy').reshape(-1, 3); xyz = xyz[np.isfinite(xyz).all(axis=1)]
        xyz = xyz[::max(1, len(xyz)//100000)]
        image_points = xyz@np.array(cloud_transform)[:3, :3].T+np.array(cloud_transform)[:3, 3]
        base_points = image_points@base_cam[:3, :3].T+base_cam[:3, 3]
        relative = base_points-np.array(origin); n = np.array(axes[2])
        uv = relative@np.array(axes[:2]).T*1000
        near = (abs(uv[:, 0]-start[0]-length/2)<length/2+30)&(abs(uv[:, 1]-start[1])<35)
        residual = relative[near]@n*1000
        measurements['cloud_plane_check'] = {'points': len(residual),
            'median_residual_mm': None if not len(residual) else float(np.median(residual)),
            'abs_residual_p95_mm': None if not len(residual) else float(np.quantile(abs(residual), .95))}
    else: measurements['cloud_plane_check'] = {'status': 'missing image-from-cloud transform; raw data retained'}
    result = {'sample_id': sample['sample_id'], 'split': sample['split'], 'repeat_group': sample['repeat_group'],
              'commanded_depth_mm': sample['commanded_depth_mm'], 'measured_depth_paper_normal_mm': None,
              'paper_start_uv_mm':start.tolist(),'commanded_tilt_delta_uv_rad':sample.get('commanded_tilt_delta_uv_rad'),
              'commanded_axis_delta_rad':sample.get('commanded_axis_delta_rad',0.),
              'model_H_mm': None, 'metrics': measurements, 'mm_per_pixel_rectified': mm_per_pixel,
              'calibration_status': {'handeye': handeye.get('validation_scope'),
                 'paper_independently_validated': paper['absolute_plane_independently_validated'],
                 'tool_orientation': sample['tcp']['rotation_status']},
              'metric_status': 'estimate_on_reused_paper_plane; not absolute metrology acceptance',
              'raw_run_directory': str(run_dir.resolve()), 'source_sample': str(Path(sample_file).resolve()),
              'eligible_for_accepted_mapping_fit': False}
    save(output/'measurement.json', result)
    return result


def build_dataset(measurements, output):
    """Index only actual completed acquisitions; never synthesize missing rows."""
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    rows=[];seen=set();groups={}
    camera_required=['camera/image.cdr','camera/image.bin','camera/image_mono.png','camera/camera_info.json',
              'camera/cloud.cdr','camera/cloud_xyz_m.npy','camera/cloud_depth_z_m.npy',
              'camera/textured_cloud.cdr','photo_stationary_1.json','photo_stationary_2.json',
              'capture_controller_feedback.jsonl']
    automatic_required=['write_execution.json','transfer_execution.json','write_raw_force.jsonl',
                        'transfer_raw_force.jsonl','write_controller_feedback.jsonl','transfer_controller_feedback.jsonl']
    for path in measurements:
        data=json.loads(Path(path).read_text());sid=data['sample_id'];run=Path(data['raw_run_directory'])
        if sid in seen:raise ValueError('duplicate sample id: '+sid)
        if data['split'] not in ('train','holdout'):raise ValueError('invalid split')
        group=data['repeat_group']
        if group in groups and groups[group]!=data['split']:raise ValueError('repeat group leaks across train/holdout')
        groups[group]=data['split'];seen.add(sid)
        if not (run/'sample.json').is_file():raise ValueError('raw sample files missing: sample.json')
        sample=json.loads((run/'sample.json').read_text())
        if sample.get('sample_id')!=sid:raise ValueError('sample identity mismatch')
        manual=sample.get('acquisition_kind')=='manual_photo_of_executed_fixed_pose_line'
        required=camera_required if manual else camera_required+automatic_required
        missing=[name for name in required if not (run/name).is_file()]
        if missing:raise ValueError('raw sample files missing: '+str(missing))
        result=json.loads((run/'result.json').read_text())
        if result['status']!='captured' or result['sample_id']!=sid:raise ValueError('acquisition identity/status mismatch')
        if manual:
            execution_source=Path(sample.get('write_execution_source','__missing__'))
            if not execution_source.is_file():raise ValueError('manual photo requires actual write execution')
            execution=json.loads(execution_source.read_text())
            bundle_path=execution_source.parent/'bundle.json'
            if not bundle_path.is_file():raise ValueError('manual photo requires writing sample identity')
            bundle=json.loads(bundle_path.read_text())
            if (execution.get('status')!='completed_buffered_program' or execution.get('motion_sent') is not True
                    or bundle.get('sample_id')!=sid or bundle.get('commanded_press_mm')!=data['commanded_depth_mm']):
                raise ValueError('manual photo writing identity/status/depth mismatch')
            if not execution.get('force_trace',{}).get('rows'):
                raise ValueError('manual photo requires full raw writing force rows')
            feedback=execution_source.parent/'controller_feedback.jsonl'
            if not feedback.is_file() or feedback.stat().st_size==0:
                raise ValueError('manual photo requires actual writing feedback')
            sources={'transfer_mode':'manual','manual_transfer_telemetry_recorded':False,
                     'write_execution_source':str(execution_source.resolve()),
                     'write_raw_force_source':str(execution_source.resolve()),
                     'write_raw_force_json_pointer':'/force_trace/rows',
                     'write_controller_feedback_source':str(feedback.resolve())}
        else:
            sources={'transfer_mode':'guarded_automatic','manual_transfer_telemetry_recorded':None,
                     'write_execution_source':str((run/'write_execution.json').resolve()),
                     'write_raw_force_source':str((run/'write_raw_force.jsonl').resolve()),
                     'write_raw_force_json_pointer':None,
                     'write_controller_feedback_source':str((run/'write_controller_feedback.jsonl').resolve())}
        row={'sample_id':sid,'split':data['split'],'repeat_group':group,
             'commanded_depth_mm':data['commanded_depth_mm'],'measured_depth_paper_normal_mm':data['measured_depth_paper_normal_mm'],
             'commanded_tilt_delta_uv_rad':sample.get('commanded_tilt_delta_uv_rad'),
             'commanded_axis_delta_rad':sample.get('commanded_axis_delta_rad',0.),
             'paper_start_uv_mm':sample.get('paper_start_uv_mm',[0.,0.]),
             'physical_tilt_uv_rad':sample.get('physical_tilt_uv_rad'),
             'speed_mm_s':sample['write_speed_mm_s'], 'model_H_mm':None,
             'width_mm':data['metrics']['width_mm'],'length_mm':data['metrics']['length_mm'],
             'metric_status':data['metric_status'],'raw_run_directory':str(run.resolve()),
             'measurement_path':str(Path(path).resolve()),'eligible_for_accepted_mapping_fit':data['eligible_for_accepted_mapping_fit'],
             'photo_tool0_pose_m_rad':sample.get('photo_tool0_pose_m_rad'),**sources}
        rows.append(row)
    fields=['sample_id','split','repeat_group','commanded_depth_mm','measured_depth_paper_normal_mm',
            'commanded_tilt_delta_uv_rad','commanded_axis_delta_rad','paper_start_uv_mm','physical_tilt_uv_rad','speed_mm_s','model_H_mm','width_mm','length_mm',
            'metric_status','raw_run_directory','measurement_path','eligible_for_accepted_mapping_fit',
            'photo_tool0_pose_m_rad','transfer_mode','manual_transfer_telemetry_recorded','write_execution_source',
            'write_raw_force_source','write_raw_force_json_pointer','write_controller_feedback_source']
    with (output/'samples.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    save(output/'dataset.json',{'sample_count':len(rows),'samples':rows,'height_mapping':None,
                               'pose_mapping':None,'physical_mapping_accepted':False})
    return {'sample_count':len(rows),'output':str(output)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sample', type=Path); p.add_argument('--run', type=Path)
    p.add_argument('--dataset',nargs='*',type=Path,default=None,help='index measurement.json files; empty list creates an empty measured dataset')
    p.add_argument('--output', type=Path, required=True); p.add_argument('--roi-mm', nargs=4, type=float, default=None)
    p.add_argument('--mm-per-pixel', type=float, default=.1)
    a = p.parse_args()
    if a.dataset is not None:print(json.dumps(build_dataset(a.dataset,a.output)))
    else:
        if not a.sample or not a.run:p.error('--sample and --run required for measurement')
        print(json.dumps(measure(a.sample, a.run, a.output, a.roi_mm, a.mm_per_pixel)))


if __name__=='__main__': main()
