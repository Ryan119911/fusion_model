"""Hardware adapter for one longline trial. No hardware imports until explicitly requested."""
import copy
import datetime as dt
import hashlib
import importlib
import json
from pathlib import Path
import struct
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from brush_buffered_motion import compile_program, execution_blockers
from brush_longline_calibration import build_plan, digest, read, save


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_support(robot_scripts):
    scripts = Path(robot_scripts).resolve()
    python_version = f'{sys.version_info.major}.{sys.version_info.minor}'
    paths = [Path.home()/'.dsh-analysis', scripts, Path(f'/opt/openrobots/lib/python{python_version}/site-packages')]
    sys.path[:0] = [str(p) for p in paths if p.is_dir()]
    return importlib.import_module('probe_paper_height'), importlib.import_module('calibrate_pen_tcp').capture


def record_air(robot_scripts, output):
    """Observe one already stationary suspended pose. Never moves to another pose."""
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'brush_longline_air_sample_v1', 'status': 'started', 'motion_sent': False,
              'force_tare_performed': False, 'operator_unloaded_confirmed': True}
    force = reader = watcher = None; done = threading.Event(); monitoring_errors = []
    try:
        support, capture = load_support(robot_scripts)
        support.controller_ready('192.168.1.3')
        before = capture('192.168.1.3', 30011, 1.)
        if before['max_motion_mm'] > .1 or before['max_motion_deg'] > .05:
            raise RuntimeError('reference pose is not stationary')
        from brush_live_state import ContinuousState
        reader = ContinuousState(support, '192.168.1.3'); reader.stationary()
        target = np.array(before['flange_pose_m_rad'])
        def monitor():
            try:
                while not done.is_set():
                    _, current, _ = reader.fresh(timeout=.15)
                    if np.linalg.norm(current[:3] - target[:3]) > .0001 or (Rotation.from_rotvec(target[3:]).inv() * Rotation.from_rotvec(current[3:])).magnitude() > np.deg2rad(.05):
                        raise RuntimeError('reference moved while force was being sampled')
            except BaseException as exc:
                if not done.is_set(): monitoring_errors.append(str(exc))
        watcher = threading.Thread(target=monitor, daemon=True); watcher.start()
        force = support.RawForce(); wrench, statistics = force.baseline(); force.recent()
        after = capture('192.168.1.3', 30011, 1.)
        a, b = np.array(before['flange_pose_m_rad']), np.array(after['flange_pose_m_rad'])
        if after['max_motion_mm'] > .1 or after['max_motion_deg'] > .05 or np.linalg.norm(a[:3] - b[:3]) > .0001 or (Rotation.from_rotvec(a[3:]).inv() * Rotation.from_rotvec(b[3:])).magnitude() > np.deg2rad(.05):
            raise RuntimeError('reference moved during collection')
        if not np.allclose(before['active_tcp_offset_m_rad'], 0, atol=1e-8, rtol=0) or not np.allclose(after['active_tcp_offset_m_rad'], 0, atol=1e-8, rtol=0):
            raise ValueError('air sample requires unchanged zero active TCP')
        done.set(); watcher.join(timeout=1.)
        if monitoring_errors: raise RuntimeError('; '.join(monitoring_errors))
        with force.lock:
            raw = [{'received_monotonic_s': r[0], 'received_unix_s': r[1], 'source_unix_s': r[2], 'wrench_n_nm': list(r[3])} for r in force.rows]
        save(out / 'raw_force.json', raw)
        report.update(status='recorded', recorded_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                      pose_m_rad=b.tolist(), wrench_n_nm=np.asarray(wrench).tolist(), statistics=statistics,
                      before=before, after=after, raw_force_sha256=sha(out / 'raw_force.json'))
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        done.set()
        if watcher is not None: watcher.join(timeout=1.)
        if reader is not None: reader.close()
        if force is not None: force.close()
        save(out / 'sample.json', report)
    return report


def fit_force(samples, holdouts, maximum_error_n, maximum_translation_m, output):
    if not np.isfinite([maximum_error_n, maximum_translation_m]).all() or not 0 < maximum_error_n < 100 or not 0 < maximum_translation_m <= .5:
        raise ValueError('finite reference-error limit (0,100)N and translation domain (0,0.5]m required')
    samples, holdouts = [Path(p) for p in samples], [Path(p) for p in holdouts]
    if set(p.resolve() for p in samples) & set(p.resolve() for p in holdouts):
        raise ValueError('force holdouts must be independent files')
    records = [read(p) for p in samples + holdouts]
    if len(samples) < 4 or len(holdouts) < 2:
        raise ValueError('at least 4 independent affine training poses and 2 held-out poses required')
    for p, r in zip(samples + holdouts, records):
        if r.get('schema') != 'brush_longline_air_sample_v1' or r.get('status') != 'recorded' or r.get('operator_unloaded_confirmed') is not True:
            raise ValueError('only measured unloaded samples are accepted')
        if sha(p.parent / 'raw_force.json') != r['raw_force_sha256']:
            raise ValueError('raw force sample changed')
        age = (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(r['recorded_at'])).total_seconds()
        if not 0 <= age <= 1800: raise ValueError('air sample older than 30 minutes')
    poses = np.array([r['pose_m_rad'] for r in records], float)
    wrench = np.array([r['wrench_n_nm'] for r in records], float)
    if poses.shape != (len(records), 6) or wrench.shape != poses.shape or not np.isfinite(np.r_[poses.ravel(), wrench.ravel()]).all():
        raise ValueError('invalid measured force/pose array')
    baseline = Rotation.from_rotvec(poses[:len(samples),3:]).mean()
    relative = (baseline.inv() * Rotation.from_rotvec(poses[:,3:])).as_rotvec() / np.deg2rad(4)
    design = np.column_stack((np.ones(len(relative)), relative))
    if np.linalg.matrix_rank(design[:len(samples)]) < 4:
        raise ValueError('air reference poses do not identify an affine orientation model')
    coefficients = np.linalg.lstsq(design[:len(samples)], wrench[:len(samples)], rcond=None)[0]
    error = np.linalg.norm((design @ coefficients - wrench)[len(samples):,:3], axis=1)
    low, high = relative[:len(samples)].min(0), relative[:len(samples)].max(0)
    covered = bool(np.all(relative[len(samples):] >= low - 1e-8) and np.all(relative[len(samples):] <= high + 1e-8))
    anchor = poses[:len(samples),:3].mean(0)
    translation_covered = bool(np.all(np.linalg.norm(poses[:,:3] - anchor, axis=1) <= maximum_translation_m))
    passed = bool(error.max() <= maximum_error_n and covered and translation_covered)
    fit = {'basis_kind':'affine', 'baseline_rotation_rad':baseline.as_rotvec().tolist(), 'coefficients':coefficients.tolist(), 'offset':[0.]*6}
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    validation = {'passed':passed, 'local_fit_sha256':digest(fit), 'maximum_holdout_force_error_n':float(error.max()),
                  'reference_error_limit_n':maximum_error_n, 'holdout_rotation_covered':covered, 'holdout_translation_covered':translation_covered,
                  'source_samples':[{'path':str(p.resolve()),'sha256':sha(p)} for p in samples+holdouts],
                  'scope':'local affine orientation reference with stationary holdouts; dynamic force residual still needs trial inspection'}
    save(out / 'validation.json', validation)
    record = {'schema':'brush_longline_force_reference_v1', 'passed':passed,
              'recorded_at':min(dt.datetime.fromisoformat(r['recorded_at']) for r in records).isoformat(), 'local_fit':fit,
              'normalized_rotation_min':low.tolist(), 'normalized_rotation_max':high.tolist(),
              'position_reference_base_m':anchor.tolist(), 'maximum_translation_m':maximum_translation_m,
              'validation_evidence':{'path':'validation.json','sha256':sha(out / 'validation.json')}}
    save(out / 'force_reference.json', record)
    result = {'status':'force_reference_fitted' if passed else 'force_reference_validation_failed', 'passed':passed,
              'maximum_holdout_force_error_n':float(error.max()), 'force_reference':str((out / 'force_reference.json').resolve()), 'motion_sent':False}
    save(out / 'RESULT.json', result)
    return result


def checked_plan(path):
    plan = read(path)
    expected = build_plan(plan['config'], plan['kind'], plan['trial_id'], plan['split'], plan['repeat_group'])
    if plan != expected:
        raise ValueError('plan differs from deterministic source configuration; regenerate it')
    return plan


def reference_wrenches(record, points, directory, now=None):
    """Consume an independently validated local force fit, never invent a baseline."""
    if record.get('schema') != 'brush_longline_force_reference_v1' or record.get('passed') is not True:
        raise ValueError('validated posture-dependent force reference required')
    stamp = dt.datetime.fromisoformat(record['recorded_at'])
    if stamp.tzinfo is None:
        raise ValueError('force reference timestamp needs timezone')
    age = ((now or dt.datetime.now(dt.timezone.utc)) - stamp).total_seconds()
    if not 0 <= age <= 1800:
        raise ValueError('force reference must be from the last 30 minutes')
    fit = record['local_fit']
    evidence = record['validation_evidence']
    evidence_path = Path(directory) / evidence['path']
    if sha(evidence_path) != evidence['sha256']:
        raise ValueError('force validation evidence changed')
    validation = read(evidence_path)
    if validation.get('passed') is not True or validation.get('local_fit_sha256') != digest(fit):
        raise ValueError('independent force validation does not bind this fit')
    baseline = Rotation.from_rotvec(fit['baseline_rotation_rad'])
    poses = np.array([p['pose_m_rad'] for p in points])
    x = (baseline.inv() * Rotation.from_rotvec(poses[:, 3:])).as_rotvec() / np.deg2rad(4)
    low, high = np.array(record['normalized_rotation_min']), np.array(record['normalized_rotation_max'])
    if low.shape != (3,) or high.shape != (3,) or not np.isfinite(np.r_[low, high]).all() or np.any(low > high) or np.any(x < low - 1e-8) or np.any(x > high + 1e-8):
        raise ValueError('planned orientation leaves measured force-reference domain')
    anchor = np.asarray(record['position_reference_base_m'], float)
    radius = float(record['maximum_translation_m'])
    if anchor.shape != (3,) or not np.isfinite(anchor).all() or not np.isfinite(radius) or radius <= 0 or np.any(np.linalg.norm(poses[:, :3] - anchor, axis=1) > radius):
        raise ValueError('planned translation leaves validated force-reference region')
    a, b, c = x.T
    if fit['basis_kind'] == 'affine':
        basis = np.column_stack((np.ones(len(x)), a, b, c))
    elif fit['basis_kind'] == 'quadratic':
        basis = np.column_stack((np.ones(len(x)), a, b, c, a*a, b*b, c*c, a*b, a*c, b*c))
    else:
        raise ValueError('force fit must use affine or quadratic basis')
    coeff = np.array(fit['coefficients'], float)
    offset = np.array(fit.get('offset', [0.] * 6), float)
    if coeff.shape != (basis.shape[1], 6) or offset.shape != (6,) or not np.isfinite(np.r_[coeff.ravel(), offset]).all():
        raise ValueError('invalid force fit coefficients')
    return basis @ coeff + offset


def prepare_live(plan_path, reference_path, watchdog_path, guard_host, robot_scripts, output,
                 support=None, capture=None, audit=None):
    plan = checked_plan(plan_path)
    if plan['config']['geometry_status'] != 'measured_current_setup':
        raise ValueError('replace example/historical geometry and confirm measured_current_setup before live preflight')
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    report = {'status': 'preflight_started', 'motion_sent': False, 'trial_id': plan['trial_id']}
    save(out / 'plan.json', plan)
    try:
        if support is None:
            support, capture = load_support(robot_scripts)
        if audit is None:
            from write_gong_1mm import audit_points
            audit = audit_points
        from brush_live_state import ContinuousState
        import brush_buffered_motion as compiler
        watchdog = read(watchdog_path)
        if watchdog.get('passed') is not True or watchdog.get('compiler_sha256') != sha(compiler.__file__):
            raise ValueError('watchdog hardware record must match the unchanged production compiler')
        reference = read(reference_path)
        points = copy.deepcopy(plan['points'])
        refs = reference_wrenches(reference, points, Path(reference_path).parent)
        support.controller_ready('192.168.1.3')
        offset = np.asarray(capture('192.168.1.3', 30011, 1.)['active_tcp_offset_m_rad'])
        if not np.allclose(offset, np.zeros(6), atol=1e-8, rtol=0):
            raise ValueError('expected unchanged zero active controller TCP')
        reader = ContinuousState(support, '192.168.1.3')
        try:
            q, actual, _ = reader.stationary()
        finally:
            reader.close()
        planned = np.array(points[0]['pose_m_rad'])
        if np.linalg.norm(actual[:3] - planned[:3]) > .0005 or (Rotation.from_rotvec(actual[3:]).inv() * Rotation.from_rotvec(planned[3:])).magnitude() > np.deg2rad(.2):
            raise ValueError('move to the displayed start-air pose before preparing; no automatic start connector is sent')
        # The first command is the measured start. Audit this exact full route.
        points[0]['pose_m_rad'] = actual.tolist()
        refs = reference_wrenches(reference, points, Path(reference_path).parent)
        for point, wrench in zip(points, refs):
            point['unloaded_wrench_n_nm'] = wrench.tolist()
        audited = audit(points, q, offset, support)
        script, manifest = compile_program(points, guard_host, 50071)
        evidence = {}
        records = {
            'physical_mapping': {'passed': True, 'scope': 'explicit geometric H/alpha/beta, measured setup; not old model parameter calibration',
                                 'config_sha256': plan['config_sha256']},
            'force_reference': {'passed': True, 'reference': reference,
                                'scope': 'independently validated fresh local posture-dependent unloaded reference'},
            'blended_path_audit': {**audited, 'program_sha256': manifest['program_sha256']},
        }
        for name, value in records.items():
            value['route_sha256'] = manifest['route_sha256']
            save(out / (name + '.json'), value)
            evidence[name] = {'path': name + '.json', 'sha256': sha(out / (name + '.json'))}
        duration = sum(np.linalg.norm(np.array(b['pose_m_rad'][:3]) - a['pose_m_rad'][:3]) / b['speed_m_s'] for a, b in zip(points, points[1:]))
        budget = max(60., duration * 2 + 60.)
        if budget > 1800:
            raise ValueError('route exceeds existing 1800-second runtime budget')
        bundle = {'sample_id': plan['trial_id'], 'points': points, 'start_pose_m_rad': actual.tolist(),
                  'active_tcp_offset_m_rad': offset.tolist(), 'maximum_runtime_s': budget,
                  'physical_mapping_verified': True, 'force_reference_verified': True,
                  'controller_watchdog_hardware_validated': True, 'evidence': evidence,
                  'force_stop_n': plan['config']['motion']['force_stop_n']}
        blockers = execution_blockers(bundle, manifest, out)
        if blockers:
            raise ValueError('; '.join(blockers))
        save(out / 'bundle.json', bundle); save(out / 'manifest.json', manifest)
        (out / 'program.script.preview').write_text(script)
        prepared = {'schema': 'brush_longline_prepared_v1', 'trial_id': plan['trial_id'], 'prepared_at': dt.datetime.now(dt.timezone.utc).isoformat(),
                    'guard_host': guard_host, 'guard_port': 50071, 'plan_sha256': sha(out / 'plan.json'),
                    'bundle_sha256': sha(out / 'bundle.json'), 'watchdog': {'path': str(Path(watchdog_path).resolve()), 'sha256': sha(watchdog_path)},
                    'force_reference_recorded_at': reference['recorded_at'], 'compiler_sha256': sha(compiler.__file__)}
        save(out / 'prepared.json', prepared)
        report.update(status='preflight_passed_no_motion', waypoints=len(points), prepared=str((out / 'prepared.json').resolve()))
    except BaseException as exc:
        report.update(status='preflight_failed_no_motion', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save(out / 'RESULT.json', report)
    return report


class Camera:
    """USB/RTSP via OpenCV, or a ROS Image topic with an optional trigger service."""
    def __init__(self, config):
        self.config = config; self.last = None; self.error = None
        self.lock = threading.Lock(); self.done = threading.Event(); self.index = 0
        self.node = self.cap = self.context = self.executor = None
        self.thread = threading.Thread(target=self._receive, daemon=True)

    def start(self):
        self.thread.start()
        if self.config.get('trigger'):
            deadline = time.monotonic() + 10.
            while not hasattr(self, 'trigger_client') and self.error is None and time.monotonic() < deadline:
                time.sleep(.01)
            self.snapshot()
        else:
            self.frame(timeout=10.)
        return self

    def _store(self, image, source_ns=None):
        if image is None or image.ndim not in (2, 3) or image.size == 0:
            raise RuntimeError('invalid camera frame')
        with self.lock:
            self.index += 1
            self.last = (image.copy(), {'sequence': self.index, 'host_unix_s': time.time(),
                         'host_monotonic_s': time.monotonic(), 'source_stamp_ns': source_ns})

    def _receive(self):
        try:
            if self.config['kind'] == 'opencv':
                import cv2
                self.cap = cv2.VideoCapture(self.config['device'])
                for key, prop in [('width', cv2.CAP_PROP_FRAME_WIDTH), ('height', cv2.CAP_PROP_FRAME_HEIGHT), ('fps', cv2.CAP_PROP_FPS)]:
                    if key in self.config: self.cap.set(prop, self.config[key])
                if not self.cap.isOpened(): raise RuntimeError('cannot open camera device')
                while not self.done.is_set():
                    ok, image = self.cap.read()
                    if not ok: raise RuntimeError('camera frame read failed')
                    self._store(image)
            elif self.config['kind'] == 'ros':
                import rclpy
                from sensor_msgs.msg import Image
                from rclpy.qos import qos_profile_sensor_data
                from rclpy.context import Context
                from rclpy.executors import SingleThreadedExecutor
                from brush_capture_measure import image_array
                self.context = Context(); rclpy.init(args=[], context=self.context)
                self.node = rclpy.create_node('brush_longline_camera_' + str(id(self)), context=self.context)
                self.executor = SingleThreadedExecutor(context=self.context); self.executor.add_node(self.node)
                def callback(msg):
                    try:
                        image = image_array(msg)
                        if msg.encoding.lower() == 'rgb8': image = image[:, :, ::-1]
                        self._store(image, msg.header.stamp.sec * 1000000000 + msg.header.stamp.nanosec)
                    except BaseException as exc:
                        self.error = exc; self.done.set()
                self.node.create_subscription(Image, self.config['topic'], callback, qos_profile_sensor_data)
                trigger = self.config.get('trigger')
                if trigger:
                    from rosidl_runtime_py.utilities import get_service
                    service = get_service(trigger['service_type'])
                    client = self.node.create_client(service, trigger['service'])
                    if not client.wait_for_service(timeout_sec=5.): raise RuntimeError('camera trigger service unavailable')
                    self.trigger_client, self.trigger_type = client, service
                while not self.done.is_set():
                    self.executor.spin_once(timeout_sec=.05)
            else:
                raise ValueError('camera kind must be opencv or ros')
        except BaseException as exc:
            self.error = exc

    def frame(self, timeout=.01, after_sequence=None):
        deadline = time.monotonic() + timeout
        while True:
            if self.error is not None: raise RuntimeError('camera failed: ' + str(self.error))
            with self.lock:
                value = self.last
                if value is not None and (after_sequence is None or value[1]['sequence'] > after_sequence):
                    if time.monotonic() - value[1]['host_monotonic_s'] <= self.config.get('maximum_frame_age_s', .5):
                        return value[0].copy(), dict(value[1])
            if time.monotonic() >= deadline: raise TimeoutError('camera frame missing/stale')
            time.sleep(.005)

    def health(self):
        if self.error is not None: raise RuntimeError('camera failed: ' + str(self.error))
        with self.lock:
            metadata = None if self.last is None else self.last[1]
            if metadata is None or time.monotonic() - metadata['host_monotonic_s'] > self.config.get('maximum_frame_age_s', .5):
                raise TimeoutError('camera frame missing/stale')

    def snapshot(self):
        previous = self.index
        if self.config.get('trigger'):
            if self.node is None or not hasattr(self, 'trigger_client'): raise RuntimeError('camera trigger not ready')
            future = self.trigger_client.call_async(self.trigger_type.Request())
            deadline = time.monotonic() + 10.
            while not future.done() and time.monotonic() < deadline: time.sleep(.01)
            if not future.done(): raise TimeoutError('camera trigger timed out')
            if future.exception() is not None: raise RuntimeError(str(future.exception()))
        return self.frame(timeout=10., after_sequence=previous)

    def close(self):
        self.done.set()
        if self.cap is not None: self.cap.release()
        self.thread.join(timeout=2.)
        if self.executor is not None: self.executor.shutdown(timeout_sec=1.)
        if self.node is not None: self.node.destroy_node()
        if self.context is not None and self.context.ok(): self.context.shutdown()


class CameraRecording:
    def __init__(self, config, output, camera_factory=Camera):
        self.config, self.output, self.factory = config, Path(output), camera_factory
        self.side = self.overhead = None; self.error = None; self.done = threading.Event(); self.ready = threading.Event(); self.thread = None

    def start(self):
        import cv2
        self.side = self.factory(self.config['side']); self.side.start()
        self.overhead = self.factory(self.config['overhead']); self.overhead.start()
        frame, meta = self.overhead.snapshot()
        if not cv2.imwrite(str(self.output / 'before.png'), frame): raise RuntimeError('cannot save before image')
        save(self.output / 'before_capture.json', meta)
        frame, _ = self.side.frame(timeout=2.)
        if not cv2.imwrite(str(self.output / 'tip_before.png'), frame): raise RuntimeError('cannot save tip state')
        self.thread = threading.Thread(target=self._record, args=(frame.shape,), daemon=True)
        self.thread.start()
        if not self.ready.wait(5.) or self.error is not None:
            raise RuntimeError('side recording failed to start: ' + str(self.error))

    def _record(self, shape):
        import cv2
        writer = cv2.VideoWriter(str(self.output / 'side.avi'), cv2.VideoWriter_fourcc(*'MJPG'),
                                 float(self.config['side'].get('fps', 25)), (shape[1], shape[0]), True)
        try:
            if not writer.isOpened(): raise RuntimeError('cannot open side video writer')
            with (self.output / 'side_frames.jsonl').open('x') as stream:
                prior = None; frame_index = 0
                while not self.done.is_set():
                    image, metadata = self.side.frame(timeout=.5, after_sequence=prior)
                    prior = metadata['sequence']
                    if image.shape[:2] != shape[:2]: raise RuntimeError('side camera dimensions changed')
                    if image.ndim == 2: image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                    writer.write(image)
                    stream.write(json.dumps({'video_frame_index': frame_index, **metadata}) + '\n'); stream.flush()
                    frame_index += 1
                    self.ready.set()
        except BaseException as exc:
            if not self.done.is_set(): self.error = exc
        finally:
            writer.release()

    def health(self):
        if self.error is not None: raise RuntimeError('side recording failed: ' + str(self.error))
        self.side.health()

    def finish(self):
        import cv2
        self.health()
        frame, metadata = self.overhead.snapshot()
        if not cv2.imwrite(str(self.output / 'after.png'), frame): raise RuntimeError('cannot save after image')
        save(self.output / 'after_capture.json', metadata)

    def close(self):
        self.done.set()
        if self.thread is not None: self.thread.join(timeout=2.)
        for camera in (self.side, self.overhead):
            if camera is not None: camera.close()


def camera_check(config_path, duration, output):
    if not np.isfinite(duration) or not 0 < duration <= 30: raise ValueError('camera check duration must be in (0,30] seconds')
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    recording = CameraRecording(read(config_path), out)
    result = {'motion_sent':False, 'camera_capture_attempted':True, 'status':'started'}
    try:
        recording.start(); deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            recording.health(); time.sleep(.02)
        recording.finish(); result['status'] = 'camera_check_completed'
    except BaseException as exc:
        result.update(status='camera_check_failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        recording.close(); save(out / 'RESULT.json', result)
    return result


def run_trial(prepared_path, output, execute=False, cameras_path=None, robot_scripts=None,
              executor=None, recording_factory=CameraRecording, support=None, capture=None):
    prepared_path = Path(prepared_path)
    prepared = read(prepared_path); directory = prepared_path.parent
    if prepared.get('schema') != 'brush_longline_prepared_v1': raise ValueError('unsupported prepared record')
    for name in ('plan', 'bundle'):
        if sha(directory / (name + '.json')) != prepared[name + '_sha256']: raise ValueError(name + ' changed after preflight')
    plan, bundle = read(directory / 'plan.json'), read(directory / 'bundle.json')
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    save(out / 'plan.json', plan)
    if not execute:
        result = {'status': 'preview_only', 'trial_id': plan['trial_id'], 'motion_sent': False,
                  'camera_triggered': False, 'start_air_pose_m_rad': bundle['points'][0]['pose_m_rad']}
        save(out / 'RESULT.json', result); return result
    result = {'status': 'started', 'trial_id': plan['trial_id'], 'motion_sent': False, 'camera_triggered': False}
    recording = None
    try:
        if cameras_path is None or robot_scripts is None: raise ValueError('--execute requires --cameras and --robot-scripts')
        import brush_buffered_motion as compiler
        if prepared['compiler_sha256'] != sha(compiler.__file__): raise ValueError('production compiler changed')
        watch = prepared['watchdog']
        if sha(watch['path']) != watch['sha256']: raise ValueError('watchdog evidence changed')
        now = dt.datetime.now(dt.timezone.utc)
        for stamp in (prepared['prepared_at'], prepared['force_reference_recorded_at']):
            if not 0 <= (now - dt.datetime.fromisoformat(stamp)).total_seconds() <= 1800: raise ValueError('preflight/reference stale; prepare again')
        _, manifest = compile_program(bundle['points'], prepared['guard_host'], prepared['guard_port'])
        errors = execution_blockers(bundle, manifest, directory)
        if errors: raise ValueError('; '.join(errors))
        if support is None:
            support, capture = load_support(robot_scripts)
        if executor is None:
            from brush_buffered_guard import execute_buffered
            executor = execute_buffered
        def stationary_pose_record(name):
            sample = capture('192.168.1.3', 30011, 1.)
            if sample['max_motion_mm'] > .1 or sample['max_motion_deg'] > .05:
                raise RuntimeError('robot not stationary at camera exposure')
            save(out / name, sample)
        stationary_pose_record('before_pose.json')
        recording = recording_factory(read(cameras_path), out); recording.start()
        result['camera_triggered'] = True
        namespace = SimpleNamespace(**vars(support))
        original_fresh, original_force = support.fresh, support.RawForce
        with (out / 'controller_feedback.jsonl').open('x') as states, (out / 'raw_force.jsonl').open('x') as forces:
            from collections import deque
            class Rows(deque):
                def append(self, row):
                    super().append(row)
                    forces.write(json.dumps({'received_monotonic_s': row[0], 'received_unix_s': row[1],
                        'source_unix_s': row[2], 'wrench_n_nm': list(row[3])}) + '\n')
            class Force(original_force):
                def __init__(self):
                    super().__init__()
                    with self.lock:
                        old = list(self.rows); self.rows = Rows(maxlen=self.rows.maxlen)
                        for row in old: self.rows.append(row)
            def fresh(reader, *args, **kwargs):
                result = original_fresh(reader, *args, **kwargs)
                recording.health()
                states.write(json.dumps({'trial_id': plan['trial_id'], 'host_unix_s': time.time(),
                    'host_monotonic_s': time.monotonic(), 'controller_time_s': struct.unpack_from('>d', reader.last_frame, 4)[0],
                    'actual_q_rad': result[0].tolist(), 'actual_tool0_pose_m_rad': result[1].tolist(), 'tcp_velocity': result[2].tolist()}) + '\n')
                return result
            namespace.fresh, namespace.RawForce = fresh, Force
            executor(bundle, directory, prepared['guard_host'], prepared['guard_port'], out / 'execution.json', namespace, capture)
        execution = read(out / 'execution.json')
        result['motion_sent'] = execution.get('motion_sent', False)
        if execution.get('status') != 'completed_buffered_program': raise RuntimeError('line did not complete')
        stationary_pose_record('after_pose.json')
        recording.finish()
        result.update(status='completed_line_and_capture', camera_triggered=True)
    except BaseException as exc:
        if (out / 'execution.json').exists(): result['motion_sent'] = read(out / 'execution.json').get('motion_sent', False)
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if recording is not None: recording.close()
        save(out / 'RESULT.json', result)
    return result
