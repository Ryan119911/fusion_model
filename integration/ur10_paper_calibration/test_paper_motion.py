"""Deterministic regression tests; no robot, ROS node, or network access."""
from contextlib import redirect_stdout
import copy
import hashlib
import io
import json
from pathlib import Path
import socket
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

import calibrate_paper_plane_auto as app
import probe_paper_height as probe
import test_paper_plane_auto as collector_tests
import paper_ur_support as transport
import recover_paper_start as recovery


class StartRecoveryTests(unittest.TestCase):
    def run_recovery(self, execute, orientation_change=0):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            calibration = root / 'tcp.json'
            calibration.write_text('{}')
            previous = root / 'previous.json'
            previous.write_text(json.dumps({'baseline': {'median_n_nm': [0] * 6}, 'active_tcp_offset_m_rad': [0] * 6}))
            tip = np.array([0, 0, .329])
            orientation = np.array([np.pi, 0, 0])
            point = np.array([.5, -.2, .0206])
            current_orientation = orientation + [0, orientation_change, 0]
            state = {'tcp': app.tcp_target(point, current_orientation, np.zeros(6), tip), 'moves': [], 'rotations': []}
            initial = state['tcp'].copy()
            session = {'rectangle': app.rectangle_points([[.4, -.3, .02], [.6, -.1, .02]]),
                       'accepted_points': [], 'corners': [{'pen_tip_base_m': [.4, -.3, .02]}, {'pen_tip_base_m': [.6, -.1, .02]}],
                       'tcp_calibration': str(calibration), 'tcp_calibration_sha256': hashlib.sha256(calibration.read_bytes()).hexdigest(),
                       'active_tcp_offset_m_rad': [0] * 6, 'flange_to_pen_tip_m': tip.tolist(),
                       'tcp_orientation_rotvec': orientation.tolist(), 'robot_identity': {'host': 'test'},
                       'attempts': [{'measurement': str(previous)}]}
            source = root / 'session.json'; source.write_text(json.dumps(session))
            def read_state(*args):
                return np.zeros(6), state['tcp'].copy(), probe.tip_position(state['tcp'], np.zeros(6), tip)
            def move(reader, force, baseline, host, target, speed, contact_allowed):
                self.assertFalse(contact_allowed)
                self.assertTrue(np.array_equal(baseline, np.zeros(6)))
                self.assertLessEqual(np.linalg.norm(target[:3] - state['tcp'][:3]), .002000001)
                state['moves'].append(target.copy()); state['tcp'] = target.copy()
            def rotate(reader, force, baseline, host, endpoint, target_orientation, speed, acceleration, entry):
                before = probe.tip_position(state['tcp'], np.zeros(6), tip)
                after = probe.tip_position(endpoint, np.zeros(6), tip)
                self.assertGreaterEqual(before[2], .039999)
                self.assertTrue(np.allclose(before, after))
                state['rotations'].append(endpoint.copy()); state['tcp'] = endpoint.copy()
            class Resource:
                def __init__(self, *args):
                    pass
                def close(self):
                    pass
                def baseline(self):
                    if probe.tip_position(state['tcp'], np.zeros(6), tip)[2] < .039999:
                        raise AssertionError('must not tare at a loaded contact')
                    return np.zeros(6), {'free_clearance_verified': True}
            argv = ['recover', '--session', str(source)] + (['--execute'] if execute else [])
            with patch.object(recovery.Path, 'home', return_value=root), \
                    patch.object(app, 'capture', return_value={'active_tcp_offset_m_rad': [0] * 6}), \
                    patch.object(app, 'state_tip', side_effect=read_state), \
                    patch.object(probe, 'controller_ready', return_value={}), \
                    patch.object(probe, 'RealtimeReader', Resource), patch.object(probe, 'RawForce', Resource), \
                    patch.object(probe, 'model_gate', return_value={}), patch.object(probe, 'force_level', return_value=.1), \
                    patch.object(probe, 'force_snapshot', return_value={}), patch.object(probe, 'move_step', side_effect=move), \
                    patch.object(app, 'travel_segment', side_effect=rotate), \
                    patch.object(probe, 'wait_stopped'), patch('sys.argv', argv), redirect_stdout(io.StringIO()):
                recovery.main()
            report = json.loads(next((root / '.handoff/reports').glob('*.json')).read_text())
            return state, initial, report

    def test_recovery_only_raises_two_finite_segments_using_old_baseline(self):
        state, initial, report = self.run_recovery(True)
        self.assertEqual(len(state['moves']), 2)
        self.assertTrue(np.allclose(state['tcp'], initial + [0, 0, .004, 0, 0, 0]))
        self.assertEqual(report['status'], 'raised_4mm')

    def test_recovery_dry_run_sends_no_motion(self):
        state, initial, report = self.run_recovery(False)
        self.assertEqual(state['moves'], [])
        self.assertFalse(report['motion_sent'])
        self.assertEqual(report['status'], 'dry_run_pass')

    def test_small_orientation_change_is_corrected_only_after_high_lift(self):
        state, initial, report = self.run_recovery(True, orientation_change=.02)
        self.assertGreater(len(state['rotations']), 1)
        self.assertEqual(report['status'], 'original_orientation_restored_at_clearance')
        self.assertTrue(np.allclose(state['tcp'][3:], [np.pi, 0, 0]))


class ScriptTransportTests(unittest.TestCase):
    def tearDown(self):
        probe.close_commands()

    def test_scripts_share_a_drained_connection_without_implicit_retry(self):
        class Socket:
            def __init__(self):
                self.data, self.closed, self.reads = [], False, 0
            def settimeout(self, value):
                pass
            def recv(self, size):
                if self.closed:
                    raise OSError('closed')
                if self.reads == 0:
                    self.reads += 1
                    return bytes(100)
                time.sleep(.001)
                raise socket.timeout()
            def sendall(self, data):
                self.data.append(data)
            def shutdown(self, how):
                pass
            def close(self):
                self.closed = True
        sock = Socket()
        with patch.object(transport.socket, 'create_connection', return_value=sock) as connect:
            probe.command('test', 'move1\n')
            probe.command('test', 'move2\n')
            probe.stop('test')
            connection = probe._command_connections['test']
            self.assertTrue(connection.received.wait(.2))
            self.assertGreaterEqual(connection.received_bytes, 79)
            self.assertEqual(sock.data, [b'move1\n', b'move2\n', b'stopl(0.05)\n'])
            connect.assert_called_once()
            probe.close_commands()
            self.assertTrue(sock.closed)

    def test_send_failure_is_reported_without_resending_motion(self):
        class FailingConnection:
            def __init__(self, host):
                self.sends = 0
            def send(self, text):
                self.sends += 1
                raise OSError('send failed')
            def close(self):
                pass
        with patch.object(probe, 'ScriptConnection', FailingConnection):
            with self.assertRaisesRegex(OSError, 'send failed'):
                probe.command('test', 'one endpoint\n')
            self.assertNotIn('test', probe._command_connections)


class Clock:
    def __init__(self, step=.02):
        self.now, self.step = 0., step

    def __call__(self):
        self.now += self.step
        return self.now


class ContactStopTests(unittest.TestCase):
    def test_stationary_window_rejects_motion_and_averages_encoder_jitter(self):
        samples = []
        waits = []
        def state(*args, **kwargs):
            n = len(samples); samples.append(n)
            waits.append(kwargs['max_wait_seconds'])
            tcp = np.r_[.5 + (-1 if n % 2 else 1) * .000035, -.2, .35, np.pi, 0, 0]
            return np.zeros(6), tcp, np.zeros(6) if n >= 3 else np.ones(6)
        with patch.object(probe, 'fresh', state):
            _, tcp, _ = probe.stationary_state(None)
        self.assertGreaterEqual(len(samples), 12)
        self.assertLess(abs(tcp[0] - .5), .000005)
        self.assertGreater(waits[0], 1)
        self.assertTrue(all(0 < wait <= 3 for wait in waits))

    def test_stationary_pause_discards_earlier_window_without_weakening_motion_watchdog(self):
        clock = {'now': 0., 'frames': 0}
        def tick():
            clock['now'] += .001
            return clock['now']
        def state(*args, **kwargs):
            clock['frames'] += 1
            if clock['frames'] == 6:
                clock['now'] += .2
            return np.zeros(6), np.r_[.5, -.2, .35, np.pi, 0, 0], np.zeros(6)
        with patch.object(probe.time, 'monotonic', tick), patch.object(probe, 'fresh', side_effect=state):
            probe.stationary_state(None)
        self.assertGreaterEqual(clock['frames'], 14)

    def test_deceleration_then_quiet_window_records_contact(self):
        clock, diagnostics, frames = Clock(), [], []
        def state(*args, **kwargs):
            speed = .0003 if len(frames) < 3 else 0.
            frames.append(speed)
            return np.zeros(6), np.r_[.5, -.2, .35, np.pi, 0, 0], np.array([0, 0, speed, 0, 0, 0])
        with patch.object(probe.time, 'monotonic', clock), \
                patch.object(probe, 'fresh', state), \
                patch.object(probe, 'force_level', return_value=1.1), \
                patch.object(probe, 'command', side_effect=AssertionError('must only observe')):
            _, pose, velocity = probe.wait_stopped(None, None, None, diagnostics)
        self.assertGreaterEqual(len(diagnostics), 12)
        self.assertEqual(np.linalg.norm(velocity), 0)
        self.assertAlmostEqual(pose[2], .35)

    def test_continuous_motion_has_bounded_wait(self):
        with patch.object(probe.time, 'monotonic', Clock()), \
                patch.object(probe, 'fresh', return_value=(np.zeros(6), np.zeros(6), np.ones(6))), \
                patch.object(probe, 'force_level', return_value=.9):
            with self.assertRaisesRegex(RuntimeError, 'did not settle'):
                probe.wait_stopped(None, None, None, [], timeout=.3)

    def test_force_and_stream_faults_abort_observation(self):
        for function, error in [('force_level', probe.ForceGuardError('5 N', {})),
                                ('fresh', TimeoutError('no state'))]:
            with self.subTest(function=function), \
                    patch.object(probe, 'force_level', return_value=1.1), \
                    patch.object(probe, 'fresh', return_value=(np.zeros(6), np.zeros(6), np.zeros(6))), \
                    patch.object(probe, function, side_effect=error), \
                    patch.object(probe, 'command') as command:
                with self.assertRaises(type(error)):
                    probe.wait_stopped(None, None, None, [])
                command.assert_not_called()


class RecoveryTests(unittest.TestCase):
    def test_vertical_arrival_uses_z_gate_separately_from_valid_xy_residual(self):
        target = np.r_[.5, -.2, .35, np.pi, 0, 0]
        actual = target.copy(); actual[:3] += [.00006, 0, -.0000487]
        class Reader:
            class Socket:
                def settimeout(self, value):
                    pass
            sock = Socket()
        with patch.object(probe.time, 'monotonic', Clock(.01)), \
                patch.object(probe, 'force_level', return_value=.1), patch.object(probe, 'command') as command, \
                patch.object(probe, 'motion_state', return_value=(np.zeros(6), actual, np.zeros(6))):
            self.assertFalse(probe.move_step(Reader(), None, None, 'test', target, .0003, False))
        command.assert_called_once()

    def test_measured_static_residual_passes_but_larger_residual_does_not(self):
        target = np.r_[.5, -.2, .35, np.pi, 0, 0]
        class Reader:
            class Socket:
                def settimeout(self, value):
                    pass
            sock = Socket()
        self.assertLess(probe.VERTICAL_ARRIVAL_TOLERANCE_M, .0001)
        for residual in (.0000504, .00009):
            actual = target.copy(); actual[2] -= residual
            with self.subTest(residual=residual), patch.object(probe.time, 'monotonic', Clock(.01)), \
                    patch.object(probe, 'force_level', return_value=.1), patch.object(probe, 'command'), \
                    patch.object(probe, 'motion_state', return_value=(np.zeros(6), actual, np.zeros(6))):
                if residual < probe.VERTICAL_ARRIVAL_TOLERANCE_M:
                    self.assertFalse(probe.move_step(Reader(), None, None, 'test', target, .0003, False))
                else:
                    with self.assertRaises(probe.MotionGuardError):
                        probe.move_step(Reader(), None, None, 'test', target, .0003, False)

    def test_probe_drift_records_actual_error_without_weakening_gates(self):
        target = np.r_[.5, -.2, .35, np.pi, 0, 0]
        class Reader:
            class Socket:
                def settimeout(self, value):
                    pass
            sock = Socket()
        for kind in ('xy', 'rotation'):
            actual = target.copy()
            if kind == 'xy':
                actual[0] += .00021
            else:
                actual[3] += .002
            with self.subTest(kind=kind), \
                    patch.object(probe, 'force_level', return_value=0), \
                    patch.object(probe, 'command') as command, \
                    patch.object(probe, 'motion_state', return_value=(np.zeros(6), actual, np.zeros(6))):
                with self.assertRaises(probe.MotionGuardError) as caught:
                    probe.move_step(Reader(), None, None, 'test', target, .0003, True)
                trace = caught.exception.diagnostics['state_trace'][-1]
                self.assertGreater(trace['xy_error_mm'], .2) if kind == 'xy' else self.assertGreater(trace['rotation_error_deg'], .1)
                command.assert_called_once()

    def test_movel_script_contains_one_complete_program(self):
        script = probe.finite_movel_program('paper_step', np.zeros(6), .0005, .001)
        self.assertEqual(script.splitlines()[0], 'def paper_step():')
        self.assertEqual(script.splitlines()[-1], 'end')
        self.assertEqual(len(script.splitlines()), 3)
        self.assertEqual(script.count('movel('), 1)
        for target in ([0] * 5, [float('nan')] * 6):
            with self.assertRaises(ValueError):
                probe.finite_movel_program('paper_step', target, .0005, .001)

    def run_recovery(self, remaining_m):
        events = []
        target = np.r_[.5, -.2, .35, np.pi, 0, 0]
        actual = target.copy(); actual[2] += remaining_m
        class Reader:
            def connect(self, **kwargs):
                events.append('connect')
        recovery = []
        with patch.object(probe, 'fresh', side_effect=[TimeoutError('stream'), (np.zeros(6), actual, np.zeros(6))]), \
                patch.object(probe, 'stop', side_effect=lambda *args: events.append('stop')), \
                patch.object(probe, 'command', side_effect=lambda host, text: events.append(text)), \
                patch.object(probe, 'force_level', return_value=.05), redirect_stdout(io.StringIO()):
            probe.motion_state(Reader(), None, None, 'test-host', target, 'same endpoint', recovery)
        return events, recovery

    def test_completed_endpoint_is_not_commanded_again(self):
        events, recovery = self.run_recovery(.00001)
        self.assertEqual(events, ['stop', 'connect'])
        self.assertFalse(recovery[0]['command_reissued'])

    def test_interrupted_segment_stops_before_retrying_same_endpoint(self):
        events, recovery = self.run_recovery(.001)
        self.assertEqual(events, ['stop', 'connect', 'same endpoint'])
        self.assertTrue(recovery[0]['command_reissued'])

    def test_repeated_outage_stops_without_third_retry(self):
        with patch.object(probe, 'fresh', side_effect=TimeoutError('stream')), \
                patch.object(probe, 'stop') as stop, patch.object(probe, 'command') as command:
            with self.assertRaisesRegex(RuntimeError, 'repeatedly'):
                probe.motion_state(None, None, None, 'test', np.zeros(6), 'move', [{}, {}])
        stop.assert_called_once_with('test')
        command.assert_not_called()

    def test_slow_segment_can_finish_after_old_five_second_timeout(self):
        clock, target = Clock(.1), np.r_[.5, -.2, .35, np.pi, 0, 0]
        def state(*args):
            actual = target.copy()
            velocity = np.zeros(6)
            if clock.now < 5.5:
                actual[2] += .001
                velocity[2] = .0001
            return np.zeros(6), actual, velocity
        class Reader:
            class Socket:
                def settimeout(self, value):
                    pass
            sock = Socket()
        with patch.object(probe.time, 'monotonic', clock), \
                patch.object(probe, 'force_level', return_value=0), \
                patch.object(probe, 'motion_state', state), patch.object(probe, 'command'):
            app.travel_segment(Reader(), None, None, 'test', target, target[3:], .001, .0005, {})
        self.assertGreater(clock.now, 5)


class SessionTests(unittest.TestCase):
    def test_real_preflight_drift_still_aborts_before_commanding(self):
        tip = np.array([0, 0, .329])
        point = np.array([.5, -.2, .05])
        orientation = np.array([np.pi, 0, 0])
        tcp = app.tcp_target(point, orientation, np.zeros(6), tip)
        changed = tcp.copy(); changed[0] += .0002
        class Force:
            def baseline(self):
                return np.zeros(6), {}
        with patch.object(app, 'state_tip', side_effect=[(np.zeros(6), tcp, point), (np.zeros(6), changed, point)]), \
                patch.object(probe, 'controller_ready', return_value={}), \
                patch.object(probe, 'model_gate', return_value={}), \
                patch.object(probe, 'force_snapshot', return_value={}), \
                patch.object(probe, 'command') as command:
            with self.assertRaisesRegex(RuntimeError, '预检期间机器人移动'):
                app.travel(None, Force(), 'test', np.zeros(6), tip, point + [0, 0, .002], orientation, [])
        command.assert_not_called()

    def test_travel_within_existing_arrival_gate_does_not_command_noise_correction(self):
        tip = np.array([0, 0, .329])
        point = np.array([.5, -.2, .05])
        orientation = np.array([np.pi, 0, 0])
        tcp = app.tcp_target(point, orientation, np.zeros(6), tip)
        log = []
        with patch.object(app, 'state_tip', return_value=(np.zeros(6), tcp, point)), \
                patch.object(probe, 'command') as command, \
                patch.object(probe, 'controller_ready', side_effect=AssertionError('no new path needed')):
            app.travel(None, None, 'test', np.zeros(6), tip, point + [.000075, 0, 0], orientation, log)
        command.assert_not_called()
        self.assertEqual(log[-1]['status'], 'already_at_target')

    def session(self):
        plan = app.rectangle_points([[.4, -.3, .02], [.6, -.1, .02]])
        return {'rectangle': plan, 'accepted_points': [
            {'index': p['index'], 'role': p['role'], 'pen_tip_base_m': [*p['xy_base_m'], .02]}
            for p in plan['points']]}

    def test_duplicate_missing_and_moved_points_cannot_be_fitted(self):
        valid = self.session()
        app.validate_session(valid, require_complete=True)
        for change in ('duplicate', 'missing', 'moved', 'role'):
            bad = copy.deepcopy(valid)
            if change == 'duplicate':
                bad['accepted_points'][-1] = bad['accepted_points'][0]
            elif change == 'missing':
                bad['accepted_points'].pop()
            elif change == 'moved':
                bad['accepted_points'][0]['pen_tip_base_m'][0] += .002
            else:
                bad['accepted_points'][-1]['role'] = 'training'
            with self.subTest(change=change), self.assertRaises(ValueError):
                app.validate_session(bad, require_complete=True)

    def test_controller_checks_requested_host_and_same_ros_robot(self):
        replies = ['Robotmode: RUNNING', 'Safetystatus: NORMAL', 'STOPPED', 'UR10', '2022300602']
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'config').mkdir()
            (root / 'config/ur10_factory_calibration.yaml').write_text('hash: ' + probe.CALIBRATION_HASH)
            with patch.object(probe, 'ROOT', root), patch.object(probe, 'dashboard', side_effect=replies) as query:
                self.assertEqual(probe.controller_ready('specific-host')['serial'], probe.ROBOT_SERIAL)
                self.assertTrue(all(c.kwargs['ip'] == 'specific-host' for c in query.call_args_list))
            with patch.object(probe, 'ROOT', root), \
                    patch.object(probe, 'dashboard', side_effect=[*replies[:4], 'wrong-serial']):
                with self.assertRaisesRegex(RuntimeError, 'expected UR10'):
                    probe.controller_ready('specific-host')

    def test_existing_geometry_fault_and_full_resume_integration(self):
        with redirect_stdout(io.StringIO()):
            collector_tests.main()


if __name__ == '__main__':
    unittest.main()
