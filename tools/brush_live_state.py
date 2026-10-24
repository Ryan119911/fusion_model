"""Continuously drain UR state during slow FK/ROS preflight work.

Receive-only, no reconnect, no motion command. Any transport failure latches.
"""
from collections import deque
import struct
import threading
import time
import numpy as np
from scipy.spatial.transform import Rotation


class ContinuousState:
    def __init__(self, support, host):
        self.support = support
        self.reader = support.RealtimeReader(host,30013)
        self.condition = threading.Condition()
        self.closed = False
        self.error = None
        self.latest = None
        self.sequence = self.delivered = 0
        self.thread = threading.Thread(target=self._receive,daemon=True)
        self.thread.start()

    def _receive(self):
        previous_source = None
        while not self.closed:
            try:
                sample = self.support.fresh(self.reader,max_wait_seconds=3 if previous_source is None else .15)
                source = struct.unpack_from('>d',self.reader.last_frame,4)[0]
                if not np.isfinite(source) or (previous_source is not None and source<=previous_source):
                    raise RuntimeError('controller state source timestamp did not advance')
                previous_source = source
                with self.condition:
                    self.latest = (time.monotonic(),tuple(np.asarray(x).copy() for x in sample))
                    self.sequence += 1
                    self.condition.notify_all()
            except BaseException as exc:
                with self.condition:
                    self.error = exc
                    self.condition.notify_all()
                return

    def fresh(self,timeout=.15):
        deadline = time.monotonic()+timeout
        with self.condition:
            while True:
                if self.error is not None:
                    raise RuntimeError('continuous state stream failed: '+str(self.error)) from self.error
                if self.closed:
                    raise RuntimeError('state stream closed')
                if self.latest is not None and self.sequence>self.delivered:
                    received,sample = self.latest
                    if time.monotonic()-received>.15:
                        raise RuntimeError('continuous state stale >150 ms')
                    self.delivered = self.sequence
                    return tuple(x.copy() for x in sample)
                remaining = deadline-time.monotonic()
                if remaining<=0:
                    raise TimeoutError('no new continuous state within deadline')
                self.condition.wait(remaining)

    def stationary(self):
        deadline=time.monotonic()+3;window=deque(maxlen=9)
        while time.monotonic()<deadline:
            sample=self.fresh(timeout=3 if self.latest is None else .15)
            if np.linalg.norm(sample[2][:3])>.0001 or np.linalg.norm(sample[2][3:])>.001:
                window.clear();continue
            window.append(sample)
            if len(window)<9:continue
            positions=np.array([x[1][:3] for x in window])
            rotations=Rotation.from_rotvec([x[1][3:] for x in window]);rotation=rotations.mean()
            if np.max(np.linalg.norm(positions-positions.mean(axis=0),axis=1))>.0001:continue
            if np.max((rotation.inv()*rotations).magnitude())>np.deg2rad(.05):continue
            return (np.mean([x[0] for x in window],axis=0),
                    np.r_[positions.mean(axis=0),rotation.as_rotvec()],
                    np.mean([x[2] for x in window],axis=0))
        raise RuntimeError('no stationary state window')

    def close(self):
        with self.condition:
            self.closed=True;self.condition.notify_all()
        self.thread.join(timeout=3.2)
        self.reader.close()
