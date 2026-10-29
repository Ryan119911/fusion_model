"""Force-increment guard local to the operator-authorized brush ink trial."""
import math
import numpy as np


class TrialForceStop(RuntimeError):
    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        super().__init__('trial force increment reached %.3f N threshold (raw peak %.3f N, filtered %.3f N)' %
                         (diagnostics['threshold_n'],diagnostics['raw_peak_delta_n'],diagnostics['filtered_delta_n']))


def trial_force_level(force, baseline, threshold_n):
    # Operator decision 2026-10-28: every brush program now uses a 100 N force
    # increment stop, replacing the earlier 10 N default. Each run still records
    # the threshold it used in the bundle and in its evidence.
    if not math.isfinite(threshold_n) or not 1 <= threshold_n <= 100:
        raise ValueError('trial stop threshold must be finite and between 1 and 100 N')
    baseline = np.asarray(baseline,dtype=float)
    if baseline.shape != (6,) or not np.isfinite(baseline).all():
        raise ValueError('invalid force reference baseline')
    rows = force.recent()  # retains original source/receipt freshness checks
    values = np.asarray([r[3] for r in rows],dtype=float)
    if values.ndim != 2 or values.shape[1] != 6 or len(values)==0 or not np.isfinite(values).all():
        raise RuntimeError('invalid force stream')
    delta = values[:,:3]-baseline[:3]
    peak = float(np.linalg.norm(delta,axis=1).max())
    filtered = float(np.linalg.norm(np.median(delta[-5:],axis=0)))
    if peak >= threshold_n:
        raise TrialForceStop({'threshold_n':threshold_n,'raw_peak_delta_n':peak,
                              'filtered_delta_n':filtered,'baseline_n_nm':baseline.tolist(),
                              'latest_wrench_n_nm':values[-1].tolist()})
    return filtered
