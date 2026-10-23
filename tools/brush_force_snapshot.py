"""Bounded raw-force snapshot with the operator-selected 1.5 s age limit."""
from itertools import islice
import time
import numpy as np

FORCE_DATA_MAX_AGE_S=1.5


class ForceFreshnessError(RuntimeError):
    def __init__(self,message,diagnostics):
        super().__init__(message)
        self.diagnostics=diagnostics


def recent_force(force):
    # Read just ten rows and measure their age in the same critical section.
    # Copying the entire history and timestamping after releasing the lock can
    # misclassify an aged snapshot while the producer keeps receiving frames.
    with force.lock:
        rows=list(islice(reversed(force.rows),10));rows.reverse()
        observed=time.monotonic()
        if not rows or observed-rows[-1][0]>FORCE_DATA_MAX_AGE_S:
            raise ForceFreshnessError('raw force stream stale >1.5 s',{
                'observed_monotonic_s':observed,'observed_unix_s':time.time(),
                'latest_receipt_monotonic_s':rows[-1][0] if rows else None,
                'latest_receipt_unix_s':rows[-1][1] if rows else None,
                'latest_source_unix_s':rows[-1][2] if rows else None,
                'receipt_age_s':observed-rows[-1][0] if rows else None,
                'receipt_limit_s':FORCE_DATA_MAX_AGE_S,'snapshot_rows':len(rows),'buffer_rows':len(force.rows)})
    age=rows[-1][1]-rows[-1][2]
    if not -.05<=age<=FORCE_DATA_MAX_AGE_S:raise RuntimeError('raw force source timestamp stale or invalid')
    if not np.all(np.isfinite([r[3] for r in rows])):raise RuntimeError('nonfinite force measurement')
    return rows
