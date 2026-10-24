# Small-angle unloaded reference acquisition

`tools/brush_unloaded_calibration.py` is a separate air-only acquisition entry.
It does not alter `brush_buffered_guard.py`, `brush_buffered_motion.py`, their
writing evidence requirements, controller payload/TCP, or the sensor tare.

The first-reference measurement cannot depend on an already verified wrench
map for the same new orientation. This entry instead monitors **raw wrench
change from an actually measured session-start baseline**, gravity included.
That is a different observation from a gravity-corrected contact force. A
completed air acquisition never sets `contact_force_map_verified` or writing
`force_reference_verified` to true.

Before sending a finite program, the entry requires a complete numeric route
audit and both existing attachment self-collision models, with the supplied
sampled routes bound by direct comparison. The natural tip must stay within
0.5 mm of a fixed anchor and 35–45 mm above the recorded paper plane throughout
the numeric audit. Tool CAD clearance is at least20 mm. Each commanded rotation
step is at most2 degrees; the total local rotation remains within10 degrees.
The baseline TCP and paper validation states remain those of their sources.

Live execution reuses the production `health_tick`: raw force increment stop
at10 N, source freshness, host scheduling, TCP route envelope, TCP speed, joint
limits, and unwrapped joint speed stop at0.60 rad/s. An additional live tip-cell
and10-degree orientation check applies. The generated controller program is
the original finite `movel` program with its150 ms callback watchdog; the
existing hardware watchdog result must match that compiler. Failure closes
the callback and sends the original `stopl(0.5)` fallback. A failed sweep stops
without automatically retrying its remaining jobs.

The dedicated air entry now adds an explicit `t` to its MoveL commands. This
avoids estimating pure-rotation runtime from near-zero flange translation.
Each duration has margin for translation, angular change and joint displacement;
the complete predicted duration must fit the original120-second segment budget.
Only the MoveL timing arguments differ from the source compiler output: removing
those arguments must restore that exact output. The controller watchdog scaffold
is unchanged. The manifest labels the original compiler digest as belonging to
the **source** program; it does not mislabel that digest as the timed program's
digest. The actual timed source is saved and transmitted directly. Production
writing still uses its original compiler and transitions.

After a verified final pose and stationary window, the entry collects three
seconds of force data while continuously checking pose movement within0.1 mm
and0.05 degrees. Controller velocities are retained. There is no extra
single-frame velocity veto during this static window: stationary windows and
continuous pose bounds provide the acceptance used by the existing touch-zero
collector. An initially introduced0.020 rad/s live angular-speed veto was
removed after it interrupted tiny corrections while measured joint rates
remained below0.014 rad/s; the original joint-speed protection remains active.

Each reference has full raw force JSONL, controller pose/joint/velocity JSONL,
program manifest, execution status, static-window timing, and measured wrench.
The session records source code and the actual initial raw baseline. Resuming
requires a fresh measured start, re-audited remaining routes, and new collision
evidence. Completed references retain their original source directories.

The prepared23 October session entry is:

```
/usr/bin/python3 -B outputs/brush_mapping_20261023/depth_batch_d006_d009_d012/run_unloaded_sweep.py
```

Default mode is preview only. `--preflight` reads without motion. `--execute`
uses the saved checked routes and stops on the first error. Saved session
geometry is not an automatic authorization for a later changed installation.

`tools/validate_brush_unloaded_sweep.py` fits only training static observations
and checks four independent held-out poses, return drift, and raw moving-force
records paired with actual pose times. Its result is scoped to the observed
air trajectories and anchor. It does not accept contact dynamics, new cable
routing, unobserved positions, the brush H mapping, or a paper/TCP calibration.
