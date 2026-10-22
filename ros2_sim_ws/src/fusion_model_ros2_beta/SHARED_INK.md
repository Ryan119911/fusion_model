# Shared V16 forward ink (2026-09-19)

The Beta ink topic now replays the original model renderer's cumulative neural
patches, not the separate opaque Bezier approximation. Each glyph has one
replaceable marker. Samples are scheduled against the same stroke's commanded
trajectory, and glyph stroke IDs are distinct. Clearing deletes cached markers.

No brush radius, network weights or CoppeliaSim code were changed. The current
inversion protocol keeps scaled xy fixed, optimizes H, and derives angles; it
is not a free six-field optimization. Online/on-demand generation and offline
cached generation use the same generator and the same forward export.

Cache artifacts: `neural_ink.npz`, `neural_ink_audit.json`,
`executed_forward.png`. Cache reuse checks CSV, checkpoint, exporter, renderer
and ink-stream hashes. World coordinates preserve pixel centres and flip image
y into paper y, with no second glyph-size normalization.

Verification on MU: run `test/verify_shared_ink_on_mu.py` with ROS sourced and
an isolated ROS_DOMAIN_ID. This builds and validates IK but never spins timers
or publishes a joint trajectory. It checks final marker opacity reconstruction,
no early second glyph, and clear behavior. `--layout horizontal` tests the other
layout. Unit tests use pytest. This does not replace visual RViz acceptance.

120 mm cached Wuhan: forward-stream max error <= 5.96e-8, coordinate error
<= 1.12e-8 m. Shape-target ink area ratios remain 1.976 and 2.059: renderer
parity is NOT target-shape accuracy. These targets are synthesized base renders,
not independent reference images. Pixel scale is the existing assumed 290 mm
reference divided by 96 pixels, not newly measured physical calibration.
Playback follows commanded time, not measured force/contact feedback.

Restart the Beta launch to load updated Python modules:

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
source install_official/setup.bash
ros2 launch fusion_model_ros2_beta ur10_brush_ros2_beta.launch.py
```

Do not run a second launch while the old Beta launch still owns its controller
and web port. Stop that launch with Ctrl+C first. The Beta launch web port is 18082.
