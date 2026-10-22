"""Necessary offline checks; a pass is NOT robot or visual acceptance."""
import math


def check_candidate(reference, candidate):
    """Inspect ordered control samples, never connecting separate strokes.

    Units follow inversion CSV: XY source units, z millimetres, angles radians.
    No arbitrary speed limit is inferred from untimed samples.
    """
    fields = ('x', 'y', 'z', 'alpha', 'beta', 'gamma')
    ids = lambda rows: [(str(r['stroke_id']), str(r['point_id'])) for r in rows]
    reasons = []
    if not reference or ids(reference) != ids(candidate):
        return {'eligible_for_further_validation': False,
                'reasons': ['empty_or_changed_sample_sequence']}
    for rows in (reference, candidate):
        try:
            finite = all(math.isfinite(float(r[k])) for r in rows for k in fields)
        except (ValueError, KeyError, TypeError):
            finite = False
        if not finite:
            return {'eligible_for_further_validation': False,
                    'reasons': ['invalid_control_values']}
    reversed_segments = []
    collapsed_segments = []
    max_depth_step = 0.0
    max_angle_step = 0.0
    angle_steps = dict.fromkeys(('alpha', 'beta', 'gamma'), 0.0)
    for i in range(1, len(candidate)):
        a, b = candidate[i-1:i+1]
        ra, rb = reference[i-1:i+1]
        if str(a['stroke_id']) != str(b['stroke_id']):
            continue
        dx, dy = (float(b[k])-float(a[k]) for k in ('x', 'y'))
        rx, ry = (float(rb[k])-float(ra[k]) for k in ('x', 'y'))
        if math.hypot(rx, ry) > 1e-9:
            if math.hypot(dx, dy) <= 1e-9:
                collapsed_segments.append(i)
            elif dx*rx + dy*ry < 0:
                reversed_segments.append(i)
        max_depth_step = max(max_depth_step, abs(float(b['z'])-float(a['z'])))
        for k in ('alpha', 'beta', 'gamma'):
            d = float(b[k])-float(a[k])
            step = abs(math.atan2(math.sin(d), math.cos(d)))
            angle_steps[k] = max(angle_steps[k], step)
            max_angle_step = max(max_angle_step, step)
    if reversed_segments:
        reasons.append('within_stroke_segment_reversal')
    if collapsed_segments:
        reasons.append('within_stroke_segment_collapse')
    return dict(eligible_for_further_validation=not reasons, reasons=reasons,
                reversed_segment_end_indices=reversed_segments,
                collapsed_segment_end_indices=collapsed_segments,
                max_depth_step_mm=max_depth_step, max_wrapped_angle_step_rad=max_angle_step,
                max_wrapped_step_by_angle_rad=angle_steps,
                robot_safe=None, visual_accepted=None,
                note='Necessary geometry checks only; untimed samples cannot establish velocity or collision safety.')
