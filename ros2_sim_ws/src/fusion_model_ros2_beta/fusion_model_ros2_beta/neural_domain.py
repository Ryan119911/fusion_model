"""V16 pose-domain checks on actual neural inputs, never on rendered pixels.

Limits transcribed from paper_bbsmg_general_v16_pose_dense.summary.json.
They are training coverage bounds, not physical safety or accuracy guarantees.
"""
from contextlib import contextmanager
import numpy as np

LIMITS = {
    'H_mm': (11., 20.),
    'alpha_rad': (0., 0.1745329201221466),
    'beta_rad': (0., 0.0872664600610733),
    'gamma_relative_rad': (-0.5235987755982988, 0.5235987755982988),
}
CONTRACT = 'v16_actual_dense_pose_domain_v1'


@contextmanager
def capture_inputs(renderer):
    captured = []
    hook = renderer.bbsmg.register_forward_pre_hook(
        lambda model, inputs: captured.append(inputs[0]))
    try:
        yield captured
    finally:
        hook.remove()


def domain_report(inputs, normalization):
    values = np.asarray(inputs)
    names, scales = normalization['feature_names'], normalization['scales']
    if values.ndim != 2 or values.shape[1] != len(names) or not len(values):
        raise ValueError('invalid or empty actual neural inputs')
    if (len(scales) != len(names) or not np.all(np.isfinite(values))
            or not np.all(np.isfinite(scales)) or np.any(np.asarray(scales) <= 0)):
        raise ValueError('nonfinite or invalid neural inputs')
    raw = values * np.asarray(scales)
    features = {}
    for name, (lo, hi) in LIMITS.items():
        v = raw[:, names.index(name)]
        outside = (v < lo-1e-6) | (v > hi+1e-6)
        features[name] = dict(min=float(v.min()), max=float(v.max()),
            training_limits=[lo, hi], outside_count=int(outside.sum()))
    return dict(contract=CONTRACT, dense_count=len(raw), features=features,
                passed=all(f['outside_count'] == 0 for f in features.values()))


def domain_residual(inputs, normalization):
    """Fixed twelve residuals even when densification changes count.

Observes the unchanged renderer, retaining XY/pose/heading derivatives. This
is a soft optimization aid; a separate strict exported-CSV audit is mandatory.
"""
    import torch
    names = normalization['feature_names']
    raw = inputs * inputs.new_tensor(normalization['scales'])
    terms = []
    for name, (lo, hi) in LIMITS.items():
        v = raw[:, names.index(name)]
        violation = (torch.relu(lo-v) + torch.relu(v-hi))/(hi-lo)
        # Extrema alone give no correction signal to other offending samples.
        # RMS covers every dense input, without changing residual length or
        # making the penalty grow merely because more samples were inserted.
        # Smooth zero avoids sqrt(0)'s undefined gradient inside the domain.
        terms.append(torch.sqrt(violation.square().mean()+1e-12)-1e-6)
        terms.extend((torch.relu(lo-v.min())/(hi-lo),
                      torch.relu(v.max()-hi)/(hi-lo)))
    return torch.stack(terms)


def domain_margins(inputs, normalization, bins=16):
    """Local extrema covering every dense input, with fixed dimensions.

Global extrema hide other simultaneously active bounds. Overlapping index
bins cover all inputs even as sweep counts change; exact audit still decides.
"""
    import torch
    if bins < 1 or len(inputs) < 1:
        raise ValueError('nonempty inputs and positive bins required')
    raw=inputs*inputs.new_tensor(normalization['scales'])
    names=normalization['feature_names']
    terms=[]
    for name,(lo,hi) in LIMITS.items():
        for i in range(bins):
            start=i*len(raw)//bins
            stop=max(start+1,((i+1)*len(raw)+bins-1)//bins)
            v=raw[start:stop,names.index(name)]
            terms.extend(((v.min()-lo)/(hi-lo),(hi-v.max())/(hi-lo)))
    return torch.stack(terms)
