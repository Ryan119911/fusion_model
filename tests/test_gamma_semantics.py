import math

import torch
import torch.nn as nn


class _Recorder(nn.Module):
    def __init__(self):
        super().__init__()
        self.last = None

    def forward(self, params):
        self.last = params.detach().clone()
        return torch.zeros((len(params), 1, 16, 16), dtype=params.dtype, device=params.device)


def test_absolute_forward_gamma_is_converted_to_local_rotation():
    from models.paper_fusion_renderer import PaperDynamicConfig, PaperFusionRenderer

    recorder = _Recorder()
    renderer = PaperFusionRenderer(
        recorder,
        {
            "input_dim": 6,
            "scales": [20.0, 1.0, 1.0, math.pi, 128.0, 128.0],
            "regression_angle_basis": "paper_declared_radian",
        },
        image_size=16,
        dynamic=PaperDynamicConfig(
            gamma_mode="relative_to_heading",
            render_max_step_px=2.0,
        ),
    )
    xy = torch.tensor([[4.0, 4.0], [4.0, 8.0]])
    posture = torch.tensor([[15.0, 0.0, 0.0], [15.0, 0.0, 0.0]])
    stroke_ids = torch.zeros(2, dtype=torch.long)
    # Absolute forward heading for a vertical stroke is +pi/2.  The local
    # gamma seen by the decoder should therefore be approximately zero.
    gamma = torch.full((2,), math.pi / 2.0)
    renderer(xy, posture, stroke_ids, gamma)
    assert recorder.last is not None
    assert float(recorder.last[:, 3].abs().max()) < 1e-5
