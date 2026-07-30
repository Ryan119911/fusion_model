import numpy as np
import torch

from models.style_refiner import StyleRefinerUNet
from tools.build_kaishu_style_dataset import geometry_features
from tools.train_kaishu_style_refiner import (
    grouped_split,
    loss_components,
    ranked_adaptation_split,
)


def test_style_refiner_shape_and_geometry_gate_initialization():
    model = StyleRefinerUNet(base_channels=8, depth=2, dropout=0)
    values = torch.rand(2, 4, 32, 32)
    prediction = model(values)
    assert prediction.shape == (2, 1, 32, 32)
    assert torch.allclose(
        prediction / values[:, :1].clamp_min(1e-6),
        torch.full_like(prediction, torch.sigmoid(torch.tensor(4.0))),
        atol=1e-4,
    )
    zeros = values.clone()
    zeros[:, 0] = 0
    assert torch.count_nonzero(model(zeros)) == 0


def test_soft_support_mode_retains_bounded_antialiased_geometry():
    model = StyleRefinerUNet(
        base_channels=8,
        depth=2,
        dropout=0,
        support_mode="mask_or_soft",
    )
    values = torch.zeros(1, 4, 32, 32)
    values[:, 0, 10:22, 10:22] = 1
    values[:, 3, 8:24, 8:24] = 0.25
    prediction = model(values)
    support = torch.maximum(values[:, :1], values[:, 3:4])
    active = support > 0
    assert torch.allclose(
        prediction[active] / support[active],
        torch.full_like(
            prediction[active], torch.sigmoid(torch.tensor(4.0))
        ),
        atol=1e-4,
    )
    assert torch.count_nonzero(prediction[:, :, :8]) == 0
    assert torch.count_nonzero(prediction[:, :, 8:10, 8:24]) > 0


def test_geometry_features_are_finite():
    gray = np.zeros((32, 32), dtype=np.float32)
    gray[8:24, 13:19] = 1
    features = geometry_features(gray)
    assert features.shape == (4, 32, 32)
    assert np.isfinite(features).all()
    assert features.min() >= 0 and features.max() <= 1


def test_grouped_split_excludes_heldout_and_sources_do_not_overlap():
    characters = np.asarray(["甲", "乙", "武", "丙", "丁", "武"])
    sources = np.asarray(["a", "a", "w1", "b", "c", "w2"])
    train, val, heldout = grouped_split(characters, sources, "武", 0.34, 3)
    assert set(characters[heldout]) == {"武"}
    assert "武" not in set(characters[train])
    assert "武" not in set(characters[val])
    assert not set(sources[train]).intersection(sources[val])


def test_ranked_adaptation_split(tmp_path):
    audit = tmp_path / "audit.json"
    audit.write_text(
        '{"ranked_candidates":[{"image_path":"a.jpg"},{"image_path":"b.jpg"}]}',
        encoding="utf-8",
    )
    sources = np.asarray(["a.jpg", "c.jpg", "b.jpg"])
    adapt, test = ranked_adaptation_split(np.arange(3), sources, str(audit), 1)
    assert adapt.tolist() == [0]
    assert test.tolist() == [1, 2]


def test_style_loss_penalizes_faint_ink_and_reports_balance():
    target = torch.zeros(2, 1, 32, 32)
    target[:, :, 8:24, 10:22] = 0.8
    geometry = (target > 0).float()
    exact = loss_components(target, target, geometry)
    faint = loss_components(target * 0.5, target, geometry)
    assert exact["ink_balance_score"] == 1
    assert faint["ink_ratio"] < 0.51
    assert faint["under_ink"] > 0
    assert faint["ink_loss"] > exact["ink_loss"]
    assert faint["local_ink_loss"] > exact["local_ink_loss"]
    assert faint["loss"] > exact["loss"]
