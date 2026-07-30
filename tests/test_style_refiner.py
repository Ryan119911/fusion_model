import numpy as np
import torch

from models.style_refiner import StyleRefinerUNet
from tools.build_kaishu_style_dataset import geometry_features
from tools.train_kaishu_style_refiner import grouped_split, ranked_adaptation_split


def test_style_refiner_shape_and_identity_initialization():
    model = StyleRefinerUNet(base_channels=8, depth=2, dropout=0)
    values = torch.rand(2, 4, 32, 32)
    prediction = model(values)
    assert prediction.shape == (2, 1, 32, 32)
    assert torch.allclose(prediction, values[:, :1], atol=1e-4)


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
