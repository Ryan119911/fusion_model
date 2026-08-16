import numpy as np

from utils.kaishu_style_features import (
    STYLE_FEATURE_CHANNEL_NAMES,
    build_style_features,
)
from utils.types import (
    CharacterTrajectory,
    PointState,
    StrokeTrajectory,
    TrajectoryPoint,
)


def _trajectory() -> CharacterTrajectory:
    points = [
        TrajectoryPoint(0, 0, 0.20, 0.25, 18.0, 0.0, 0.0, 0.0, PointState.DOWN),
        TrajectoryPoint(0, 1, 0.50, 0.50, 15.0, 0.0, 0.0, 0.0, PointState.MOVE),
        TrajectoryPoint(0, 2, 0.80, 0.75, 11.0, 0.0, 0.0, 0.0, PointState.UP),
    ]
    return CharacterTrajectory("测", [StrokeTrajectory(0, points)])


def test_v16_features_include_footprint_pressure_and_speed_channels():
    gray = np.zeros((128, 128), dtype=np.float32)
    gray[28:101, 60:68] = 1.0
    features, metrics = build_style_features(gray, _trajectory())

    assert features.shape == (len(STYLE_FEATURE_CHANNEL_NAMES), 128, 128)
    assert np.isfinite(features).all()
    assert features[4].max() > 0.0  # local footprint width
    assert features[7].max() > 0.0  # z-derived pressure proxy
    assert features[11].max() > 0.0  # displacement-derived speed proxy
    assert 0.0 <= metrics["trajectory_target_coverage"] <= 1.0
    assert 0.0 <= metrics["support_dice"] <= 1.0
