import numpy as np

from fusion_model_ros2_beta.gamma_semantics import (
    absolute_to_local,
    forward_headings,
    local_to_absolute,
)


def test_local_gamma_roundtrip_across_multiple_strokes():
    xy = np.array([[0., 0.], [1., 0.], [1., 1.], [2., 0.], [2., -1.]])
    strokes = np.array([0, 0, 0, 1, 1])
    local = np.array([-.2, -.1, .1, .2, .3])
    absolute = local_to_absolute(local, xy, strokes)
    np.testing.assert_allclose(absolute_to_local(absolute, xy, strokes), local)
    np.testing.assert_allclose(
        forward_headings(xy, strokes), [0., np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2]
    )


def test_model_local_gamma_stays_inside_training_limit():
    local = np.linspace(-np.pi / 6, np.pi / 6, 9)
    xy = np.column_stack((np.arange(9), np.zeros(9)))
    recovered = absolute_to_local(local_to_absolute(local, xy, np.zeros(9)), xy, np.zeros(9))
    assert np.max(np.abs(recovered)) <= np.pi / 6 + 1e-12
