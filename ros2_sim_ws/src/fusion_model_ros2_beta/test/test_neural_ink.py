from types import SimpleNamespace as NS
import numpy as np
import pytest
from fusion_model_ros2_beta.neural_ink import frame_quads, schedule_frames


def test_pixels_keep_opacity_orientation_and_metric_size():
    frame = np.array([[0., .2], [1., .001]])
    vertices, alpha = frame_quads(frame, [10., 20.], .003)
    np.testing.assert_array_equal(alpha, [.2, 1., .001])
    np.testing.assert_allclose(vertices[0].mean(axis=0), [10.003, 20.])
    np.testing.assert_allclose(vertices[1].mean(axis=0), [10., 19.997])
    for triangle in vertices.reshape(-1, 3, 2):
        assert np.linalg.det(np.array([triangle[1]-triangle[0], triangle[2]-triangle[0]])) > 0


def test_order_and_glyph_identity_exclude_airborne_targets():
    def target(x, state, stroke):
        return NS(point=NS(x=x, y=0., state=state, stroke_id=stroke))
    targets = [target(0,3,0), target(0,1,0), target(1,2,0),
               target(0,3,100000), target(0,1,100000), target(1,2,100000)]
    xy = np.array([[0.,0.], [.5,0.], [1.,0.]])
    first = schedule_frames(xy, [0,0,0], targets, 0)
    second = schedule_frames(xy, [0,0,0], targets, 100000)
    assert np.all(np.diff(first)>=0)
    assert first[-1] < second[0]
    assert np.searchsorted(second, 3, side='right') == 0
    with pytest.raises(ValueError):
        schedule_frames(xy, [8,8,8], targets, 0)


def test_empty_frame():
    vertices, alpha = frame_quads(np.zeros((3,3)), [0.,0.], .003)
    assert vertices.shape == (0,6,2)
    assert len(alpha) == 0
