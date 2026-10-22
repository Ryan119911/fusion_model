import numpy as np
import pytest
from fusion_model_ros2_beta.joint_continuity import continuous_joint_path, nearest_equivalent


def test_ik_wrap_boundary_keeps_short_motion():
    result = nearest_equivalent(np.radians([-179]), np.radians([179]), [-2*np.pi], [2*np.pi])
    np.testing.assert_allclose(np.degrees(result), [181])


def test_global_plan_avoids_greedy_limit_reset():
    values = np.radians([[0], [170], [250], [350], [7], [60]])
    result = continuous_joint_path(values, [0], [-2*np.pi], [2*np.pi])
    assert result[0, 0] == 0
    assert np.max(abs(np.diff(result[:, 0])[1:])) <= np.pi
    np.testing.assert_allclose(np.exp(1j*result), np.exp(1j*values), atol=1e-12)


def test_unavoidable_limit_reset_is_rejected():
    values = np.radians([[0], [0], [100], [200], [300], [400], [500], [600], [700], [800]])
    with pytest.raises(ValueError, match="replan IK branch"):
        continuous_joint_path(values, [0], [-2*np.pi], [2*np.pi])
