import numpy as np
import pytest
from fusion_model_ros2_beta.domain_seed import fit_gamma_seed


def test_dense_gamma_seed_satisfies_training_interval():
    matrix = np.eye(3)
    sweep = np.array([[1.,0.,0.], [.5,.5,0.], [0.,1.,0.], [0.,.5,.5], [0.,0.,1.]])
    heading = np.array([-.2,-.1,0.,.1,.2])
    points, logits = fit_gamma_seed(matrix,sweep,heading,np.full(3,3.14),return_node_logits=True)
    assert np.all(np.abs(sweep@points-heading) <= np.pi/6)
    np.testing.assert_allclose(np.pi*np.tanh(matrix@logits),points)


def test_conflicting_dense_constraints_are_rejected():
    with pytest.raises(ValueError, match='linear_relaxation_feasible=False'):
        fit_gamma_seed(np.eye(1),np.ones((2,1)),np.array([-1.,1.]),np.zeros(1))
