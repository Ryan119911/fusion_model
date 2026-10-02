import numpy as np
import pytest
from fusion_model_ros2_beta.original_image_metrics import ssim


def test_identical_image_ssim():
    x=np.random.default_rng(7).random((128,128))
    assert abs(ssim(x,x)-1.)<1e-14


def test_ssim_matches_standard_library():
    metrics=pytest.importorskip('skimage.metrics')
    x=np.random.default_rng(7).random((128,128));y=np.clip(x+.1,0,1)
    assert abs(ssim(x,y)-metrics.structural_similarity(x,y,data_range=1.,win_size=7))<1e-12
