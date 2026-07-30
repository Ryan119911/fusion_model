import numpy as np

from tools.snap_trajectory_to_target import (
    nearest_skeleton_displacements,
    smooth_displacements,
    thin_binary,
)


def test_thinning_and_snap_reduce_distance():
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:24, 14:19] = True
    skeleton = thin_binary(mask)
    assert 8 <= int(skeleton.sum()) <= 20
    xy = np.asarray([[10.0, 10.0], [11.0, 16.0], [10.0, 22.0]])
    displacement, before = nearest_skeleton_displacements(xy, skeleton, 8.0)
    smoothed = smooth_displacements(
        displacement, np.zeros(len(xy), dtype=np.int64), sigma=0.5
    )
    after_xy = xy + smoothed
    _, after = nearest_skeleton_displacements(after_xy, skeleton, 100.0)
    assert float(after.mean()) < float(before.mean())
    assert float(np.linalg.norm(smoothed, axis=1).max()) <= 8.0
