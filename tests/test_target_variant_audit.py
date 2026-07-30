import numpy as np

from tools.audit_character_target_variants import binary_overlap


def test_binary_overlap_identical_and_disjoint():
    left = np.zeros((8, 8), dtype=np.float32)
    left[1:3, 1:3] = 1
    identical = binary_overlap(left, left)
    assert identical["dice_to_canonical"] == 1.0
    assert identical["iou_to_canonical"] == 1.0

    right = np.zeros_like(left)
    right[5:7, 5:7] = 1
    disjoint = binary_overlap(left, right)
    assert disjoint["dice_to_canonical"] < 1e-5
    assert disjoint["iou_to_canonical"] < 1e-5
