from tools.calibrate_style_support import select_support_scale


def candidate(scale, heldout_mse, heldout_balance, mse, iou, balance):
    return {
        "scale": scale,
        "heldout": {
            "mse": heldout_mse,
            "ink_balance_score": heldout_balance,
        },
        "canonical": {
            "mse": mse,
            "iou": iou,
            "ink_balance_score": balance,
        },
    }


def test_support_selection_rejects_heldout_regression():
    values = [
        candidate(1.0, 0.00745, 0.96, 0.0183, 0.747, 0.91),
        candidate(1.5, 0.00748, 0.98, 0.0176, 0.752, 0.93),
        candidate(2.0, 0.00800, 0.99, 0.0173, 0.753, 0.95),
    ]
    selected, audited = select_support_scale(
        values, max_heldout_mse_regression=0.0001
    )
    assert selected["scale"] == 1.5
    assert [item["eligible"] for item in audited] == [True, True, False]


def test_support_selection_falls_back_to_baseline():
    values = [
        candidate(1.0, 0.007, 0.97, 0.018, 0.75, 0.95),
        candidate(1.5, 0.008, 0.90, 0.017, 0.74, 0.90),
    ]
    selected, _ = select_support_scale(values)
    assert selected["scale"] == 1.0
