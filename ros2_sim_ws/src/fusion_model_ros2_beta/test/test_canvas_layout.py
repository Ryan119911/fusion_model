import math
import pytest
from fusion_model_ros2_beta.canvas_layout import plan_canvas


def test_legacy_vertical_area_matches_glyph_aspect():
    plan = plan_canvas([(1,1), (1.15584753755,1)], .8, .4, 'vertical')
    assert plan[0]['font_size_m'] == pytest.approx(.1625)
    assert plan[1]['font_size_m'] == pytest.approx(.18782522485)
    assert plan[0]['center_y_m'] > plan[1]['center_y_m']
    assert plan[0]['center_y_m']-plan[1]['center_y_m'] == pytest.approx(.2075)


def test_single_stroke_does_not_expand_minor_noise():
    plan = plan_canvas([(1, .00001)], .52, .32, 'vertical')
    assert plan[0]['font_size_m'] == pytest.approx(.29)


def test_reference_boundary_is_canonicalized_for_offline_backend():
    plan = plan_canvas([(1, 1)], .52, .32, 'horizontal')
    assert plan[0]['font_size_m'] == .29
    assert plan[0]['font_size_m'] <= .29


@pytest.mark.parametrize('minor', [0., .00001, .09003576666264389, .119])
def test_single_stroke_uses_virtual_square_at_both_requested_sizes(minor):
    vertical = plan_canvas([(1.,minor)], .4, .19, 'vertical')
    horizontal = plan_canvas([(1.,minor)], .25, .4, 'horizontal')
    assert vertical[0]['font_size_m'] == pytest.approx(.16)
    assert horizontal[0]['font_size_m'] == pytest.approx(.22)


def test_horizontal_and_width_changes():
    small = plan_canvas([(1,1), (1,1)], .4, .32, 'horizontal')
    large = plan_canvas([(1,1), (1,1)], .52, .32, 'horizontal')
    assert small[0]['font_size_m'] < large[0]['font_size_m']
    assert large[0]['center_x_m'] < large[1]['center_x_m']


@pytest.mark.parametrize('width,height', [(float('nan'), .3), (.4,0), (.9,.3), (.4,.5), (.01,.01), (.8,.4)])
def test_invalid_or_model_unsupported_area(width,height):
    with pytest.raises(ValueError):
        plan_canvas([(1,1)], width, height, 'horizontal')
