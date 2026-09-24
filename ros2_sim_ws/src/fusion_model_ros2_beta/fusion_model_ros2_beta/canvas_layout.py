"""Legacy rectangular glyph cells; output physical sizes for offline inversion."""
import math


def plan_canvas(spans, width, height, layout, *, paper_width=.8, paper_height=.4,
                gap=.045, margin=.018, reference=.29):
    if layout not in ('horizontal', 'vertical'):
        raise ValueError('无效排版方向')
    if not spans or not all(math.isfinite(v) and v > 0 for v in (width, height)):
        raise ValueError('书写区域长宽必须为大于零的有限数值')
    if width > paper_width or height > paper_height:
        raise ValueError('书写区域不能超过纸面尺寸')
    cell_margin = min(margin, max(.001, gap/3))
    axis = width if layout == 'horizontal' else height
    glyph_axis = (axis-2*cell_margin-gap*(len(spans)-1))/len(spans)
    if glyph_axis <= 2*cell_margin:
        raise ValueError('书写区域太小，无法容纳文字和字间距')
    cell_axis = glyph_axis+2*cell_margin
    cw, ch = (cell_axis, height) if layout == 'horizontal' else (width, cell_axis)
    result = []
    for index, (sx, sy) in enumerate(spans):
        dominant = max(sx, sy)
        if not all(math.isfinite(v) and v >= 0 for v in (sx,sy)) or dominant <= 1e-9:
            raise ValueError('轨迹 XY 范围无效')
        nx = sx if sx > .12*dominant else dominant
        ny = sy if sy > .12*dominant else dominant
        size = dominant*min((cw-2*cell_margin)/nx, (ch-2*cell_margin)/ny)
        if size <= 0 or size > reference+1e-12:
            raise ValueError(f'该区域推算字号 {size*1000:.1f} mm 超出当前反演范围（上限 {reference*1000:.0f} mm），请缩小区域')
        # Floating-point layout arithmetic can produce
        # 0.29000000000000004 for the supported 0.29 m boundary.  Keep the
        # validated tolerance above, then canonicalize the value passed to the
        # offline inversion backend so an exactly-full canvas remains usable.
        size = min(size, reference)
        raw_gap = max(0., gap-2*cell_margin)
        x = -width/2+cw/2+index*(cw+raw_gap) if layout == 'horizontal' else 0.
        y = height/2-ch/2-index*(ch+raw_gap) if layout == 'vertical' else 0.
        result.append(dict(font_size_m=size, center_x_m=x, center_y_m=y,
                           writing_width_m=width, writing_height_m=height))
    return result
