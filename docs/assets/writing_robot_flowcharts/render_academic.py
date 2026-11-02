from pathlib import Path
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.font_manager import FontProperties

HERE = Path(__file__).resolve().parent
GRAPHS = json.loads((HERE / 'flowcharts.json').read_text())
FONT = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
BLUE = '#EAF3FC'
GREEN = '#EAF5EE'
EDGE = '#2E4053'
TEXT = '#172B3A'

# Explicit node frames and Manhattan routes keep every connector orthogonal.
LAYOUTS = {
    '01_system': {
        'size': (1000, 1150),
        'nodes': {
            'input': (180, 40, 520, 70),
            'inverse': (180, 160, 520, 90),
            'candidates': (180, 300, 520, 90),
            'mapping': (180, 440, 520, 90),
            'write': (180, 580, 520, 90),
            'observe': (180, 720, 520, 90),
            'measure': (180, 860, 520, 90),
            'update': (180, 1000, 520, 90),
        },
        'routes': {
            ('update', 'mapping'): [(700, 1045), (880, 1045), (880, 485), (700, 485)],
        },
        'labels': {
            ('observe', 'measure'): (510, 835),
            ('update', 'mapping'): (790, 465),
        },
    },
    '02_stroke_loop': {
        'size': (1480, 1320),
        'nodes': {
            'target': (180, 40, 500, 90),
            'calibration': (180, 180, 500, 90),
            'generate': (180, 320, 500, 120),
            'write': (180, 490, 500, 70),
            'measure': (180, 610, 500, 100),
            'accept': (230, 755, 400, 150),
            'keep': (180, 970, 500, 90),
            'correct': (850, 770, 400, 120),
            'valid': (850, 965, 400, 150),
            'recalibrate': (850, 1170, 400, 90),
        },
        'routes': {
            ('accept', 'correct'): [(630, 830), (850, 830)],
            ('valid', 'generate'): [(1250, 1040), (1360, 1040), (1360, 380), (680, 380)],
            ('recalibrate', 'calibration'): [(1250, 1215), (1420, 1215), (1420, 225), (680, 225)],
        },
        'labels': {
            ('accept', 'keep'): (465, 938),
            ('accept', 'correct'): (740, 808),
            ('valid', 'generate'): (1040, 357),
            ('valid', 'recalibrate'): (1085, 1142),
        },
    },
    '03_experiment': {
        'size': (1490, 1510),
        'nodes': {
            'conditions': (180, 40, 560, 110),
            'design': (180, 200, 560, 120),
            'collect': (180, 370, 560, 90),
            'measure': (180, 510, 560, 110),
            'separable': (240, 670, 440, 150),
            'fit': (180, 885, 560, 90),
            'holdout': (180, 1025, 560, 90),
            'pass': (240, 1170, 440, 140),
            'glyph': (180, 1370, 560, 90),
            'investigate': (900, 915, 400, 140),
        },
        'routes': {
            ('separable', 'investigate'): [(680, 745), (1100, 745), (1100, 915)],
            ('pass', 'investigate'): [(680, 1240), (1100, 1240), (1100, 1055)],
            ('investigate', 'conditions'): [(1300, 985), (1410, 985), (1410, 95), (740, 95)],
        },
        'labels': {
            ('separable', 'fit'): (495, 854),
            ('separable', 'investigate'): (800, 723),
            ('pass', 'glyph'): (505, 1340),
            ('pass', 'investigate'): (810, 1218),
        },
    },
}


def intersects_frame(p, q, frame):
    x, y, w, h = frame
    inset = 1.0
    if p[0] == q[0]:
        return x + inset < p[0] < x + w - inset and max(min(p[1], q[1]), y + inset) < min(max(p[1], q[1]), y + h - inset)
    return y + inset < p[1] < y + h - inset and max(min(p[0], q[0]), x + inset) < min(max(p[0], q[0]), x + w - inset)


for graph in GRAPHS:
    layout = LAYOUTS[graph['name']]
    width, height = layout['size']
    frames = layout['nodes']
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    for a, b, label, kind in graph['edges']:
        if (a, b) in layout['routes']:
            points = layout['routes'][(a, b)]
        else:
            x, y, w, h = frames[a]
            xx, yy, ww, hh = frames[b]
            assert x + w / 2 == xx + ww / 2, (a, b)
            points = [(x + w / 2, y + h), (xx + ww / 2, yy)]
        for p, q in zip(points, points[1:]):
            assert p[0] == q[0] or p[1] == q[1], (a, b, p, q)
            for key, frame in frames.items():
                if key not in (a, b):
                    assert not intersects_frame(p, q, frame), (a, b, key)
        tip = points[-1]
        prev = points[-2]
        distance = math.dist(prev, tip)
        ux = (tip[0] - prev[0]) / distance
        uy = (tip[1] - prev[1]) / distance
        base = (tip[0] - ux * 12, tip[1] - uy * 12)
        shaft = points[:-1] + [base]
        ax.plot([p[0] for p in shaft], [p[1] for p in shaft], color=EDGE, linewidth=1.6, solid_capstyle='butt', solid_joinstyle='miter', zorder=1)
        arrow = [tip, (base[0] - uy * 5.5, base[1] + ux * 5.5), (base[0] + uy * 5.5, base[1] - ux * 5.5)]
        ax.add_patch(Polygon(arrow, closed=True, facecolor=EDGE, edgecolor=EDGE, linewidth=0.5, zorder=2))
        if label:
            lx, ly = layout['labels'][(a, b)]
            ax.text(lx, ly, label, ha='center', va='center', fontproperties=FONT, fontsize=16, color=TEXT,
                    bbox=dict(facecolor='white', edgecolor='none', pad=2.5), zorder=4)

    for key, label, kind in graph['nodes']:
        x, y, w, h = frames[key]
        fill = GREEN if kind in ('planned', 'decision', 'end') else BLUE
        if kind == 'decision':
            shape = Polygon([(x + w / 2, y), (x + w, y + h / 2), (x + w / 2, y + h), (x, y + h / 2)], closed=True,
                            facecolor=fill, edgecolor=EDGE, linewidth=1.4, zorder=3)
        else:
            shape = Rectangle((x, y), w, h, facecolor=fill, edgecolor=EDGE, linewidth=1.4, zorder=3)
        ax.add_patch(shape)
        lines = label.split('\n')
        for i, line in enumerate(lines):
            yy = y + h / 2 + (i - (len(lines) - 1) / 2) * 30
            ax.text(x + w / 2, yy, line, ha='center', va='center', fontproperties=FONT, fontsize=18, color=TEXT, zorder=4)

    stem = HERE / (graph['name'] + '_academic')
    plt.rcParams['svg.fonttype'] = 'none'
    fig.savefig(stem.with_suffix('.svg'), facecolor='white', metadata={'Date': None})
    fig.savefig(stem.with_suffix('.png'), dpi=180, facecolor='white')
    plt.close(fig)

    mmd = ['%%{init: {"theme": "base", "flowchart": {"curve": "stepAfter", "nodeSpacing": 45, "rankSpacing": 55}, "themeVariables": {"fontFamily": "Noto Sans CJK SC, sans-serif", "fontSize": "18px", "lineColor": "#2E4053"}}}%%', 'flowchart TB']
    for key, label, kind in graph['nodes']:
        text = label.replace('\n', '<br/>')
        mmd.append(f'    {key}{{"{text}"}}' if kind == 'decision' else f'    {key}["{text}"]')
    for a, b, label, kind in graph['edges']:
        annotation = f'|{label}|' if label else ''
        mmd.append(f'    {a} -->{annotation} {b}')
    mmd.extend([
        '    classDef blue fill:#EAF3FC,stroke:#2E4053,stroke-width:1.6px,color:#172B3A,rx:0,ry:0;',
        '    classDef green fill:#EAF5EE,stroke:#2E4053,stroke-width:1.6px,color:#172B3A,rx:0,ry:0;',
    ])
    for cls in ('blue', 'green'):
        keys = [key for key, _, kind in graph['nodes'] if ('green' if kind in ('planned', 'decision', 'end') else 'blue') == cls]
        if keys:
            mmd.append(f'    class {",".join(keys)} {cls};')
    stem.with_suffix('.mmd').write_text('\n'.join(mmd) + '\n')

print('Rendered 3 academic flowcharts: aligned frames; all connector segments horizontal or vertical; no connector crosses another node.')
