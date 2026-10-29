#!/usr/bin/env python3
"""Read the frozen complete gong trajectory. Local files only; no robot access."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'outputs/gong_brush_a4_20261016_315mm/scaled_initial_pose.numeric.csv'
REVIEW = ROOT/'outputs/gong_offline_review_20261021'
EXPECTED = '3fa5df4f20919678818e28cc8fa82530873cf420cad1fbb881a6275f624c60ef'


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def load(include_interpolated=False, include_robot_candidate=False):
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if digest != EXPECTED:
        raise ValueError('frozen source trajectory changed: '+digest)
    source = read_csv(SOURCE)
    paper = read_csv(REVIEW/'organized_controls.csv')
    counts = [sum(int(r['stroke_id']) == i for r in source) for i in range(3)]
    keys = lambda rows: [(int(r['stroke_id']), int(r['point_id'])) for r in rows]
    if len(source) != 16 or counts != [6, 4, 6] or keys(source) != keys(paper):
        raise ValueError('source/paper controls do not preserve the complete 6+4+6 route')
    result = {
        'mode': 'local_read_only', 'motion_sent': False, 'robot_connection_attempted': False,
        'source_csv': str(SOURCE), 'source_sha256': digest, 'control_counts': counts,
        'paper_controls_csv': str(REVIEW/'organized_controls.csv'),
        'model_checkpoint': str(ROOT/'outputs/paper_bbsmg_general_v16_pose_dense/bbsmg_best.pt'),
        'handoff': str(ROOT/'docs/交接_笔触模型完整轨迹与工字任务.md'),
        'semantics': {
            'source_x_y': 'dataset source coordinates, not mm and not robot Base coordinates',
            'source_z': 'model H in mm; not robot Z or measured brush compression',
            'angles': 'rad; model parameters, not controller Euler angles',
            'source_gamma': 'absolute native-canvas angle, y down',
            'paper_x_m_paper_y_m': 'existing paper-plane coordinates in metres, x right/y up; before trial translation',
            'gamma_conversion': 'source canvas absolute -> tangent local -> paper absolute, already recorded separately; convert once',
            'state': '0/1/2 stroke start/body/end, not a measured air/contact switch',
            'source_route': 'complete contact control points; lift/transfer/approach generated separately',
            'model': 'V16 B-BSMG predicts local footprint from posture; selected trajectory already exists; no retraining/inversion needed',
        },
        'source_rows': source, 'paper_controls': paper,
    }
    if include_interpolated:
        result['model_interpolated'] = {
            'source': str(REVIEW/'model_interpolated.csv'),
            'scope': '1166 model visualization samples; no robot timing or executable lift/transfer path',
            'rows': read_csv(REVIEW/'model_interpolated.csv'),
        }
    if include_robot_candidate:
        path = ROOT/'outputs/brush_mapping_20261027/gong_current_mount_runtime_geometry/candidate.json'
        result['robot_candidate'] = {'source': str(path),
            'scope': 'planned historical-start trial with approach/lift/transfer; not execution-ready; read current evidence and actual state',
            'data': json.loads(path.read_text())}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', action='store_true', help='all original CSV columns and organized columns as JSON; numeric CSV cells remain strings')
    parser.add_argument('--include-interpolated', action='store_true')
    parser.add_argument('--include-robot-candidate', action='store_true')
    args = parser.parse_args()
    data = load(args.include_interpolated, args.include_robot_candidate)
    if args.json or args.include_interpolated or args.include_robot_candidate:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return
    print('完整三笔“工”：6+4+6 控制点；只读本地文件，不连接或驱动机械臂。')
    print('原始轨迹：'+data['source_csv'])
    print('纸面字段：'+data['paper_controls_csv'])
    print('源 x/y 不是 mm；H(mm)不是实际下压量；角度(rad)不是机器人欧拉角。')
    print('stroke,point,source_x,source_y,H_mm,alpha_rad,beta_rad,gamma_canvas_abs_rad,paper_x_m,paper_y_m,gamma_local_rad,gamma_paper_abs_rad')
    for s, p in zip(data['source_rows'], data['paper_controls']):
        values = [s[k] for k in ('stroke_id','point_id','x','y','z','alpha','beta','gamma')]
        values += [p[k] for k in ('paper_x_m','paper_y_m','gamma_local_rad','gamma_absolute_paper_rad')]
        print(','.join(values))
    print('完整交接：'+data['handoff'])
    print('加 --json 查看全部原始字段；加 --include-interpolated / --include-robot-candidate 分别读取插值或机器人候选。')


if __name__ == '__main__':
    main()
