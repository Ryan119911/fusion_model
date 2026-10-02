"""Create a human-readable report from completed, hashed A/B/C evidence."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path


def create(root):
    root=Path(root)
    report=json.loads((root/'verification.json').read_text())
    if not report['complete']: raise ValueError('full A/B/C and both actual canvas sizes are not complete')
    passed=report['accepted']
    lines=['# V16 原图—UR10 腕部运动完整对照验收','',
        '结论：'+('通过离线验收。' if passed else '未通过整体验收。字迹质量与腕部运动分别评价，不能凭减少转动注册正式轨迹。'),'',
        '固定原图 wu_kaishu_target.png、初值、冻结 V16、16 步/order 11；520×320 mm 当前画布推导单格/三格中央格字号。',
        'A 无新增旋转代价；B 仅绝对工具姿态权重10；C 同时加入真实腕部权重10，并独立优化提笔转换。',
        '关节运动来自工厂标定 UR10 连续 IK，不是 gamma 代替，也不是真实机械臂测量。所有差值不取模。',
        '没有发布运动、没有切换正式默认配置、没有修改 CoppeliaSim。','',
        '## 原图质量与机器人转动','',
        '| 字号(mm) | 组 | IoU | SSIM | MSE | q4(°) | q5(°) | q6(°) | 合计(°) | 最大内部步长(°) | 规划时长(s) | 机器人约束 | 字迹绝对门槛 |',
        '|---:|:--|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--|:--|']
    stroke_rows=[]
    for experiment in report['experiments']:
        for r in experiment['results']:
            fmt=lambda x:f'{x:.3f}' if x is not None else '—'
            joints=r['q4_q5_q6_deg']
            step=math.degrees(r['max_internal_step_rad']) if r['max_internal_step_rad'] is not None else None
            lines.append('| '+ ' | '.join([fmt(r['font_size_m']*1000),r['case'],f"{r['iou']:.6f}",
                f"{r['ssim']:.6f}",f"{r['mse']:.6f}",*[fmt(v) for v in joints],fmt(r['wrist_deg']),
                fmt(step),fmt(r['duration_s']),'通过' if r['robot_passed'] else '未通过',
                '通过' if r['absolute_visual_passed'] else '未通过'])+' |')
    lines+=['','## 接触/提笔/初始接近分项','',
        '每格为 q4/q5/q6（度）及三者合计。初始接近不计入纯提笔。','',
        '| 字号(mm) | 组 | 接触 | 两笔之间纯提笔 | 初始接近 |',
        '|---:|:--|:--|:--|:--|']
    for experiment in report['experiments']:
        for r in experiment['results']:
            phases=[]
            for key in ('contact_deg','pure_penup_deg','approach_deg'):
                vals=r[key]
                phases.append(' / '.join(f'{v:.2f}' for v in vals)+f'；合计 {sum(vals):.2f}' if vals else '未生成可用完整路径')
            lines.append(f"| {r['font_size_m']*1000:.3f} | {r['case']} | "+' | '.join(phases)+' |')
    lines+=['','## 联合代价及提笔优化消融','']
    for experiment in report['experiments']:
        cases={r['case']:r for r in experiment['cases']};c=cases['C'];p=c['planning']
        rows={r['case']:r for r in experiment['results']};a,b,cr=(rows[k] for k in 'ABC')
        lines.append(f"### {cr['font_size_m']*1000:.3f} mm\n")
        if all(r['wrist_deg'] is not None for r in (a,b,cr)):
            lines.append(f"C 对 A 腕部减少 {(1-cr['wrist_deg']/a['wrist_deg'])*100:.2f}%，对 B 减少 {(1-cr['wrist_deg']/b['wrist_deg'])*100:.2f}%；C-A IoU 差 {cr['iou']-a['iou']:+.6f}。\n")
        if p['feasible']:
            lines.append(f"只优化提笔（同一 C 接触轨迹）：腕部 {p['legacy']['wrist_total_deg']:.3f}° → {p['penup_optimized']['wrist_total_deg']:.3f}°。全部接触姿态/关节逐项未变，逐笔回零={p['penup_audit']['per_stroke_home_reset']}。\n")
            lines.append(f"反馈 IK 与最终规划源控制点关节增量最大差：{p['feedback_vs_planned_source_joint_delta_error_rad']:.8g} rad。\n")
        else:lines.append('C 完整规划被拒绝：'+p.get('error','unknown')+'\n')
        feedback=c['lm']['diagnostics']['actual_ur10_wrist_feedback']
        lines.append('进入反演的腕部残差 Jacobian 分项范数：`'+json.dumps(feedback['residual_jacobian_field_norms'])+'`。\n')
    lines+=['## 逐笔画原图残差','',
        '以下按源笔画近邻 Voronoi 分区，报告 IoU、额外/缺失前景像素和质心偏移；不是人工真实笔画标注。','',
        '| 字号(mm) | 组 | 笔画ID | IoU | 额外墨(px) | 缺墨(px) | 质心偏移(px) |',
        '|---:|:--|---:|---:|---:|---:|---:|']
    for experiment in report['experiments']:
        for c in experiment['cases']:
            for stroke in c['stroke_residuals']['strokes']:
                row=dict(font_mm=experiment['canvas']['font_size_m']*1000,case=c['case'],
                    stroke_id=stroke['stroke_id'],iou=stroke['iou'],extra_ink_px=stroke['extra_ink_px'],
                    missing_ink_px=stroke['missing_ink_px'],centroid_offset_px=stroke['centroid_offset_px'])
                stroke_rows.append(row)
                lines.append(f"| {row['font_mm']:.3f} | {row['case']} | {row['stroke_id']} | {row['iou']:.6f} | {row['extra_ink_px']} | {row['missing_ink_px']} | {row['centroid_offset_px']:.3f} |")
    lines+=['','## 约束和边界','',
        '各 case 的 result.json 保存严格IK、接触FK误差、工厂标定关节限位、无取模连续性、速度、现有奇异门槛、全链归一化 Jacobian 最小奇异值和碰撞检查。',
        '碰撞范围是当前连杆包络、纸桌及注册工具/相机OBB；不构成完整场景网格自碰撞或硬件安全认证。',
        '真实关节反馈含接触点与跨笔端点增量；独立提笔规划处理路径绕行附加运动。',
        '未验证真实加速度、力矩、压力和机器人实际执行；初始关节固定为当前规划安全初值，不是本次读取的实际硬件关节。',
        '绝对字迹门槛 IoU≥0.95、SSIM≥0.90；相对下降≤0.01不能替代绝对门槛。高背景SSIM不能替代前景IoU。','',
        '## 复现与证据','',
        '完整说明、ROS显式参数和命令见 README_wrist_validation.md。执行 `bash run_tests.sh`、`bash run_full_abc.sh`、`sha256sum -c verification.sha256`。',
        '所有输出路径以 *_result.json 指向的 candidate 为准，不使用目录修改时间选旧缓存或未完成结果。',
        'verification.json 保存图像、代码、测试、CSV、模型流和几何资源SHA256；verification.sha256保存报告本身SHA。','']
    (root/'acceptance_report.md').write_text('\n'.join(lines),encoding='utf-8')
    with (root/'stroke_residuals.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(stroke_rows[0]));w.writeheader();w.writerows(stroke_rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    create(parser.parse_args().output)
