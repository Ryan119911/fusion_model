# 中文注释：本文件基于渲染残差优化轨迹控制点，使生成图像贴近目标书法图。
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
from PIL import Image

from utils.types import TrajectoryPoint, StrokeTrajectory, CharacterTrajectory, PointState
from models.fusion_renderer import FusionRenderer
from optim.chebyshev import fit_3d_nodes_from_points, parameterize_3d
from optim.lm import lm_solve, LMResult

from models.geometry import trajectory_bounds


# 中文注释：保存轨迹优化后的样本、图像、代价和迭代信息。
@dataclass
class TrajectoryOptimizationResult:
    order: int
    lm_result: LMResult
    optimized_sample: CharacterTrajectory
    target_image: np.ndarray
    rendered_image: np.ndarray


# 中文注释：读取目标灰度图并缩放到指定画布尺寸。
def load_target_image(path: str, image_size: int = 128) -> np.ndarray:
    img = Image.open(path).convert("L").resize((image_size, image_size))
    arr = np.array(img, dtype=np.float32) / 255.0

    # 统一极性：
    # 白底黑字 -> 黑底白字
    # 目标统一为：背景=0，墨迹=1。
    if arr.mean() > 0.5:
        arr = 1.0 - arr

    return arr


# 中文注释：从样本轨迹中提取 xyz 点序列。
def sample_to_xyz(sample: CharacterTrajectory) -> np.ndarray:
    pts = sample.all_points()
    if len(pts) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray([[p.x, p.y, p.z] for p in pts], dtype=np.float64)


# 中文注释：从样本轨迹中提取姿态角序列。
def sample_angles(sample: CharacterTrajectory) -> np.ndarray:
    pts = sample.all_points()
    if len(pts) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray([[p.alpha, p.beta, p.gamma] for p in pts], dtype=np.float64)

# 中文注释：把单条笔画拆分为 xyz 位置和姿态角。
def stroke_to_xyz_angles(stroke: StrokeTrajectory) -> Tuple[np.ndarray, np.ndarray]:
    pts = stroke.sorted_points()
    xyz = np.asarray([[p.x, p.y, p.z] for p in pts], dtype=np.float64)
    angles = np.asarray([[p.alpha, p.beta, p.gamma] for p in pts], dtype=np.float64)
    return xyz, angles


# 中文注释：用每笔 Chebyshev 节点重建完整轨迹样本。
def rebuild_sample_per_stroke_cheb(
    template: CharacterTrajectory,
    order: int,
    freeze_angles: bool = True,
) -> CharacterTrajectory:
    new_points: List[TrajectoryPoint] = []

    for stroke in template.sorted_strokes():
        pts = stroke.sorted_points()
        n = len(pts)

        if n == 0:
            continue

        xyz0, ang0 = stroke_to_xyz_angles(stroke)

        # 避免 order >= 点数导致不必要振荡
        stroke_order = min(order, max(1, n - 1))

        x_nodes, y_nodes, z_nodes = fit_3d_nodes_from_points(xyz0, stroke_order)
        xyz_rec = parameterize_3d(
            x_nodes,
            y_nodes,
            z_nodes,
            num_samples=n,
        )

        if freeze_angles:
            ang_rec = ang0.copy()
        else:
            a_nodes, b_nodes, g_nodes = fit_3d_nodes_from_points(ang0, stroke_order)
            ang_rec = parameterize_3d(
                a_nodes,
                b_nodes,
                g_nodes,
                num_samples=n,
            )

        for i, p in enumerate(pts):
            new_points.append(
                TrajectoryPoint(
                    stroke_id=p.stroke_id,
                    point_id=p.point_id,
                    x=float(xyz_rec[i, 0]),
                    y=float(xyz_rec[i, 1]),
                    z=float(xyz_rec[i, 2]),
                    alpha=float(ang_rec[i, 0]),
                    beta=float(ang_rec[i, 1]),
                    gamma=float(ang_rec[i, 2]),
                    state=PointState.from_value(int(p.state)),
                )
            )

    bucket: Dict[int, List[TrajectoryPoint]] = {}
    for p in new_points:
        bucket.setdefault(p.stroke_id, []).append(p)

    strokes = [
        StrokeTrajectory(
            stroke_id=sid,
            points=sorted(pts, key=lambda q: q.point_id),
        )
        for sid, pts in sorted(bucket.items(), key=lambda x: x[0])
    ]

    return CharacterTrajectory(
        character=template.character,
        strokes=strokes,
        meta=dict(template.meta),
    )

# 中文注释：为每笔轨迹构建 xyz 控制节点和优化向量。
def build_per_stroke_xyz_decision(
    template: CharacterTrajectory,
    order: int,
):
    """
    返回：
      decision0: 所有 stroke 的 x/y/z Chebyshev nodes 拼起来
      specs: 每一笔的解码元信息
    """
    parts = []
    specs = []

    offset = 0

    for stroke in template.sorted_strokes():
        pts = stroke.sorted_points()
        n_pts = len(pts)
        if n_pts == 0:
            continue

        xyz0 = np.asarray(
            [[p.x, p.y, p.z] for p in pts],
            dtype=np.float64,
        )
        ang0 = np.asarray(
            [[p.alpha, p.beta, p.gamma] for p in pts],
            dtype=np.float64,
        )

        stroke_order = min(order, max(1, n_pts - 1))
        x_nodes, y_nodes, z_nodes = fit_3d_nodes_from_points(xyz0, stroke_order)

        vec = np.concatenate([x_nodes, y_nodes, z_nodes], axis=0).astype(np.float64)
        parts.append(vec)

        size = len(vec)
        specs.append({
            "stroke_id": stroke.stroke_id,
            "points": pts,
            "n_pts": n_pts,
            "order": stroke_order,
            "node_count": stroke_order + 1,
            "offset": offset,
            "size": size,
            "angles": ang0,
        })
        offset += size

    if len(parts) == 0:
        return np.zeros((0,), dtype=np.float64), specs

    return np.concatenate(parts, axis=0).astype(np.float64), specs

# 中文注释：把每笔优化向量解码为轨迹采样点。
def decode_per_stroke_xyz_decision(
    template: CharacterTrajectory,
    decision_vec: np.ndarray,
    specs: List[Dict[str, Any]],
    x_lo: float,
    x_hi: float,
    y_lo: float,
    y_hi: float,
    z_lo: float,
    z_hi: float,
) -> CharacterTrajectory:
    new_points: List[TrajectoryPoint] = []

    for spec in specs:
        off = spec["offset"]
        size = spec["size"]
        n_node = spec["node_count"]
        n_pts = spec["n_pts"]
        pts = spec["points"]
        ang = spec["angles"]

        local = decision_vec[off:off + size]

        x_nodes = local[:n_node]
        y_nodes = local[n_node:2 * n_node]
        z_nodes = local[2 * n_node:3 * n_node]

        xyz = parameterize_3d(
            x_nodes,
            y_nodes,
            z_nodes,
            num_samples=n_pts,
        )

        xyz[:, 0] = np.clip(xyz[:, 0], x_lo, x_hi)
        xyz[:, 1] = np.clip(xyz[:, 1], y_lo, y_hi)
        xyz[:, 2] = np.clip(xyz[:, 2], z_lo, z_hi)

        for i, p in enumerate(pts):
            new_points.append(
                TrajectoryPoint(
                    stroke_id=p.stroke_id,
                    point_id=p.point_id,
                    x=float(xyz[i, 0]),
                    y=float(xyz[i, 1]),
                    z=float(xyz[i, 2]),
                    alpha=float(ang[i, 0]),
                    beta=float(ang[i, 1]),
                    gamma=float(ang[i, 2]),
                    state=PointState.from_value(int(p.state)),
                )
            )

    bucket: Dict[int, List[TrajectoryPoint]] = {}
    for p in new_points:
        bucket.setdefault(p.stroke_id, []).append(p)

    strokes = [
        StrokeTrajectory(
            stroke_id=sid,
            points=sorted(pts, key=lambda q: q.point_id),
        )
        for sid, pts in sorted(bucket.items(), key=lambda x: x[0])
    ]

    return CharacterTrajectory(
        character=template.character,
        strokes=strokes,
        meta=dict(template.meta),
    )

# 中文注释：把每笔 xyz 和姿态角控制节点拼成 6D 优化向量。
def stack_decision_vector_6d(
    x_nodes: np.ndarray,
    y_nodes: np.ndarray,
    z_nodes: np.ndarray,
    a_nodes: np.ndarray,
    b_nodes: np.ndarray,
    g_nodes: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(x_nodes, dtype=np.float64),
            np.asarray(y_nodes, dtype=np.float64),
            np.asarray(z_nodes, dtype=np.float64),
            np.asarray(a_nodes, dtype=np.float64),
            np.asarray(b_nodes, dtype=np.float64),
            np.asarray(g_nodes, dtype=np.float64),
        ],
        axis=0,
    ).astype(np.float64)


# 中文注释：把 6D 优化向量拆回每笔控制节点。
def unstack_decision_vector_6d(vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vec = np.asarray(vec, dtype=np.float64).reshape(-1)
    if len(vec) % 6 != 0:
        raise ValueError("decision vector length must be divisible by 6")
    n = len(vec) // 6
    return (
        vec[:n],
        vec[n:2 * n],
        vec[2 * n:3 * n],
        vec[3 * n:4 * n],
        vec[4 * n:5 * n],
        vec[5 * n:],
    )


# 中文注释：根据重建的 xyz 和姿态角生成新的轨迹样本。
def rebuild_sample_from_xyz_angles(
    template: CharacterTrajectory,
    xyz: np.ndarray,
    angles: np.ndarray,
) -> CharacterTrajectory:
    xyz = np.asarray(xyz, dtype=np.float64)
    angles = np.asarray(angles, dtype=np.float64)

    template_points = template.all_points()
    if len(template_points) != len(xyz):
        raise ValueError("xyz length does not match template points")
    if len(template_points) != len(angles):
        raise ValueError("angles length does not match template points")

    new_points: List[TrajectoryPoint] = []
    for i, p in enumerate(template_points):
        new_points.append(
            TrajectoryPoint(
                stroke_id=p.stroke_id,
                point_id=p.point_id,
                x=float(xyz[i, 0]),
                y=float(xyz[i, 1]),
                z=float(xyz[i, 2]),
                alpha=float(angles[i, 0]),
                beta=float(angles[i, 1]),
                gamma=float(angles[i, 2]),
                state=PointState.from_value(int(p.state)),
            )
        )

    bucket: Dict[int, List[TrajectoryPoint]] = {}
    for p in new_points:
        bucket.setdefault(p.stroke_id, []).append(p)

    strokes = [
        StrokeTrajectory(
            stroke_id=sid,
            points=sorted(pts, key=lambda q: q.point_id),
        )
        for sid, pts in sorted(bucket.items(), key=lambda x: x[0])
    ]
    return CharacterTrajectory(
        character=template.character,
        strokes=strokes,
        meta=dict(template.meta),
    )


# 中文注释：封装目标图像残差、正则项和 LM 优化流程。
class TrajectoryOptimizer:
    # 中文注释：初始化对象并保存后续处理所需的配置和成员变量。
    def __init__(
        self,
        renderer: FusionRenderer,
        z_reg_weight: float = 1e-3,
        angle_reg_weight: float = 1e-4,
        render_samples: int = 128,
        fg_weight: float = 15.0,
        under_ink_weight: float = 35.0,
        ink_weight: float = 40.0,
        structure_weight: float = 2.0,
        stroke_presence_weight: float = 50.0,
        stroke_keep_ratio: float = 0.65,
    ):
        self.renderer = renderer
        self.z_reg_weight = z_reg_weight
        self.angle_reg_weight = angle_reg_weight
        self.render_samples = render_samples

        # 像素/结构约束权重
        self.fg_weight = fg_weight
        self.under_ink_weight = under_ink_weight
        self.ink_weight = ink_weight
        self.structure_weight = structure_weight

        # 防止某些笔画被优化到消失
        self.stroke_presence_weight = stroke_presence_weight
        self.stroke_keep_ratio = stroke_keep_ratio


    # 中文注释：创建给定参数化方式下的残差函数。
    def _residual_fn_factory(
        self,
        template: CharacterTrajectory,
        target_image: np.ndarray,
        order: int,
        fixed_bounds: Tuple[float, float, float, float],
        stroke_specs: List[Dict[str, Any]],
        decision0_ref: np.ndarray,
    ):
        xyz0 = sample_to_xyz(template)
        if len(xyz0) == 0:
            raise ValueError("Template trajectory is empty")

        ang_init = sample_angles(template)
        z_init = xyz0[:, 2].copy()

        # 固定整字 bounds，限制 LM 不要把轨迹推到初始画布外太多
        min_x, max_x, min_y, max_y = fixed_bounds
        bw = max(max_x - min_x, 1e-6)
        bh = max(max_y - min_y, 1e-6)

        xy_margin_ratio = 0.03   # 先保守：允许超出初始 bbox 3%
        x_lo = min_x - xy_margin_ratio * bw
        x_hi = max_x + xy_margin_ratio * bw
        y_lo = min_y - xy_margin_ratio * bh
        y_hi = max_y + xy_margin_ratio * bh

        z_lo = max(0.0, float(np.percentile(z_init, 1)) - 0.25)
        z_hi = float(np.percentile(z_init, 99)) + 0.25

        num_points = len(template.all_points())

        target = np.asarray(target_image, dtype=np.float64)

        # 初始 forward 渲染，使用固定 bounds
        initial_out = self.renderer.render_character(
            template,
            fixed_bounds=fixed_bounds,
        )
        initial_render = np.asarray(
            initial_out["character_image"],
            dtype=np.float64,
        )

        # 记录每一笔初始渲染强度，用于防止笔画消失
        initial_stroke_means: Dict[int, float] = {}
        for sid, s_out in initial_out.get("strokes", {}).items():
            img = np.asarray(s_out["stroke_image"], dtype=np.float64)
            initial_stroke_means[sid] = float(img.mean())

        fg_mask = target > 0.1

        pixel_weights = 1.0 + self.fg_weight * target
        sqrt_pixel_weights = np.sqrt(pixel_weights)

        target_mean = float(target.mean())

        # 中文注释：定义当前示例或优化流程使用的残差函数。
        def residual_fn(decision_vec: np.ndarray) -> np.ndarray:
            sample_cur = decode_per_stroke_xyz_decision(
                template=template,
                decision_vec=decision_vec,
                specs=stroke_specs,
                x_lo=x_lo,
                x_hi=x_hi,
                y_lo=y_lo,
                y_hi=y_hi,
                z_lo=z_lo,
                z_hi=z_hi,
            )

            out = self.renderer.render_character(
                sample_cur,
                fixed_bounds=fixed_bounds,
            )

            rendered = np.asarray(
                out["character_image"],
                dtype=np.float64,
            )

            # 1. 前景加权像素误差
            diff = rendered - target
            weighted_pix_res = diff * sqrt_pixel_weights

            # 2. 欠画惩罚
            if np.any(fg_mask):
                under = np.maximum(target[fg_mask] - rendered[fg_mask], 0.0)
                under_ink_res = np.sqrt(self.under_ink_weight) * under
            else:
                under_ink_res = np.zeros((0,), dtype=np.float64)

            # 3. 全局墨量约束
            rendered_mean = float(rendered.mean())
            ink_res = np.asarray(
                [self.ink_weight * (rendered_mean - target_mean)],
                dtype=np.float64,
            )

            # 4. 初始结构保持
            structure_res = self.structure_weight * (rendered - initial_render)

            # 5. 逐笔保留约束
            stroke_presence = []
            cur_strokes = out.get("strokes", {})

            for sid, init_mean in initial_stroke_means.items():
                if init_mean <= 1e-8:
                    continue

                if sid not in cur_strokes:
                    loss = self.stroke_keep_ratio * init_mean
                else:
                    cur_img = np.asarray(
                        cur_strokes[sid]["stroke_image"],
                        dtype=np.float64,
                    )
                    cur_mean = float(cur_img.mean())
                    loss = max(
                        self.stroke_keep_ratio * init_mean - cur_mean,
                        0.0,
                    )

                stroke_presence.append(self.stroke_presence_weight * loss)

            if len(stroke_presence) > 0:
                stroke_presence_res = np.asarray(
                    stroke_presence,
                    dtype=np.float64,
                )
            else:
                stroke_presence_res = np.zeros((0,), dtype=np.float64)

            # 6. per-stroke decision 正则
            decision_reg = np.sqrt(self.z_reg_weight) * 0.01 * (
                decision_vec - decision0_ref
            )

            return np.concatenate(
                [
                    weighted_pix_res.reshape(-1),
                    under_ink_res.reshape(-1),
                    ink_res.reshape(-1),
                    structure_res.reshape(-1),
                    stroke_presence_res.reshape(-1),
                    decision_reg.reshape(-1),
                ],
                axis=0,
            )

        return residual_fn


    # 中文注释：执行多阶数轨迹优化并返回最佳结果。
    def optimize(
        self,
        template: CharacterTrajectory,
        target_image: np.ndarray,
        order: int = 5,
        damping: float = 1e-2,
        max_steps: int = 30,
    ) -> TrajectoryOptimizationResult:
        print("[CHECK] TrajectoryOptimizer.optimize() called", flush=True)        

        xyz0 = sample_to_xyz(template)
        if len(xyz0) == 0:
            raise ValueError("Template trajectory is empty")
        

        z_ref = xyz0[:, 2].copy()
        z_lo = max(0.0, float(np.percentile(z_ref, 1)) - 0.25)
        z_hi = float(np.percentile(z_ref, 99)) + 0.25

        
        fixed_bounds = trajectory_bounds(template)


        min_x, max_x, min_y, max_y = fixed_bounds
        bw = max(max_x - min_x, 1e-6)
        bh = max(max_y - min_y, 1e-6)

        xy_margin_ratio = 0.03
        x_lo = min_x - xy_margin_ratio * bw
        x_hi = max_x + xy_margin_ratio * bw
        y_lo = min_y - xy_margin_ratio * bh
        y_hi = max_y + xy_margin_ratio * bh


        ang0 = sample_angles(template)

        decision0, stroke_specs = build_per_stroke_xyz_decision(
            template,
            order=order,
        )

        residual_fn = self._residual_fn_factory(
            template,
            target_image,
            order,
            fixed_bounds=fixed_bounds,
            stroke_specs=stroke_specs,
            decision0_ref=decision0.copy(),
        )


        lm_result = lm_solve(
            residual_fn,
            x0=decision0,
            damping=damping,
            max_steps=max_steps,
        )

        # 中文注释：限制 LM 更新步长，避免单次迭代造成过大的轨迹突变。
        def limit_lm_step(step_vec: np.ndarray) -> np.ndarray:
            max_abs = float(np.max(np.abs(step_vec))) if step_vec.size > 0 else 0.0
            max_step = 0.01 * max(bw, bh)

            if max_abs <= max_step or max_abs < 1e-12:
                return step_vec

            return step_vec * (max_step / max_abs)

        hist = getattr(lm_result, "history", {})
        cost_hist = hist.get("cost", [])
        damp_hist = hist.get("damping", [])

        if len(cost_hist) > 0:
            print(f"[CHECK] lm initial cost={cost_hist[0]:.6f}", flush=True)
            print(f"[CHECK] lm final history cost={cost_hist[-1]:.6f}", flush=True)
            print(f"[CHECK] lm cost history={cost_hist}", flush=True)
            print(f"[CHECK] lm damping history={damp_hist}", flush=True)

        delta_norm = np.linalg.norm(lm_result.x - decision0)
        rel_delta = delta_norm / (np.linalg.norm(decision0) + 1e-8)

        msg = getattr(lm_result, "message", "")
        print(f"[CHECK] lm success={lm_result.success}, message={msg}", flush=True)
        print(f"[CHECK] decision delta norm={delta_norm:.6e}, rel_delta={rel_delta:.6e}", flush=True)
        print(f"[CHECK] final cost={lm_result.final_cost:.6f}", flush=True)


        # 中文注释：将优化向量解码为轨迹并立即渲染，便于评估候选结果。
        def decode_and_render(decision_vec: np.ndarray):
            sample_candidate = decode_per_stroke_xyz_decision(
                template=template,
                decision_vec=decision_vec,
                specs=stroke_specs,
                x_lo=x_lo,
                x_hi=x_hi,
                y_lo=y_lo,
                y_hi=y_hi,
                z_lo=z_lo,
                z_hi=z_hi,
            )

            rendered_candidate = self.renderer.render_character(
                sample_candidate,
                fixed_bounds=fixed_bounds,
            )["character_image"]

            return sample_candidate, np.asarray(rendered_candidate, dtype=np.float64)

        

        target = np.asarray(target_image, dtype=np.float64)
        initial_sample = template
        initial_render = np.asarray(
            self.renderer.render_character(
                template,
                fixed_bounds=fixed_bounds,
            )["character_image"],
            dtype=np.float64,
        )

        fg_mask = target > 0.1

        # 中文注释：根据目标图和渲染图的差异为候选轨迹打分。
        def candidate_score(rendered: np.ndarray) -> float:
            global_diff = float(np.abs(rendered - target).mean())

            if np.any(fg_mask):
                fg_diff = float(np.abs(rendered[fg_mask] - target[fg_mask]).mean())
                fg_render_mean = float(rendered[fg_mask].mean())
                fg_target_mean = float(target[fg_mask].mean())
                fg_under = max(fg_target_mean - fg_render_mean, 0.0)
            else:
                fg_diff = global_diff
                fg_under = 0.0

            mean_gap = abs(float(rendered.mean()) - float(target.mean()))

            score = (
            0.25 * global_diff
                + 1.00 * fg_diff
                + 0.80 * fg_under
                + 0.50 * mean_gap
            )
            return score

        per_stroke_decoded_sample = rebuild_sample_per_stroke_cheb(
            template,
            order=order,
            freeze_angles=True,
        )

        per_stroke_decoded_render = np.asarray(
            self.renderer.render_character(
                per_stroke_decoded_sample,
                fixed_bounds=fixed_bounds,
            )["character_image"],
            dtype=np.float64,
        )

        print(
            f"[CHECK] per-stroke decoded0 score={candidate_score(per_stroke_decoded_render):.6f}, "
            f"mean={per_stroke_decoded_render.mean():.6f}",
            flush=True,
        )
        print(
            f"[CHECK] per-stroke decoded0 global diff={float(np.abs(per_stroke_decoded_render - target).mean()):.6f}",
            flush=True,
        )

        # 注意：decoded0 检查必须放在 candidate_score 定义之后
        decoded0_sample, decoded0_render = decode_and_render(decision0)
        decoded0_score = candidate_score(decoded0_render)
        initial_score = candidate_score(initial_render)

        print(
            f"[CHECK] direct initial score={initial_score:.6f}, "
            f"mean={initial_render.mean():.6f}",
            flush=True,
        )
        print(
            f"[CHECK] decoded decision0 score={decoded0_score:.6f}, "
            f"mean={decoded0_render.mean():.6f}",
            flush=True,
        )
        print(
            f"[CHECK] decoded0 global diff={float(np.abs(decoded0_render - target).mean()):.6f}",
            flush=True,
        )

        best_alpha = 0.0
        best_score = candidate_score(initial_render)
        best_sample = initial_sample
        best_rendered = initial_render

        raw_step = lm_result.x - decision0
        step = limit_lm_step(raw_step)

        print(
            f"[CHECK] raw_step_norm={np.linalg.norm(raw_step):.6e}, "
            f"limited_step_norm={np.linalg.norm(step):.6e}",
            flush=True,
        )
        alphas = [
            0.0,
            0.005, 0.01, 0.02, 0.03, 0.05,
            0.075, 0.10, 0.15, 0.20, 0.25,
            0.35, 0.50, 0.75, 1.00,
            1.25, 1.50, 2.00, 3.00, 4.00,
        ]

        print(f"[CHECK] initial selection score={best_score:.6f}", flush=True)

        for alpha in alphas[1:]:
            candidate_x = decision0 + alpha * step
            sample_candidate, rendered_candidate = decode_and_render(candidate_x)
            score = candidate_score(rendered_candidate)
            mean_gap = abs(float(rendered_candidate.mean()) - float(target.mean()))

            print(
                f"[CHECK] alpha={alpha:.4f}, "
                f"score={score:.6f}, "
                f"mean={rendered_candidate.mean():.6f}, "
                f"mean_gap={mean_gap:.6f}",
                flush=True,
            )

            # 建议先用 0.006；如果想更贴近 target mean，改成 0.003
            max_mean_gap = 0.003

            if mean_gap > max_mean_gap:
                print(
                    f"[CHECK] alpha={alpha:.4f} skipped by mean_gap={mean_gap:.6f}",
                    flush=True,
                )
                continue

            if score < best_score:
                best_score = score
                best_alpha = alpha
                best_sample = sample_candidate
                best_rendered = rendered_candidate

        print(
            f"[CHECK] selected alpha={best_alpha:.4f}, "
            f"best_score={best_score:.6f}",
            flush=True,
        )

        if best_alpha == 0.0:
            print("[CHECK] initial forward accepted; LM update rejected.", flush=True)
        else:
            print("[CHECK] optimized accepted", flush=True)

        optimized_sample = best_sample
        rendered = best_rendered


        

        # 初始 forward
        # initial_render = self.renderer.render_character(template, fixed_bounds=fixed_bounds)["character_image"]

        # x_nodes, y_nodes, z_nodes, a_nodes, b_nodes, g_nodes = unstack_decision_vector_6d(lm_result.x)

        # xyz_opt = parameterize_3d(
        #     x_nodes,
        #     y_nodes,
        #     z_nodes,
        #     num_samples=len(template.all_points()),
        # )

        # angles_opt = parameterize_3d(
        #     a_nodes,
        #     b_nodes,
        #     g_nodes,
        #     num_samples=len(template.all_points()),
        # )

        # optimized_sample = rebuild_sample_from_xyz_angles(
        #     template,
        #     xyz_opt,
        #     angles_opt,
        # )

        # rendered = self.renderer.render_character(optimized_sample)["character_image"]

        # 使用前景区域作为接受/回退判断，避免 global diff 误导
        

        target = np.asarray(target_image, dtype=np.float64)
        initial_render = np.asarray(initial_render, dtype=np.float64)
        rendered = np.asarray(rendered, dtype=np.float64)

        fg_mask = target > 0.1

        init_global_diff = float(np.abs(initial_render - target).mean())
        opt_global_diff = float(np.abs(rendered - target).mean())

        if np.any(fg_mask):
            init_fg_diff = float(np.abs(initial_render[fg_mask] - target[fg_mask]).mean())
            opt_fg_diff = float(np.abs(rendered[fg_mask] - target[fg_mask]).mean())
        else:
            init_fg_diff = init_global_diff
            opt_fg_diff = opt_global_diff

        print(f"[CHECK] initial global diff={init_global_diff:.6f}", flush=True)
        print(f"[CHECK] optimized global diff={opt_global_diff:.6f}", flush=True)
        print(f"[CHECK] initial foreground diff={init_fg_diff:.6f}", flush=True)
        print(f"[CHECK] optimized foreground diff={opt_fg_diff:.6f}", flush=True)
        print(f"[CHECK] initial mean={initial_render.mean():.6f}, optimized mean={rendered.mean():.6f}, target mean={target.mean():.6f}", flush=True)

        

        # 如果优化后前景更差，直接回退到初始轨迹
        eps = 1e-3

        if opt_fg_diff > init_fg_diff + eps:
            print(
                "[WARN] Optimized foreground is worse than initial. "
                "Fallback to initial trajectory.",
                flush=True,
            )
            optimized_sample = template
            rendered = initial_render
        else:
            if best_alpha == 0.0:
                print("[CHECK] final accepted: initial forward", flush=True)
            else:
                print("[CHECK] final accepted: optimized trajectory", flush=True)

        return TrajectoryOptimizationResult(
            order=order,
            lm_result=lm_result,
            optimized_sample=optimized_sample,
            target_image=target_image,
            rendered_image=rendered,
        )


# 中文注释：作为脚本直接运行时，从这里进入命令行流程或示例测试。
if __name__ == "__main__":
    print("trajectory_optimizer.py provides 6D TrajectoryOptimizer for closed-loop optimization.")
# 使用说明：该模块实现了融合笔触模型的首版反向优化闭环。
# TrajectoryOptimizer.optimize() 接收一个初始整字轨迹模板和一张目标字符图，先用 fit_3d_nodes_from_points() 把离散轨迹压缩成 Chebyshev 节点；
# 随后通过 lm_solve() 迭代优化这些节点。
# 每次迭代中，_residual_fn_factory() 都会把当前决策向量恢复为整条 3D 轨迹，重建 CharacterTrajectory，交给 FusionRenderer 渲染成字符图，再与 target_image 做像素级残差比较；
# 同时附加一个对 z 轴偏移的正则项，以避免优化过程出现不受约束的深度漂移。优化完成后，会返回 TrajectoryOptimizationResult，其中包含 LM 收敛信息、优化后的轨迹样本以及最终渲染图。
# 当前首版实现默认保持 alpha/beta/gamma 不变，只优化 x/y/z 节点；后续如果你需要，也可以在此基础上把旋转角一并纳入决策变量。
