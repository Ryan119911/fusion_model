from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
from PIL import Image

from utils.types import TrajectoryPoint, StrokeTrajectory, CharacterTrajectory, PointState
from models.fusion_renderer import FusionRenderer
from optim.chebyshev import fit_3d_nodes_from_points, parameterize_3d
from optim.lm import lm_solve, LMResult

from models.geometry import trajectory_bounds


@dataclass
class TrajectoryOptimizationResult:
    order: int
    lm_result: LMResult
    optimized_sample: CharacterTrajectory
    target_image: np.ndarray
    rendered_image: np.ndarray


def load_target_image(path: str, image_size: int = 128) -> np.ndarray:
    img = Image.open(path).convert("L").resize((image_size, image_size))
    arr = np.array(img, dtype=np.float32) / 255.0

    # 统一极性：
    # 白底黑字 -> 黑底白字
    # 目标统一为：背景=0，墨迹=1。
    if arr.mean() > 0.5:
        arr = 1.0 - arr

    return arr


def sample_to_xyz(sample: CharacterTrajectory) -> np.ndarray:
    pts = sample.all_points()
    if len(pts) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray([[p.x, p.y, p.z] for p in pts], dtype=np.float64)


def sample_angles(sample: CharacterTrajectory) -> np.ndarray:
    pts = sample.all_points()
    if len(pts) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray([[p.alpha, p.beta, p.gamma] for p in pts], dtype=np.float64)


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


class TrajectoryOptimizer:
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


    def _residual_fn_factory(
        self,
        template: CharacterTrajectory,
        target_image: np.ndarray,
        order: int,
        fixed_bounds: Tuple[float, float, float, float],
    ):
        xyz0 = sample_to_xyz(template)
        if len(xyz0) == 0:
            raise ValueError("Template trajectory is empty")

        ang_init = sample_angles(template)
        z_init = xyz0[:, 2].copy()

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

        def residual_fn(decision_vec: np.ndarray) -> np.ndarray:
            x_nodes, y_nodes, z_nodes, a_nodes, b_nodes, g_nodes = unstack_decision_vector_6d(decision_vec)

            xyz = parameterize_3d(
                x_nodes,
                y_nodes,
                z_nodes,
                num_samples=num_points,
            )

            # temp

            xyz[:, 2] = np.clip(xyz[:, 2], z_lo, z_hi)

            # angles = parameterize_3d(
            #     a_nodes,
            #     b_nodes,
            #     g_nodes,
            #     num_samples=num_points,
            # )
            angles = ang_init.copy()

            sample_cur = rebuild_sample_from_xyz_angles(
                template,
                xyz,
                angles,
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

            # 6. z 正则
            z_reg = np.sqrt(self.z_reg_weight) * (xyz[:, 2] - z_init)

            # 7. angle 正则
            angle_reg = np.sqrt(self.angle_reg_weight) * (
                angles.reshape(-1) - ang_init.reshape(-1)
            )

            return np.concatenate(
                [
                    weighted_pix_res.reshape(-1),
                    under_ink_res.reshape(-1),
                    ink_res.reshape(-1),
                    structure_res.reshape(-1),
                    stroke_presence_res.reshape(-1),
                    z_reg.reshape(-1),
                    angle_reg.reshape(-1),
                ],
                axis=0,
            )

        return residual_fn


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

        ang0 = sample_angles(template)

        # 初始 6D Chebyshev 节点
        x_nodes0, y_nodes0, z_nodes0 = fit_3d_nodes_from_points(xyz0, order)
        a_nodes0, b_nodes0, g_nodes0 = fit_3d_nodes_from_points(ang0, order)

        decision0 = stack_decision_vector_6d(
            x_nodes0, y_nodes0, z_nodes0,
            a_nodes0, b_nodes0, g_nodes0,
        )
        

        residual_fn = self._residual_fn_factory(
            template,
            target_image,
            order,
            fixed_bounds=fixed_bounds,
        )

        lm_result = lm_solve(
            residual_fn,
            x0=decision0,
            damping=damping,
            max_steps=max_steps,
        )

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


        def decode_and_render(decision_vec: np.ndarray):
            x_nodes, y_nodes, z_nodes, a_nodes, b_nodes, g_nodes = unstack_decision_vector_6d(decision_vec)

            xyz = parameterize_3d(
                x_nodes,
                y_nodes,
                z_nodes,
                num_samples=len(template.all_points()),
            )

            # temp

            xyz[:, 2] = np.clip(xyz[:, 2], z_lo, z_hi)

            # angles = parameterize_3d(
            #     a_nodes,
            #     b_nodes,
            #     g_nodes,
            #     num_samples=len(template.all_points()),
            # )
            angles = ang0.copy()

            sample_candidate = rebuild_sample_from_xyz_angles(
                template,
                xyz,
                angles,
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


        alphas = [
            0.0,
            0.005, 0.01, 0.02, 0.03, 0.05,
            0.075, 0.10, 0.15, 0.20, 0.25,
            0.35, 0.50, 0.75, 1.00
        ]

        best_alpha = 0.0
        best_score = candidate_score(initial_render)
        best_sample = initial_sample
        best_rendered = initial_render

        step = lm_result.x - decision0

        print(f"[CHECK] initial selection score={best_score:.6f}", flush=True)

        for alpha in alphas[1:]:
            candidate_x = decision0 + alpha * step
            sample_candidate, rendered_candidate = decode_and_render(candidate_x)
            score = candidate_score(rendered_candidate)

            print(
                f"[CHECK] alpha={alpha:.2f}, "
                f"score={score:.6f}, "
                f"mean={rendered_candidate.mean():.6f}",
                flush=True,
            )

            if score < best_score:
                best_score = score
                best_alpha = alpha
                best_sample = sample_candidate
                best_rendered = rendered_candidate

        print(
            f"[CHECK] selected alpha={best_alpha:.2f}, "
            f"best_score={best_score:.6f}",
            flush=True,
        )

        if best_alpha == 0.0:
            print("[CHECK] No beneficial LM update found; using initial forward.", flush=True)
        else:
            print("[CHECK] Using interpolated optimized trajectory.", flush=True)

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
            print("[CHECK] optimized accepted", flush=True)

        return TrajectoryOptimizationResult(
            order=order,
            lm_result=lm_result,
            optimized_sample=optimized_sample,
            target_image=target_image,
            rendered_image=rendered,
        )


if __name__ == "__main__":
    print("trajectory_optimizer.py provides 6D TrajectoryOptimizer for closed-loop optimization.")
# 使用说明：该模块实现了融合笔触模型的首版反向优化闭环。
# TrajectoryOptimizer.optimize() 接收一个初始整字轨迹模板和一张目标字符图，先用 fit_3d_nodes_from_points() 把离散轨迹压缩成 Chebyshev 节点；
# 随后通过 lm_solve() 迭代优化这些节点。
# 每次迭代中，_residual_fn_factory() 都会把当前决策向量恢复为整条 3D 轨迹，重建 CharacterTrajectory，交给 FusionRenderer 渲染成字符图，再与 target_image 做像素级残差比较；
# 同时附加一个对 z 轴偏移的正则项，以避免优化过程出现不受约束的深度漂移。优化完成后，会返回 TrajectoryOptimizationResult，其中包含 LM 收敛信息、优化后的轨迹样本以及最终渲染图。
# 当前首版实现默认保持 alpha/beta/gamma 不变，只优化 x/y/z 节点；后续如果你需要，也可以在此基础上把旋转角一并纳入决策变量。
