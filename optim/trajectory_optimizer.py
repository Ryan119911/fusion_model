from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
from PIL import Image

from utils.types import TrajectoryPoint, StrokeTrajectory, CharacterTrajectory, PointState
from models.fusion_renderer import FusionRenderer
from optim.chebyshev import fit_3d_nodes_from_points, parameterize_3d
from optim.lm import lm_solve, LMResult


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
    ):
        self.renderer = renderer
        self.z_reg_weight = z_reg_weight
        self.angle_reg_weight = angle_reg_weight
        self.render_samples = render_samples

    def _residual_fn_factory(
        self,
        template: CharacterTrajectory,
        target_image: np.ndarray,
        order: int,
    ):
        xyz0 = sample_to_xyz(template)
        if len(xyz0) == 0:
            raise ValueError("Template trajectory is empty")

        ang_init = sample_angles(template)
        z_init = xyz0[:, 2].copy()

        num_points = len(template.all_points())

        def residual_fn(decision_vec: np.ndarray) -> np.ndarray:
            x_nodes, y_nodes, z_nodes, a_nodes, b_nodes, g_nodes = unstack_decision_vector_6d(decision_vec)

            xyz = parameterize_3d(x_nodes, y_nodes, z_nodes, num_samples=num_points)
            angles = parameterize_3d(a_nodes, b_nodes, g_nodes, num_samples=num_points)

            sample_cur = rebuild_sample_from_xyz_angles(template, xyz, angles)
            rendered = self.renderer.render_character(sample_cur)["character_image"]

            pix_res = (rendered - target_image).reshape(-1).astype(np.float64)

            z_reg = np.sqrt(self.z_reg_weight) * (xyz[:, 2] - z_init)
            angle_reg = np.sqrt(self.angle_reg_weight) * (angles.reshape(-1) - ang_init.reshape(-1))

            return np.concatenate([pix_res, z_reg, angle_reg], axis=0)

        return residual_fn

    def optimize(
        self,
        template: CharacterTrajectory,
        target_image: np.ndarray,
        order: int = 5,
        damping: float = 1e-2,
        max_steps: int = 30,
    ) -> TrajectoryOptimizationResult:
        xyz0 = sample_to_xyz(template)
        if len(xyz0) == 0:
            raise ValueError("Template trajectory is empty")

        ang0 = sample_angles(template)

        # 初始 6D Chebyshev 节点
        x_nodes0, y_nodes0, z_nodes0 = fit_3d_nodes_from_points(xyz0, order)
        a_nodes0, b_nodes0, g_nodes0 = fit_3d_nodes_from_points(ang0, order)

        decision0 = stack_decision_vector_6d(
            x_nodes0, y_nodes0, z_nodes0,
            a_nodes0, b_nodes0, g_nodes0,
        )

        residual_fn = self._residual_fn_factory(template, target_image, order)

        lm_result = lm_solve(
            residual_fn,
            x0=decision0,
            damping=damping,
            max_steps=max_steps,
        )

        x_nodes, y_nodes, z_nodes, a_nodes, b_nodes, g_nodes = unstack_decision_vector_6d(lm_result.x)

        xyz_opt = parameterize_3d(x_nodes, y_nodes, z_nodes, num_samples=len(template.all_points()))
        angles_opt = parameterize_3d(a_nodes, b_nodes, g_nodes, num_samples=len(template.all_points()))

        optimized_sample = rebuild_sample_from_xyz_angles(template, xyz_opt, angles_opt)
        rendered = self.renderer.render_character(optimized_sample)["character_image"]

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
