from dataclasses import dataclass, field
from typing import Callable, Optional, Dict, Any, List
import numpy as np


@dataclass
class LMResult:
    x: np.ndarray
    success: bool
    num_steps: int
    final_cost: float
    message: str = ""
    history: Dict[str, List[float]] = field(default_factory=dict)


def squared_cost(residual: np.ndarray) -> float:
    r = np.asarray(residual, dtype=np.float64).reshape(-1)
    return 0.5 * float(np.dot(r, r))


def numerical_jacobian(residual_fn: Callable[[np.ndarray], np.ndarray], x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    r0 = np.asarray(residual_fn(x), dtype=np.float64).reshape(-1)
    m = r0.shape[0]
    n = x.shape[0]
    J = np.zeros((m, n), dtype=np.float64)
    for j in range(n):
        xp = x.copy()
        xm = x.copy()
        xp[j] += eps
        xm[j] -= eps
        rp = np.asarray(residual_fn(xp), dtype=np.float64).reshape(-1)
        rm = np.asarray(residual_fn(xm), dtype=np.float64).reshape(-1)
        J[:, j] = (rp - rm) / (2.0 * eps)
    return J


def lm_solve(residual_fn: Callable[[np.ndarray], np.ndarray], x0: np.ndarray, jacobian_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None, damping: float = 1e-2, damping_up: float = 10.0, damping_down: float = 0.3, max_steps: int = 50, xtol: float = 1e-6, ftol: float = 1e-8, gtol: float = 1e-8, eps_jac: float = 1e-5) -> LMResult:
    x = np.asarray(x0, dtype=np.float64).reshape(-1).copy()
    mu = float(damping)
    history = {"cost": [], "damping": []}

    r = np.asarray(residual_fn(x), dtype=np.float64).reshape(-1)
    cost = squared_cost(r)
    history["cost"].append(cost)
    history["damping"].append(mu)

    for step in range(1, max_steps + 1):
        J = jacobian_fn(x) if jacobian_fn is not None else numerical_jacobian(residual_fn, x, eps=eps_jac)
        J = np.asarray(J, dtype=np.float64)
        g = J.T @ r
        if np.linalg.norm(g, ord=np.inf) < gtol:
            return LMResult(x=x, success=True, num_steps=step - 1, final_cost=cost, message="Gradient tolerance reached", history=history)

        A = J.T @ J
        D = np.diag(np.diag(A))
        H = A + mu * (D + 1e-12 * np.eye(A.shape[0], dtype=np.float64))

        try:
            delta = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return LMResult(x=x, success=False, num_steps=step - 1, final_cost=cost, message="Linear solve failed", history=history)

        if np.linalg.norm(delta) < xtol * (np.linalg.norm(x) + xtol):
            return LMResult(x=x, success=True, num_steps=step - 1, final_cost=cost, message="Step tolerance reached", history=history)

        x_new = x + delta
        r_new = np.asarray(residual_fn(x_new), dtype=np.float64).reshape(-1)
        cost_new = squared_cost(r_new)

        if cost_new < cost:
            if abs(cost - cost_new) < ftol * (1.0 + cost):
                x = x_new
                cost = cost_new
                history["cost"].append(cost)
                history["damping"].append(mu)
                return LMResult(x=x, success=True, num_steps=step, final_cost=cost, message="Function tolerance reached", history=history)
            x = x_new
            r = r_new
            cost = cost_new
            mu = max(mu * damping_down, 1e-12)
        else:
            mu = mu * damping_up

        history["cost"].append(cost)
        history["damping"].append(mu)

    return LMResult(x=x, success=False, num_steps=max_steps, final_cost=cost, message="Maximum steps reached", history=history)


if __name__ == "__main__":
    # 拟合 x ≈ 3 的简单示例: residual = [x-3]
    def residual_fn(v: np.ndarray) -> np.ndarray:
        return np.array([v[0] - 3.0], dtype=np.float64)

    result = lm_solve(residual_fn, x0=np.array([0.0], dtype=np.float64))
    print("success:", result.success)
    print("x:", result.x)
    print("final_cost:", result.final_cost)
# 使用说明：该模块实现了一个通用的 Levenberg–Marquardt 最小二乘求解器。
# lm_solve() 接收残差函数 residual_fn(x)，并可选接收解析 Jacobian；
# 如果未提供 Jacobian，则会通过 numerical_jacobian() 使用中心差分近似计算。
# 求解器内部按 \(J^TJ+\mu\,diag(J^TJ)\) 的形式构造阻尼正规方程，并根据代价是否下降自适应调整阻尼系数 \(\mu\)。
# LMResult 会统一返回最终参数、是否收敛、步数、最终代价与完整历史，便于在 trajectory_optimizer.py 中记录优化过程和调试收敛行为。
