from typing import List, Tuple
import math

import numpy as np


def cgl_nodes(order: int) -> np.ndarray:
    if order < 1:
        raise ValueError("order must be >= 1")
    k = np.arange(order + 1, dtype=np.float64)
    return np.cos(np.pi * k / order)


def barycentric_weights(order: int) -> np.ndarray:
    if order < 1:
        raise ValueError("order must be >= 1")
    w = np.ones(order + 1, dtype=np.float64)
    w[0] = 0.5
    w[-1] = 0.5 * ((-1.0) ** order)
    for i in range(1, order):
        w[i] = (-1.0) ** i
    return w


def barycentric_interpolate(t: np.ndarray, nodes: np.ndarray, values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    nodes = np.asarray(nodes, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    out = np.zeros_like(t, dtype=np.float64)
    for i, ti in enumerate(t):
        diff = ti - nodes
        close = np.where(np.abs(diff) < 1e-12)[0]
        if len(close) > 0:
            out[i] = values[int(close[0])]
        else:
            tmp = weights / diff
            out[i] = np.sum(tmp * values) / np.sum(tmp)
    return out


def normalize_time_grid(num_samples: int) -> np.ndarray:
    if num_samples < 2:
        return np.array([0.0], dtype=np.float64)
    return np.linspace(-1.0, 1.0, num_samples, dtype=np.float64)


def parameterize_1d(node_values: np.ndarray, num_samples: int) -> np.ndarray:
    node_values = np.asarray(node_values, dtype=np.float64)
    order = len(node_values) - 1
    nodes = cgl_nodes(order)
    weights = barycentric_weights(order)
    t = normalize_time_grid(num_samples)
    return barycentric_interpolate(t, nodes, node_values, weights)


def parameterize_3d(x_nodes: np.ndarray, y_nodes: np.ndarray, z_nodes: np.ndarray, num_samples: int) -> np.ndarray:
    x = parameterize_1d(x_nodes, num_samples)
    y = parameterize_1d(y_nodes, num_samples)
    z = parameterize_1d(z_nodes, num_samples)
    return np.stack([x, y, z], axis=-1)


def resample_sequence(values: np.ndarray, target_len: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        src_t = np.linspace(-1.0, 1.0, len(values), dtype=np.float64)
        dst_t = np.linspace(-1.0, 1.0, target_len, dtype=np.float64)
        return np.interp(dst_t, src_t, values)
    else:
        cols = [resample_sequence(values[:, i], target_len) for i in range(values.shape[1])]
        return np.stack(cols, axis=-1)


def fit_nodes_from_sequence(values: np.ndarray, order: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    target_len = order + 1
    return resample_sequence(values, target_len)


def fit_3d_nodes_from_points(points_xyz: np.ndarray, order: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    points_xyz = np.asarray(points_xyz, dtype=np.float64)
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError("points_xyz must have shape [N,3]")
    nodes_xyz = fit_nodes_from_sequence(points_xyz, order)
    return nodes_xyz[:, 0], nodes_xyz[:, 1], nodes_xyz[:, 2]


def stack_decision_vector(x_nodes: np.ndarray, y_nodes: np.ndarray, z_nodes: np.ndarray) -> np.ndarray:
    return np.concatenate([np.asarray(x_nodes), np.asarray(y_nodes), np.asarray(z_nodes)], axis=0).astype(np.float64)


def unstack_decision_vector(vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    vec = np.asarray(vec, dtype=np.float64)
    if vec.ndim != 1 or len(vec) % 3 != 0:
        raise ValueError("decision vector length must be divisible by 3")
    n = len(vec) // 3
    return vec[:n], vec[n:2 * n], vec[2 * n:]


if __name__ == "__main__":
    order = 4
    nodes = cgl_nodes(order)
    weights = barycentric_weights(order)
    vals = np.array([0.0, 0.5, 1.0, 0.5, 0.0], dtype=np.float64)
    t = normalize_time_grid(16)
    y = barycentric_interpolate(t, nodes, vals, weights)
    print("nodes:", nodes)
    print("interp shape:", y.shape)
# 使用说明：该模块实现了基于 Chebyshev-Gauss-Lobatto (CGL) 节点的轨迹参数化工具。
# cgl_nodes() 与 barycentric_weights() 用于生成伪光谱节点及其重心权重；barycentric_interpolate() 用于在任意归一化时间点上恢复连续轨迹；
# parameterize_1d() 和 parameterize_3d() 则分别用于按节点值生成一维或三维轨迹序列。
# fit_nodes_from_sequence() 和 fit_3d_nodes_from_points() 可用于把已有离散轨迹压缩为有限个 CGL 决策变量；
# stack_decision_vector() 与 unstack_decision_vector() 用于在优化器内部把 x/y/z 三组节点打包成单个参数向量。
# 后续在 lm.py 和 trajectory_optimizer.py 中，会直接使用这些函数把连续轨迹转换成可优化的有限维决策变量。
