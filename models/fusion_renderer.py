# 中文注释：本文件融合动态笔刷模型和 B-BSMG 神经网络，将轨迹渲染为笔画/字符图像。
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch

from utils.types import StrokeTrajectory, CharacterTrajectory, DynamicBrushState
from models.dynamic_brush import DynamicBrushModel
from models.geometry import dynamic_state_to_bbsmg_input, normalize_trajectory_xy
from models.bbsmg import BBSMG, normalize_bbsmg_inputs
from models.geometry import (
    dynamic_state_to_bbsmg_input,
    normalize_trajectory_xy,
    normalize_trajectory_xy_with_bounds,
)


NORM_PADDING = 4


# 中文注释：把数组或张量移动到渲染器所在设备并设定类型。
def _to_device_tensor(x: List[float], device) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)


# 中文注释：计算二维折线长度。
def polyline_length(points: List[Tuple[float, float]]) -> float:
    total = 0.0
    for i in range(1, len(points)):
        dx = points[i][0] - points[i - 1][0]
        dy = points[i][1] - points[i - 1][1]
        total += float((dx * dx + dy * dy) ** 0.5)
    return total


# 中文注释：把局部笔触图像按中心位置贴到大画布上。
def paste_patch(canvas: np.ndarray, patch: np.ndarray, x0: int, y0: int) -> np.ndarray:
    """
    保留工具函数：如果后续改回局部 patch 粘贴式渲染可以继续用。
    当前 stroke-level 版本暂时不依赖它。
    """
    h, w = patch.shape[-2], patch.shape[-1]
    H, W = canvas.shape[-2], canvas.shape[-1]

    x1 = min(x0 + w, W)
    y1 = min(y0 + h, H)

    sx0 = max(0, -x0)
    sy0 = max(0, -y0)
    dx0 = max(0, x0)
    dy0 = max(0, y0)

    if dx0 >= x1 or dy0 >= y1:
        return canvas

    canvas[dy0:y1, dx0:x1] = np.maximum(
        canvas[dy0:y1, dx0:x1],
        patch[
            sy0:sy0 + (y1 - dy0),
            sx0:sx0 + (x1 - dx0),
        ],
    )

    return canvas


# 中文注释：负责把轨迹、动态笔刷和神经笔触生成器融合成渲染图。
class FusionRenderer:
    # 中文注释：初始化对象并保存后续处理所需的配置和成员变量。
    def __init__(
        self,
        image_size: int = 128,
        device: str = "cpu",
        input_dim: int = 5,
        latent_dim: int = 128,
        base_channels: int = 64,
        brush: Optional[DynamicBrushModel] = None,
    ):
        self.image_size = image_size
        self.device = device
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.input_normalization: Optional[Dict[str, Any]] = None

        self.brush = brush if brush is not None else DynamicBrushModel()

        self.bbsmg = BBSMG(
            input_dim=input_dim,
            latent_dim=latent_dim,
            base_channels=base_channels,
            image_size=image_size,
        ).to(device)

        self.bbsmg.eval()

    # 中文注释：加载已训练的 B-BSMG 权重并切换到评估模式。
    def set_input_normalization(self, normalization: Dict[str, Any]) -> None:
        scales = normalization.get("scales")
        if scales is None or len(scales) != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} normalization scales, got {scales}"
            )
        self.input_normalization = dict(normalization)

    def load_input_normalization_from_npz(self, npz_path: str) -> None:
        """Reconstruct normalization for legacy checkpoints."""
        path = Path(npz_path)
        if not path.exists():
            raise FileNotFoundError(f"Training NPZ not found: {npz_path}")
        data = np.load(path, allow_pickle=True)
        inputs = np.asarray(data["inputs"], dtype=np.float32)
        if inputs.ndim != 2 or inputs.shape[1] != self.input_dim:
            raise ValueError(
                f"NPZ inputs shape {inputs.shape} is incompatible with input_dim={self.input_dim}"
            )
        scales = np.ones((self.input_dim,), dtype=np.float32)
        scales[0] = max(float(np.nanmax(inputs[:, 0])), 1.0)
        if self.input_dim > 3:
            scales[3:] = float(self.image_size)
        self.set_input_normalization({
            "version": 1,
            "input_dim": self.input_dim,
            "scales": scales.tolist(),
            "source": str(path),
        })

    def load_weights(
        self,
        ckpt_path: str,
        normalization_npz: Optional[str] = None,
    ) -> None:
        path = Path(ckpt_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        data = torch.load(path, map_location=self.device)

        if isinstance(data, dict) and data.get("input_normalization") is not None:
            self.set_input_normalization(data["input_normalization"])
        elif normalization_npz is not None:
            self.load_input_normalization_from_npz(normalization_npz)
        else:
            raise RuntimeError(
                "This checkpoint does not contain input_normalization. "
                "Pass normalization_npz with the NPZ used to train it."
            )

        if isinstance(data, dict) and "model_state" in data:
            state = data["model_state"]
        elif isinstance(data, dict) and "model_state_dict" in data:
            state = data["model_state_dict"]
        elif isinstance(data, dict) and "state_dict" in data:
            state = data["state_dict"]
        else:
            state = data

        self.bbsmg.load_state_dict(state)
        self.bbsmg.eval()

    # 中文注释：把单个动态笔刷状态渲染为局部笔触贴片。
    @torch.no_grad()
    def render_state(
        self,
        state: DynamicBrushState,
        x0: Optional[float] = None,
        y0: Optional[float] = None,
    ) -> np.ndarray:
        """
        5D 渲染模式：
            h, alpha, beta, x0, y0

        只有 input_dim=5 时才应使用。
        """
        if self.input_dim != 5:
            raise RuntimeError(
                f"render_state() only supports input_dim=5, "
                f"but current input_dim={self.input_dim}. "
                f"Use render_state_with_stroke_features() instead."
            )

        bbin = dynamic_state_to_bbsmg_input(
            state,
            x0=x0,
            y0=y0,
        )

        inp = _to_device_tensor(bbin.as_list(), self.device)
        inp = normalize_bbsmg_inputs(inp, self.input_normalization)
        pred = self.bbsmg(inp)[0, 0].detach().cpu().numpy().astype(np.float32)

        return pred

    # 中文注释：使用额外笔画特征渲染单个状态对应的局部笔触。
    @torch.no_grad()
    def render_state_with_stroke_features(
        self,
        state: DynamicBrushState,
        norm_points: List[Tuple[float, float]],
    ) -> np.ndarray:
        """
        10D 渲染模式：
            h, alpha, beta, x0, y0, x1, y1, dx, dy, length

        其中 norm_points 必须是 normalize_trajectory_xy()
        后得到的当前笔画点序列。
        """
        if len(norm_points) == 0:
            return np.zeros((self.image_size, self.image_size), dtype=np.float32)

        x0, y0 = norm_points[0]
        x1, y1 = norm_points[-1]
        dx = x1 - x0
        dy = y1 - y0
        length = polyline_length(norm_points)

        bbin = dynamic_state_to_bbsmg_input(
            state,
            x0=float(x0),
            y0=float(y0),
        )

        if self.input_dim == 10:
            inp_list = [
                float(bbin.h),
                float(bbin.alpha),
                float(bbin.beta),
                float(x0),
                float(y0),
                float(x1),
                float(y1),
                float(dx),
                float(dy),
                float(length),
            ]
        elif self.input_dim == 5:
            inp_list = bbin.as_list()
        else:
            raise ValueError(f"Unsupported B-BSMG input_dim: {self.input_dim}")

        inp = _to_device_tensor(inp_list, self.device)
        inp = normalize_bbsmg_inputs(inp, self.input_normalization)
        pred = self.bbsmg(inp)[0, 0].detach().cpu().numpy().astype(np.float32)

        return pred

    # 中文注释：渲染一条笔画，并返回画布及中间状态。
    def render_stroke(
        self,
        stroke: StrokeTrajectory,
        norm_points: Optional[List[Tuple[float, float]]] = None,
    ) -> Dict[str, Any]:
        """
        Stroke-level 渲染。

        当前训练逻辑是：
            一笔的第一个 DynamicBrushState + 笔画几何特征 -> 整笔 stroke image

        因此每个 stroke 只渲染一次。
        """
        states = self.brush.simulate_stroke(
            stroke,
            reset_brush=True,
        )

        if len(states) == 0:
            stroke_img = np.zeros(
                (self.image_size, self.image_size),
                dtype=np.float32,
            )
            return {
                "states": states,
                "patches": [],
                "stroke_image": stroke_img,
            }

        state0 = states[0]

        if norm_points is not None and len(norm_points) > 0:
            patch = self.render_state_with_stroke_features(
                state0,
                norm_points,
            )
        else:
            # 仅兼容 5D 老模型
            patch = self.render_state(state0)

        stroke_img = patch.astype(np.float32)

        return {
            "states": states,
            "patches": [patch],
            "stroke_image": stroke_img,
        }

    # 中文注释：渲染完整字符的所有笔画。
    def render_character(
    self,
    sample: CharacterTrajectory,
    fixed_bounds: Optional[Tuple[float, float, float, float]] = None,
) -> Dict[str, Any]:
        """
        整字渲染。

        与 build_pseudo_pairs.py 保持一致：
        先用 normalize_trajectory_xy() 得到每笔在 128x128 画布中的几何信息，
        再传给 B-BSMG。
        """
        canvas = np.zeros(
            (self.image_size, self.image_size),
            dtype=np.float32,
        )
        stroke_outputs: Dict[int, Dict[str, Any]] = {}

        strokes = sample.sorted_strokes()

        if fixed_bounds is None:
            norm_strokes = normalize_trajectory_xy(
                sample,
                canvas_size=self.image_size,
                padding=NORM_PADDING,
            )
        else:
            norm_strokes = normalize_trajectory_xy_with_bounds(
                sample,
                bounds=fixed_bounds,
                canvas_size=self.image_size,
                padding=NORM_PADDING,
            )

        for stroke_order, stroke in enumerate(strokes):
            sid = stroke.stroke_id

            norm_points = None
            if stroke_order < len(norm_strokes):
                norm_points = norm_strokes[stroke_order]

            out = self.render_stroke(
                stroke,
                norm_points=norm_points,
            )

            stroke_img = out["stroke_image"]
            canvas = np.maximum(canvas, stroke_img)
            stroke_outputs[sid] = out

        return {
            "character_image": canvas,
            "strokes": stroke_outputs,
        }


# 中文注释：作为脚本直接运行时，从这里进入命令行流程或示例测试。
if __name__ == "__main__":
    renderer = FusionRenderer(
        image_size=128,
        device="cpu",
        input_dim=10,
        latent_dim=128,
        base_channels=64,
    )
    print("FusionRenderer initialized.")
