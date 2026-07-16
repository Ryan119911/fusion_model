from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from models.bbsmg import BBSMG, normalize_bbsmg_inputs
from models.dynamic_brush import DynamicBrushModel
from models.geometry import (
    compute_heading,
    normalize_trajectory_xy,
    normalize_trajectory_xy_with_bounds,
)
from utils.feature_schema import (
    build_stroke_features,
    checkpoint_schema,
    get_feature_schema,
    normalization_for_inputs,
    read_npz_schema,
)
from utils.types import CharacterTrajectory, StrokeTrajectory


NORM_PADDING = 4


def _load_checkpoint(path: str, map_location: Any) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class FusionRenderer:
    def __init__(
        self,
        image_size: int = 128,
        device: str = "cpu",
        input_dim: int = 10,
        feature_schema: str = "stroke10_v1",
        latent_dim: int = 128,
        base_channels: int = 64,
        out_channels: int = 1,
        use_tanh: bool = False,
        brush: Optional[DynamicBrushModel] = None,
    ):
        self.image_size = image_size
        self.device = torch.device(
            device if device == "cpu" or torch.cuda.is_available() else "cpu"
        )
        self.input_dim = input_dim
        self.feature_schema = feature_schema
        schema = get_feature_schema(feature_schema)
        if schema.input_dim != input_dim:
            raise ValueError("feature_schema and input_dim disagree")
        self.input_normalization: Optional[Dict[str, Any]] = None
        self.brush = brush or DynamicBrushModel()
        self.bbsmg = BBSMG(
            input_dim=input_dim,
            latent_dim=latent_dim,
            base_channels=base_channels,
            out_channels=out_channels,
            image_size=image_size,
            use_tanh=use_tanh,
        ).to(self.device)
        self.bbsmg.eval()

    def set_input_normalization(self, normalization: Dict[str, Any]) -> None:
        scales = normalization.get("scales")
        if scales is None or len(scales) != self.input_dim:
            raise ValueError("Invalid B-BSMG input normalization")
        schema = normalization.get("feature_schema")
        if schema and schema != self.feature_schema:
            raise ValueError(
                f"Normalization schema {schema} does not match renderer schema {self.feature_schema}"
            )
        self.input_normalization = dict(normalization)

    def load_input_normalization_from_npz(self, npz_path: str) -> None:
        data = np.load(npz_path, allow_pickle=True)
        inputs = np.asarray(data["inputs"], dtype=np.float32)
        schema = read_npz_schema(data, inputs.shape[1])
        if schema != self.feature_schema:
            raise ValueError(
                f"NPZ schema {schema} does not match renderer schema {self.feature_schema}"
            )
        self.set_input_normalization(
            normalization_for_inputs(inputs, schema, self.image_size)
        )

    def load_weights(
        self,
        checkpoint_path: str,
        normalization_npz: Optional[str] = None,
    ) -> None:
        checkpoint = _load_checkpoint(checkpoint_path, map_location=self.device)
        schema = checkpoint_schema(checkpoint, self.input_dim)
        if schema != self.feature_schema:
            raise ValueError(
                f"Checkpoint schema {schema} does not match renderer schema {self.feature_schema}"
            )
        normalization = (
            checkpoint.get("input_normalization")
            if isinstance(checkpoint, dict)
            else None
        )
        if normalization:
            self.set_input_normalization(normalization)
        elif normalization_npz:
            self.load_input_normalization_from_npz(normalization_npz)
        else:
            raise RuntimeError(
                "Legacy checkpoint has no input normalization; pass --normalization_npz"
            )
        if isinstance(checkpoint, dict):
            state = next(
                (
                    checkpoint[key]
                    for key in ("model_state", "model_state_dict", "state_dict")
                    if key in checkpoint
                ),
                checkpoint,
            )
        else:
            state = checkpoint
        self.bbsmg.load_state_dict(state)
        self.bbsmg.eval()

    @torch.no_grad()
    def render_stroke(
        self,
        stroke: StrokeTrajectory,
        normalized_points: List[Tuple[float, float]],
    ) -> Dict[str, Any]:
        points = stroke.sorted_points()
        if not points or not normalized_points:
            image = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            return {"states": [], "patches": [], "stroke_image": image}
        headings = compute_heading([(point.x, point.y) for point in points])
        states = self.brush.simulate_stroke(stroke, theta0=headings[0])
        features = build_stroke_features(
            self.feature_schema, points[0], states[0], normalized_points
        )
        tensor = torch.tensor(
            features, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        tensor = normalize_bbsmg_inputs(tensor, self.input_normalization)
        prediction = self.bbsmg(tensor)[0, 0].detach().cpu().numpy().astype(np.float32)
        return {
            "states": states,
            "features": features,
            "patches": [prediction],
            "stroke_image": prediction,
        }

    def render_character(
        self,
        sample: CharacterTrajectory,
        fixed_bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> Dict[str, Any]:
        canvas = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        normalized = (
            normalize_trajectory_xy(
                sample, canvas_size=self.image_size, padding=NORM_PADDING
            )
            if fixed_bounds is None
            else normalize_trajectory_xy_with_bounds(
                sample,
                fixed_bounds,
                canvas_size=self.image_size,
                padding=NORM_PADDING,
            )
        )
        outputs: Dict[int, Dict[str, Any]] = {}
        for stroke, points in zip(sample.sorted_strokes(), normalized):
            output = self.render_stroke(stroke, points)
            canvas = np.maximum(canvas, output["stroke_image"])
            outputs[stroke.stroke_id] = output
        return {"character_image": canvas, "strokes": outputs}
