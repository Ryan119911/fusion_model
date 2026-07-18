from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class DataConfig:
    root_dir: str = "data"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    cache_dir: str = "data/cache"
    trajectory_csv: str = "data/raw/trajectories.csv"
    json_dir: str = "data/raw/json_files"
    data_csv: str = "data/raw/data.csv"
    dictionary_txt: str = "data/raw/makemeahanzi/dictionary.txt"
    graphics_txt: str = "data/raw/makemeahanzi/graphics.txt"
    image_dir: str = "data/raw/images"
    image_ext: str = ".jpg"
    chirography_filter: Optional[str] = None
    z_min: float = 0.0
    z_max: float = 4.0
    points_per_stroke: int = 128
    canvas_size: int = 128
    svg_canvas_size: int = 1024
    timestamp_column: Optional[str] = None
    validate_trajectories: bool = True


@dataclass
class LossConfig:
    weighted_mse: float = 1.0
    ssim: float = 0.3
    dice: float = 0.3
    cldice: float = 0.05
    edge: float = 0.1
    structure: float = 0.05
    ink: float = 0.1
    positive_weight: float = 4.0
    cldice_iterations: int = 10


@dataclass
class TrainConfig:
    seed: int = 42
    device: str = "cuda"
    batch_size: int = 16
    num_workers: int = 4
    lr: float = 3e-4
    weight_decay: float = 1e-6
    epochs: int = 30
    log_interval: int = 20
    save_interval: int = 10
    output_dir: str = "outputs/bbsmg"
    amp: bool = True
    amp_dtype: str = "float16"
    amp_init_scale: float = 1024.0
    amp_growth_interval: int = 2000
    amp_backoff_factor: float = 0.5
    amp_max_consecutive_nonfinite_steps: int = 16
    pin_memory: bool = True
    persistent_workers: bool = True
    target_cache_dir: Optional[str] = "data/cache/npz_arrays"
    split_strategy: str = "group"
    split_group_key: str = "sample_id"
    val_ratio: float = 0.1
    split_manifest: Optional[str] = None
    lr_factor: float = 0.5
    lr_patience: int = 3
    min_lr: float = 1e-6
    gradient_clip_norm: float = 1.0
    character_loss_weight: float = 0.0
    loss: LossConfig = field(default_factory=LossConfig)


@dataclass
class BBSMGConfig:
    input_dim: int = 10
    feature_schema: str = "stroke10_v1"
    latent_dim: int = 128
    base_channels: int = 64
    out_channels: int = 1
    image_size: int = 128
    use_tanh: bool = False


@dataclass
class DynamicBrushConfig:
    mode: str = "disabled"
    calibration_path: Optional[str] = None
    kw: float = 0.02
    kd: float = 0.02
    dt: float = 0.01
    width_poly_degree: int = 2
    drag_poly_degree: int = 2
    offset_poly_degree: int = 3
    snap_clip_min: float = 0.0


@dataclass
class OptimConfig:
    cheb_order_min: int = 3
    cheb_order_max: int = 4
    lm_damping: float = 5e-2
    lm_max_steps: int = 20
    jacobian_epsilon: float = 1e-5
    render_samples_per_stroke: int = 128
    optimize_angles: bool = False
    xy_margin_ratio: float = 0.03
    z_margin: float = 0.25
    angle_margin_radians: float = 0.35
    xyz_reg_weight: float = 1e-4
    z_reg_weight: float = 1e-2
    angle_reg_weight: float = 1e-2


@dataclass
class FusionBrushConfig:
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    bbsmg: BBSMGConfig = field(default_factory=BBSMGConfig)
    dynamic_brush: DynamicBrushConfig = field(default_factory=DynamicBrushConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)


def get_default_config() -> FusionBrushConfig:
    return FusionBrushConfig()


def _update_dataclass(instance: Any, updates: Dict[str, Any], prefix: str = "") -> Any:
    if not isinstance(updates, dict):
        raise TypeError(f"Configuration section '{prefix or '<root>'}' must be a mapping")
    fields = getattr(instance, "__dataclass_fields__", {})
    for key, value in updates.items():
        path = f"{prefix}.{key}" if prefix else key
        if key not in fields:
            raise KeyError(f"Unknown configuration key: {path}")
        current = getattr(instance, key)
        if hasattr(current, "__dataclass_fields__"):
            _update_dataclass(current, value, path)
        else:
            setattr(instance, key, value)
    return instance


def _validate_config(cfg: FusionBrushConfig) -> None:
    from utils.feature_schema import get_feature_schema

    schema = get_feature_schema(cfg.bbsmg.feature_schema)
    if schema.input_dim != cfg.bbsmg.input_dim:
        raise ValueError(
            f"bbsmg.feature_schema={schema.name} requires input_dim={schema.input_dim}"
        )
    if cfg.dynamic_brush.mode not in {"disabled", "heuristic", "calibrated"}:
        raise ValueError("dynamic_brush.mode must be disabled, heuristic, or calibrated")
    if cfg.dynamic_brush.mode == "calibrated" and not cfg.dynamic_brush.calibration_path:
        raise ValueError("dynamic_brush.calibration_path is required in calibrated mode")
    if cfg.train.split_strategy not in {"group", "random"}:
        raise ValueError("train.split_strategy must be group or random")
    if cfg.train.amp_dtype not in {"float16", "bfloat16"}:
        raise ValueError("train.amp_dtype must be float16 or bfloat16")
    if cfg.train.amp_init_scale <= 0.0:
        raise ValueError("train.amp_init_scale must be positive")
    if cfg.train.amp_growth_interval < 1:
        raise ValueError("train.amp_growth_interval must be >= 1")
    if not 0.0 < cfg.train.amp_backoff_factor < 1.0:
        raise ValueError("train.amp_backoff_factor must be in (0, 1)")
    if cfg.train.amp_max_consecutive_nonfinite_steps < 1:
        raise ValueError(
            "train.amp_max_consecutive_nonfinite_steps must be >= 1"
        )
    if not 0.0 <= cfg.train.val_ratio < 1.0:
        raise ValueError("train.val_ratio must be in [0, 1)")
    if cfg.train.save_interval < 1:
        raise ValueError("train.save_interval must be >= 1")
    if cfg.train.batch_size < 1 or cfg.train.num_workers < 0:
        raise ValueError("train.batch_size must be >= 1 and num_workers must be >= 0")
    if cfg.train.character_loss_weight < 0.0:
        raise ValueError("train.character_loss_weight must be non-negative")
    if cfg.train.character_loss_weight > 0.0 and cfg.train.split_strategy != "group":
        raise ValueError("Character-level loss requires train.split_strategy=group")
    if not 0.0 < cfg.train.lr or not 0.0 < cfg.train.min_lr:
        raise ValueError("Learning rates must be positive")


def load_config(path: Optional[str] = None) -> FusionBrushConfig:
    cfg = get_default_config()
    if path is not None:
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path_obj, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
        _update_dataclass(cfg, data)
    _validate_config(cfg)
    return cfg


def ensure_dirs(cfg: FusionBrushConfig) -> None:
    for path in (
        cfg.data.root_dir,
        cfg.data.raw_dir,
        cfg.data.processed_dir,
        cfg.data.cache_dir,
        cfg.train.output_dir,
    ):
        Path(path).mkdir(parents=True, exist_ok=True)
