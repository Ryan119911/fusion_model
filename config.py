# 中文注释：本文件集中定义项目配置结构，负责从 YAML 读取参数并创建必要目录。
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
import yaml


# 中文注释：数据路径、画布尺寸和书体筛选等数据相关配置。
@dataclass
class DataConfig:
    root_dir: str = "data"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    cache_dir: str = "data/cache"
    trajectory_csv: str = "data/raw/trajectories.csv"
    json_dir: str = "data/raw/json_files"
    # 主数据表：列为 img_path / text / author / chirography / location
    data_csv: str = "data/raw/data.csv"
    dictionary_txt: str = "data/raw/makemeahanzi/dictionary.txt"
    graphics_txt: str = "data/raw/makemeahanzi/graphics.txt"
    image_dir: str = "data/raw/images"           # img_path 相对此目录解析
    image_ext: str = ".jpg"                      # 由 .png 改为 .jpg
    chirography_filter: Optional[str] = "楷"               # 书体筛选（楷/行/草/隶/篆…）
    z_min: float = 0.15
    z_max: float = 1.0
    points_per_stroke: int = 16
    canvas_size: int = 128
    svg_canvas_size: int = 1024
    character_trajectory_padding: int = 16
    # 旧的 label_dir / json_ext 已删除：标注改由 data_csv 提供


# 中文注释：训练过程的随机种子、设备、批大小、学习率和输出目录配置。
@dataclass
class TrainConfig:
    seed: int = 42
    device: str = "cuda"
    batch_size: int = 64  #临时改小
    num_workers: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-6
    epochs: int = 100  #临时改小 100
    log_interval: int = 20
    save_interval: int = 5  #临时改小 5
    output_dir: str = "outputs"


# 中文注释：B-BSMG 网络结构相关配置。
@dataclass
class BBSMGConfig:
    input_dim: int = 10
    latent_dim: int = 256
    base_channels: int = 64
    out_channels: int = 1
    image_size: int = 128
    use_tanh: bool = False


@dataclass
class CharacterGeneratorConfig:
    input_channels: int = 6
    base_channels: int = 32
    out_channels: int = 1
    image_size: int = 128
    depth: int = 4
    dropout: float = 0.1
    use_tanh: bool = False
    prior_strength: float = 0.75
    prior_channel: int = 1
    prior_threshold: float = 0.70
    prior_sharpness: float = 10.0


# 中文注释：动态笔刷模型的多项式阶数和物理近似参数配置。
@dataclass
class DynamicBrushConfig:
    kw: float = 0.02
    kd: float = 0.02
    dt: float = 0.01
    width_poly_degree: int = 2
    drag_poly_degree: int = 2
    offset_poly_degree: int = 3
    snap_clip_min: float = 0.0


# 中文注释：轨迹优化器的阶数、阻尼、采样数和正则权重配置。
@dataclass
class OptimConfig:
    cheb_order_min: int = 3
    cheb_order_max: int = 4
    lm_damping: float = 5e-2
    lm_max_steps: int = 20
    render_samples_per_stroke: int = 128
    z_reg_weight: float = 1e-2
    angle_reg_weight: float = 1e-2


# 中文注释：项目总配置，组合数据、训练、模型和优化配置。
@dataclass
class FusionBrushConfig:
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    bbsmg: BBSMGConfig = field(default_factory=BBSMGConfig)
    character_generator: CharacterGeneratorConfig = field(default_factory=CharacterGeneratorConfig)
    dynamic_brush: DynamicBrushConfig = field(default_factory=DynamicBrushConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)


# 中文注释：返回一份默认配置对象，作为未提供 YAML 时的基准。
def get_default_config() -> FusionBrushConfig:
    return FusionBrushConfig()


# 中文注释：将字典中的配置项递归写入 dataclass，保留未覆盖的默认值。
def _update_dataclass(instance, updates: Dict[str, Any]):
    for key, value in updates.items():
        if not hasattr(instance, key):
            continue
        current = getattr(instance, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _update_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance


# 中文注释：读取 YAML 配置文件并合并到默认配置。
def load_config(path: Optional[str] = None) -> FusionBrushConfig:
    cfg = get_default_config()
    if path is None:
        return cfg
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path_obj, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return _update_dataclass(cfg, data)


# 中文注释：确保数据缓存、处理结果和训练输出目录存在。
def ensure_dirs(cfg: FusionBrushConfig) -> None:
    Path(cfg.data.root_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.data.raw_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.data.processed_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.data.cache_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.train.output_dir).mkdir(parents=True, exist_ok=True)
