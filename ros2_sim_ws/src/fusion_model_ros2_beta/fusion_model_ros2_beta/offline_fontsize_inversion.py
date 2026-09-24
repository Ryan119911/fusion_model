"""Offline V16 font-size inversion for the ROS 2 Beta package.

This module is deliberately independent from rclpy.  It prepares a scaled
target canvas and scaled x/y initial pose, invokes the model repository's
single-character V16 inversion in a separate Python environment, and exports
one physical, glyph-centred trajectory.  ROS consumes that result without
normalising a glyph into a runtime cell or changing its brush posture.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from .trajectory_catalog import TrajectoryEntry, character_directory_name


ProgressCallback = Callable[[str], None]
_LANCZOS = getattr(Image, "Resampling", Image).LANCZOS
V16_CHECKPOINT_SHA256 = "30d5e0c37dc7b7913c02fea26930babafa46a4a07b5f698520df277a04d5b717"
MODEL_WEIGHT_VERSION = "V16"
INVERSION_PIPELINE = "paper_psoc_lm_v42_fused_pose"
INVERSION_INTERFACE_VERSION = "ros2_offline_fontsize_v3"


@dataclass(frozen=True)
class OfflineInversionConfig:
    model_root: Path
    python_executable: Path
    bbsmg_checkpoint: Path
    cache_root: Path
    reference_font_size_m: float = 0.29
    image_size: int = 128
    padding: int = 16
    timeout_s: int = 1800
    device: str = "cuda"
    order: int = 5
    max_steps: int = 15
    optimization_size: int = 64
    point_batch_size: int = 64
    expected_checkpoint_sha256: str = V16_CHECKPOINT_SHA256
    joint_foreground_weight: float = 1.0
    joint_hard_neural_domain: bool = False
    joint_xy_target_skeleton_weight: float = 0.0
    joint_xy_target_skeleton_max_distance_px: float = 12.0
    joint_xy_target_skeleton_threshold: float = 0.35

    def validate(self) -> None:
        if not self.model_root.is_dir():
            raise FileNotFoundError(f"model root does not exist: {self.model_root}")
        if not self.python_executable.is_file():
            raise FileNotFoundError(
                f"offline Python does not exist: {self.python_executable}"
            )
        if not self.bbsmg_checkpoint.is_file():
            raise FileNotFoundError(
                f"V16 B-BSMG checkpoint does not exist: {self.bbsmg_checkpoint}"
            )
        if not math.isfinite(self.reference_font_size_m) or self.reference_font_size_m <= 0:
            raise ValueError("reference_font_size_m must be positive and finite")
        if self.image_size <= 0 or self.padding < 0:
            raise ValueError("image_size and padding are invalid")
        if 2 * self.padding >= self.image_size:
            raise ValueError("padding leaves no model image area")
        if self.timeout_s <= 0:
            raise ValueError("offline inversion timeout must be positive")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_pose_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        required = {
            "character",
            "sample_id",
            "stroke_id",
            "point_id",
            "x",
            "y",
            "z",
            "alpha",
            "beta",
            "gamma",
            "state",
        }
        missing = sorted(required - set(fields))
        if missing:
            raise ValueError(f"pose CSV is missing columns {missing}: {path}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"pose CSV has no rows: {path}")
    return fields, rows


def _xy_reference(rows: list[dict[str, str]]) -> tuple[float, float, float]:
    values = [(float(row["x"]), float(row["y"])) for row in rows]
    if not all(math.isfinite(x) and math.isfinite(y) for x, y in values):
        raise ValueError("pose CSV contains non-finite x/y")
    x_min = min(value[0] for value in values)
    x_max = max(value[0] for value in values)
    y_min = min(value[1] for value in values)
    y_max = max(value[1] for value in values)
    span = max(x_max - x_min, y_max - y_min)
    if span <= 1.0e-9:
        raise ValueError("pose CSV has no usable x/y extent")
    return 0.5 * (x_min + x_max), 0.5 * (y_min + y_max), span


def scale_initial_pose_csv(
    source: Path,
    destination: Path,
    scale: float,
) -> tuple[float, float, float]:
    """Scale x/y about the source glyph centre while retaining its full pose."""

    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("font-size scale must be positive and finite")
    fields, rows = _read_pose_rows(source)
    center_x, center_y, source_span = _xy_reference(rows)
    for row in rows:
        row["x"] = repr(center_x + (float(row["x"]) - center_x) * scale)
        row["y"] = repr(center_y + (float(row["y"]) - center_y) * scale)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return center_x, center_y, source_span


def scale_complete_target_image(
    source: Path,
    destination: Path,
    scale: float,
    image_size: int,
) -> None:
    """Uniformly scale the complete target canvas and keep it centred."""

    if not math.isfinite(scale) or not 0.0 < scale <= 1.0:
        raise ValueError("target image scale must be in (0, 1]")
    with Image.open(source) as opened:
        image = opened.convert("L").resize((image_size, image_size), _LANCZOS)
    pixels = image.load()
    corners = [
        int(pixels[0, 0]),
        int(pixels[image_size - 1, 0]),
        int(pixels[0, image_size - 1]),
        int(pixels[image_size - 1, image_size - 1]),
    ]
    background = sorted(corners)[len(corners) // 2]
    scaled_size = max(1, min(image_size, int(round(image_size * scale))))
    scaled = image.resize((scaled_size, scaled_size), _LANCZOS)
    canvas = Image.new("L", (image_size, image_size), color=background)
    offset = ((image_size - scaled_size) // 2, (image_size - scaled_size) // 2)
    canvas.paste(scaled, offset)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def export_physical_trajectory(
    inverted_csv: Path,
    destination: Path,
    *,
    source_center_x: float,
    source_center_y: float,
    source_span: float,
    reference_font_size_m: float,
    font_size_m: float,
    inversion_label: str = "v16_v42_fused_pose",
) -> int:
    """Convert model-source x/y units once into glyph-centred metres."""

    fields, rows = _read_pose_rows(inverted_csv)
    metres_per_source_unit = reference_font_size_m / source_span
    for extra in ("xy_unit", "font_size_m", "offline_inversion"):
        if extra not in fields:
            fields.append(extra)
    for row in rows:
        row["x"] = repr(
            (float(row["x"]) - source_center_x) * metres_per_source_unit
        )
        row["y"] = repr(
            (float(row["y"]) - source_center_y) * metres_per_source_unit
        )
        row["xy_unit"] = "m"
        row["font_size_m"] = repr(font_size_m)
        row["offline_inversion"] = inversion_label
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


class OfflineFontSizeGenerator:
    """Generate and cache a full-pose V16 trajectory for one physical size."""

    def __init__(self, config: OfflineInversionConfig) -> None:
        config.validate()
        self.config = config
        self._checkpoint_sha256 = _sha256(config.bbsmg_checkpoint)
        expected = config.expected_checkpoint_sha256.strip().lower()
        if expected and self._checkpoint_sha256.lower() != expected:
            raise RuntimeError(
                "V16 B-BSMG checkpoint SHA256 mismatch: "
                f"{self._checkpoint_sha256} != {expected}"
            )

    def identity(self) -> dict:
        """Return the explicit model/pipeline identity exposed by the API."""

        return {
            "interface_version": INVERSION_INTERFACE_VERSION,
            "model_weight_version": MODEL_WEIGHT_VERSION,
            "checkpoint": str(self.config.bbsmg_checkpoint),
            "checkpoint_sha256": self._checkpoint_sha256,
            "inversion_pipeline": INVERSION_PIPELINE,
            "fixed_fields": ["x", "y"],
            "optimized_fields": ["H"],
            "derived_fields": ["alpha", "beta", "gamma"],
            "ros_ink_renderer": "shared V16 PaperFusionRenderer incremental stream",
        }

    def _ensure_neural_ink(self, output_dir: Path) -> None:
        exporter = Path(__file__).with_name('export_neural_ink.py')
        audit_path = output_dir / 'neural_ink_audit.json'
        stream = output_dir / 'neural_ink.npz'
        if audit_path.is_file() and stream.is_file():
            audit = json.loads(audit_path.read_text(encoding='utf-8'))
            if (audit.get('physical_csv_sha256') == _sha256(output_dir/'physical_trajectory.csv')
                    and audit.get('checkpoint_sha256') == self._checkpoint_sha256
                    and audit.get('exporter_sha256') == _sha256(exporter)
                    and audit.get('renderer_source_sha256') == _sha256(self.config.model_root/'models/paper_fusion_renderer.py')
                    and audit.get('stream_sha256') == _sha256(stream)):
                return
        with (output_dir/'neural_ink_export.log').open('w', encoding='utf-8') as log:
            result = subprocess.run([
                str(self.config.python_executable), str(exporter),
                '--model-root', str(self.config.model_root), '--output-dir', str(output_dir),
            ], cwd=self.config.model_root, stdout=log, stderr=subprocess.STDOUT,
               timeout=self.config.timeout_s, check=False)
        if result.returncode:
            raise RuntimeError(f'统一笔触导出失败：{output_dir}/neural_ink_export.log')

    @staticmethod
    def _validate_source_contract(entry: TrajectoryEntry) -> None:
        assert entry.trajectory_csv is not None
        _, rows = _read_pose_rows(entry.trajectory_csv)
        prototypes = {
            str(row.get("prototype", "")).strip() for row in rows
            if str(row.get("prototype", "")).strip()
        }
        if prototypes != {INVERSION_PIPELINE}:
            raise RuntimeError(
                f"“{entry.character}”基础轨迹协议不是 {INVERSION_PIPELINE}: "
                f"{sorted(prototypes) or ['missing']}"
            )

    @staticmethod
    def _shape_target(entry: TrajectoryEntry) -> Path:
        """Use the image rendered by the matching v42 trajectory when present.

        Some database target selections have the right Unicode label but the
        wrong glyph form (for example simplified ``汉`` paired with ``漢``).
        The completed v42 render is generated from the exact trajectory that
        ROS must preserve, so it is the authoritative shape target for a
        font-size re-inversion.  Older entries fall back to their catalog
        target.
        """

        if entry.output_dir is not None:
            rendered = entry.output_dir / "inversion_rendered.png"
            if rendered.is_file():
                return rendered
        assert entry.target_image is not None
        return entry.target_image

    @staticmethod
    def _validate_fused_report(path: Path, character: str) -> dict:
        if not path.is_file():
            raise RuntimeError(f"离线反演缺少质量报告：{path}")
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"离线反演质量报告不可读：{path}") from error
        if report.get("format") != INVERSION_PIPELINE:
            raise RuntimeError(
                f"“{character}”未生成全库同款 v42 fused-pose 轨迹"
            )
        if report.get("fixed_xy") is not True:
            raise RuntimeError(f"“{character}”反演改变了字号缩放后的中心线")
        if set(report.get("optimized_fields", [])) != {"H"}:
            raise RuntimeError(f"“{character}”反演自由优化字段不是仅 H")
        if not {"alpha", "beta", "gamma"}.issubset(
            set(report.get("derived_fields", []))
        ):
            raise RuntimeError(f"“{character}”没有重新推导三个旋转角")
        safety = report.get("trajectory_safety", {})
        if safety.get("safe") is not True:
            raise RuntimeError(f"“{character}”离线轨迹未通过安全检查")
        metrics = report.get("metrics", {})
        for name in ("plain_mse", "dice_at_0.5", "ink_ratio_at_0.5"):
            value = metrics.get(name)
            if value is None or not math.isfinite(float(value)):
                raise RuntimeError(f"“{character}”缺少有效质量指标 {name}")
        return report

    def _cache_directory(
        self,
        entry: TrajectoryEntry,
        font_size_m: float,
    ) -> tuple[Path, str, str]:
        assert entry.trajectory_csv is not None
        assert entry.target_image is not None
        shape_target = self._shape_target(entry)
        source_digest = _sha256(entry.trajectory_csv)
        target_digest = _sha256(shape_target)
        config_value = {
            "interface_version": INVERSION_INTERFACE_VERSION,
            "model_weight_version": MODEL_WEIGHT_VERSION,
            "pipeline": INVERSION_PIPELINE,
            "source_sha256": source_digest,
            "target_sha256": target_digest,
            "font_size_m": round(float(font_size_m), 9),
            "reference_font_size_m": self.config.reference_font_size_m,
            "image_size": self.config.image_size,
            "padding": self.config.padding,
            "checkpoint_sha256": self._checkpoint_sha256,
            "order": self.config.order,
            "max_steps": self.config.max_steps,
            "optimization_size": self.config.optimization_size,
            "device": self.config.device,
            "fixed_fields": ["x", "y"],
            "optimized_fields": ["H"],
            "derived_fields": ["alpha", "beta", "gamma"],
        }
        cache_digest = hashlib.sha256(
            json.dumps(config_value, sort_keys=True).encode("utf-8")
        ).hexdigest()
        millimetres = int(round(font_size_m * 1000.0))
        directory = (
            self.config.cache_root
            / character_directory_name(entry.character)
            / f"font_{millimetres:04d}mm_{cache_digest[:12]}"
        )
        return directory, source_digest, target_digest

    def generate(
        self,
        entry: TrajectoryEntry,
        font_size_m: float,
        progress: Optional[ProgressCallback] = None,
    ) -> TrajectoryEntry:
        if entry.trajectory_csv is None or not entry.trajectory_csv.is_file():
            raise FileNotFoundError(f"source trajectory is unavailable for {entry.character}")
        if entry.target_image is None or not entry.target_image.is_file():
            raise FileNotFoundError(f"target image is unavailable for {entry.character}")
        if not entry.sample_id:
            raise ValueError(f"sample_id is unavailable for {entry.character}")
        self._validate_source_contract(entry)
        if not math.isfinite(font_size_m) or font_size_m <= 0.0:
            raise ValueError("font size must be positive and finite")
        if font_size_m > self.config.reference_font_size_m + 1.0e-12:
            raise ValueError(
                f"字号不能超过离线标定上限 "
                f"{self.config.reference_font_size_m * 1000:.0f} mm"
            )

        output_dir, source_digest, target_digest = self._cache_directory(
            entry, font_size_m
        )
        physical_csv = output_dir / "physical_trajectory.csv"
        scaled_target = output_dir / "scaled_target.png"
        metadata_path = output_dir / "offline_inversion.json"
        if physical_csv.is_file() and scaled_target.is_file() and metadata_path.is_file():
            self._ensure_neural_ink(output_dir)
            if progress:
                progress(f"“{entry.character}”{font_size_m * 1000:.0f} mm 离线轨迹命中缓存")
            return self._entry(entry, physical_csv, scaled_target, output_dir, font_size_m)

        output_dir.mkdir(parents=True, exist_ok=True)
        scale = font_size_m / self.config.reference_font_size_m
        shape_target = self._shape_target(entry)
        initial_pose = output_dir / "scaled_initial_pose.csv"
        if progress:
            progress(f"正在缩放“{entry.character}”的 v42 基准字形和物理中心线")
        center_x, center_y, source_span = scale_initial_pose_csv(
            entry.trajectory_csv,
            initial_pose,
            scale,
        )
        scale_complete_target_image(
            shape_target,
            scaled_target,
            scale,
            self.config.image_size,
        )

        script = self.config.model_root / "tools" / "invert_paper_trajectory.py"
        if not script.is_file():
            raise FileNotFoundError(f"V16 inversion script does not exist: {script}")
        command = [
            str(self.config.python_executable),
            "-u",
            str(script),
            "--trajectory_csv",
            str(entry.trajectory_csv),
            "--initial_pose_csv",
            str(initial_pose),
            "--initial_pose_xy_source",
            "csv",
            "--target_image",
            str(scaled_target),
            "--bbsmg_ckpt",
            str(self.config.bbsmg_checkpoint),
            "--character",
            entry.character,
            "--sample_id",
            entry.sample_id,
            "--output_dir",
            str(output_dir),
            "--output_stem",
            "inversion",
            "--device",
            self.config.device,
            "--image_size",
            str(self.config.image_size),
            "--padding",
            str(self.config.padding),
            "--order",
            str(self.config.order),
            "--max_steps",
            str(self.config.max_steps),
            "--damping",
            "0.05",
            "--optimization_size",
            str(self.config.optimization_size),
            "--point_batch_size",
            str(self.config.point_batch_size),
            "--pixel_weight",
            "12.0",
            "--h_smoothness_weight",
            "0.2",
            "--h_point_velocity_weight",
            "5.0",
            "--h_point_acceleration_weight",
            "10.0",
            "--h_prior_weight",
            "0.001",
            "--field_mode",
            "h_only",
            "--fused_pose_from_height",
            "--cap_order_to_points",
            "--dynamic_profile",
            "wang2020_figure4_digitized_v1",
            "--pixels_per_model_unit",
            "20.0",
            "--footprint_longitudinal_scale",
            "0.22",
            "--footprint_transverse_scale",
            "0.262",
            "--render_max_step_px",
            "2.0",
        ]
        if progress:
            progress(
                f"正在用全库同款 v42 fused-pose 反演“{entry.character}”："
                "固定字号中心线、优化 H、重新推导 alpha/beta/gamma"
            )
        log_path = output_dir / "offline_process.log"
        environment = dict(os.environ)
        environment["PYTHONUNBUFFERED"] = "1"
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                cwd=self.config.model_root,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=self.config.timeout_s,
                check=False,
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"“{entry.character}”字号离线反演失败（exit={result.returncode}），"
                f"日志：{log_path}"
            )
        inverted_csv = output_dir / "inversion_trajectory.csv"
        if not inverted_csv.is_file():
            raise RuntimeError(f"离线反演没有生成轨迹：{inverted_csv}")
        report = self._validate_fused_report(
            output_dir / "inversion_report.json", entry.character
        )
        point_count = export_physical_trajectory(
            inverted_csv,
            physical_csv,
            source_center_x=center_x,
            source_center_y=center_y,
            source_span=source_span,
            reference_font_size_m=self.config.reference_font_size_m,
            font_size_m=font_size_m,
        )
        metadata = {
            "format": INVERSION_INTERFACE_VERSION,
            "model_weight_version": MODEL_WEIGHT_VERSION,
            "inversion_pipeline": INVERSION_PIPELINE,
            "character": entry.character,
            "sample_id": entry.sample_id,
            "font_size_m": font_size_m,
            "reference_font_size_m": self.config.reference_font_size_m,
            "scale": scale,
            "source_trajectory": str(entry.trajectory_csv),
            "catalog_target": str(entry.target_image),
            "shape_target": str(shape_target),
            "source_sha256": source_digest,
            "target_sha256": target_digest,
            "checkpoint": str(self.config.bbsmg_checkpoint),
            "checkpoint_sha256": self._checkpoint_sha256,
            "fixed_fields": ["x", "y"],
            "optimized_fields": ["H"],
            "derived_fields": ["alpha", "beta", "gamma"],
            "quality_metrics": report.get("metrics", {}),
            "physical_xy": True,
            "point_count": point_count,
            "command": command,
        }
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._ensure_neural_ink(output_dir)
        if progress:
            progress(f"“{entry.character}”{font_size_m * 1000:.0f} mm 离线轨迹已生成")
        return self._entry(entry, physical_csv, scaled_target, output_dir, font_size_m)

    def _entry(
        self,
        source: TrajectoryEntry,
        trajectory: Path,
        target: Path,
        output_dir: Path,
        font_size_m: float,
    ) -> TrajectoryEntry:
        metadata = dict(source.metadata)
        metadata.update(
            {
                "source": "v16_v42_fused_pose_offline_fontsize",
                "interface_version": INVERSION_INTERFACE_VERSION,
                "model_weight_version": MODEL_WEIGHT_VERSION,
                "checkpoint_sha256": self._checkpoint_sha256,
                "inversion_pipeline": INVERSION_PIPELINE,
                "coordinate_frame": "glyph_center_m",
                "xy_unit": "m",
                "font_size_m": font_size_m,
                "gamma_relative_to_path": False,
                "offline_full_pose_inversion": True,
                "shape_preserving_v42_fused_pose": True,
                "neural_ink_path": str(output_dir/'neural_ink.npz'),
                "neural_ink_audit": str(output_dir/'neural_ink_audit.json'),
            }
        )
        return TrajectoryEntry(
            character=source.character,
            status="ready",
            trajectory_csv=trajectory,
            target_image=target,
            sample_id=source.sample_id,
            output_dir=output_dir,
            point_count=source.point_count,
            # v42 ``inversion_rendered.png`` uses light ink on a dark
            # background.  Let the evaluator detect polarity and perform its
            # normal foreground alignment instead of assuming black-on-white.
            target_is_normalized=False,
            message="V16/v42 shape-preserving offline font-size trajectory",
            quality="v16_v42_fontsize_reinverted",
            metadata=metadata,
        )
