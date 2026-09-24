"""Six-field V16 inversion against a hash-pinned original target image.

This backend is separate from the historical reconstructed-A experiment.  It
proves the canonical image came from the original source, then makes every
physical-size target directly from that source crop with one resample.
"""
from __future__ import annotations

from dataclasses import asdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
import PIL
from PIL import Image

from .inversion_candidate_checks import check_candidate
from .joint_candidate import FIELDS, V16_CHECKPOINT_SHA256, load_joint_candidate
from .joint_fontsize_inversion import build_joint_command, scale_reference_canvas
from .offline_fontsize_inversion import (
    OfflineFontSizeGenerator,
    _read_pose_rows,
    _sha256,
    export_physical_trajectory,
    scale_initial_pose_csv,
)
from .target_provenance import render_scaled_original_target, validate_original_target_record


INTERFACE = "v16_original_target_joint_fontsize_v2"
MANIFEST_FORMAT = "v16_original_target_manifest_v1"


def _resolved(path: Any) -> Path:
    return Path(str(path)).expanduser().resolve()


class OriginalTargetFontSizeGenerator:
    """Generate complete V16 controls using only an original-target contract."""

    def __init__(self, config, references: list[Mapping[str, Any]], *, allow_screened: bool = False):
        self._exporter = OfflineFontSizeGenerator(config)
        self.config = config
        self.allow_screened = allow_screened
        if self._exporter._checkpoint_sha256 != V16_CHECKPOINT_SHA256:
            raise ValueError("original-target generator requires the pinned V16 checkpoint")
        if (config.image_size, config.padding, config.reference_font_size_m) != (128, 16, 0.29):
            raise ValueError("original targets require the validated 128/16/0.29m model frame")
        if config.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if (not math.isfinite(config.joint_foreground_weight)
                or not 0.0 <= config.joint_foreground_weight <= 12.0):
            raise ValueError("joint foreground weight must be finite and between 0 and 12")
        if (not math.isfinite(config.joint_xy_target_skeleton_weight)
                or config.joint_xy_target_skeleton_weight < 0.0):
            raise ValueError("target skeleton weight must be finite and non-negative")
        if (not math.isfinite(config.joint_xy_target_skeleton_max_distance_px)
                or config.joint_xy_target_skeleton_max_distance_px <= 0.0):
            raise ValueError("target skeleton distance must be positive and finite")
        if (not math.isfinite(config.joint_xy_target_skeleton_threshold)
                or not 0.0 < config.joint_xy_target_skeleton_threshold < 1.0):
            raise ValueError("target skeleton threshold must be in (0,1)")
        self.references: dict[tuple[str, str, str], dict[str, Any]] = {}
        for item in references:
            record = dict(item)
            key = (str(record.get("character", "")), str(record.get("sample_id", "")),
                   str(record.get("source_sha256", "")).lower())
            if len(key[0]) != 1 or not key[1] or len(key[2]) != 64:
                raise ValueError("invalid original-target reference identity")
            if key in self.references:
                raise ValueError("ambiguous original-target reference")
            size = float(record.get("font_size_m", float("nan")))
            if not math.isfinite(size) or not 0.0 < size <= config.reference_font_size_m:
                raise ValueError("invalid original-target reference physical size")
            self.references[key] = record

    def identity(self) -> dict[str, Any]:
        return {
            "interface_version": INTERFACE,
            "model_weight_version": "V16",
            "foreground_loss_weight": self.config.joint_foreground_weight,
            "target_skeleton_weight": self.config.joint_xy_target_skeleton_weight,
            "inversion_pipeline": "V16 six-field original-target inversion",
            "checkpoint_sha256": V16_CHECKPOINT_SHA256,
            "optimized_fields": sorted(FIELDS),
            "fixed_fields": [],
            "derived_fields": [],
            "runtime_scaling": False,
            "mode": "experimental_original_target_generation",
            "legacy_fallback": False,
            "historical_A_used": False,
            "visual_accepted": False,
            "reference_count": len(self.references),
            "allows_experimental_screened_targets": self.allow_screened,
        }

    def _validated_record(self, entry, source: Path) -> tuple[dict[str, Any], dict[str, Any]]:
        source_digest = _sha256(source)
        record = self.references.get((entry.character, entry.sample_id, source_digest))
        if record is None:
            raise ValueError(
                "no pinned original target for this character/sample/source; "
                "no A, catalog, rendered-image or legacy fallback"
            )
        if _resolved(record["source_trajectory"]) != source:
            raise ValueError("original-target manifest pins a different source trajectory path")
        provenance = validate_original_target_record(
            record, image_size=self.config.image_size, padding=self.config.padding,
            allow_screened=self.allow_screened,
        )
        if (provenance["character"], provenance["sample_id"]) != (
            entry.character, entry.sample_id
        ):
            raise ValueError("original target identity changed")
        return record, provenance

    @staticmethod
    def _seed(record: Mapping[str, Any], entry, source: Path, provenance: Mapping[str, Any]):
        folder_value = record.get("seed_folder")
        if not folder_value:
            return None, None
        folder = _resolved(folder_value)
        seed_entry = load_joint_candidate(folder, require_training_domain=True)
        if (seed_entry.character, seed_entry.sample_id) != (entry.character, entry.sample_id):
            raise ValueError("original-target seed identity mismatch")
        metadata = json.loads((folder / "offline_inversion.json").read_text(encoding="utf-8"))
        if metadata.get("format") != INTERFACE:
            raise ValueError("seed was not generated by the original-target contract")
        if _resolved(metadata.get("source_trajectory")) != source:
            raise ValueError("seed source coordinate frame mismatch")
        seed_provenance = metadata.get("target_provenance", {})
        for key in ("original_target_sha256", "normalized_target_sha256",
                    "target_source_sha256", "source_annotation_sha256"):
            if seed_provenance.get(key) != provenance.get(key):
                raise ValueError(f"seed original-target provenance mismatch: {key}")
        return folder / "inversion_trajectory.csv", float(metadata["font_size_m"])

    def generate(self, entry, font_size_m: float, progress=None):
        size = float(font_size_m)
        if not math.isfinite(size) or not 0.0 < size <= self.config.reference_font_size_m:
            raise ValueError("physical size must be positive and at most 290 mm")
        if entry.trajectory_csv is None or not entry.sample_id:
            raise ValueError("missing source trajectory or sample id")
        source = Path(entry.trajectory_csv).expanduser().resolve()
        record, provenance = self._validated_record(entry, source)
        _, original_rows = _read_pose_rows(source)
        if any(
            row["character"] != entry.character
            or row["sample_id"] != entry.sample_id
            or int(float(row["state"])) == 3
            for row in original_rows
        ):
            raise ValueError("source must contain only this sample and contact trajectories")
        seed, seed_size = self._seed(record, entry, source, provenance)
        implementation_files = [
            Path(__file__),
            Path(__file__).with_name("target_provenance.py"),
            Path(__file__).with_name("joint_fontsize_inversion.py"),
            Path(__file__).with_name("joint_optimizer.py"),
            Path(__file__).with_name("domain_seed.py"),
            Path(__file__).with_name("constrained_step.py"),
            Path(__file__).with_name("neural_domain.py"),
            Path(__file__).with_name("joint_candidate.py"),
            Path(__file__).with_name("run_joint_inversion.py"),
            Path(__file__).with_name("export_neural_ink.py"),
            self.config.model_root / "models/paper_fusion_renderer.py",
            self.config.model_root / "tools/invert_paper_trajectory.py",
            self.config.model_root / "utils/image_preprocessing.py",
            self.config.model_root / "datasets/calligraphy_image_dataset.py",
        ]
        identity = {
            "interface": INTERFACE,
            "character": entry.character,
            "sample_id": entry.sample_id,
            "source_sha256": provenance["source_sha256"],
            "original_target_sha256": provenance["original_target_sha256"],
            "normalized_target_sha256": provenance["normalized_target_sha256"],
            "target_source_sha256": provenance["target_source_sha256"],
            "source_annotation_sha256": provenance["source_annotation_sha256"],
            "glyph_identity": provenance["glyph_identity"],
            "allow_screened": self.allow_screened,
            "target_preprocessor_runtime": {
                "python": f"{sys.version_info.major}.{sys.version_info.minor}",
                "numpy": np.__version__,
                "pillow": PIL.__version__,
            },
            "reference_size_m": float(record["font_size_m"]),
            "requested_size_m": size,
            "checkpoint_sha256": V16_CHECKPOINT_SHA256,
            "seed_sha256": _sha256(seed) if seed else None,
            "seed_size_m": seed_size,
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(self.config).items()
            },
            "implementation": {str(path): _sha256(path) for path in implementation_files},
        }
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        output = (
            Path(self.config.cache_root).expanduser().resolve()
            / f"char_u{ord(entry.character):04x}"
            / f"original_target_{key[:24]}"
        )
        marker = output / "original_target_generation_complete.json"
        if output.exists():
            if not marker.is_file() or json.loads(marker.read_text(encoding="utf-8")) != identity:
                raise RuntimeError(f"incomplete or conflicting original-target generation; inspect {output}")
            return load_joint_candidate(output, require_training_domain=True)
        output.mkdir(parents=True, exist_ok=False)
        (output / "original_target_generation_request.json").write_text(
            json.dumps(identity, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        initial = output / "scaled_initial_pose.csv"
        scaled_target = output / "scaled_target.png"
        cx, cy, span = scale_initial_pose_csv(
            source, initial, size / self.config.reference_font_size_m
        )
        if seed:
            seed_fields, rows = _read_pose_rows(seed)
            if [(row["stroke_id"], row["point_id"]) for row in rows] != [
                (row["stroke_id"], row["point_id"]) for row in original_rows
            ]:
                raise ValueError("seed sample sequence mismatch")
            for row in rows:
                row["x"] = repr(cx + (float(row["x"]) - cx) * size / seed_size)
                row["y"] = repr(cy + (float(row["y"]) - cy) * size / seed_size)
            with initial.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=seed_fields)
                writer.writeheader()
                writer.writerows(rows)
        normalized_target = Path(provenance["normalized_target_image"])
        ratio = size / float(record["font_size_m"])
        scaling = render_scaled_original_target(
            provenance, scaled_target, ratio,
            image_size=self.config.image_size, padding=self.config.padding,
        )
        legacy_target = output / "scaled_target_two_resamples_comparison.png"
        scale_reference_canvas(
            normalized_target,
            legacy_target,
            ratio,
            image_size=self.config.image_size,
        )
        with Image.open(scaled_target) as opened:
            direct_pixels = np.asarray(opened.convert("L"), dtype=np.int16)
        with Image.open(legacy_target) as opened:
            legacy_pixels = np.asarray(opened.convert("L"), dtype=np.int16)
        difference = np.abs(direct_pixels - legacy_pixels)
        scaling["two_resample_comparison"] = {
            "different_pixels": int(np.count_nonzero(difference)),
            "max_abs_8bit": int(difference.max()),
            "mean_abs_8bit": float(difference.mean()),
            "mse_8bit": float(np.mean(np.square(difference.astype(np.float64)))),
            "two_resample_sha256": _sha256(legacy_target),
        }
        (output / "target_scaling_comparison.json").write_text(
            json.dumps(scaling, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        command = build_joint_command(
            self.config, source, initial, scaled_target, output,
            entry.character, entry.sample_id,
        )
        if progress:
            progress(
                f"正在以原始目标图对‘{entry.character}’ {size * 1000:.3f} mm "
                "进行 V16 六维模型反演"
            )
        environment = dict(
            os.environ,
            BETA_INVERSION_MODEL_ROOT=str(self.config.model_root),
            PYTHONUNBUFFERED="1",
        )
        with (output / "joint_process.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                command,
                cwd=self.config.model_root,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=self.config.timeout_s,
                check=True,
            )
        report = json.loads((output / "inversion_report.json").read_text(encoding="utf-8"))
        if set(report["optimized_fields"]) != FIELDS or "--fused_pose_from_height" in command:
            raise ValueError("model did not return six independently optimized fields")
        _, before = _read_pose_rows(initial)
        _, after = _read_pose_rows(output / "inversion_trajectory.csv")
        checks = check_candidate(before, after)
        if (not checks["eligible_for_further_validation"]
                or [row["state"] for row in before] != [row["state"] for row in after]):
            raise ValueError("generated trajectory failed geometry/state checks")
        export_physical_trajectory(
            output / "inversion_trajectory.csv",
            output / "physical_trajectory.csv",
            source_center_x=cx,
            source_center_y=cy,
            source_span=span,
            reference_font_size_m=self.config.reference_font_size_m,
            font_size_m=size,
            inversion_label="v16_original_target_joint_pose_metric_units",
        )
        metadata = {
            "format": INTERFACE,
            "character": entry.character,
            "sample_id": entry.sample_id,
            "font_size_m": size,
            "reference_font_size_m": self.config.reference_font_size_m,
            "source_trajectory": str(source),
            "source_sha256": provenance["source_sha256"],
            "checkpoint": str(self.config.bbsmg_checkpoint),
            "checkpoint_sha256": V16_CHECKPOINT_SHA256,
            "command": command,
            "optimized_fields": sorted(FIELDS),
            "fixed_fields": [],
            "derived_fields": [],
            "quality_metrics": report["metrics"],
            "target_provenance": provenance,
            "target_scaling": scaling,
            "historical_A_used": False,
            "visual_accepted": False,
        }
        (output / "offline_inversion.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output / "control_change_audit.json").write_text(
            json.dumps(
                {
                    "candidate_checks": checks,
                    "optimized_fields": sorted(FIELDS),
                    "metrics": report["metrics"],
                    "target_basis": "original_image",
                    "historical_A_used": False,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        self._exporter._ensure_neural_ink(output)
        result = load_joint_candidate(output, require_training_domain=True)
        with marker.open("x", encoding="utf-8") as handle:
            json.dump(identity, handle, ensure_ascii=False, indent=2)
        return result
