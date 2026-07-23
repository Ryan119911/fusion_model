import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ensure_dirs, load_config
from datasets.calligraphy_image_dataset import CalligraphyImageDataset
from datasets.character_dataset import CHARACTER_DATA_FORMAT
from datasets.trajectory_dataset import load_trajectory_csv
from utils.character_features import SPATIAL_CHANNEL_NAMES, extract_character_spatial_maps
from utils.character_alignment import align_target_to_trajectory, alignment_metrics
from utils.character_script import CharacterScriptMapper, SCRIPT_MODES
from utils.image_preprocessing import letterbox_character_image, load_character_image
from utils.structure_mask import (
    STRUCTURE_TARGET_MODE,
    build_structure_mask,
)


TARGET_PADDING = 4


def rasterize_character(
    normalized_strokes,
    canvas_size: int,
    width: int = 5,
) -> np.ndarray:
    image = Image.new("L", (canvas_size, canvas_size), 0)
    draw = ImageDraw.Draw(image)
    for points in normalized_strokes:
        if len(points) >= 2:
            draw.line(points, fill=255, width=width, joint="curve")
        elif len(points) == 1:
            x, y = points[0]
            radius = max(width // 2, 1)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
    return np.asarray(image, dtype=np.float32) / 255.0


def _normalize_candidate_path(value: Any) -> str:
    return str(value or "").replace("\\", "/").removeprefix("./")


def load_candidate_exclusions(path: Optional[str]) -> List[Dict[str, Any]]:
    if not path:
        return []
    exclusion_path = Path(path)
    if not exclusion_path.exists():
        raise FileNotFoundError(f"Candidate exclusion file not found: {path}")
    with open(exclusion_path, "r", encoding="utf-8") as file:
        values = json.load(file)
    if not isinstance(values, list):
        raise ValueError("Candidate exclusion JSON must contain a list")
    result = []
    for index, value in enumerate(values):
        if not isinstance(value, dict) or not value.get("image_path"):
            raise ValueError(
                f"Candidate exclusion #{index} requires an image_path object field"
            )
        result.append(dict(value))
    return result


def candidate_exclusion(
    candidate: Dict[str, Any],
    character: str,
    exclusions: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    candidate_path = _normalize_candidate_path(candidate.get("image_path"))
    candidate_bbox = candidate.get("bbox")
    for exclusion in exclusions:
        excluded_character = exclusion.get("character")
        if excluded_character and str(excluded_character) != str(character):
            continue
        excluded_path = _normalize_candidate_path(exclusion.get("image_path"))
        if (
            excluded_path != candidate_path
            and not candidate_path.endswith("/" + excluded_path)
        ):
            continue
        excluded_bbox = exclusion.get("bbox")
        if excluded_bbox is not None:
            if candidate_bbox is None or len(candidate_bbox) != len(excluded_bbox):
                continue
            if any(
                abs(float(actual) - float(expected)) > 1e-3
                for actual, expected in zip(candidate_bbox, excluded_bbox)
            ):
                continue
        return exclusion
    return None


def build_image_index(
    dataset: CalligraphyImageDataset,
    mapper: Optional[CharacterScriptMapper] = None,
) -> Tuple[DefaultDict[str, List[Dict[str, Any]]], Dict[str, int]]:
    result: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    mapping_counts = Counter()
    for item in dataset.index:
        character = item.get("character")
        if character:
            annotation_character = str(character)
            try:
                mapped_character = (
                    mapper.convert(annotation_character) if mapper else annotation_character
                )
            except ValueError:
                mapping_counts["mapping_errors"] += 1
                continue
            mapped_item = dict(item)
            mapped_item["annotation_character"] = annotation_character
            mapped_item["mapped_target_character"] = mapped_character
            result[mapped_character].append(mapped_item)
            mapping_counts["boxes"] += 1
            if mapped_character != annotation_character:
                mapping_counts["remapped_boxes"] += 1
    mapping_counts["characters"] = len(result)
    return result, dict(mapping_counts)


def target_quality_failures(
    metrics: Dict[str, float],
    thresholds: Dict[str, float],
) -> List[str]:
    failures = []
    if metrics["coverage"] < thresholds["min_alignment_coverage"]:
        failures.append("coverage_below_threshold")
    if metrics["support_dice"] < thresholds["min_support_dice"]:
        failures.append("support_dice_below_threshold")
    if (
        metrics["target_outside_support_fraction"]
        > thresholds["max_outside_support_fraction"]
    ):
        failures.append("outside_support_above_threshold")
    area_ratio = metrics["target_to_support_area_ratio"]
    if area_ratio < thresholds["min_target_support_area_ratio"]:
        failures.append("target_support_area_ratio_too_small")
    if area_ratio > thresholds["max_target_support_area_ratio"]:
        failures.append("target_support_area_ratio_too_large")
    if metrics["target_ink_fraction"] > thresholds["max_target_ink_fraction"]:
        failures.append("target_ink_above_threshold")
    if (
        metrics["foreground_bbox_fill_fraction"]
        > thresholds["max_foreground_bbox_fill_fraction"]
    ):
        failures.append("foreground_bbox_fill_above_threshold")
    if metrics["border_ink_fraction"] > thresholds["max_border_ink_fraction"]:
        failures.append("border_ink_above_threshold")
    return failures


def save_audit_panel(
    path: Path,
    spatial_maps: np.ndarray,
    unregistered_target: np.ndarray,
    aligned_target: np.ndarray,
    structure_target: np.ndarray,
) -> None:
    """Save proximity, raw/aligned grayscale, structure mask and RGB overlap."""
    proximity = np.clip(spatial_maps[1], 0.0, 1.0)
    centerline = np.clip(spatial_maps[0], 0.0, 1.0)
    panels = [proximity, unregistered_target, aligned_target, structure_target]
    gray_panels = [
        np.repeat((np.clip(panel, 0.0, 1.0) * 255).astype(np.uint8)[..., None], 3, axis=2)
        for panel in panels
    ]
    overlap = np.zeros((*aligned_target.shape, 3), dtype=np.uint8)
    overlap[..., 0] = (np.clip(structure_target, 0.0, 1.0) * 255).astype(np.uint8)
    overlap[..., 1] = (centerline * 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate([*gray_panels, overlap], axis=1), mode="RGB").save(path)


def select_best_real_target(
    image_dataset: CalligraphyImageDataset,
    candidates: List[Dict[str, Any]],
    spatial_maps: np.ndarray,
    canvas_size: int,
    max_target_candidates: int,
    max_registered_candidates: int,
    quality_thresholds: Dict[str, float],
    structure_threshold: float,
    min_component_pixels: int,
    candidate_exclusions: List[Dict[str, Any]],
) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, np.ndarray]]:
    """Clean candidates, register matches, and return the best structure mask."""
    excluded_candidates = []
    retained_candidates = []
    character = str(candidates[0].get("mapped_target_character") or "") if candidates else ""
    for candidate in candidates:
        exclusion = candidate_exclusion(candidate, character, candidate_exclusions)
        if exclusion is None:
            retained_candidates.append(candidate)
        else:
            excluded_candidates.append({
                "image_path": candidate.get("image_path"),
                "bbox": candidate.get("bbox"),
                "reason": exclusion.get("reason"),
            })
    candidates = retained_candidates
    if not candidates:
        raise ValueError("All real target candidates were manually excluded")
    if max_target_candidates > 0:
        candidates = candidates[:max_target_candidates]
    prepared = []
    errors = 0
    for selected in candidates:
        try:
            image_sample = image_dataset._build_sample(selected)
            target, transform = letterbox_character_image(
                image_sample["image"],
                canvas_size=canvas_size,
                padding=TARGET_PADDING,
                crop_foreground=True,
            )
            raw_metrics = alignment_metrics(target, spatial_maps[0], spatial_maps[1])
            prepared.append((raw_metrics["score"], target, transform, selected, raw_metrics))
        except (OSError, ValueError, RuntimeError) as error:
            errors += 1
            print(f"[WARN] Target candidate failed: {selected.get('image_path')}: {error}")
    if not prepared:
        raise ValueError("No readable target candidates remain for this character")

    prepared.sort(key=lambda item: item[0], reverse=True)
    registration_pool = prepared[:max(max_registered_candidates, 1)]
    best = None
    passing_registered_candidates = 0
    for raw_score, target, transform, selected, raw_metrics in registration_pool:
        try:
            aligned_gray, registration = align_target_to_trajectory(
                target,
                centerline=spatial_maps[0],
                proximity=spatial_maps[1],
            )
            structure_target, structure_info = build_structure_mask(
                aligned_gray,
                threshold=structure_threshold,
                min_component_pixels=min_component_pixels,
            )
        except ValueError:
            errors += 1
            continue
        registered_metrics = alignment_metrics(
            structure_target,
            spatial_maps[0],
            spatial_maps[1],
        )
        failures = target_quality_failures(registered_metrics, quality_thresholds)
        if not failures:
            passing_registered_candidates += 1
        rank = (not failures, float(registered_metrics["score"]))
        if best is None or rank > best[0]:
            best = (
                rank,
                structure_target,
                aligned_gray,
                target,
                transform,
                selected,
                raw_metrics,
                registration,
                failures,
                structure_info,
            )
    if best is None:
        raise ValueError("No registered target candidate produced a structure mask")

    (
        _, structure_target, aligned_gray, unregistered, transform, selected,
        raw_metrics, registration, selected_failures, structure_info,
    ) = best
    info = {
        "image_path": str(selected.get("image_path")),
        "json_path": str(selected.get("json_path")),
        "bbox": selected.get("bbox"),
        "annotation_character": selected.get("annotation_character"),
        "mapped_target_character": selected.get("mapped_target_character"),
        "canvas_transform": transform,
        "registration": registration,
        "target_candidates_considered": len(prepared),
        "target_candidates_registered": len(registration_pool),
        "target_candidates_passing_quality": passing_registered_candidates,
        "selected_candidate_quality_failures": selected_failures,
        "target_candidate_errors": errors,
        "target_candidates_excluded": len(excluded_candidates),
        "excluded_candidates": excluded_candidates,
        "unregistered_alignment": raw_metrics,
        "structure_cleanup": structure_info,
        "structure_alignment": alignment_metrics(
            structure_target,
            spatial_maps[0],
            spatial_maps[1],
        ),
    }
    audit = {
        "unregistered_target": unregistered,
        "aligned_target": aligned_gray,
        "structure_target": structure_target,
    }
    return structure_target, info, audit


def main(args) -> None:
    if not 0.0 <= args.min_alignment_coverage <= 1.0:
        raise ValueError("--min_alignment_coverage must be in [0, 1]")
    if args.max_target_candidates < 0:
        raise ValueError("--max_target_candidates must be non-negative")
    if args.max_registered_candidates < 1:
        raise ValueError("--max_registered_candidates must be positive")
    for name in (
        "min_support_dice",
        "max_outside_support_fraction",
        "min_target_support_area_ratio",
        "max_target_support_area_ratio",
        "max_target_ink_fraction",
        "max_foreground_bbox_fill_fraction",
        "max_border_ink_fraction",
    ):
        value = float(getattr(args, name))
        if value < 0.0:
            raise ValueError(f"--{name} must be non-negative")
    for name in (
        "min_support_dice",
        "max_outside_support_fraction",
        "max_target_ink_fraction",
        "max_foreground_bbox_fill_fraction",
        "max_border_ink_fraction",
    ):
        if float(getattr(args, name)) > 1.0:
            raise ValueError(f"--{name} must not exceed 1")
    if args.min_target_support_area_ratio > args.max_target_support_area_ratio:
        raise ValueError("minimum target/support area ratio exceeds maximum")
    if args.audit_limit_per_status < 0:
        raise ValueError("--audit_limit_per_status must be non-negative")
    if not 0.0 < args.structure_threshold < 1.0:
        raise ValueError("--structure_threshold must satisfy 0 < value < 1")
    if args.min_component_pixels < 0:
        raise ValueError("--min_component_pixels must be non-negative")
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    if args.chirography is not None:
        cfg.data.chirography_filter = args.chirography

    char_cfg = cfg.character_generator
    canvas_size = int(char_cfg.image_size)
    trajectory_padding = (
        int(args.trajectory_padding)
        if args.trajectory_padding is not None
        else int(cfg.data.character_trajectory_padding)
    )
    if char_cfg.input_channels != len(SPATIAL_CHANNEL_NAMES):
        raise ValueError(
            f"character_generator.input_channels={char_cfg.input_channels}, but the spatial "
            f"schema contains {len(SPATIAL_CHANNEL_NAMES)} channels"
        )
    if args.target_image and not args.target_character:
        raise ValueError("--target_image requires --target_character")
    candidate_exclusions = load_candidate_exclusions(args.exclude_candidates_json)
    if candidate_exclusions:
        print(
            f"[INFO] Loaded {len(candidate_exclusions)} manual target candidate exclusion(s)"
        )

    trajectories = load_trajectory_csv(args.trajectory_csv or cfg.data.trajectory_csv)
    if args.character:
        trajectories = [sample for sample in trajectories if sample.character == args.character]
    if not trajectories:
        raise RuntimeError("No trajectory samples matched the requested character filter")
    print(f"[INFO] Loaded {len(trajectories)} character trajectories")

    image_dataset = None
    image_index: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    script_mapping_stats: Dict[str, int] = {}
    if Path(cfg.data.image_dir).exists() and Path(cfg.data.json_dir).exists():
        script_mapper = CharacterScriptMapper(args.target_script)
        image_dataset = CalligraphyImageDataset(
            image_dir=cfg.data.image_dir,
            json_dir=cfg.data.json_dir,
            image_ext=cfg.data.image_ext,
            image_size=None,
            grayscale=True,
            padding=0.0,
            data_csv=getattr(cfg.data, "data_csv", None),
            chirography_filter=getattr(cfg.data, "chirography_filter", None),
        )
        image_index, script_mapping_stats = build_image_index(image_dataset, script_mapper)
        print(f"[INFO] Indexed real targets for {len(image_index)} characters")
        print(
            f"[INFO] Target script={args.target_script}; "
            f"remapped boxes={script_mapping_stats.get('remapped_boxes', 0)}"
        )
    else:
        print("[WARN] Calligraphy image folders are absent; synthetic whole-character targets will be used")

    override_target = None
    override_transform = None
    if args.target_image:
        override_target, override_transform = load_character_image(
            args.target_image,
            canvas_size=canvas_size,
            padding=TARGET_PADDING,
        )
        print(
            f"[INFO] External target {args.target_image} will supervise "
            f"character {args.target_character!r}"
        )

    inputs: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    metadata: List[Dict[str, Any]] = []
    source_counts: DefaultDict[str, int] = defaultdict(int)
    alignment_coverages: List[float] = []
    rejected_pairs: List[Dict[str, Any]] = []
    quality_thresholds = {
        "min_alignment_coverage": float(args.min_alignment_coverage),
        "min_support_dice": float(args.min_support_dice),
        "max_outside_support_fraction": float(args.max_outside_support_fraction),
        "min_target_support_area_ratio": float(args.min_target_support_area_ratio),
        "max_target_support_area_ratio": float(args.max_target_support_area_ratio),
        "max_target_ink_fraction": float(args.max_target_ink_fraction),
        "max_foreground_bbox_fill_fraction": float(
            args.max_foreground_bbox_fill_fraction
        ),
        "max_border_ink_fraction": float(args.max_border_ink_fraction),
    }
    output_path = Path(args.output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_dir = (
        Path(args.audit_dir)
        if args.audit_dir
        else output_path.parent / f"{output_path.stem}_audit"
    )
    audit_counts = Counter()
    audit_manifest: List[Dict[str, Any]] = []
    excluded_candidate_count = 0
    skipped = 0

    for sample_index, trajectory in enumerate(trajectories):
        character = trajectory.character
        if not character:
            skipped += 1
            continue
        try:
            spatial_maps, normalized_strokes = extract_character_spatial_maps(
                trajectory,
                canvas_size=canvas_size,
                padding=trajectory_padding,
                line_width=args.trajectory_width,
            )
        except ValueError as error:
            print(f"[SKIP] sample={trajectory.meta.get('sample_id')}: {error}")
            skipped += 1
            continue

        source_type = "synthetic"
        source_info: Dict[str, Any] = {}
        structure_info: Dict[str, Any] = {}
        audit_payload: Optional[Dict[str, np.ndarray]] = None
        if override_target is not None and character == args.target_character:
            aligned_gray, registration = align_target_to_trajectory(
                override_target.copy(),
                centerline=spatial_maps[0],
                proximity=spatial_maps[1],
            )
            target, structure_info = build_structure_mask(
                aligned_gray,
                threshold=args.structure_threshold,
                min_component_pixels=args.min_component_pixels,
            )
            source_type = "external"
            source_info = {
                "image_path": str(args.target_image),
                "canvas_transform": override_transform,
                "registration": registration,
                "structure_cleanup": structure_info,
            }
            audit_payload = {
                "unregistered_target": override_target.copy(),
                "aligned_target": aligned_gray,
                "structure_target": target,
            }
        elif image_dataset is not None and image_index.get(character):
            candidates = image_index[character]
            try:
                target, source_info, audit_payload = select_best_real_target(
                    image_dataset,
                    candidates,
                    spatial_maps,
                    canvas_size,
                    max_target_candidates=args.max_target_candidates,
                    max_registered_candidates=args.max_registered_candidates,
                    quality_thresholds=quality_thresholds,
                    structure_threshold=args.structure_threshold,
                    min_component_pixels=args.min_component_pixels,
                    candidate_exclusions=candidate_exclusions,
                )
            except ValueError as error:
                rejected_pairs.append({
                    "character": character,
                    "sample_id": trajectory.meta.get("sample_id"),
                    "reason": "target_selection_failed",
                    "error": str(error),
                })
                skipped += 1
                continue
            source_type = "real"
            structure_info = dict(source_info.get("structure_cleanup") or {})
            excluded_candidate_count += int(
                source_info.get("target_candidates_excluded", 0)
            )
        else:
            if args.require_real_target:
                rejected_pairs.append({
                    "character": character,
                    "sample_id": trajectory.meta.get("sample_id"),
                    "reason": "no_real_target_candidate",
                })
                skipped += 1
                continue
            target = rasterize_character(
                normalized_strokes,
                canvas_size=canvas_size,
                width=args.synthetic_width,
            )
            structure_info = {
                "mode": STRUCTURE_TARGET_MODE,
                "threshold": None,
                "min_component_pixels": 0,
                "foreground_pixels_before": int((target >= 0.5).sum()),
                "foreground_pixels_after": int((target >= 0.5).sum()),
                "foreground_fraction": float((target >= 0.5).mean()),
                "components_total": None,
                "components_removed": 0,
                "pixels_removed": 0,
            }

        centerline_mask = spatial_maps[0] >= 0.5
        trajectory_target_coverage = float((target[centerline_mask] >= 0.5).mean())
        alignment = alignment_metrics(target, spatial_maps[0], spatial_maps[1])
        failures = (
            target_quality_failures(alignment, quality_thresholds)
            if source_type in {"real", "external"}
            else []
        )
        audit_status = "rejected" if failures else "accepted"
        if (
            audit_payload is not None
            and audit_counts[audit_status] < args.audit_limit_per_status
        ):
            audit_number = audit_counts[audit_status]
            codepoints = "-".join(f"U+{ord(value):04X}" for value in character)
            audit_name = f"{audit_number:04d}_{codepoints}.png"
            audit_path = audit_dir / audit_status / audit_name
            save_audit_panel(
                audit_path,
                spatial_maps,
                audit_payload["unregistered_target"],
                audit_payload["aligned_target"],
                audit_payload["structure_target"],
            )
            audit_counts[audit_status] += 1
            audit_manifest.append({
                "character": character,
                "status": audit_status,
                "path": str(audit_path),
                "failed_checks": failures,
                "alignment_metrics": alignment,
                "image_path": source_info.get("image_path"),
                "annotation_character": source_info.get("annotation_character"),
                "mapped_target_character": source_info.get("mapped_target_character"),
                "structure_cleanup": structure_info,
            })
        if failures:
            rejected_pairs.append({
                "character": character,
                "sample_id": trajectory.meta.get("sample_id"),
                "target_source": source_type,
                "image_path": source_info.get("image_path"),
                "annotation_character": source_info.get("annotation_character"),
                "mapped_target_character": source_info.get("mapped_target_character"),
                "reason": "target_quality_failed",
                "failed_checks": failures,
                "alignment_metrics": alignment,
                "quality_thresholds": quality_thresholds,
            })
            skipped += 1
            continue
        alignment_coverages.append(trajectory_target_coverage)

        # Float16 is sufficient for normalized conditioning/target maps and
        # keeps a full 128x128 six-channel corpus practical in host memory.
        inputs.append(spatial_maps.astype(np.float16))
        targets.append(np.clip(target, 0.0, 1.0).astype(np.float16)[None, ...])
        metadata.append({
            "character": character,
            "sample_id": trajectory.meta.get("sample_id"),
            "trajectory_sample_index": sample_index,
            "num_strokes": len(trajectory.sorted_strokes()),
            "trajectory_padding": trajectory_padding,
            "trajectory_width": args.trajectory_width,
            "trajectory_target_coverage": trajectory_target_coverage,
            "alignment_metrics": alignment,
            "target_source": source_type,
            "target_mode": STRUCTURE_TARGET_MODE,
            "structure_cleanup": structure_info,
            **source_info,
        })
        source_counts[source_type] += 1

    rejection_path = output_path.with_suffix(".rejected.json")
    with open(rejection_path, "w", encoding="utf-8") as file:
        json.dump(rejected_pairs, file, ensure_ascii=False, indent=2)
    rejection_summary = Counter(item["reason"] for item in rejected_pairs)
    quality_failure_summary = Counter(
        failure
        for item in rejected_pairs
        for failure in item.get("failed_checks", [])
    )
    summary_path = output_path.with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump({
            "format_version": CHARACTER_DATA_FORMAT,
            "target_script": args.target_script,
            "script_mapping": script_mapping_stats,
            "accepted": len(inputs),
            "rejected": len(rejected_pairs),
            "rejection_reasons": dict(rejection_summary),
            "quality_failures": dict(quality_failure_summary),
            "quality_thresholds": quality_thresholds,
            "target_mode": STRUCTURE_TARGET_MODE,
            "structure_threshold": args.structure_threshold,
            "min_component_pixels": args.min_component_pixels,
            "candidate_exclusion_file": args.exclude_candidates_json,
            "excluded_target_candidates": excluded_candidate_count,
        }, file, ensure_ascii=False, indent=2)
    if audit_manifest:
        audit_dir.mkdir(parents=True, exist_ok=True)
        with open(audit_dir / "audit_manifest.json", "w", encoding="utf-8") as file:
            json.dump(audit_manifest, file, ensure_ascii=False, indent=2)
    if not inputs:
        raise RuntimeError(
            "No character-level pairs passed target cleanup/alignment. "
            f"See {rejection_path}"
        )
    if override_target is not None and source_counts["external"] == 0:
        raise RuntimeError(
            f"The external target was not used because character "
            f"{args.target_character!r} was absent from the selected trajectories"
        )

    np.savez_compressed(
        output_path,
        inputs=np.stack(inputs),
        targets=np.stack(targets),
        meta=np.asarray(metadata, dtype=object),
        channel_names=np.asarray(SPATIAL_CHANNEL_NAMES),
        format_version=np.asarray(CHARACTER_DATA_FORMAT),
        trajectory_padding=np.asarray(trajectory_padding, dtype=np.int32),
        trajectory_width=np.asarray(args.trajectory_width, dtype=np.int32),
        preprocessing_version=np.asarray("clean_register_script_structure_v3"),
        min_alignment_coverage=np.asarray(args.min_alignment_coverage, dtype=np.float32),
        target_script=np.asarray(args.target_script),
        quality_thresholds=np.asarray(json.dumps(quality_thresholds, sort_keys=True)),
        target_mode=np.asarray(STRUCTURE_TARGET_MODE),
        structure_threshold=np.asarray(args.structure_threshold, dtype=np.float32),
        min_component_pixels=np.asarray(args.min_component_pixels, dtype=np.int32),
    )
    print(f"[DONE] Character pairs saved to: {output_path}")
    print(
        f"[DONE] inputs shape: "
        f"({len(inputs)}, {len(SPATIAL_CHANNEL_NAMES)}, {canvas_size}, {canvas_size})"
    )
    print(f"[DONE] channels: {', '.join(SPATIAL_CHANNEL_NAMES)}")
    print(
        f"[DONE] trajectory normalization: padding={trajectory_padding}, "
        f"width={args.trajectory_width}"
    )
    print(f"[DONE] targets shape: ({len(targets)}, 1, {canvas_size}, {canvas_size})")
    print(
        f"[DONE] target mode: {STRUCTURE_TARGET_MODE}, "
        f"threshold={args.structure_threshold:.3f}, "
        f"min_component_pixels={args.min_component_pixels}"
    )
    print(f"[DONE] target sources: {dict(source_counts)}; skipped={skipped}")
    print(
        f"[DONE] target quality filter: rejected={len(rejected_pairs)}, "
        f"report={rejection_path}, summary={summary_path}"
    )
    print(f"[DONE] rejection reasons: {dict(rejection_summary)}")
    print(f"[DONE] quality failures: {dict(quality_failure_summary)}")
    print(f"[DONE] manually excluded target candidates: {excluded_candidate_count}")
    if audit_manifest:
        print(f"[DONE] audit panels: {dict(audit_counts)}, directory={audit_dir}")
    print(
        "[CHECK] trajectory/target coverage: "
        f"mean={np.mean(alignment_coverages):.4f}, "
        f"median={np.median(alignment_coverages):.4f}, "
        f"min={np.min(alignment_coverages):.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build six-channel spatial trajectory maps for the whole-character U-Net"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--trajectory_csv", default=None)
    parser.add_argument("--output_npz", default="data/processed/character_train.npz")
    parser.add_argument("--character", default=None, help="Optional exact character filter")
    parser.add_argument("--target_character", default=None, help="Character receiving --target_image")
    parser.add_argument("--target_image", default=None, help="Optional external whole-character target")
    parser.add_argument("--chirography", default=None)
    parser.add_argument("--trajectory_width", type=int, default=3)
    parser.add_argument("--trajectory_padding", type=int, default=None)
    parser.add_argument("--synthetic_width", type=int, default=5)
    parser.add_argument("--require_real_target", action="store_true")
    parser.add_argument("--target_script", choices=SCRIPT_MODES, default="traditional")
    parser.add_argument("--min_alignment_coverage", type=float, default=0.55)
    parser.add_argument("--min_support_dice", type=float, default=0.45)
    parser.add_argument("--max_outside_support_fraction", type=float, default=0.35)
    parser.add_argument("--min_target_support_area_ratio", type=float, default=0.30)
    parser.add_argument("--max_target_support_area_ratio", type=float, default=1.70)
    parser.add_argument("--max_target_ink_fraction", type=float, default=0.45)
    parser.add_argument("--max_foreground_bbox_fill_fraction", type=float, default=0.55)
    parser.add_argument("--max_border_ink_fraction", type=float, default=0.02)
    parser.add_argument("--max_target_candidates", type=int, default=64)
    parser.add_argument("--max_registered_candidates", type=int, default=8)
    parser.add_argument(
        "--structure_threshold",
        type=float,
        default=0.35,
        help="Binarization threshold applied after target registration",
    )
    parser.add_argument(
        "--min_component_pixels",
        type=int,
        default=8,
        help="Remove isolated target-mask components smaller than this area",
    )
    parser.add_argument(
        "--exclude_candidates_json",
        default=None,
        help="Optional JSON list of exact source image/bbox candidates to exclude",
    )
    parser.add_argument("--audit_dir", default=None)
    parser.add_argument("--audit_limit_per_status", type=int, default=40)
    main(parser.parse_args())
