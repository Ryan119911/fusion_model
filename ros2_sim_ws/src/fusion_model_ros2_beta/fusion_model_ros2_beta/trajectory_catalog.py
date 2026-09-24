"""Discover character-specific trajectories produced by brush-model batches.

The batch inversion in the main project is intentionally independent from this
ROS 2 package.  The V16 baseline writes sharded manifests plus one
``char_uXXXX/pose_refined.csv`` per completed character.  V17 writes
``pose_top1.csv`` (or its rank-1 candidate) and a quality-gate summary.  This
module reads both contracts without importing or modifying the CoppeliaSim
driver, so a newly completed trajectory becomes selectable after the next
catalog scan.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


def character_directory_name(character: str) -> str:
    """Return the portable directory name used by the batch trajectory job."""

    text = str(character)
    return "char_" + "_".join(f"u{ord(value):04x}" for value in text)


def character_from_directory_name(directory_name: str) -> Optional[str]:
    """Decode the ``char_uXXXX[_uXXXX]`` name used by V17 outputs."""

    text = str(directory_name)
    if not text.startswith("char_"):
        return None
    tokens = text[len("char_") :].split("_")
    if not tokens or any(
        not token.startswith("u") or len(token) < 2 for token in tokens
    ):
        return None
    try:
        return "".join(chr(int(token[1:], 16)) for token in tokens)
    except (ValueError, OverflowError):
        return None


@dataclass(frozen=True)
class TrajectoryEntry:
    """One database character and the files needed for a ROS 2 replay."""

    character: str
    status: str
    trajectory_csv: Optional[Path] = None
    target_image: Optional[Path] = None
    sample_id: Optional[str] = None
    output_dir: Optional[Path] = None
    point_count: int = 0
    target_is_normalized: bool = False
    message: str = ""
    quality: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return (
            self.status == "ready"
            and self.trajectory_csv is not None
            and self.trajectory_csv.is_file()
            and self.target_image is not None
            and self.target_image.is_file()
            and bool(self.sample_id)
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "character": self.character,
            "status": "ready" if self.ready else self.status,
            "ready": self.ready,
            "sample_id": self.sample_id,
            "point_count": self.point_count,
            "trajectory_csv": str(self.trajectory_csv) if self.trajectory_csv else None,
            "target_image": str(self.target_image) if self.target_image else None,
            "target_is_normalized": self.target_is_normalized,
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "message": self.message,
            "quality": self.quality,
            "source": self.metadata.get("source"),
            "v17_export_eligible": self.metadata.get("v17_export_eligible"),
            "v17_withheld_reasons": self.metadata.get("v17_withheld_reasons", []),
            "v17_iou": self.metadata.get("v17_iou"),
        }


def _resolve_path(root: Path, value: Any) -> Optional[Path]:
    if not value:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = root / path
    return path


def _record_output_dir(root: Path, record: Mapping[str, Any]) -> Optional[Path]:
    output_dir = _resolve_path(root, record.get("output_dir"))
    if output_dir is not None:
        return output_dir
    csv_path = _resolve_path(root, record.get("pose_csv"))
    if csv_path is not None:
        return csv_path.parent
    return None


def _read_csv_characters(path: Path) -> Dict[str, tuple[str, int]]:
    """Index characters in an optional fallback multi-character CSV."""

    if not path.is_file():
        return {}
    result: Dict[str, tuple[str, int]] = {}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                character = str(row.get("character", "")).strip()
                if not character:
                    continue
                sample_id = str(row.get("sample_id", "")).strip()
                _, count = result.get(character, ("", 0))
                result[character] = (sample_id, count + 1)
    except (OSError, csv.Error, UnicodeError):
        return {}
    return result


def _read_pose_csv_metadata(path: Path) -> tuple[Optional[str], Optional[str], int]:
    """Read the character, sample id, and point count from a pose CSV."""

    if not path.is_file():
        return None, None, 0
    character: Optional[str] = None
    sample_id: Optional[str] = None
    count = 0
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row_character = str(row.get("character", "")).strip()
                if row_character and character is None:
                    character = row_character
                row_sample = str(row.get("sample_id", "")).strip()
                if row_sample and sample_id is None:
                    sample_id = row_sample
                count += 1
    except (OSError, csv.Error, UnicodeError):
        return None, None, 0
    return character, sample_id, count


def _manifest_paths(root: Path) -> list[Path]:
    """Return all root and shard manifests, oldest first.

    The V16 full run was produced by independent workers.  Its small root
    manifest is only a partial early run, while the newer
    ``manifest_shard_*`` files contain the complete batch.  Processing by
    modification time lets the latest record for a character win without
    destructively merging files in the model-output directory.
    """

    paths = [path for path in root.glob("manifest*.jsonl") if path.is_file()]

    def modified(path: Path) -> int:
        try:
            return path.stat().st_mtime_ns
        except OSError:
            return 0

    return sorted(paths, key=lambda path: (modified(path), path.name))


def _read_database_characters(path: Path, style: str) -> set[str]:
    """Read the unique character set from the database's text snippets.

    ``data.csv.text`` is a source inscription/snippet, not a single label.
    The target database is therefore the set of non-whitespace characters
    occurring in all rows of the selected calligraphic style.
    """

    if not path.is_file():
        return set()
    result: set[str] = set()
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                text = str(row.get("text", "")).strip()
                chirography = str(row.get("chirography", "")).strip()
                if not text or (style and style not in chirography):
                    continue
                result.update(value for value in text if not value.isspace())
    except (OSError, csv.Error, UnicodeError):
        return set()
    return result


class TrajectoryCatalog:
    """Read the batch manifest and expose only replayable trajectories."""

    def __init__(
        self,
        root: Path | str,
        *,
        fallback_csv: Path | str | None = None,
        fallback_target: Path | str | None = None,
        fallback_character: str | None = None,
        database_csv: Path | str | None = None,
        database_style: str = "楷",
        target_overrides: Optional[Mapping[str, Path | str]] = None,
        allow_v17_ineligible: bool = False,
    ) -> None:
        self.root = Path(root).expanduser()
        self.fallback_csv = (
            Path(fallback_csv).expanduser() if fallback_csv is not None else None
        )
        self.fallback_target = (
            Path(fallback_target).expanduser()
            if fallback_target is not None
            else None
        )
        self.fallback_character = (
            str(fallback_character).strip() if fallback_character else None
        )
        self.database_csv = (
            Path(database_csv).expanduser() if database_csv is not None else None
        )
        self.database_style = str(database_style)
        self.target_overrides = {
            str(key): Path(value).expanduser()
            for key, value in (target_overrides or {}).items()
        }
        self.allow_v17_ineligible = bool(allow_v17_ineligible)

    def _target_for(
        self,
        character: str,
        record_target: Optional[Path],
        output_dir: Optional[Path],
    ) -> tuple[Optional[Path], bool]:
        override = self.target_overrides.get(character)
        if override is not None:
            return override, False
        if record_target is not None:
            # Batch targets are saved as the already-normalized 128x128 target.
            return record_target, record_target.name == "target.png"
        if (
            self.fallback_target is not None
            and (
                self.fallback_character is None
                or character == self.fallback_character
            )
        ):
            return self.fallback_target, False
        if output_dir is not None:
            candidate = output_dir / "target.png"
            if candidate.is_file():
                return candidate, True
        return None, False

    def _entry_from_record(self, record: Mapping[str, Any]) -> Optional[TrajectoryEntry]:
        character = str(record.get("character", "")).strip()
        if not character:
            return None
        output_dir = _record_output_dir(self.root, record)
        trajectory = _resolve_path(self.root, record.get("pose_csv"))
        if trajectory is None and output_dir is not None:
            trajectory = output_dir / "pose_refined.csv"
        record_target = _resolve_path(self.root, record.get("target_image"))
        target, target_is_normalized = self._target_for(
            character, record_target, output_dir
        )
        status = str(record.get("status", "unavailable"))
        if trajectory is not None and trajectory.is_file() and target is not None and target.is_file():
            status = "ready"
        elif status == "completed":
            status = "generating"
        sample_value = record.get("trajectory_sample_id")
        sample_id = (
            str(sample_value).strip()
            if sample_value is not None and str(sample_value).strip()
            else None
        )
        return TrajectoryEntry(
            character=character,
            status=status,
            trajectory_csv=trajectory,
            target_image=target,
            sample_id=sample_id,
            output_dir=output_dir,
            point_count=int(record.get("trajectory_point_count", 0) or 0),
            target_is_normalized=target_is_normalized,
            message=str(record.get("error", "") or ""),
            quality=(str(record.get("quality", "")).strip() or None),
            metadata=dict(record),
        )

    def _entry_from_v17_directory(
        self, output_dir: Path
    ) -> Optional[TrajectoryEntry]:
        """Read one completed V17 ``pose_top1.csv`` directory.

        V17 deliberately has no root ``manifest.jsonl`` because its four
        shards write independently.  The per-character directory and its
        candidate summary are therefore the atomic catalog contract.
        """

        character = character_from_directory_name(output_dir.name)
        trajectory = output_dir / "pose_top1.csv"
        trajectory_source = "v17_pose_top1"
        if character is None:
            return None

        summary_path = output_dir / "v17" / "candidate_summary.json"
        if not summary_path.is_file():
            summary_path = output_dir / "candidate_summary.json"
        summary: Dict[str, Any] = {}
        if summary_path.is_file():
            try:
                loaded = json.loads(summary_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    summary = loaded
            except (OSError, UnicodeError, json.JSONDecodeError):
                summary = {}

        candidate: Mapping[str, Any] = {}
        candidates = summary.get("candidates", [])
        if isinstance(candidates, list) and candidates:
            ranked = [item for item in candidates if isinstance(item, dict)]
            candidate = next(
                (item for item in ranked if int(item.get("rank", 0) or 0) == 1),
                ranked[0] if ranked else {},
            )
        if not trajectory.is_file():
            # One V17 shard can finish the three candidate optimisations and
            # write candidate_summary.json before the convenience copy to
            # pose_top1.csv.  The rank-1 candidate is still a valid diagnostic
            # source, so expose it with the same quality gate instead of
            # silently losing the character from the catalog.
            candidate_values = []
            estimate_csv = candidate.get("estimate_csv")
            if estimate_csv:
                candidate_values.append(_resolve_path(output_dir, estimate_csv))
            label = str(candidate.get("label", "")).strip()
            if label:
                candidate_values.append(
                    output_dir / "v17" / label / "candidate_trajectory.csv"
                )
            candidate_path = next(
                (path for path in candidate_values if path is not None and path.is_file()),
                None,
            )
            if candidate_path is None:
                return None
            trajectory = candidate_path
            trajectory_source = "v17_rank1_candidate"

        csv_character, sample_id, point_count = _read_pose_csv_metadata(trajectory)
        if csv_character and csv_character != character:
            return TrajectoryEntry(
                character=character,
                status="unavailable",
                trajectory_csv=trajectory,
                output_dir=output_dir,
                point_count=point_count,
                message=(
                    "V17 directory/CSV character mismatch: "
                    f"directory={character!r}, csv={csv_character!r}"
                ),
                quality="invalid",
                metadata={"source": trajectory_source},
            )
        eligible_value = candidate.get("export_eligible")
        eligible = bool(eligible_value) if eligible_value is not None else None
        reasons_value = candidate.get("withheld_reasons", [])
        reasons = [str(value) for value in reasons_value] if isinstance(
            reasons_value, list
        ) else ([str(reasons_value)] if reasons_value else [])
        iou = (candidate.get("image_metrics", {}) or {}).get("iou_at_0.5")
        target, target_is_normalized = self._target_for(
            character, output_dir / "target.png", output_dir
        )
        files_ready = (
            trajectory.is_file()
            and target is not None
            and target.is_file()
            and bool(sample_id)
        )
        if eligible is True:
            status = "ready" if files_ready else "generating"
            quality = "v17_eligible"
            message = "V17 rank-1 trajectory passed its export gate."
        elif eligible is False:
            status = (
                "ready"
                if files_ready and self.allow_v17_ineligible
                else "low_quality"
                if files_ready
                else "generating"
            )
            quality = "v17_ineligible"
            message = "V17 rank-1 withheld: " + ", ".join(reasons or ["quality_gate"])
            if self.allow_v17_ineligible and files_ready:
                message += " (loaded because allow_v17_ineligible=true)"
        else:
            status = "unverified" if files_ready else "generating"
            quality = "v17_unverified"
            message = "V17 pose_top1 exists but candidate_summary.json has no rank-1 gate."

        metadata: Dict[str, Any] = {
            "source": trajectory_source,
            "gamma_relative_to_path": True,
            "v17_summary_path": str(summary_path) if summary_path.is_file() else None,
            "v17_export_eligible": eligible,
            "v17_withheld_reasons": reasons,
            "v17_iou": iou,
            "v17_candidate_rank": candidate.get("rank"),
            "v17_candidate_score": (candidate.get("score", {}) or {}).get("total"),
        }
        return TrajectoryEntry(
            character=character,
            status=status,
            trajectory_csv=trajectory,
            target_image=target,
            sample_id=sample_id or f"{character}_v17",
            output_dir=output_dir,
            point_count=point_count,
            target_is_normalized=target_is_normalized,
            message=message,
            quality=quality,
            metadata=metadata,
        )

    def scan(self) -> Dict[str, TrajectoryEntry]:
        """Scan the current manifest and output folders.

        The scan is deliberately cheap and stateless.  The web page calls it
        for every lookup, which means the running batch generator can add new
        characters without restarting the ROS 2 process.
        """

        entries: Dict[str, TrajectoryEntry] = {}
        for manifest in _manifest_paths(self.root):
            try:
                with manifest.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        entry = self._entry_from_record(record)
                        if entry is not None:
                            entries[entry.character] = entry
            except (OSError, UnicodeError):
                pass

        if self.root.is_dir():
            output_characters = {
                entry.output_dir: entry.character
                for entry in entries.values()
                if entry.output_dir is not None
            }
            for output_dir in self.root.glob("char_*"):
                has_v17_contract = (
                    (output_dir / "pose_top1.csv").is_file()
                    or (output_dir / "v17" / "candidate_summary.json").is_file()
                    or (output_dir / "candidate_summary.json").is_file()
                )
                if has_v17_contract:
                    v17_entry = self._entry_from_v17_directory(output_dir)
                    if v17_entry is not None:
                        entries[v17_entry.character] = v17_entry
                        continue
                trajectory = output_dir / "pose_refined.csv"
                if not trajectory.is_file():
                    continue
                record_target = output_dir / "target.png"
                character = output_characters.get(output_dir, "")
                if not character:
                    # Unknown folders are not made selectable because the
                    # manifest is the authoritative character-to-target map.
                    continue
                target, target_is_normalized = self._target_for(
                    character, record_target, output_dir
                )
                previous = entries.get(character)
                metadata = dict(previous.metadata) if previous else {}
                metadata.setdefault("source", "v16_pose_refined")
                metadata.setdefault("gamma_relative_to_path", True)
                entries[character] = TrajectoryEntry(
                    character=character,
                    status="ready" if target is not None and target.is_file() else "generating",
                    trajectory_csv=trajectory,
                    target_image=target,
                    sample_id=previous.sample_id if previous else f"{character}_fake_sim",
                    output_dir=output_dir,
                    point_count=previous.point_count if previous else 0,
                    target_is_normalized=target_is_normalized,
                    message=previous.message if previous else "",
                    quality=(
                        previous.quality
                        if previous and previous.quality
                        else "v16_batch"
                    ),
                    metadata=metadata,
                )

        if self.fallback_csv is not None:
            fallback_characters = _read_csv_characters(self.fallback_csv)
            for character, (sample_id, count) in fallback_characters.items():
                if character in entries and entries[character].ready:
                    continue
                target, target_is_normalized = self._target_for(
                    character, None, self.fallback_csv.parent
                )
                entries[character] = TrajectoryEntry(
                    character=character,
                    status=(
                        "ready"
                        if target is not None and target.is_file()
                        else "unavailable"
                    ),
                    trajectory_csv=self.fallback_csv,
                    target_image=target,
                    sample_id=sample_id or f"{character}_fake_sim",
                    output_dir=self.fallback_csv.parent,
                    point_count=count,
                    target_is_normalized=target_is_normalized,
                    message="fallback trajectory CSV (used because V17 is not export-eligible)",
                    quality="fallback",
                    metadata={"source": "v16_fallback", "gamma_relative_to_path": True},
                )

        if self.database_csv is not None:
            for character in _read_database_characters(
                self.database_csv, self.database_style
            ):
                if character in entries:
                    continue
                entries[character] = TrajectoryEntry(
                    character=character,
                    status="pending",
                    message="database character is waiting for batch trajectory generation",
                )
        return entries

    def resolve(self, character: str) -> Optional[TrajectoryEntry]:
        """Return a ready entry, or an unavailable entry with its reason."""

        text = str(character).strip()
        if not text:
            return None
        return self.scan().get(text)

    def summary(self, query: str = "", limit: int = 50) -> Dict[str, Any]:
        entries = self.scan()
        query = str(query).strip()
        selected: Iterable[TrajectoryEntry] = entries.values()
        if query:
            selected = (
                entry for entry in selected if query in entry.character
            )
        selected = sorted(selected, key=lambda entry: entry.character)
        selected_list = list(selected)[: max(1, int(limit))]
        return {
            "root": str(self.root),
            "ready_count": sum(entry.ready for entry in entries.values()),
            "generating_count": sum(
                entry.status in {"generating", "pending"}
                for entry in entries.values()
            ),
            "unavailable_count": sum(
                not entry.ready and entry.status not in {"generating", "pending"}
                for entry in entries.values()
            ),
            "v17_low_quality_count": sum(
                entry.quality == "v17_ineligible" for entry in entries.values()
            ),
            "total_known": len(entries),
            "query": query,
            "entries": [entry.as_dict() for entry in selected_list],
        }
