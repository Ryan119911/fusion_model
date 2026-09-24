import json
import os
import tempfile
import unittest
from pathlib import Path

from fusion_model_ros2_beta.trajectory_catalog import (
    TrajectoryCatalog,
    character_directory_name,
    character_from_directory_name,
)


@unittest.skipIf(
    os.name == "nt",
    "The managed Windows runner denies temporary test directories; run on ROS 2 Linux.",
)
class TrajectoryCatalogTest(unittest.TestCase):
    def temporary_directory(self):
        # The managed Windows test runner may deny the system TEMP directory.
        if os.name == "nt":
            return tempfile.TemporaryDirectory(
                dir=Path(__file__).resolve().parents[4] / "tmp"
            )
        return tempfile.TemporaryDirectory()

    def test_completed_manifest_entry_is_ready(self):
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            output = root / character_directory_name("一")
            output.mkdir()
            (output / "pose_refined.csv").write_text("character,sample_id\n一,一_fake_sim\n", encoding="utf-8")
            (output / "target.png").write_bytes(b"target")
            record = {
                "character": "一",
                "status": "completed",
                "output_dir": str(output),
                "pose_csv": str(output / "pose_refined.csv"),
                "target_image": str(output / "target.png"),
                "trajectory_sample_id": "一_fake_sim",
                "trajectory_point_count": 1,
            }
            (root / "manifest.jsonl").write_text(
                json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8"
            )

            entry = TrajectoryCatalog(root).resolve("一")

            self.assertIsNotNone(entry)
            assert entry is not None
            self.assertTrue(entry.ready)
            self.assertTrue(entry.target_is_normalized)
            self.assertEqual(entry.sample_id, "一_fake_sim")

    def test_failed_entry_is_not_selectable(self):
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            record = {
                "character": "二",
                "status": "missing_trajectory",
                "error": "no trajectory sample",
            }
            (root / "manifest.jsonl").write_text(
                json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8"
            )

            entry = TrajectoryCatalog(root).resolve("二")

            self.assertIsNotNone(entry)
            assert entry is not None
            self.assertFalse(entry.ready)
            self.assertEqual(entry.status, "missing_trajectory")

    def test_fallback_target_is_limited_to_fallback_character(self):
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            csv_path = root / "fallback.csv"
            target = root / "wu.png"
            target.write_bytes(b"target")
            csv_path.write_text(
                "character,sample_id\n武,武_fake_sim\n一,一_fake_sim\n",
                encoding="utf-8",
            )
            catalog = TrajectoryCatalog(
                root / "batch",
                fallback_csv=csv_path,
                fallback_target=target,
                fallback_character="武",
            )

            wu = catalog.resolve("武")
            one = catalog.resolve("一")

            self.assertIsNotNone(wu)
            self.assertTrue(wu.ready if wu is not None else False)
            self.assertIsNotNone(one)
            self.assertFalse(one.ready if one is not None else True)

    def test_database_character_waits_for_batch_output(self):
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            database = root / "data.csv"
            database.write_text(
                "img_path,text,author,chirography,location\n"
                "sample.png,三 二,author,楷书,train\n",
                encoding="utf-8",
            )

            entry = TrajectoryCatalog(
                root / "batch", database_csv=database, database_style="楷"
            ).resolve("三")

            self.assertIsNotNone(entry)
            assert entry is not None
            self.assertEqual(entry.status, "pending")
            self.assertFalse(entry.ready)
            self.assertEqual(
                TrajectoryCatalog(
                    root / "batch", database_csv=database, database_style="楷"
                ).summary()["total_known"],
                2,
            )

    def test_v17_pose_top1_is_quality_gated_and_can_be_explicitly_overridden(self):
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            output = root / character_directory_name("一")
            (output / "v17").mkdir(parents=True)
            (output / "pose_top1.csv").write_text(
                "character,sample_id\n一,一_v17\n", encoding="utf-8"
            )
            (output / "target.png").write_bytes(b"target")
            (output / "v17" / "candidate_summary.json").write_text(
                json.dumps(
                    {
                        "format": "paper_pose_multisolution_v17",
                        "candidates": [
                            {
                                "rank": 1,
                                "export_eligible": False,
                                "withheld_reasons": ["image_iou_below_threshold"],
                                "image_metrics": {"iou_at_0.5": 0.69},
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            entry = TrajectoryCatalog(root).resolve("一")
            self.assertIsNotNone(entry)
            assert entry is not None
            self.assertEqual(entry.status, "low_quality")
            self.assertFalse(entry.ready)
            self.assertEqual(entry.quality, "v17_ineligible")
            self.assertEqual(entry.metadata["gamma_relative_to_path"], True)

            overridden = TrajectoryCatalog(
                root, allow_v17_ineligible=True
            ).resolve("一")
            self.assertIsNotNone(overridden)
            assert overridden is not None
            self.assertTrue(overridden.ready)

    def test_v17_directory_name_round_trip(self):
        self.assertEqual(
            character_from_directory_name(character_directory_name("武一")), "武一"
        )

    def test_v17_rank_one_candidate_is_used_when_pose_top1_copy_is_missing(self):
        with self.temporary_directory() as temporary:
            root = Path(temporary)
            output = root / character_directory_name("下")
            candidate_dir = output / "v17" / "m1"
            candidate_dir.mkdir(parents=True)
            candidate_csv = candidate_dir / "candidate_trajectory.csv"
            candidate_csv.write_text(
                "character,sample_id\n下,下_v17\n", encoding="utf-8"
            )
            (output / "target.png").write_bytes(b"target")
            (output / "v17" / "candidate_summary.json").write_text(
                json.dumps(
                    {
                        "candidates": [
                            {
                                "rank": 1,
                                "label": "m1",
                                "estimate_csv": str(candidate_csv),
                                "export_eligible": False,
                                "withheld_reasons": ["image_iou_below_threshold"],
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            entry = TrajectoryCatalog(root).resolve("下")
            self.assertIsNotNone(entry)
            assert entry is not None
            self.assertEqual(entry.metadata["source"], "v17_rank1_candidate")
            self.assertEqual(entry.sample_id, "下_v17")
            self.assertFalse(entry.ready)


if __name__ == "__main__":
    unittest.main()
