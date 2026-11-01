import csv
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.prepare_dataset import PipelineConfig, run_pipeline


def _write_image(path: Path, value: int = 100) -> None:
    image = np.full((32, 48, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


class PrepareDatasetTests(unittest.TestCase):
    def test_pipeline_is_deterministic_and_writes_manifests(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            raw = tmp_path / "raw"
            raw.mkdir()
            _write_image(raw / "first.jpg", 80)
            _write_image(raw / "second.jpg", 160)
            (raw / "not-an-image.txt").write_text("ignored", encoding="utf-8")

            config = PipelineConfig(
                raw_dir=raw,
                output_dir=tmp_path / "processed",
                manifest_dir=tmp_path / "manifests",
                labels_dir=tmp_path / "labels",
                max_side=16,
            )
            first_summary = run_pipeline(config)
            first_manifest = (config.manifest_dir / "dataset_manifest.json").read_text()

            second_summary = run_pipeline(config, clean=True)
            second_manifest = (config.manifest_dir / "dataset_manifest.json").read_text()

            self.assertEqual(first_summary["source_count"], 2)
            self.assertEqual(first_summary["included_count"], 2)
            self.assertEqual(second_summary["split_counts"], first_summary["split_counts"])
            self.assertEqual(second_manifest, first_manifest)
            self.assertTrue((config.manifest_dir / "dataset_manifest.csv").exists())

            records = json.loads(second_manifest)
            for record in records:
                output = tmp_path / record["output_path"]
                image = cv2.imread(str(output))
                self.assertIsNotNone(image)
                self.assertLessEqual(max(image.shape[:2]), 16)

    def test_exact_duplicates_are_excluded_and_invalid_labels_are_kept_for_review(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            raw = tmp_path / "raw"
            labels = tmp_path / "labels"
            raw.mkdir()
            labels.mkdir()
            _write_image(raw / "a_original.jpg")
            (raw / "z_duplicate.jpg").write_bytes((raw / "a_original.jpg").read_bytes())
            (labels / "a_original.txt").write_text("bad label\n", encoding="utf-8")

            config = PipelineConfig(
                raw_dir=raw,
                output_dir=tmp_path / "processed",
                manifest_dir=tmp_path / "manifests",
                labels_dir=labels,
            )
            summary = run_pipeline(config)

            self.assertEqual(summary["included_count"], 1)
            self.assertEqual(summary["rejected_count"], 1)
            with (config.manifest_dir / "dataset_manifest.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(
                {row["annotation_status"] for row in rows if row["processing_status"] == "included"},
                {"invalid"},
            )
            self.assertTrue((config.output_dir / "invalid_labels" / "a_original.txt").exists())


if __name__ == "__main__":
    unittest.main()
