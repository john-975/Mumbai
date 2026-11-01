"""Build a clean, reproducible dataset from the raw image directory.

The pipeline deliberately keeps ``dataset/raw`` immutable.  It performs the
following steps for every supported image:

* discovers images recursively;
* verifies that OpenCV can decode them;
* removes exact duplicate files from the prepared set;
* records the existing image-quality gate result without throwing away
  images that need manual review;
* resizes large images while preserving their aspect ratio;
* copies valid YOLO detection labels when they exist; and
* writes deterministic train/validation/test splits and CSV/JSON manifests.

This module uses only the dependencies already required by the project.  Run
it from the repository root with::

    python -m src.prepare_dataset --clean

The command is intentionally conservative: a quality warning is represented
as ``review`` in the manifest.  It is not treated as evidence that an image
is unusable for annotation or exploration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cv2

try:  # Works both as ``python -m src.prepare_dataset`` and direct execution.
    from .quality_gate import check_image_quality
except ImportError:  # pragma: no cover - exercised by direct CLI use.
    from quality_gate import check_image_quality


SUPPORTED_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
)
SPLITS = ("train", "val", "test")
MANIFEST_FIELDS = (
    "record_id",
    "source_path",
    "source_sha256",
    "source_bytes",
    "output_path",
    "output_sha256",
    "split",
    "processing_status",
    "quality_status",
    "quality_reasons",
    "annotation_status",
    "annotation_path",
    "width",
    "height",
    "processed_width",
    "processed_height",
    "laplacian_variance",
    "mean_brightness",
    "dark_pixel_fraction",
    "bright_pixel_fraction",
    "error",
)


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for one dataset preparation run."""

    raw_dir: Path
    output_dir: Path
    manifest_dir: Path
    labels_dir: Optional[Path] = None
    max_side: int = 1600
    jpeg_quality: int = 95
    seed: int = 42
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    def __post_init__(self) -> None:
        if self.max_side <= 0:
            raise ValueError("max_side must be greater than zero")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        ratios = (self.train_ratio, self.val_ratio, self.test_ratio)
        if any(r < 0 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
            raise ValueError("train, val and test ratios must be non-negative and sum to 1")


def discover_images(raw_dir: Path) -> List[Path]:
    """Return supported image files in deterministic relative-path order."""

    if not raw_dir.exists():
        raise FileNotFoundError("Raw dataset directory does not exist: %s" % raw_dir)
    if not raw_dir.is_dir():
        raise NotADirectoryError("Raw dataset path is not a directory: %s" % raw_dir)

    return sorted(
        (
            path
            for path in raw_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ),
        key=lambda path: path.relative_to(raw_dir).as_posix().lower(),
    )


def sha256_file(path: Path) -> str:
    """Hash a file in chunks so large photographs do not fill memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_record_id(relative_path: Path) -> str:
    """Convert a relative path into a stable, filesystem-safe record id."""

    value = relative_path.with_suffix("").as_posix()
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return value or "image"


def _unique_record_id(relative_path: Path, used: Set[str]) -> str:
    record_id = _safe_record_id(relative_path)
    if record_id in used:
        suffix = hashlib.sha1(relative_path.as_posix().encode("utf-8")).hexdigest()[:8]
        record_id = "%s_%s" % (record_id, suffix)
    used.add(record_id)
    return record_id


def _json_safe(value):
    """Convert OpenCV/numpy scalar values to JSON-compatible primitives."""

    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _quality_record(quality_result: dict) -> Tuple[str, List[str], Dict[str, object]]:
    metrics = {
        key: _json_safe(value)
        for key, value in quality_result.get("metrics", {}).items()
    }
    reasons = [str(reason) for reason in quality_result.get("reasons", [])]
    # The quality gate is a safety signal.  Its provisional thresholds should
    # not destroy source data, so all decoded images remain eligible for the
    # prepared set and warnings become review metadata.
    status = "ready" if quality_result.get("status") == "READY" else "review"
    return status, reasons, metrics


def _preprocess_image(image, max_side: int):
    """Resize an image to a maximum side while preserving all content."""

    height, width = image.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_side:
        return image

    scale = float(max_side) / float(longest_side)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def _find_label(label_root: Optional[Path], source_path: Path) -> Optional[Path]:
    if label_root is None or not label_root.exists():
        return None
    candidates = sorted(label_root.rglob(source_path.stem + ".txt"))
    return candidates[0] if candidates else None


def _validate_yolo_label(label_path: Path) -> Tuple[str, str]:
    """Validate a YOLO detection label without changing its contents."""

    try:
        text = label_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return "invalid", "Could not read label: %s" % exc

    for line_number, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        values = line.split()
        if len(values) != 5:
            return "invalid", "Line %d must contain class and four box values" % line_number
        try:
            class_id = int(values[0])
            box = [float(value) for value in values[1:]]
        except ValueError:
            return "invalid", "Line %d contains non-numeric values" % line_number
        if class_id < 0 or any(value < 0.0 or value > 1.0 for value in box):
            return "invalid", "Line %d has values outside YOLO's normalized range" % line_number
        if box[2] == 0.0 or box[3] == 0.0:
            return "invalid", "Line %d has a zero-sized bounding box" % line_number
    return "valid", ""


def _split_for_path(relative_path: Path, config: PipelineConfig) -> str:
    """Assign a path to a deterministic split without global random state."""

    key = "%d:%s" % (config.seed, relative_path.as_posix())
    bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) / float(2**32)
    if bucket < config.train_ratio:
        return "train"
    if bucket < config.train_ratio + config.val_ratio:
        return "val"
    return "test"


def _empty_record(record_id: str, source_path: Path, config: PipelineConfig) -> dict:
    return {
        "record_id": record_id,
        "source_path": source_path.relative_to(config.raw_dir).as_posix(),
        "source_sha256": "",
        "source_bytes": source_path.stat().st_size if source_path.exists() else 0,
        "output_path": "",
        "output_sha256": "",
        "split": "",
        "processing_status": "rejected",
        "quality_status": "",
        "quality_reasons": "",
        "annotation_status": "missing",
        "annotation_path": "",
        "width": "",
        "height": "",
        "processed_width": "",
        "processed_height": "",
        "laplacian_variance": "",
        "mean_brightness": "",
        "dark_pixel_fraction": "",
        "bright_pixel_fraction": "",
        "error": "",
    }


def _write_image(path: Path, image, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = cv2.imwrite(
        str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    if not written:
        raise OSError("OpenCV could not write image: %s" % path)


def _clean_generated_output(output_dir: Path, manifest_dir: Path) -> None:
    """Remove only directories owned by this pipeline."""

    generated_markers = {
        output_dir: output_dir / ".pipeline-generated",
        manifest_dir: manifest_dir / ".pipeline-generated",
    }
    for directory, marker in generated_markers.items():
        if not directory.exists():
            continue
        if not marker.exists():
            raise OSError(
                "Refusing to remove a directory without a pipeline marker: %s. "
                "Remove it manually or run without --clean." % directory
            )
        shutil.rmtree(directory)


def _make_output_dirs(config: PipelineConfig) -> None:
    for split in SPLITS:
        (config.output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (config.output_dir / split / "labels").mkdir(parents=True, exist_ok=True)
    (config.output_dir / "invalid_labels").mkdir(parents=True, exist_ok=True)
    config.manifest_dir.mkdir(parents=True, exist_ok=True)


def _write_manifests(records: Sequence[dict], config: PipelineConfig) -> None:
    config.manifest_dir.mkdir(parents=True, exist_ok=True)
    csv_path = config.manifest_dir / "dataset_manifest.csv"
    json_path = config.manifest_dir / "dataset_manifest.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    json_path.write_text(json.dumps(list(records), indent=2) + "\n", encoding="utf-8")


def _summary(records: Sequence[dict], config: PipelineConfig) -> dict:
    included = [record for record in records if record["processing_status"] == "included"]
    rejected = [record for record in records if record["processing_status"] != "included"]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "raw_dir": config.raw_dir.as_posix(),
        "output_dir": config.output_dir.as_posix(),
        "manifest_dir": config.manifest_dir.as_posix(),
        "config": {
            "max_side": config.max_side,
            "jpeg_quality": config.jpeg_quality,
            "seed": config.seed,
            "train_ratio": config.train_ratio,
            "val_ratio": config.val_ratio,
            "test_ratio": config.test_ratio,
        },
        "source_count": len(records),
        "included_count": len(included),
        "rejected_count": len(rejected),
        "processing_status_counts": dict(Counter(record["processing_status"] for record in records)),
        "quality_status_counts": dict(Counter(record["quality_status"] for record in included)),
        "annotation_status_counts": dict(Counter(record["annotation_status"] for record in included)),
        "split_counts": dict(Counter(record["split"] for record in included)),
        "rejection_reasons": dict(Counter(record["error"] for record in rejected)),
    }


def run_pipeline(config: PipelineConfig, clean: bool = False) -> dict:
    """Prepare the dataset and return the generated summary."""

    if clean:
        _clean_generated_output(config.output_dir, config.manifest_dir)
    _make_output_dirs(config)

    files = discover_images(config.raw_dir)
    records: List[dict] = []
    seen_hashes: Dict[str, str] = {}
    used_ids: Set[str] = set()

    for source_path in files:
        relative_path = source_path.relative_to(config.raw_dir)
        record_id = _unique_record_id(relative_path, used_ids)
        record = _empty_record(record_id, source_path, config)

        try:
            source_hash = sha256_file(source_path)
            record["source_sha256"] = source_hash
            if source_hash in seen_hashes:
                record["error"] = "Exact duplicate of %s" % seen_hashes[source_hash]
                records.append(record)
                continue
            seen_hashes[source_hash] = relative_path.as_posix()

            image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                record["error"] = "Image could not be decoded by OpenCV"
                records.append(record)
                continue

            quality_result = check_image_quality(image)
            quality_status, quality_reasons, metrics = _quality_record(quality_result)
            record["quality_status"] = quality_status
            record["quality_reasons"] = "; ".join(quality_reasons)
            record["width"] = metrics.get("width", image.shape[1])
            record["height"] = metrics.get("height", image.shape[0])
            for metric_name in (
                "laplacian_variance",
                "mean_brightness",
                "dark_pixel_fraction",
                "bright_pixel_fraction",
            ):
                record[metric_name] = metrics.get(metric_name, "")

            split = _split_for_path(relative_path, config)
            processed = _preprocess_image(image, config.max_side)
            output_path = config.output_dir / split / "images" / (record_id + ".jpg")
            _write_image(output_path, processed, config.jpeg_quality)
            record["output_path"] = output_path.relative_to(config.output_dir.parent).as_posix()
            record["output_sha256"] = sha256_file(output_path)
            record["processed_width"] = int(processed.shape[1])
            record["processed_height"] = int(processed.shape[0])
            record["split"] = split
            record["processing_status"] = "included"

            label_path = _find_label(config.labels_dir, source_path)
            if label_path is not None:
                annotation_status, annotation_error = _validate_yolo_label(label_path)
                record["annotation_status"] = annotation_status
                if annotation_status == "valid":
                    destination = config.output_dir / split / "labels" / (record_id + ".txt")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(label_path, destination)
                    record["annotation_path"] = destination.relative_to(
                        config.output_dir.parent
                    ).as_posix()
                else:
                    invalid_destination = config.output_dir / "invalid_labels" / (record_id + ".txt")
                    shutil.copy2(label_path, invalid_destination)
                    record["error"] = annotation_error
                    record["annotation_path"] = invalid_destination.relative_to(
                        config.output_dir.parent
                    ).as_posix()
            records.append(record)
        except (OSError, ValueError, cv2.error) as exc:
            record["error"] = str(exc)
            records.append(record)

    _write_manifests(records, config)
    summary = _summary(records, config)
    (config.manifest_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (config.manifest_dir / ".pipeline-generated").write_text(
        "Generated by src.prepare_dataset. Do not place source data here.\n",
        encoding="utf-8",
    )
    (config.output_dir / "README.md").write_text(_output_readme(config, summary), encoding="utf-8")
    (config.output_dir / ".pipeline-generated").write_text(
        "Generated by src.prepare_dataset. Do not place source data here.\n",
        encoding="utf-8",
    )
    return summary


def _output_readme(config: PipelineConfig, summary: dict) -> str:
    return """# Prepared dataset

This directory was generated by `python -m src.prepare_dataset`.

## Layout

- `train/images`, `val/images`, `test/images`: deterministic image splits.
- `train/labels`, `val/labels`, `test/labels`: valid YOLO detection labels,
  when supplied in `dataset/labels`.
- `invalid_labels`: labels that failed validation and were not used.
- `../manifests/dataset_manifest.csv`: one row per discovered source image.

The raw images are not modified. Images that decode successfully are included
even when the provisional quality gate reports a warning. Use the manifest's
`quality_status` and `quality_reasons` fields to select or review images
before training. `ready` means the current preliminary gate passed; it does
not mean that the drone is correctly assembled.

## Current run

- Source images: {source_count}
- Included images: {included_count}
- Rejected/duplicate images: {rejected_count}
- Quality statuses: {quality_status_counts}
- Splits: {split_counts}
- Annotations: {annotation_status_counts}
""".format(**summary)


def _default_dataset_root() -> Path:
    """Support both the repository's ``dataset`` name and ``datasets``."""

    for candidate in (Path("dataset"), Path("datasets")):
        if candidate.exists():
            return candidate
    return Path("dataset")


def build_parser() -> argparse.ArgumentParser:
    dataset_root = _default_dataset_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=dataset_root / "raw")
    parser.add_argument("--output-dir", type=Path, default=dataset_root / "processed")
    parser.add_argument("--manifest-dir", type=Path, default=dataset_root / "manifests")
    parser.add_argument("--labels-dir", type=Path, default=dataset_root / "labels")
    parser.add_argument("--max-side", type=int, default=1600)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing generated output and manifests before processing.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = PipelineConfig(
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            manifest_dir=args.manifest_dir,
            labels_dir=args.labels_dir,
            max_side=args.max_side,
            jpeg_quality=args.jpeg_quality,
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )
        summary = run_pipeline(config, clean=args.clean)
    except (FileNotFoundError, NotADirectoryError, ValueError, OSError) as exc:
        parser.error(str(exc))
        return 2

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point.
    sys.exit(main())
