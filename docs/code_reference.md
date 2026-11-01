# Code Reference

## `src/prepare_dataset.py`

This is the command-line entry point and orchestration layer for data
preparation.

### Configuration

`PipelineConfig` is an immutable configuration object. It validates the
maximum image side, JPEG quality, and split ratios before processing begins.

### Public functions

| Function | Responsibility |
|---|---|
| `discover_images(raw_dir)` | Recursively find supported files in stable order. |
| `sha256_file(path)` | Hash file bytes in chunks. |
| `run_pipeline(config, clean=False)` | Execute the complete preparation process and return a summary. |
| `build_parser()` | Build the CLI argument parser. |
| `main(argv=None)` | Validate CLI arguments, run the pipeline, and print JSON summary. |

### Internal functions

| Function | Responsibility |
|---|---|
| `_safe_record_id` / `_unique_record_id` | Create stable filesystem-safe record IDs. |
| `_quality_record` | Convert the quality-gate result into manifest values. |
| `_preprocess_image` | Resize while preserving aspect ratio. |
| `_find_label` | Locate a label matching an image stem. |
| `_validate_yolo_label` | Check YOLO row structure and normalized values. |
| `_split_for_path` | Assign a deterministic split from seed and source path. |
| `_write_manifests` | Write CSV and JSON per-image records. |
| `_summary` | Aggregate status, split, annotation, and error counts. |

## `src/quality_gate.py`

This module provides the preliminary technical-quality checks used by the
preparation pipeline and future inference code.

| Symbol | Responsibility |
|---|---|
| `MIN_IMAGE_WIDTH`, `MIN_IMAGE_HEIGHT` | Minimum source dimensions. |
| `MIN_LAPLACIAN_VARIANCE` | Blur/detail threshold. |
| `MIN_MEAN_BRIGHTNESS`, `MAX_MEAN_BRIGHTNESS` | Overall brightness bounds. |
| `MAX_DARK_PIXEL_FRACTION` | Maximum nearly-black pixel share. |
| `MAX_BRIGHT_PIXEL_FRACTION` | Maximum nearly-white pixel share. |
| `check_image_quality(image)` | Measure an array and return status, reasons, and metrics. |
| `check_image_file(image_path)` | Load an image and call `check_image_quality`. |

The quality gate returns `READY` or `UNABLE_TO_VERIFY`. The preparation layer
uses the lower-case manifest values `ready` and `review` so a warning can be
tracked without discarding a decodable source image.

## `tests/test_prepare_dataset.py`

The tests cover the meaningful pipeline contracts:

- processing creates manifests and downsizes images;
- identical inputs produce identical split assignments and manifests;
- exact duplicates are excluded from the prepared image set; and
- malformed labels are preserved in `invalid_labels` and marked invalid.

They use Python's standard `unittest` runner, so no extra test dependency is
required:

```bash
./.venv/bin/python -m unittest discover -s tests -v
```

## Data ownership and mutation rules

The source directories are read-only inputs from the pipeline's perspective:

- `dataset/raw`: original photographs;
- `dataset/annotated`: human annotation workspace;
- `dataset/labels`: supplied labels.

Generated data belongs in `dataset/processed` and `dataset/manifests`.
`--clean` is intended for those generated directories only.
