# Pluto Drone Propeller Inspection System

A computer-vision project to inspect the propeller assembly of a stationary Pluto 1.2 drone using photographs captured by a mobile phone.

## Objectives

- Detect the four motor positions.
- Detect missing propellers.
- Classify propeller types.
- Check propeller placement against the verified reference configuration.
- Identify visible mounting defects and flag uncertain cases for manual inspection.

## Technology

- Python
- OpenCV
- NumPy
- Matplotlib

## Project Structure

- `src/`: Computer-vision implementation
- `dataset/raw/`: Original photographs; never modified by the pipeline
- `dataset/annotated/`: Annotated images
- `dataset/labels/`: Dataset labels
- `dataset/processed/`: Resized, organized image splits and validated labels
- `dataset/manifests/`: CSV/JSON records for every discovered source image
- `docs/`: Project documentation
- `tests/`: Tests
- `results/`: Generated inspection results

## Dataset preparation

Build the complete clean/preprocessed dataset from the raw photographs:

```bash
./.venv/bin/python -m src.prepare_dataset --clean
```

The pipeline recursively discovers image files, checks that they decode,
removes exact duplicates, preserves quality-gate warnings as `review` records,
resizes images to a maximum side of 1600 pixels, validates optional YOLO
detection labels, and creates deterministic `train`, `val`, and `test` splits.
The raw images are never overwritten. See `dataset/manifests/dataset_manifest.csv`
for the source-to-output mapping, quality metrics, and annotation status.

## Documentation

- [`docs/architecture.md`](docs/architecture.md): system stages, boundaries,
  result states, and safety rules.
- [`docs/data_pipeline.md`](docs/data_pipeline.md): exact cleaning logic,
  thresholds, file layout, manifest contract, and command reference.
- [`docs/flowcharts.md`](docs/flowcharts.md): Mermaid flowcharts for the
  preparation, quality-gate, and future inspection paths.
- [`docs/code_reference.md`](docs/code_reference.md): module and function
  responsibilities.

## Current Status

Dataset preparation is implemented. Drone localization, propeller detection,
classification, and verified reference comparison remain under development.

## Safety

Visual inspection cannot independently verify every mechanical defect or actual motor rotation direction.
