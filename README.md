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
- `dataset/raw/`: Original photographs
- `dataset/annotated/`: Annotated images
- `dataset/labels/`: Dataset labels
- `docs/`: Project documentation
- `tests/`: Tests
- `results/`: Generated inspection results

## Current Status

Initial project setup. The inspection algorithm and dataset are under development.

## Safety

Visual inspection cannot independently verify every mechanical defect or actual motor rotation direction.
