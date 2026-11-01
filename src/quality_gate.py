"""Preliminary technical-quality checks for inspection photographs.

The quality gate measures whether an image is technically suitable for visual
inspection. It does not detect a drone, locate motors, identify propellers, or
prove correct assembly. Thresholds are provisional and must be calibrated
against reviewed Pluto photographs.

The public functions return a dictionary with this shape::

    {
        "status": "READY" or "UNABLE_TO_VERIFY",
        "reasons": ["..."],
        "metrics": {"width": ..., "laplacian_variance": ...}
    }

The dataset preparation layer records a failed quality gate as ``review``
rather than deleting an otherwise decodable source image.
"""

from pathlib import Path

import cv2
import numpy as np


# Initial thresholds for experimentation.
# These must be calibrated against real Pluto photographs.
MIN_IMAGE_WIDTH = 640
MIN_IMAGE_HEIGHT = 480

MIN_LAPLACIAN_VARIANCE = 50.0

MIN_MEAN_BRIGHTNESS = 40.0
MAX_MEAN_BRIGHTNESS = 220.0

MAX_DARK_PIXEL_FRACTION = 0.35
MAX_BRIGHT_PIXEL_FRACTION = 0.35


def check_image_quality(image: np.ndarray):
    """
    Evaluate whether an image is suitable for inspection.

    Returns:
        dict containing:
            status: READY or UNABLE_TO_VERIFY
            reasons: list of quality failures
            metrics: measured image properties

    Note:
        Passing this gate does NOT mean the drone is correctly
        assembled. It only means the image passed these checks.
    """

    if image is None or not isinstance(image, np.ndarray):
        return {
            "status": "UNABLE_TO_VERIFY",
            "reasons": ["Image could not be loaded."],
            "metrics": {}
        }

    if image.size == 0:
        return {
            "status": "UNABLE_TO_VERIFY",
            "reasons": ["Image is empty."],
            "metrics": {}
        }

    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        return {
            "status": "UNABLE_TO_VERIFY",
            "reasons": ["Unsupported image format."],
            "metrics": {}
        }

    height, width = gray.shape

    # Image quality measurements
    laplacian_variance = float(
        cv2.Laplacian(gray, cv2.CV_64F).var()
    )

    mean_brightness = float(np.mean(gray))

    dark_pixel_fraction = float(
        np.mean(gray < 10)
    )

    bright_pixel_fraction = float(
        np.mean(gray > 245)
    )

    metrics = {
        "width": width,
        "height": height,
        "laplacian_variance": round(laplacian_variance, 2),
        "mean_brightness": round(mean_brightness, 2),
        "dark_pixel_fraction": round(dark_pixel_fraction, 4),
        "bright_pixel_fraction": round(bright_pixel_fraction, 4)
    }

    reasons = []

    # Resolution check
    if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
        reasons.append(
            "Image resolution is too low. Capture a higher-resolution image."
        )

    # Blur check
    if laplacian_variance < MIN_LAPLACIAN_VARIANCE:
        reasons.append(
            "Image may be blurry. Hold the phone steady and refocus."
        )

    # Overall brightness check
    if mean_brightness < MIN_MEAN_BRIGHTNESS:
        reasons.append(
            "Image is too dark. Improve the lighting."
        )

    elif mean_brightness > MAX_MEAN_BRIGHTNESS:
        reasons.append(
            "Image may be overexposed. Reduce the exposure."
        )

    # Clipping checks
    if dark_pixel_fraction > MAX_DARK_PIXEL_FRACTION:
        reasons.append(
            "Too many pixels are nearly black. Check shadows and exposure."
        )

    if bright_pixel_fraction > MAX_BRIGHT_PIXEL_FRACTION:
        reasons.append(
            "Too many pixels are nearly white. Reduce glare or exposure."
        )

    return {
        "status": (
            "READY" if not reasons else "UNABLE_TO_VERIFY"
        ),
        "reasons": reasons,
        "metrics": metrics
    }


def check_image_file(image_path: Path):
    """Load an image from disk and evaluate its quality."""

    image = cv2.imread(str(image_path))

    if image is None:
        return {
            "status": "UNABLE_TO_VERIFY",
            "reasons": [
                f"Could not open image: {image_path}"
            ],
            "metrics": {}
        }

    return check_image_quality(image)
