#!/usr/bin/env python3

import csv
import time
from pathlib import Path
from collections import Counter

import cv2
import torch
from ultralytics import YOLO


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset/raw/drone_dataset"
MODEL_PATH = PROJECT_ROOT / "models/yolo26n.pt"
OUTPUT_DIR = PROJECT_ROOT / "results/yolo"

ANNOTATED_DIR = OUTPUT_DIR / "annotated_conf025"

THRESHOLDS = [0.05, 0.10, 0.25, 0.40, 0.50]

INFERENCE_CONFIDENCE = min(THRESHOLDS)
IMAGE_SIZE = 640
DEVICE = "cpu"
ANNOTATION_THRESHOLD = 0.25
MAX_ANNOTATED_WIDTH = 1600

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ============================================================
# Helpers
# ============================================================

def get_images(folder):
    return sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def annotate_image(image_path, boxes, names, output_path):
    """
    Draw detections with confidence >= ANNOTATION_THRESHOLD.
    Resize the output for easier inspection and reduced disk usage.
    Original dataset images are never modified.
    """
    image = cv2.imread(str(image_path))

    if image is None:
        return False

    height, width = image.shape[:2]
    scale = min(1.0, MAX_ANNOTATED_WIDTH / width)

    if scale < 1.0:
        image = cv2.resize(
            image,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )

    for box in boxes:
        if box["confidence"] < ANNOTATION_THRESHOLD:
            continue

        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]

        x1 = int(x1 * scale)
        y1 = int(y1 * scale)
        x2 = int(x2 * scale)
        y2 = int(y2 * scale)

        label = (
            f'{names[box["class_id"]]} '
            f'{box["confidence"]:.2f}'
        )

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        text_y = max(20, y1 - 7)
        cv2.putText(
            image,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(str(output_path), image)


# ============================================================
# Main benchmark
# ============================================================

def main():
    if not DATASET_DIR.is_dir():
        raise SystemExit(f"Dataset folder not found: {DATASET_DIR}")

    if not MODEL_PATH.is_file():
        raise SystemExit(f"Local model weights not found: {MODEL_PATH}")

    images = get_images(DATASET_DIR)

    if not images:
        raise SystemExit(f"No supported images found in: {DATASET_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

    # Limit CPU threads for this dual-core laptop.
    torch.set_num_threads(2)

    print("=" * 65)
    print("YOLO PRETRAINED DETECTION BENCHMARK")
    print("=" * 65)
    print(f"Model:       {MODEL_PATH}")
    print(f"Dataset:     {DATASET_DIR}")
    print(f"Images:      {len(images)}")
    print(f"Device:      {DEVICE}")
    print(f"Image size:  {IMAGE_SIZE}")
    print(f"Thresholds:  {THRESHOLDS}")
    print("Internet:    Not required for local inference")
    print("=" * 65)

    model = YOLO(str(MODEL_PATH))
    class_names = model.names

    all_detections = []
    image_runs = []
    summary_rows = []

    # These counters include all predicted classes, not just drones.
    counters = {
        threshold: Counter()
        for threshold in THRESHOLDS
    }

    images_with_detections = {
        threshold: set()
        for threshold in THRESHOLDS
    }

    total_start = time.perf_counter()
    successful_images = 0
    failed_images = []

    for index, image_path in enumerate(images, start=1):
        relative_path = image_path.relative_to(DATASET_DIR).as_posix()

        print(
            f"[{index:03d}/{len(images):03d}] "
            f"{relative_path}",
            flush=True,
        )

        image_start = time.perf_counter()

        try:
            results = model.predict(
                source=str(image_path),
                imgsz=IMAGE_SIZE,
                conf=INFERENCE_CONFIDENCE,
                device=DEVICE,
                max_det=300,
                verbose=False,
                save=False,
            )

            result = results[0]
            elapsed_seconds = time.perf_counter() - image_start

            detections = []

            if result.boxes is not None and len(result.boxes) > 0:
                xyxy = result.boxes.xyxy.cpu().tolist()
                class_ids = result.boxes.cls.cpu().tolist()
                confidences = result.boxes.conf.cpu().tolist()

                for coordinates, class_value, confidence in zip(
                    xyxy, class_ids, confidences
                ):
                    class_id = int(class_value)

                    detections.append({
                        "class_id": class_id,
                        "class_name": class_names[class_id],
                        "confidence": float(confidence),
                        "x1": float(coordinates[0]),
                        "y1": float(coordinates[1]),
                        "x2": float(coordinates[2]),
                        "y2": float(coordinates[3]),
                    })

            successful_images += 1

            # Save one row per image, including images with zero detections.
            image_runs.append({
                "image": relative_path,
                "status": "success",
                "detections_at_005": len(detections),
                "wall_time_seconds": round(elapsed_seconds, 4),
                "inference_time_ms": round(
                    float(result.speed.get("inference", 0.0)), 3
                ),
                "image_width": int(result.orig_shape[1]),
                "image_height": int(result.orig_shape[0]),
                "error": "",
            })

            for detection in detections:
                for threshold in THRESHOLDS:
                    if detection["confidence"] >= threshold:
                        counters[threshold][detection["class_name"]] += 1
                        images_with_detections[threshold].add(relative_path)

                        all_detections.append({
                            "image": relative_path,
                            "threshold": threshold,
                            "class_id": detection["class_id"],
                            "class_name": detection["class_name"],
                            "confidence": round(detection["confidence"], 6),
                            "x1": round(detection["x1"], 2),
                            "y1": round(detection["y1"], 2),
                            "x2": round(detection["x2"], 2),
                            "y2": round(detection["y2"], 2),
                        })

            annotated_path = ANNOTATED_DIR / (
                image_path.stem + "_conf025.jpg"
            )

            saved = annotate_image(
                image_path,
                detections,
                class_names,
                annotated_path,
            )

            if not saved:
                print("  WARNING: Could not save annotated image.")

            visible_count = sum(
                d["confidence"] >= ANNOTATION_THRESHOLD
                for d in detections
            )

            print(
                f"  detections@0.05={len(detections)}, "
                f"detections@0.25={visible_count}, "
                f"time={elapsed_seconds:.2f}s"
            )

        except Exception as exc:
            failed_images.append({
                "image": relative_path,
                "error": repr(exc),
            })

            image_runs.append({
                "image": relative_path,
                "status": "failed",
                "detections_at_005": "",
                "wall_time_seconds": "",
                "inference_time_ms": "",
                "image_width": "",
                "image_height": "",
                "error": repr(exc),
            })

            print(f"  ERROR: {exc}")

    total_seconds = time.perf_counter() - total_start

    # Summarize each confidence threshold.
    for threshold in THRESHOLDS:
        total_detections = sum(counters[threshold].values())

        summary_rows.append({
            "threshold": threshold,
            "images_processed": successful_images,
            "images_with_at_least_one_detection":
                len(images_with_detections[threshold]),
            "images_with_zero_detections":
                successful_images - len(images_with_detections[threshold]),
            "total_detections": total_detections,
            "detections_per_processed_image": round(
                total_detections / successful_images, 4
            ) if successful_images else 0,
            "class_counts": "; ".join(
                f"{name}={count}"
                for name, count in sorted(counters[threshold].items())
            ),
        })

    write_csv(
        OUTPUT_DIR / "detections.csv",
        all_detections,
        [
            "image", "threshold", "class_id", "class_name",
            "confidence", "x1", "y1", "x2", "y2",
        ],
    )

    write_csv(
        OUTPUT_DIR / "image_runs.csv",
        image_runs,
        [
            "image", "status", "detections_at_005",
            "wall_time_seconds", "inference_time_ms",
            "image_width", "image_height", "error",
        ],
    )

    write_csv(
        OUTPUT_DIR / "threshold_summary.csv",
        summary_rows,
        [
            "threshold", "images_processed",
            "images_with_at_least_one_detection",
            "images_with_zero_detections",
            "total_detections", "detections_per_processed_image",
            "class_counts",
        ],
    )

    write_csv(
        OUTPUT_DIR / "failed_images.csv",
        failed_images,
        ["image", "error"],
    )

    print("\n" + "=" * 65)
    print("BENCHMARK SUMMARY")
    print("=" * 65)
    print(f"Images discovered:  {len(images)}")
    print(f"Images processed:   {successful_images}")
    print(f"Images failed:      {len(failed_images)}")
    print(f"Total elapsed:      {total_seconds:.2f} seconds")

    if successful_images:
        print(
            "Average wall time:  "
            f"{total_seconds / successful_images:.2f} seconds/image"
        )

    print("\nThreshold results:")

    for row in summary_rows:
        print(
            f"  conf={row['threshold']:.2f} | "
            f"detections={row['total_detections']} | "
            f"images with detections="
            f"{row['images_with_at_least_one_detection']}/"
            f"{successful_images}"
        )

    print("\nOutput files:")
    print(f"  {OUTPUT_DIR / 'detections.csv'}")
    print(f"  {OUTPUT_DIR / 'image_runs.csv'}")
    print(f"  {OUTPUT_DIR / 'threshold_summary.csv'}")
    print(f"  {OUTPUT_DIR / 'failed_images.csv'}")
    print(f"  {ANNOTATED_DIR}")

    print("\nIMPORTANT:")
    print("These are prediction statistics, NOT measured accuracy.")
    print("Ground-truth annotations are required for precision, recall,")
    print("IoU and mAP evaluation.")


if __name__ == "__main__":
    main()
