from pathlib import Path
import csv
import cv2
import numpy as np

# Dataset directory
DATASET_DIR = Path("dataset/raw/drone_dataset")

# Output report
OUTPUT_FILE = Path("results/dataset_quality_report.csv")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def analyze_image(path):
    image = cv2.imread(str(path))

    if image is None:
        return {
            "filename": path.name,
            "width": None,
            "height": None,
            "laplacian_variance": None,
            "mean_brightness": None,
            "dark_pixel_fraction": None,
            "bright_pixel_fraction": None,
            "read_status": "UNREADABLE",
        }

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return {
        "filename": path.name,
        "width": width,
        "height": height,
        "laplacian_variance": round(
            float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2
        ),
        "mean_brightness": round(float(gray.mean()), 2),
        "dark_pixel_fraction": round(
            float(np.mean(gray < 10)), 4
        ),
        "bright_pixel_fraction": round(
            float(np.mean(gray > 245)), 4
        ),
        "read_status": "READABLE",
    }


def main():
    if not DATASET_DIR.is_dir():
        raise SystemExit(
            f"Dataset directory not found: {DATASET_DIR.resolve()}"
        )

    images = sorted(
        path
        for path in DATASET_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not images:
        raise SystemExit("No supported images found.")

    results = [analyze_image(path) for path in images]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

    readable = [
        row for row in results
        if row["read_status"] == "READABLE"
    ]

    print(f"Images found: {len(images)}")
    print(f"Readable images: {len(readable)}")
    print(f"Unreadable images: {len(images) - len(readable)}")

    if readable:
        scores = [
            row["laplacian_variance"]
            for row in readable
        ]

        print("\nLaplacian variance:")
        print(f"  Minimum: {min(scores):.2f}")
        print(f"  Median:  {np.median(scores):.2f}")
        print(f"  Maximum: {max(scores):.2f}")

        for threshold in (20, 30, 40, 50, 75, 100):
            below = sum(score < threshold for score in scores)
            print(
                f"  Below {threshold:3}: "
                f"{below:3}/{len(scores)} images"
            )

    print(f"\nReport saved to: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()