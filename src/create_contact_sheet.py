from pathlib import Path
import math
import cv2
import numpy as np

DATASET_DIR = Path("dataset/raw/drone_dataset")
OUTPUT_FILE = Path("results/contact_sheet.jpg")

THUMB_WIDTH = 240
THUMB_HEIGHT = 180
LABEL_HEIGHT = 35
COLUMNS = 4

images = sorted(
    p for p in DATASET_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

if not images:
    raise SystemExit(f"No images found in {DATASET_DIR}")

rows = math.ceil(len(images) / COLUMNS)

sheet = np.full(
    (rows * (THUMB_HEIGHT + LABEL_HEIGHT),
     COLUMNS * THUMB_WIDTH, 3),
    245,
    dtype=np.uint8
)

for index, path in enumerate(images):
    image = cv2.imread(str(path))

    if image is None:
        continue

    height, width = image.shape[:2]

    scale = min(
        THUMB_WIDTH / width,
        THUMB_HEIGHT / height
    )

    new_width = int(width * scale)
    new_height = int(height * scale)

    thumbnail = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    row = index // COLUMNS
    col = index % COLUMNS

    x = col * THUMB_WIDTH + (THUMB_WIDTH - new_width) // 2
    y = row * (THUMB_HEIGHT + LABEL_HEIGHT)

    sheet[y:y + new_height, x:x + new_width] = thumbnail

    label_y = y + THUMB_HEIGHT + 23

    cv2.putText(
        sheet,
        path.name,
        (col * THUMB_WIDTH + 5, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (20, 20, 20),
        1,
        cv2.LINE_AA
    )

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

cv2.imwrite(str(OUTPUT_FILE), sheet)

print(f"Images: {len(images)}")
print(f"Contact sheet: {OUTPUT_FILE.resolve()}")
print(f"Dimensions: {sheet.shape[1]} x {sheet.shape[0]}")