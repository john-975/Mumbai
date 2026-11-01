from pathlib import Path
import cv2
import numpy as np
import math

DATASET_DIR = Path("dataset/raw/propeller_reference/type_B/type_B")
OUTPUT_FILE = Path("results/B_propeller_inspection_sheet.jpg")

# Change this list after inspecting the original contact sheet.
# These are candidate close-up photographs, not confirmed labels.
CANDIDATES = [
"IMG20261101134020.jpg",
"IMG20261101133956.jpg",
"IMG20261101134024.jpg",
"IMG20261101134059.jpg",
"IMG20261101134013.jpg",
"IMG20261101134032.jpg",
"IMG20261101134044.jpg",
"IMG20261101133954.jpg",
"IMG20261101133959.jpg",
"IMG20261101134010.jpg",

]

THUMB_W = 360
THUMB_H = 240
LABEL_H = 35
COLS = 3

loaded = []

for name in CANDIDATES:
    path = DATASET_DIR / name
    image = cv2.imread(str(path))

    if image is None:
        print(f"Skipping unreadable/missing file: {name}")
        continue

    h, w = image.shape[:2]
    scale = min(THUMB_W / w, THUMB_H / h)

    resized = cv2.resize(
        image,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_AREA,
    )

    loaded.append((name, resized))

if not loaded:
    raise SystemExit(
        f"No candidate images found in {DATASET_DIR.resolve()}"
    )

rows = math.ceil(len(loaded) / COLS)

sheet = np.full(
    (rows * (THUMB_H + LABEL_H), COLS * THUMB_W, 3),
    245,
    dtype=np.uint8,
)

for i, (name, thumb) in enumerate(loaded):
    row, col = divmod(i, COLS)

    x = col * THUMB_W + (THUMB_W - thumb.shape[1]) // 2
    y = row * (THUMB_H + LABEL_H)

    sheet[y:y + thumb.shape[0], x:x + thumb.shape[1]] = thumb

    cv2.putText(
        sheet,
        name,
        (col * THUMB_W + 5, y + THUMB_H + 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(OUTPUT_FILE), sheet)

print(f"Selected images: {len(loaded)}")
print(f"Saved: {OUTPUT_FILE.resolve()}")