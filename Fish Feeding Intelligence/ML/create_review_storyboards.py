
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(r"C:\Users\LENOVO\Desktop\Kk")

FEEDING = ROOT / "feeding after feeding" / "feeding_dataset" / "clips"
BASELINE = ROOT / "No_feeding Activity" / "clips_30s"

OUTPUT = ROOT / "outputs" / "video_review"
OUTPUT.mkdir(parents=True, exist_ok=True)

FEEDING_FILES = [
    "feeding_0_clip-9.mp4",
    "feeding_0_clip-10.mp4",
    "feeding_0_clip-11.mp4",
    "feeding_1_clip-1.mp4",
    "feeding_1_clip-2.mp4",
    "feeding_1_clip-3.mp4",
    "feeding_3_clip-1.mp4",
    "feeding_3_clip-2.mp4",
    "feeding_3_clip-3.mp4",
]

BASELINE_FILES = [
    "clip001.mp4",
    "clip002.mp4",
    "clip003.mp4",
    "clip009.mp4",
    "clip010.mp4",
    "clip011.mp4",
]

TIMESTAMPS = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 29]

TILE_WIDTH = 640
TILE_HEIGHT = 390
FRAME_HEIGHT = 360


def generate_storyboard(video_path, group):
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        print("Cannot open:", video_path)
        return

    storyboard = np.zeros(
        (4 * TILE_HEIGHT, 3 * TILE_WIDTH, 3),
        dtype=np.uint8
    )

    for index, timestamp in enumerate(TIMESTAMPS):
        capture.set(
            cv2.CAP_PROP_POS_MSEC,
            timestamp * 1000
        )

        success, frame = capture.read()

        if not success:
            continue

        frame = cv2.resize(
            frame,
            (TILE_WIDTH, FRAME_HEIGHT)
        )

        tile = np.zeros(
            (TILE_HEIGHT, TILE_WIDTH, 3),
            dtype=np.uint8
        )

        tile[30:390] = frame

        cv2.putText(
            tile,
            f"{video_path.name} | {timestamp}s",
            (12, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        row = index // 3
        col = index % 3

        y = row * TILE_HEIGHT
        x = col * TILE_WIDTH

        storyboard[
            y:y + TILE_HEIGHT,
            x:x + TILE_WIDTH
        ] = tile

    capture.release()

    output_path = OUTPUT / (
        f"{group}_{video_path.stem}_review.jpg"
    )

    cv2.imwrite(
        str(output_path),
        storyboard,
        [cv2.IMWRITE_JPEG_QUALITY, 93]
    )

    print("Created:", output_path.name)


for filename in FEEDING_FILES:
    generate_storyboard(FEEDING / filename, "feeding")

for filename in BASELINE_FILES:
    generate_storyboard(BASELINE / filename, "baseline")

print("\nCompleted!")
print("Saved to:", OUTPUT)
