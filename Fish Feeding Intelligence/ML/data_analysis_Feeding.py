from pathlib import Path
import subprocess
import json
import csv

# ============================================================
# INPUT FOLDER
# ============================================================

INPUT_FOLDER = Path(
    r"C:\Users\LENOVO\Desktop\Kk\feeding after feeding"
)

CLIP_LENGTH = 30

# ============================================================
# OUTPUT
# ============================================================

OUTPUT_FOLDER = INPUT_FOLDER / "feeding_dataset"
CLIPS_FOLDER = OUTPUT_FOLDER / "clips"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
CLIPS_FOLDER.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_FOLDER / "feeding_clips.csv"

# ============================================================
# FFMPEG PATHS
# ============================================================

FFMPEG = Path(
    r"C:\Users\LENOVO\anaconda3\envs\video_tools\Library\bin\ffmpeg.exe"
)

FFPROBE = Path(
    r"C:\Users\LENOVO\anaconda3\envs\video_tools\Library\bin\ffprobe.exe"
)

if not FFMPEG.exists():
    raise FileNotFoundError(f"FFmpeg not found: {FFMPEG}")

if not FFPROBE.exists():
    raise FileNotFoundError(f"FFprobe not found: {FFPROBE}")

# ============================================================
# GET VIDEO DURATION
# ============================================================

def get_duration(video_path):

    command = [
        str(FFPROBE),
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        str(video_path)
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True
    )

    info = json.loads(result.stdout)

    return float(info["format"]["duration"])


# ============================================================
# FIND ALL FEEDING VIDEOS
# ============================================================

videos = sorted(
    INPUT_FOLDER.glob("feeding_*.mp4")
)

print()
print("==========================================")
print(f"Found {len(videos)} feeding videos")
print("==========================================")

for video in videos:
    print(video.name)


# ============================================================
# PROCESS EVERY VIDEO
# ============================================================

rows = []

for video in videos:

    video_id = video.stem.strip()

    print()
    print("==========================================")
    print("Processing:", video.name)
    print("==========================================")

    duration = get_duration(video)

    print(f"Duration: {duration:.2f} seconds")

    start_time = 0
    clip_number = 1

    while start_time + CLIP_LENGTH <= duration:

        end_time = start_time + CLIP_LENGTH

        clip_name = (
            f"{video_id}_clip-{clip_number}.mp4"
        )

        clip_path = CLIPS_FOLDER / clip_name

        print(
            f"Creating: {clip_name} "
            f"[{start_time:.0f}s - {end_time:.0f}s]"
        )

        command = [
            str(FFMPEG),
            "-y",

            "-ss", str(start_time),

            "-i", str(video),

            "-t", str(CLIP_LENGTH),

            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",

            "-c:a", "aac",

            "-movflags", "+faststart",

            str(clip_path)
        ]

        subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )

        rows.append(
            {
                "clip_name": clip_name,
                "source_video": video.name,
                "start_sec": start_time,
                "end_sec": end_time,
                "duration_sec": CLIP_LENGTH,
                "phase": "post_feeding",
                "feeding_behavior": "UNLABELED"
            }
        )

        start_time += CLIP_LENGTH
        clip_number += 1

    remaining = duration - start_time

    if remaining > 0:
        print(
            f"Skipped final {remaining:.2f} sec "
            "(less than 30 sec)"
        )


# ============================================================
# SAVE MASTER CSV
# ============================================================

columns = [
    "clip_name",
    "source_video",
    "start_sec",
    "end_sec",
    "duration_sec",
    "phase",
    "feeding_behavior"
]

with open(
    CSV_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as csv_file:

    writer = csv.DictWriter(
        csv_file,
        fieldnames=columns
    )

    writer.writeheader()
    writer.writerows(rows)


# ============================================================
# FINISH
# ============================================================

print()
print("==========================================")
print("ALL VIDEOS COMPLETED")
print("==========================================")

print(f"Total clips created: {len(rows)}")

print()
print("Clips saved in:")
print(CLIPS_FOLDER)

print()
print("CSV saved as:")
print(CSV_PATH)