from pathlib import Path
import subprocess
import json
import sys

# =========================================================
# SETTINGS
# =========================================================

INPUT_FOLDER = Path(
    r"C:\Users\LENOVO\Desktop\Kk\No_feeding Activity\New folder"
)

OUTPUT_FOLDER = INPUT_FOLDER / "clips_30s"

CLIP_LENGTH = 30
KEEP_LAST_SHORT_CLIP = False

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


# =========================================================
# EXACT FFMPEG PATHS FROM CONDA ENVIRONMENT
# =========================================================

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

print("FFmpeg found :", FFMPEG)
print("FFprobe found:", FFPROBE)


# =========================================================
# GET VIDEO DURATION
# =========================================================

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


# =========================================================
# FIND MP4 VIDEOS
# =========================================================

videos = sorted(INPUT_FOLDER.glob("*.mp4"))

print(f"\nFound {len(videos)} videos")

clip_number = 1


# =========================================================
# PROCESS VIDEOS
# =========================================================

for video in videos:

    print("\n===================================")
    print("Processing:", video.name)
    print("===================================")

    duration = get_duration(video)

    print(f"Duration: {duration:.2f} seconds")

    start_time = 0

    while start_time < duration:

        remaining = duration - start_time

        # Do not include a final short clip
        if remaining < CLIP_LENGTH and not KEEP_LAST_SHORT_CLIP:

            print(
                f"Skipping final {remaining:.2f} seconds "
                f"of {video.name}"
            )

            break

        current_length = min(CLIP_LENGTH, remaining)

        output_name = f"clip{clip_number:03d}.mp4"

        output_path = OUTPUT_FOLDER / output_name

        command = [
            str(FFMPEG),

            "-y",

            "-ss", str(start_time),

            "-i", str(video),

            "-t", str(current_length),

            # Keep same image size.
            # No cropping / filtering / speed changes.
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",

            "-c:a", "aac",

            str(output_path)
        ]

        subprocess.run(
            command,
            check=True
        )

        print(
            f"{output_name}  |  "
            f"{start_time:.0f}s → "
            f"{start_time + current_length:.0f}s"
        )

        clip_number += 1
        start_time += CLIP_LENGTH


# =========================================================
# FINISH
# =========================================================

print("\n===================================")
print("COMPLETED")
print("===================================")

print("Total clips created:", clip_number - 1)

print("\nSaved folder:")
print(OUTPUT_FOLDER)