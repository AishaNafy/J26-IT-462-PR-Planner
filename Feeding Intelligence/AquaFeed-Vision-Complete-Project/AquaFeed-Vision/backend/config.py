
import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

# Project directories
BASE = Path(__file__).resolve().parent.parent

load_dotenv(BASE / ".env")

# Main AI project folder
PROJECT = Path(os.getenv(
    "AQUAFEED_PROJECT_ROOT",
    r"C:\Users\aisha\OneDrive\Desktop\PPT1\PPT1"
))

# YOLO model weights
WEIGHTS = Path(os.getenv(
    "AQUAFEED_YOLO_WEIGHTS",
    str(PROJECT / "goldfish_yolo26_training"
        / "yolo26l_goldfish" / "weights" / "best.pt")
))

# Processing device
DEVICE = os.getenv("AQUAFEED_DEVICE", "cpu")

# Motion features dataset
FEATURES = PROJECT / "outputs" / "motion_features.csv"

# Motion preprocessing model
PREPROCESSOR = (
    PROJECT / "outputs" / "models"
    / "motion_preprocessor.joblib"
)

# Motion autoencoder model
AUTOENCODER = (
    PROJECT / "outputs" / "models"
    / "motion_autoencoder.pt"
)

# Runtime directories
RUNTIME = BASE / "runtime"
RUNS = RUNTIME / "runs"
RUNS.mkdir(parents=True, exist_ok=True)


def ffmpeg_path():
    configured = os.getenv("AQUAFEED_FFMPEG", "")

    if configured and Path(configured).is_file():
        return configured

    found = shutil.which("ffmpeg")
    if found:
        return found

    conda = os.getenv("CONDA_PREFIX", "")

    if conda:
        p = Path(conda) / "Library" / "bin" / "ffmpeg.exe"

        if p.is_file():
            return str(p)

    return None


def system_info():
    return {
        "yolo": WEIGHTS.is_file(),
        "motion": FEATURES.is_file(),
        "autoencoder": AUTOENCODER.is_file(),
        "preprocessor": PREPROCESSOR.is_file(),
        "ffmpeg": bool(ffmpeg_path()),
        "device": DEVICE,
        "yolo_path": str(WEIGHTS),
        "validated_hunger_model": False
    }
