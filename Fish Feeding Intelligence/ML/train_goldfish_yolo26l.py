from pathlib import Path
import os

# Must be set before importing torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(
    r"C:\Users\LENOVO\Desktop\Kk"
)

DATA_YAML = (
    PROJECT_ROOT
    / "gold fish.v10i.yolo26"
    / "data.yaml"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "goldfish_yolo26_training"
)


def main():

    print("=" * 60)
    print("GOLDFISH DETECTION | YOLO26 LARGE")
    print("=" * 60)

    if not DATA_YAML.exists():
        raise FileNotFoundError(DATA_YAML)

    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        torch.cuda.empty_cache()

    # YOLO26 Large
    model = YOLO("yolo26s.pt")

    model.train(
        data=str(DATA_YAML),

        epochs=150,
        patience=30,

        # IMPORTANT for 6 GB GPU
        imgsz=640,
        batch=1,

        device=0,

        # Windows / memory safe
        workers=0,
        cache=False,

        # Mixed precision saves VRAM
        amp=True,

        single_cls=True,
        pretrained=True,

        optimizer="auto",

        # Augmentation
        degrees=5.0,
        translate=0.10,
        scale=0.40,
        fliplr=0.5,

        close_mosaic=15,

        val=True,
        plots=True,

        seed=42,
        deterministic=True,

        project=str(OUTPUT_DIR),
        name="yolo26l_goldfish",
        exist_ok=True,

        save=True,
        save_period=10,

        verbose=True,
    )


if __name__ == "__main__":
    main()