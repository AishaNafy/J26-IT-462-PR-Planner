from pathlib import Path
from functools import lru_cache
import math
import subprocess

import cv2
import numpy as np

from .config import (
    WEIGHTS, DEVICE, FEATURES, PREPROCESSOR,
    AUTOENCODER, ffmpeg_path
)

FEATURE_NAMES = [
    "flow_mean_norm_s",
    "flow_p90_norm_s",
    "flow_std_norm_s",
    "flow_energy_norm_s2",
    "flow_temporal_std",
    "flow_temporal_p90",
    "spatial_motion_entropy",
    "flow_top_left_norm_s",
    "flow_top_right_norm_s",
    "flow_bottom_left_norm_s",
    "flow_bottom_right_norm_s",
    "frame_difference_mean",
    "frame_difference_std",
]


def prepare_video(src, directory):
    suffix = src.suffix.lower()
    if suffix in (".webm", ".mov", ".mkv"):
        ffmpeg = ffmpeg_path()
        if not ffmpeg:
            raise RuntimeError(
                "FFmpeg required for this video format. "
                "Install conda-forge ffmpeg in your environment."
            )
        dest = directory / "converted.mp4"
        process = subprocess.run([
            ffmpeg, "-y", "-loglevel", "error",
            "-i", str(src), "-an",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(dest)
        ], capture_output=True, text=True, timeout=180)
        if process.returncode:
            raise RuntimeError(
                "FFmpeg conversion failed: " + process.stderr[-400:]
            )
        return dest
    return src


def measure_motion(video, progress):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError("Could not decode the video.")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    total = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0:
        cap.release()
        raise RuntimeError("Invalid video frame rate.")

    duration = total / fps
    if duration < 12:
        cap.release()
        raise ValueError("Please upload at least 12 seconds of video.")

    sample_step = max(1, round(fps / 5))
    sampled_fps = fps / sample_step
    seconds_per_step = 1.0 / sampled_fps

    prev = None
    values, means, high_values = [], [], []
    differences, entropy_values = [], []
    region_series = [[], [], [], []]
    brightness = []
    timeline = []
    frame_index = 0
    count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_time = frame_index / fps
        frame_index += 1

        # Analyse only first 30 seconds.
        if frame_time > 30:
            break

        if (frame_index - 1) % sample_step:
            continue

        h, w = frame.shape[:2]
        h_small = max(1, round(h * 320 / max(1, w)))
        frame_small = cv2.resize(frame, (320, h_small))
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        brightness.append(float(gray.mean() / 255))
        count += 1

        if prev is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag = np.linalg.norm(flow, axis=2)
            mag = mag / math.hypot(320, h_small)
            mag = mag / seconds_per_step

            values.append(mag.ravel())
            mean = float(mag.mean())
            means.append(mean)
            high_values.append(float(np.percentile(mag, 90)))

            difference = float(
                np.abs(
                    gray.astype(np.float32) -
                    prev.astype(np.float32)
                ).mean() / 255
            )
            differences.append(difference)
            timeline.append({
                "t": round(frame_time, 1),
                "motion": round(mean, 7)
            })

            hh, ww = mag.shape
            regions = [
                mag[:hh//2, :ww//2],
                mag[:hh//2, ww//2:],
                mag[hh//2:, :ww//2],
                mag[hh//2:, ww//2:]
            ]
            parts = [
                float(x.mean()) if x.size else 0
                for x in regions
            ]
            for i, item in enumerate(parts):
                region_series[i].append(item)

            weights = np.array(parts, dtype=float) + 1e-12
            weights = weights / weights.sum()
            entropy_values.append(
                float(-np.sum(weights * np.log(weights)) / np.log(4))
            )

        prev = gray

    cap.release()

    if not values:
        raise RuntimeError("Not enough video frames for motion analysis.")

    all_values = np.concatenate(values)
    mean_series = np.array(means)
    diff_series = np.array(differences)

    numbers = [
        float(all_values.mean()),
        float(np.percentile(all_values, 90)),
        float(all_values.std()),
        float(np.mean(all_values ** 2)),
        float(mean_series.std()),
        float(np.percentile(mean_series, 90)),
        float(np.mean(entropy_values)),
        *[float(np.mean(x)) for x in region_series],
        float(diff_series.mean()),
        float(diff_series.std())
    ]

    features = dict(zip(FEATURE_NAMES, numbers))
    progress("Motion analysis complete", 38)

    return {
        "features": features,
        "series": timeline,
        "duration": round(min(duration, 30), 1),
        "frames": count,
        "sampled_fps": round(sampled_fps, 2),
        "brightness": round(float(np.mean(brightness)), 4)
    }


def feature_summary(features):
    """
    Research-only comparative movement analysis.
    Does not establish actual fish hunger.
    """
    result = {
        "movement_index": None,
        "movement_level": "Unavailable",
        "behaviour_deviation": None,
        "anomaly_level": "Not assessed",
        "mse": None,
        "ae_status": "Unavailable",
        "warning": "Movement and anomaly are not hunger probabilities."
    }

    if not FEATURES.exists():
        result["warning"] = "Reference dataset not found."
        return result

    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    reference = pd.read_csv(FEATURES)
    if not all(k in reference.columns for k in FEATURE_NAMES):
        result["warning"] = "Reference feature columns differ."
        return result

    baseline = reference[
        reference["phase"] == "baseline"
    ][FEATURE_NAMES].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()

    if len(baseline) < 5:
        result["warning"] = "Too few baseline examples."
        return result

    flow_baseline = baseline["flow_mean_norm_s"].to_numpy()
    new_flow = features["flow_mean_norm_s"]
    quickness = 100 * np.mean(flow_baseline <= new_flow)

    result["movement_index"] = round(float(quickness), 1)
    result["movement_level"] = (
        "High activity" if quickness >= 80
        else "Moderate activity" if quickness >= 35
        else "Lower activity"
    )

    scaler = StandardScaler()
    baseline_scaled = scaler.fit_transform(baseline.to_numpy())
    centre = baseline_scaled.mean(axis=0)

    x = np.array(
        [[features[k] for k in FEATURE_NAMES]],
        dtype=float
    )
    current = scaler.transform(x)[0]

    distances = np.linalg.norm(baseline_scaled - centre, axis=1)
    distance = float(np.linalg.norm(current - centre))
    percentile = 100 * np.mean(distances <= distance)

    result["behaviour_deviation"] = round(float(percentile), 1)
    result["anomaly_level"] = (
        "Unusual movement pattern" if percentile >= 95
        else "Some movement changes" if percentile >= 75
        else "Within reference range"
    )

    # Optional connection to original saved autoencoder.
    # Actual architecture/feature parity requires original
    # phase1_pipeline.py to verify.
    if AUTOENCODER.exists() and PREPROCESSOR.exists():
        try:
            import joblib
            import torch
            from torch import nn

            preobj = joblib.load(PREPROCESSOR)
            if isinstance(preobj, dict):
                scaler_saved = next(
                    (preobj[k] for k in (
                        "scaler", "preprocessor", "transformer"
                    ) if k in preobj), None
                )
                order = preobj.get("feature_columns", FEATURE_NAMES)
            else:
                scaler_saved = preobj
                order = FEATURE_NAMES

            if scaler_saved is None:
                raise ValueError("Unsupported preprocessor format")

            x_saved = scaler_saved.transform(
                np.array(
                    [[features[k] for k in order]],
                    dtype=np.float32
                )
            )

            class MotionAE(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.encoder = nn.Sequential(
                        nn.Linear(13, 64), nn.ReLU(),
                        nn.Linear(64, 32), nn.ReLU(),
                        nn.Linear(32, 6)
                    )
                    self.decoder = nn.Sequential(
                        nn.Linear(6, 32), nn.ReLU(),
                        nn.Linear(32, 64), nn.ReLU(),
                        nn.Linear(64, 13)
                    )

                def forward(self, x):
                    return self.decoder(self.encoder(x))

            try:
                state = torch.load(
                    AUTOENCODER, map_location="cpu", weights_only=True
                )
            except TypeError:
                state = torch.load(AUTOENCODER, map_location="cpu")

            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            elif isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]

            model = MotionAE()
            model.load_state_dict(state)
            model.eval()
            with torch.inference_mode():
                tensor = torch.tensor(x_saved, dtype=torch.float32)
                reconstructed = model(tensor)
                mse = torch.mean((tensor - reconstructed) ** 2)
                result["mse"] = round(float(mse.item()), 5)
                result["ae_status"] = "Connected (experimental)"

        except Exception as exc:
            result["ae_status"] = "Compatibility check required"
            result["warning"] += " Autoencoder: " + str(exc)[:130]

    return result


@lru_cache(maxsize=1)
def get_yolo():
    if not WEIGHTS.exists():
        raise FileNotFoundError(
            "Trained YOLO checkpoint missing: " + str(WEIGHTS)
        )
    from ultralytics import YOLO
    return YOLO(str(WEIGHTS))


def detect_fish(video, directory, confidence, sample_count, progress):
    output = {
        "available": False,
        "count_median": None,
        "count_max": None,
        "count_min": None,
        "sampled_frames": 0,
        "average_confidence": None,
        "boxes": [],
        "image": None,
        "error": None
    }

    if not WEIGHTS.exists():
        output["error"] = "YOLO best.pt not found"
        return output

    model = get_yolo()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        output["error"] = "Cannot read video for detection"
        return output

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1
    duration = min(float(frames / fps), 30)
    timestamps = np.linspace(
        0.1, max(0.1, duration - 0.25), sample_count
    )

    counts, scores = [], []
    best_image, best_boxes = None, []
    best_count = -1

    for index, second in enumerate(timestamps):
        cap.set(cv2.CAP_PROP_POS_MSEC, float(second) * 1000)
        success, frame = cap.read()
        if not success:
            continue

        try:
            pred = model.predict(
                frame,
                conf=float(confidence),
                iou=0.5,
                imgsz=640,
                device=DEVICE,
                verbose=False
            )[0]
        except (RuntimeError, MemoryError) as exc:
            if "memory" not in str(exc).lower():
                raise
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            pred = model.predict(
                frame, conf=float(confidence), iou=0.5,
                imgsz=640, device="cpu", verbose=False
            )[0]

        rects = []
        if pred.boxes is not None:
            coords = pred.boxes.xyxy.detach().cpu().numpy()
            probs = pred.boxes.conf.detach().cpu().numpy()
            for box, score in zip(coords, probs):
                x1, y1, x2, y2 = [int(v) for v in box]
                rects.append({
                    "xyxy": [x1, y1, x2, y2],
                    "confidence": round(float(score), 3)
                })
                scores.append(float(score))

        count = len(rects)
        counts.append(count)

        if count > best_count:
            best_count = count
            best_boxes = rects
            best_image = frame.copy()

        progress(
            f"Detecting goldfish {index+1}/{sample_count}",
            45 + int((index + 1) / sample_count * 43)
        )

    cap.release()

    if not counts:
        output["error"] = "No sampled frames decoded"
        return output

    if best_image is not None:
        for box in best_boxes:
            x1, y1, x2, y2 = box["xyxy"]
            label = f"Goldfish {box['confidence']:.2f}"
            cv2.rectangle(
                best_image, (x1, y1), (x2, y2),
                (30, 170, 255), 3
            )
            cv2.putText(
                best_image, label,
                (x1, max(24, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (30, 170, 255), 2
            )

        image_path = directory / "detections.jpg"
        cv2.imwrite(str(image_path), best_image, [
            cv2.IMWRITE_JPEG_QUALITY, 90
        ])
        output["image"] = "detections.jpg"

    output.update({
        "available": True,
        "count_median": int(np.median(counts)),
        "count_min": int(min(counts)),
        "count_max": int(max(counts)),
        "sampled_frames": len(counts),
        "average_confidence": round(
            float(np.mean(scores)), 3
        ) if scores else None,
        "boxes": best_boxes
    })

    return output


def analyse_video(source, directory, confidence, samples, progress):
    progress("Preparing video", 8)
    video = prepare_video(source, directory)

    progress("Extracting movement features", 15)
    motion = measure_motion(video, progress)

    progress("Comparing behaviour", 40)
    behaviour = feature_summary(motion["features"])

    progress("Running Goldfish YOLO", 44)
    fish = detect_fish(
        video, directory, confidence, samples, progress
    )

    progress("Preparing final report", 95)
    return {
        "video": {
            "duration_seconds": motion["duration"],
            "sampled_frames": motion["frames"],
            "sampled_fps": motion["sampled_fps"]
        },
        "fish": fish,
        "behaviour": behaviour,
        "features": motion["features"],
        "timeline": motion["series"],
        "hunger": {
            "status": "NOT_VALIDATED",
            "description": (
                "No clinically or experimentally validated "
                "hunger classifier is connected. Feeding reminders "
                "are based only on the recorded schedule."
            )
        },
        "health": {
            "status": "NOT_ASSESSED",
            "description": (
                "Swimming behaviour and visible fish counts "
                "cannot establish fish health."
            )
        },
        "settings": {
            "confidence": round(float(confidence), 2),
            "yolo_frames": int(samples)
        }
    }
