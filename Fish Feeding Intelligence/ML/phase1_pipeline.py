
"""
AquaFeedLab | Phase 1 Research Pipeline
=======================================

Purpose
-------
1. Discover existing 30-second Goldfish video clips.
2. Extract quantitative video-motion features.
3. Train a self-supervised masked denoising autoencoder.
4. Export behavioural representations and diagnostics.
5. Evaluate four-class feeding-response prediction only
   when verified annotations are available.

Research limitations
--------------------
- Optical flow is not calibrated fish swimming speed.
- Reconstruction error is not a direct hunger score.
- No feeding-response labels are generated automatically.
- Supervised validation groups clips by recording/session.

Python: 3.11+
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# 01 | PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

BASELINE_DIR = Path(
    r"C:\Users\LENOVO\Desktop\Kk"
    r"\No_feeding Activity\clips_30s"
)

FEEDING_DIR = Path(
    r"C:\Users\LENOVO\Desktop\Kk"
    r"\feeding after feeding\feeding_dataset\clips"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
RESEARCH_DIR = OUTPUT_DIR / "research"
EVALUATION_DIR = OUTPUT_DIR / "evaluation"

MANIFEST_PATH = OUTPUT_DIR / "dataset_manifest.csv"
ANNOTATION_PATH = OUTPUT_DIR / "feeding_annotations.csv"
FEATURE_PATH = OUTPUT_DIR / "motion_features.csv"
ERROR_PATH = OUTPUT_DIR / "processing_errors.csv"

SEED = 42
TARGET_SAMPLE_FPS = 5.0
ANALYSIS_WIDTH = 320

SSL_EPOCHS = 100
SSL_BATCH_SIZE = 16
SSL_LEARNING_RATE = 0.001
SSL_MASK_PROBABILITY = 0.20
LATENT_DIM = 6

FEEDING_CLASSES = ["None", "Weak", "Medium", "Strong"]

# These are optical-flow / image-motion statistics.
# They are NOT fish speed measurements in cm/s.

FEATURE_COLUMNS = [
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


# ============================================================
# 02 | UTILITIES
# ============================================================

def initialize_environment():
    for directory in [
        OUTPUT_DIR,
        MODEL_DIR,
        RESEARCH_DIR,
        EVALUATION_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def save_csv(dataframe, destination):
    dataframe.to_csv(
        destination,
        index=False,
        encoding="utf-8-sig",
    )


def save_json(payload, destination):
    with open(destination, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def natural_sort_key(path):
    parts = re.split(r"(\d+)", path.name.lower())
    return [
        int(item) if item.isdigit() else item
        for item in parts
    ]


def derive_recording_id(path, phase):
    """
    feeding_0_clip-1.mp4 -> feeding_0

    The old baseline clips are named clip001.mp4 etc.
    Their original recording IDs cannot be recovered
    reliably from those filenames.
    """

    if phase == "baseline":
        return "baseline_source_unknown"

    stem = path.stem.strip()

    match = re.match(
        r"^(.*)_clip[-_](\d+)$",
        stem,
        flags=re.IGNORECASE,
    )

    if match:
        return match.group(1).strip()

    return stem


# ============================================================
# 03 | DATASET DISCOVERY AND ANNOTATIONS
# ============================================================

def prepare_dataset():
    print("\n[01/04] Preparing dataset manifest")

    sources = [
        ("baseline", BASELINE_DIR),
        ("feeding_recording", FEEDING_DIR),
    ]

    records = []

    for phase, directory in sources:

        if not directory.is_dir():
            print(f"[WARNING] Missing directory: {directory}")
            continue

        videos = sorted(
            [
                file
                for file in directory.iterdir()
                if file.is_file()
                and file.suffix.lower() == ".mp4"
            ],
            key=natural_sort_key,
        )

        print(f"{phase}: {len(videos)} clips")

        for video in videos:
            recording_id = derive_recording_id(
                video, phase
            )

            records.append({
                "clip_uid": f"{phase}::{video.name}",
                "clip_name": video.name,
                "clip_path": str(video.resolve()),
                "phase": phase,
                "recording_id": recording_id,
                "session_id": recording_id,
            })

    if not records:
        raise RuntimeError(
            "No MP4 clips found. Check the two input paths."
        )

    manifest = pd.DataFrame(records)

    if manifest["clip_uid"].duplicated().any():
        raise RuntimeError("Duplicate clip identifiers found.")

    save_csv(manifest, MANIFEST_PATH)

    # Preserve previously completed annotation fields.
    previous = {}

    if ANNOTATION_PATH.exists():
        old = pd.read_csv(
            ANNOTATION_PATH,
            dtype=str,
        ).fillna("")

        for _, row in old.iterrows():
            previous[row["clip_uid"]] = row.to_dict()

    annotations = []

    editable_fields = [
        "session_id",
        "food_observed",
        "feeding_intensity",
        "annotator",
        "notes",
    ]

    for item in records:
        row = {
            "clip_uid": item["clip_uid"],
            "clip_name": item["clip_name"],
            "phase": item["phase"],
            "recording_id": item["recording_id"],
            "session_id": item["session_id"],
            "food_observed": "",
            "feeding_intensity": "",
            "annotator": "",
            "notes": "",
        }

        if item["clip_uid"] in previous:
            saved = previous[item["clip_uid"]]

            for field in editable_fields:
                if field in saved:
                    row[field] = saved[field]

        annotations.append(row)

    save_csv(
        pd.DataFrame(annotations),
        ANNOTATION_PATH,
    )

    print(f"Manifest saved: {MANIFEST_PATH}")
    print(f"Annotations saved: {ANNOTATION_PATH}")

    return manifest


# ============================================================
# 04 | OPTICAL-FLOW FEATURE EXTRACTION
# ============================================================

def extract_clip_features(video_path):
    """
    Samples frames for analysis only.
    Original video is never modified.

    Optical-flow magnitude is normalized by:
    image diagonal and elapsed time between sampled frames.

    Units are normalized image displacement per second,
    not physical fish swimming speed.
    """

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(
            f"Cannot open video: {video_path}"
        )

    try:
        fps = float(
            capture.get(cv2.CAP_PROP_FPS)
        )

        width = int(
            capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        if fps <= 0 or width <= 0 or height <= 0:
            raise ValueError(
                "Invalid video FPS or frame dimensions."
            )

        stride = max(
            1,
            int(round(fps / TARGET_SAMPLE_FPS)),
        )

        effective_fps = fps / stride
        time_interval = stride / fps

        processing_height = max(
            64,
            round(height * ANALYSIS_WIDTH / width),
        )

        diagonal = np.hypot(
            ANALYSIS_WIDTH,
            processing_height,
        )

        previous_gray = None
        sampled_frames = 0
        frame_index = 0

        observations = []
        brightness_values = []

        while True:
            success, frame = capture.read()

            if not success:
                break

            if frame_index % stride != 0:
                frame_index += 1
                continue

            resized = cv2.resize(
                frame,
                (ANALYSIS_WIDTH, processing_height),
                interpolation=cv2.INTER_AREA,
            )

            gray = cv2.cvtColor(
                resized,
                cv2.COLOR_BGR2GRAY,
            )

            sampled_frames += 1
            brightness_values.append(
                float(np.mean(gray) / 255.0)
            )

            if previous_gray is not None:

                flow = cv2.calcOpticalFlowFarneback(
                    previous_gray,
                    gray,
                    None,
                    0.5,   # pyramid scale
                    3,     # pyramid levels
                    15,    # window size
                    3,     # iterations
                    5,     # polynomial neighbourhood
                    1.2,   # Gaussian standard deviation
                    0,
                )

                magnitude = cv2.magnitude(
                    flow[:, :, 0],
                    flow[:, :, 1],
                )

                normalized_motion = (
                    magnitude / diagonal / time_interval
                )

                frame_difference = (
                    cv2.absdiff(
                        previous_gray,
                        gray,
                    ).astype(np.float32) / 255.0
                )

                h, w = normalized_motion.shape
                mid_y, mid_x = h // 2, w // 2

                quadrants = [
                    normalized_motion[:mid_y, :mid_x],
                    normalized_motion[:mid_y, mid_x:],
                    normalized_motion[mid_y:, :mid_x],
                    normalized_motion[mid_y:, mid_x:],
                ]

                quadrant_means = np.array(
                    [
                        float(np.mean(region))
                        for region in quadrants
                    ],
                    dtype=np.float64,
                )

                probability = (
                    quadrant_means /
                    max(float(quadrant_means.sum()), 1e-12)
                )

                entropy = -np.sum(
                    probability *
                    np.log(probability + 1e-12)
                ) / np.log(4.0)

                observations.append({
                    "mean": float(
                        np.mean(normalized_motion)
                    ),
                    "p90": float(
                        np.percentile(
                            normalized_motion, 90
                        )
                    ),
                    "std": float(
                        np.std(normalized_motion)
                    ),
                    "energy": float(
                        np.mean(normalized_motion ** 2)
                    ),
                    "entropy": float(entropy),
                    "q1": float(quadrant_means[0]),
                    "q2": float(quadrant_means[1]),
                    "q3": float(quadrant_means[2]),
                    "q4": float(quadrant_means[3]),
                    "difference": float(
                        np.mean(frame_difference)
                    ),
                })

            previous_gray = gray
            frame_index += 1

        if not observations:
            raise ValueError(
                "Insufficient decoded frames for optical flow."
            )

        movement = pd.DataFrame(observations)

        features = {
            "video_fps": fps,
            "sampled_fps": effective_fps,
            "sampled_frames": sampled_frames,
            "video_width": width,
            "video_height": height,

            "flow_mean_norm_s": movement["mean"].mean(),
            "flow_p90_norm_s": movement["p90"].mean(),
            "flow_std_norm_s": movement["std"].mean(),

            "flow_energy_norm_s2": (
                movement["energy"].mean()
            ),

            "flow_temporal_std": movement["mean"].std(
                ddof=0
            ),

            "flow_temporal_p90": np.percentile(
                movement["mean"], 90
            ),

            "spatial_motion_entropy": (
                movement["entropy"].mean()
            ),

            "flow_top_left_norm_s": (
                movement["q1"].mean()
            ),

            "flow_top_right_norm_s": (
                movement["q2"].mean()
            ),

            "flow_bottom_left_norm_s": (
                movement["q3"].mean()
            ),

            "flow_bottom_right_norm_s": (
                movement["q4"].mean()
            ),

            "frame_difference_mean": (
                movement["difference"].mean()
            ),

            "frame_difference_std": (
                movement["difference"].std(ddof=0)
            ),

            # Camera/light diagnostics, excluded from
            # the model's primary feature set.
            "brightness_mean": float(
                np.mean(brightness_values)
            ),

            "brightness_std": float(
                np.std(brightness_values)
            ),
        }

        for key, value in features.items():
            if not np.isfinite(value):
                raise ValueError(
                    f"Non-finite feature: {key}"
                )

        return features

    finally:
        capture.release()


def extract_dataset_features():
    print("\n[02/04] Extracting video motion features")

    manifest = pd.read_csv(MANIFEST_PATH)

    results = []
    failures = []

    for index, row in manifest.iterrows():

        path = Path(row["clip_path"])

        print(
            f"[{index + 1}/{len(manifest)}] "
            f"{row['clip_name']}"
        )

        try:
            features = extract_clip_features(path)

            results.append({
                **row.to_dict(),
                **features,
            })

        except Exception as exc:
            failures.append({
                "clip_uid": row["clip_uid"],
                "clip_name": row["clip_name"],
                "error": str(exc),
            })

            print(f"  ERROR: {exc}")

    save_csv(
        pd.DataFrame(
            failures,
            columns=[
                "clip_uid",
                "clip_name",
                "error",
            ],
        ),
        ERROR_PATH,
    )

    if not results:
        raise RuntimeError(
            "No video features could be extracted. "
            "See processing_errors.csv."
        )

    dataframe = pd.DataFrame(results)

    save_csv(dataframe, FEATURE_PATH)

    print(f"\nProcessed successfully: {len(results)}")
    print(f"Failed: {len(failures)}")
    print(f"Feature CSV: {FEATURE_PATH}")

    return dataframe


# ============================================================
# 05 | SELF-SUPERVISED AUTOENCODER
# ============================================================

class MotionDenoisingAutoencoder(nn.Module):

    def __init__(self, input_dim, latent_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding


def train_self_supervised_model():
    print("\n[03/04] Training self-supervised model")

    dataframe = pd.read_csv(FEATURE_PATH)

    baseline = dataframe.loc[
        dataframe["phase"] == "baseline"
    ].copy()

    if len(baseline) < 8:
        print(
            "SKIPPED: At least 8 processable baseline "
            "clips are required by this prototype."
        )
        print(
            "This is an engineering minimum, not a "
            "scientifically sufficient sample size."
        )
        return

    # Fit data preparation on baseline observations only.
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    baseline_values = imputer.fit_transform(
        baseline[FEATURE_COLUMNS]
    )

    baseline_values = scaler.fit_transform(
        baseline_values
    ).astype(np.float32)

    all_values = imputer.transform(
        dataframe[FEATURE_COLUMNS]
    )

    all_values = scaler.transform(
        all_values
    ).astype(np.float32)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    print("Baseline samples:", len(baseline))
    print("Features:", len(FEATURE_COLUMNS))

    model = MotionDenoisingAutoencoder(
        input_dim=len(FEATURE_COLUMNS),
        latent_dim=LATENT_DIM,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=SSL_LEARNING_RATE,
    )

    dataset = TensorDataset(
        torch.from_numpy(baseline_values)
    )

    loader = DataLoader(
        dataset,
        batch_size=min(
            SSL_BATCH_SIZE, len(dataset)
        ),
        shuffle=True,
    )

    history = []

    for epoch in range(1, SSL_EPOCHS + 1):

        model.train()

        epoch_losses = []

        for (batch,) in loader:

            batch = batch.to(device)

            # Self-supervised masked-feature reconstruction.
            # No hunger or feeding labels are used.
            mask = (
                torch.rand_like(batch)
                < SSL_MASK_PROBABILITY
            )

            masked_input = batch.masked_fill(
                mask, 0.0
            )

            reconstruction, _ = model(
                masked_input
            )

            loss = nn.functional.mse_loss(
                reconstruction,
                batch,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(float(loss.item()))

        epoch_loss = float(np.mean(epoch_losses))

        history.append({
            "epoch": epoch,
            "training_loss": epoch_loss,
        })

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d}/{SSL_EPOCHS} "
                f"| MSE: {epoch_loss:.6f}"
            )

    # Save trained model and preprocessing.
    joblib.dump(
        {
            "imputer": imputer,
            "scaler": scaler,
            "feature_columns": FEATURE_COLUMNS,
        },
        MODEL_DIR / "motion_preprocessor.joblib",
    )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": len(FEATURE_COLUMNS),
            "latent_dim": LATENT_DIM,
            "seed": SEED,
        },
        MODEL_DIR / "motion_autoencoder.pt",
    )

    save_csv(
        pd.DataFrame(history),
        RESEARCH_DIR / "ssl_training_history.csv",
    )

    # Evaluate learned representations on all clips.
    # No input masking during inference.
    model.eval()

    with torch.no_grad():

        tensor = torch.from_numpy(
            all_values
        ).to(device)

        reconstruction, embedding = model(tensor)

        reconstruction_error = torch.mean(
            (reconstruction - tensor) ** 2,
            dim=1,
        ).cpu().numpy()

        embedding_values = (
            embedding.cpu().numpy()
        )

    representation = dataframe[
        [
            "clip_uid",
            "clip_name",
            "phase",
            "recording_id",
            "session_id",
        ]
    ].copy()

    representation[
        "baseline_reconstruction_mse"
    ] = reconstruction_error

    for dimension in range(LATENT_DIM):
        representation[
            f"embedding_{dimension + 1}"
        ] = embedding_values[:, dimension]

    output_path = (
        RESEARCH_DIR / "behaviour_embeddings.csv"
    )

    save_csv(representation, output_path)

    print("\nSelf-supervised model saved.")
    print("Behaviour embeddings:", output_path)

    print(
        "\nNOTE: Reconstruction MSE measures similarity "
        "to the learned baseline representation; "
        "it is NOT a hunger probability."
    )


# ============================================================
# 06 | FEEDING-RESPONSE VALIDATION
# ============================================================

def validation_status(status, reason):
    result = {
        "status": status,
        "reason": reason,
        "target": "Observed feeding-response intensity",
        "is_hunger_ground_truth": False,
    }

    save_json(
        result,
        RESEARCH_DIR / "response_validation.json",
    )

    print(f"\nValidation {status}: {reason}")


def validate_feeding_response():

    print("\n[04/04] Checking feeding-response validation")

    features = pd.read_csv(FEATURE_PATH)

    annotations = pd.read_csv(
        ANNOTATION_PATH,
        dtype=str,
    ).fillna("")

    required = [
        "clip_uid",
        "session_id",
        "food_observed",
        "feeding_intensity",
    ]

    for column in required:
        if column not in annotations:
            raise ValueError(
                f"Missing annotation column: {column}"
            )

    if annotations["clip_uid"].duplicated().any():
        raise ValueError(
            "Duplicate clip identifiers in annotations."
        )

    joined = features.merge(
        annotations[required],
        on="clip_uid",
        how="left",
        suffixes=("", "_annotation"),
        validate="one_to_one",
    )

    valid_labels = {
        "none": "None",
        "weak": "Weak",
        "medium": "Medium",
        "strong": "Strong",
    }

    joined["target_class"] = (
        joined["feeding_intensity"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .map(valid_labels)
    )

    observed_food = (
        joined["food_observed"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["yes", "true", "1"])
    )

    candidates = joined.loc[
        (joined["phase"] == "feeding_recording")
        & observed_food
        & joined["target_class"].notna()
    ].copy()

    if candidates.empty:
        validation_status(
            "SKIPPED",
            "No verified four-class annotations available. "
            "Videos were not automatically assigned labels.",
        )
        return

    class_counts = (
        candidates["target_class"]
        .value_counts()
        .to_dict()
    )

    missing_classes = [
        category
        for category in FEEDING_CLASSES
        if class_counts.get(category, 0) == 0
    ]

    if missing_classes:
        validation_status(
            "SKIPPED",
            "Dataset does not yet contain all four "
            f"feeding-response classes: {missing_classes}",
        )
        return

    # User-editable session IDs come from annotations.
    candidates["evaluation_group"] = (
        candidates["session_id_annotation"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    if (
        candidates["evaluation_group"] == ""
    ).any():
        validation_status(
            "SKIPPED",
            "Every labeled clip needs a valid session_id.",
        )
        return

    # Each class must be represented in at least two
    # independent recording/session groups so leaving
    # one group out does not remove the class from training.
    groups_per_class = (
        candidates.groupby("target_class")[
            "evaluation_group"
        ].nunique()
    )

    insufficient = {
        category: int(
            groups_per_class.get(category, 0)
        )
        for category in FEEDING_CLASSES
        if groups_per_class.get(category, 0) < 2
    }

    if insufficient:
        validation_status(
            "SKIPPED",
            "Each response class needs examples from at "
            "least two independent sessions. "
            f"Insufficient group coverage: {insufficient}",
        )
        return

    x = candidates[FEATURE_COLUMNS].to_numpy(
        dtype=np.float64
    )

    y = candidates["target_class"].to_numpy()

    groups = candidates[
        "evaluation_group"
    ].to_numpy()

    validator = LeaveOneGroupOut()

    # The entire preprocessing pipeline is refitted
    # inside each training fold to prevent leakage.
    def make_classifier():
        return Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=SEED,
                ),
            ),
        ])

    predictions = np.empty(
        len(candidates), dtype=object
    )

    for train_index, test_index in validator.split(
        x, y, groups
    ):

        classifier = make_classifier()

        classifier.fit(
            x[train_index],
            y[train_index],
        )

        predictions[test_index] = classifier.predict(
            x[test_index]
        )

    report = {
        "status": "COMPLETED",
        "target": "Observed feeding-response intensity",
        "is_hunger_ground_truth": False,
        "evaluation": "Leave-One-Session-Out",
        "samples": int(len(y)),
        "independent_groups": int(
            len(np.unique(groups))
        ),
        "accuracy": float(
            accuracy_score(y, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y, predictions
            )
        ),
        "macro_f1": float(
            f1_score(
                y,
                predictions,
                labels=FEEDING_CLASSES,
                average="macro",
                zero_division=0,
            )
        ),
        "classes": FEEDING_CLASSES,
        "confusion_matrix": confusion_matrix(
            y,
            predictions,
            labels=FEEDING_CLASSES,
        ).tolist(),
        "classification_report": (
            classification_report(
                y,
                predictions,
                labels=FEEDING_CLASSES,
                output_dict=True,
                zero_division=0,
            )
        ),
    }

    save_json(
        report,
        RESEARCH_DIR / "response_validation.json",
    )

    prediction_table = candidates[
        [
            "clip_uid",
            "clip_name",
            "evaluation_group",
        ]
    ].copy()

    prediction_table["actual_class"] = y
    prediction_table["predicted_class"] = predictions

    save_csv(
        prediction_table,
        EVALUATION_DIR / "response_predictions.csv",
    )

    # Final exploratory response classifier:
    # trained on all verified examples after evaluation.
    final_classifier = make_classifier()
    final_classifier.fit(x, y)

    joblib.dump(
        {
            "pipeline": final_classifier,
            "feature_columns": FEATURE_COLUMNS,
            "classes": FEEDING_CLASSES,
            "target": "feeding_response_not_hunger",
        },
        EVALUATION_DIR
        / "feeding_response_model.joblib",
    )

    print("\nFeeding-response validation completed.")
    print(f"Samples: {len(y)}")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"Macro F1: {report['macro_f1']:.4f}")
    print(
        f"Balanced accuracy: "
        f"{report['balanced_accuracy']:.4f}"
    )

    print(
        "\nImportant: these are session-held-out "
        "feeding-response metrics, not proven "
        "physiological hunger-detection accuracy."
    )


# ============================================================
# 07 | APPLICATION ENTRY POINT
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "AquaFeedLab Phase 1: Goldfish behaviour "
            "representation and feeding-response research."
        )
    )

    parser.add_argument(
        "--stage",
        choices=[
            "all",
            "prepare",
            "extract",
            "ssl",
            "validate",
        ],
        default="all",
    )

    args = parser.parse_args()

    initialize_environment()

    print("=" * 55)
    print("AQUAFEEDLAB | PHASE 1 RESEARCH")
    print("=" * 55)
    print("Project:", PROJECT_ROOT)
    print("Stage:", args.stage)

    if args.stage in ["all", "prepare"]:
        prepare_dataset()

    if args.stage in ["all", "extract"]:
        extract_dataset_features()

    if args.stage in ["all", "ssl"]:
        train_self_supervised_model()

    if args.stage in ["all", "validate"]:
        validate_feeding_response()

    print("\n" + "=" * 55)
    print("PIPELINE FINISHED")
    print("=" * 55)
    print("Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
