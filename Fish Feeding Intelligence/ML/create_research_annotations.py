
"""
AquaFeedLab
Phase 1B — Research Annotation Manager

Creates a reviewed copy of the existing 288-row
feeding_annotations.csv without modifying the original.

Outputs:
- feeding_annotations_reviewed.csv
- annotation_review_queue.csv
- annotation_summary.json
- annotation_protocol.txt

No manual input during execution.
"""

from pathlib import Path
from collections import Counter
import argparse
import csv
import json


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(r"C:\Users\LENOVO\Desktop\Kk")

DATA_DIR = ROOT / "outputs"

INPUT_CSV = DATA_DIR / "feeding_annotations.csv"

OUTPUT_CSV = DATA_DIR / "feeding_annotations_reviewed.csv"

QUEUE_CSV = DATA_DIR / "annotation_review_queue.csv"

SUMMARY_JSON = DATA_DIR / "annotation_summary.json"

PROTOCOL_FILE = DATA_DIR / "annotation_protocol.txt"


# ============================================================
# PRELIMINARY REVIEW EVIDENCE
# ============================================================

REVIEWS = {}


def register_review(
    clip_name,
    review_status,
    candidate_class="",
    visibility="UNKNOWN",
    notes="",
):
    REVIEWS[clip_name] = {
        "review_status": review_status,
        "provisional_class": candidate_class,
        "visibility_quality": visibility,
        "review_notes": notes,
        "annotation_basis": "STORYBOARD_11_FRAMES",
    }


# ------------------------------------------------------------
# A. NO-FEEDING BASELINE CLIPS
# ------------------------------------------------------------

for name in [
    "clip001.mp4",
    "clip002.mp4",
    "clip003.mp4",
    "clip009.mp4",
    "clip010.mp4",
    "clip011.mp4",
]:
    register_review(
        clip_name=name,
        review_status="BASELINE_REFERENCE",
        visibility="USABLE",
        notes=(
            "Goldfish swimming visible in sampled frames. "
            "No direct feeding event established. "
            "Not a four-class feeding-intensity sample."
        ),
    )


# ------------------------------------------------------------
# B. AMBIGUOUS FEEDING RECORDING CLIPS
# ------------------------------------------------------------

for number in [9, 10, 11]:
    register_review(
        clip_name=f"feeding_0_clip-{number}.mp4",
        review_status="FOOD_INTERACTION_UNCONFIRMED",
        visibility="USABLE",
        notes=(
            "Fish swimming/group movement visible. "
            "Food capture or consumption cannot be "
            "confirmed from sampled storyboard frames. "
            "Requires continuous-video examination."
        ),
    )


# ------------------------------------------------------------
# C. PROVISIONAL FEEDING RESPONSE CANDIDATES
# ------------------------------------------------------------

register_review(
    clip_name="feeding_1_clip-1.mp4",
    review_status="PROVISIONAL_REVIEW",
    candidate_class="Medium",
    visibility="USABLE",
    notes=(
        "Local fish grouping near the lower tank. "
        "Medium is a preliminary visual candidate, "
        "not a verified feeding-response label. "
        "Confirm actual food interaction in video."
    ),
)

register_review(
    clip_name="feeding_1_clip-2.mp4",
    review_status="PROVISIONAL_REVIEW",
    candidate_class="Strong",
    visibility="FRAMING_CHANGE",
    notes=(
        "Dense fish grouping and movement. "
        "Strong is an unverified candidate. "
        "Camera framing changes near the end. "
        "Verify food interaction using original video."
    ),
)

register_review(
    clip_name="feeding_1_clip-3.mp4",
    review_status="PROVISIONAL_REVIEW",
    candidate_class="Medium",
    visibility="FRAMING_CHANGE",
    notes=(
        "Fish grouping visible. Medium is an "
        "unverified candidate. Camera framing and "
        "viewpoint change during this clip. "
        "Check original footage before labeling."
    ),
)


# ------------------------------------------------------------
# D. LOW-VISIBILITY CLIPS
# ------------------------------------------------------------

for number in [1, 2, 3]:
    register_review(
        clip_name=f"feeding_3_clip-{number}.mp4",
        review_status="QUALITY_REVIEW",
        visibility="LOW_VISIBILITY",
        notes=(
            "Low contrast/hazy appearance and "
            "reflections limit behavioural interpretation. "
            "Hold from supervised training until reviewed."
        ),
    )


# ============================================================
# INPUT / OUTPUT FUNCTIONS
# ============================================================

def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Input annotation CSV not found: {path}"
        )

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        columns = list(reader.fieldnames or [])

    return rows, columns


def write_csv(path, rows, columns):
    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=columns,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# BUILD COMPLETE REVIEW DATASET
# ============================================================

def build_reviewed_dataset():

    input_rows, original_columns = load_csv(INPUT_CSV)

    required_columns = {
        "clip_uid",
        "clip_name",
        "phase",
        "recording_id",
        "session_id",
        "food_observed",
        "feeding_intensity",
    }

    missing = required_columns - set(original_columns)

    if missing:
        raise ValueError(
            f"Missing required input columns: {sorted(missing)}"
        )

    identifiers = [
        row["clip_uid"] for row in input_rows
    ]

    if len(identifiers) != len(set(identifiers)):
        raise ValueError(
            "Duplicate clip_uid values found in the input."
        )

    additional_columns = [
        "review_status",
        "provisional_class",
        "visibility_quality",
        "review_notes",
        "annotation_basis",
        "verification_status",
        "training_eligible",
    ]

    output_columns = list(
        dict.fromkeys(
            original_columns + additional_columns
        )
    )

    reviewed_rows = []

    for original in input_rows:

        row = dict(original)

        clip_name = row["clip_name"]
        phase = row["phase"]

        review = REVIEWS.get(clip_name)

        if review is not None:
            row.update(review)

        else:
            row.update({
                "review_status": "NOT_REVIEWED",
                "provisional_class": "",
                "visibility_quality": "NOT_ASSESSED",
                "review_notes": "",
                "annotation_basis": "",
            })

        # Preserve existing verified user labels, if any.
        # A provisional class is NEVER copied into
        # feeding_intensity automatically.

        row["verification_status"] = (
            "NOT_VERIFIED"
        )

        row["training_eligible"] = "NO"

        if phase == "baseline":

            # Do not label no-food behaviour as None.
            row["provisional_class"] = ""

            if (
                row.get("feeding_intensity", "").strip()
                in ["None", "Weak", "Medium", "Strong"]
            ):
                print(
                    "[WARNING] Baseline contains a feeding "
                    f"intensity label: {clip_name}. "
                    "Please review this record."
                )

        reviewed_rows.append(row)

    return reviewed_rows, output_columns


# ============================================================
# REVIEW QUEUE
# ============================================================

def generate_review_queue(rows):

    priorities = {
        "PROVISIONAL_REVIEW": 1,
        "FOOD_INTERACTION_UNCONFIRMED": 2,
        "QUALITY_REVIEW": 3,
        "NOT_REVIEWED": 4,
        "BASELINE_REFERENCE": 5,
    }

    queue = []

    for row in rows:

        queue.append({
            "priority": priorities.get(
                row["review_status"], 9
            ),
            "clip_name": row["clip_name"],
            "phase": row["phase"],
            "recording_id": row["recording_id"],
            "review_status": row["review_status"],
            "provisional_class": row["provisional_class"],
            "visibility_quality": row["visibility_quality"],
            "review_notes": row["review_notes"],
        })

    queue.sort(
        key=lambda row: (
            row["priority"],
            row["recording_id"],
            row["clip_name"],
        )
    )

    return queue


# ============================================================
# RESEARCH ANNOTATION PROTOCOL
# ============================================================

PROTOCOL = """
AquaFeedLab — Phase 1B Annotation Protocol
==========================================

TARGET:
Observable feeding-response intensity, not direct
physiological hunger.

VALID FEEDING RESPONSE CLASSES:
None
Weak
Medium
Strong

BASELINE RULE:
A no-food recording is not automatically a None response.

NONE RULE:
Use None only when a food stimulus is independently
confirmed and no food-directed response is observed
during the evaluated interval.

PROVISIONAL REVIEW:
Candidate classes inferred from sparse storyboard frames
must not be used as verified ground-truth training labels.

FULL VIDEO VALIDATION:
Inspect continuous footage for evidence of:
- Food presence or introduction
- Food-directed approach
- Actual food interaction / ingestion
- Response persistence
- Visibility and camera-related confounding

Do not equate high motion or clustering alone with hunger.

RESEARCH QUALITY:
Use a written class rubric, independent reviewers where
possible, and record disagreements.

SPLITTING:
Keep all clips from one original recording/session in
the same train/test partition.

HUNGER LIMITATION:
Response classes represent observed feeding motivation,
not a validated direct physiological hunger measurement.
"""


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Overwrite an existing reviewed CSV. "
            "Do not use after manually editing the output."
        ),
    )

    args = parser.parse_args()

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if OUTPUT_CSV.exists() and not args.overwrite:
        raise FileExistsError(
            f"{OUTPUT_CSV} already exists. "
            "This safety check protects your manual edits. "
            "Use --overwrite only if you intend to replace it."
        )

    rows, columns = build_reviewed_dataset()

    write_csv(
        OUTPUT_CSV,
        rows,
        columns,
    )

    queue = generate_review_queue(rows)

    write_csv(
        QUEUE_CSV,
        queue,
        [
            "priority",
            "clip_name",
            "phase",
            "recording_id",
            "review_status",
            "provisional_class",
            "visibility_quality",
            "review_notes",
        ],
    )

    statuses = Counter(
        row["review_status"] for row in rows
    )

    phases = Counter(
        row["phase"] for row in rows
    )

    summary = {
        "project": "AquaFeedLab",
        "stage": "Phase 1B",
        "total_records": len(rows),
        "phase_counts": dict(phases),
        "review_status_counts": dict(statuses),
        "verified_training_labels_added": 0,
        "note": (
            "Provisional storyboard labels are not "
            "verified ground truth."
        ),
    }

    with SUMMARY_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)

    PROTOCOL_FILE.write_text(
        PROTOCOL.strip(),
        encoding="utf-8",
    )

    print("\n" + "=" * 55)
    print("AQUAFEEDLAB | PHASE 1B COMPLETED")
    print("=" * 55)

    print(f"Total annotation rows : {len(rows)}")
    print(f"Baseline records      : {phases['baseline']}")
    print(
        "Feeding records       : "
        f"{phases['feeding_recording']}"
    )

    print("\nReview status:")
    for status, count in sorted(statuses.items()):
        print(f"  {status}: {count}")

    print("\nCreated files:")
    print(OUTPUT_CSV)
    print(QUEUE_CSV)
    print(SUMMARY_JSON)
    print(PROTOCOL_FILE)

    print(
        "\nNo provisional labels were copied into "
        "the verified feeding_intensity column."
    )


if __name__ == "__main__":
    main()
