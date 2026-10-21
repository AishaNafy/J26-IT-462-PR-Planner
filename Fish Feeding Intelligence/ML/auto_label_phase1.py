from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


ROOT = Path(r"C:\Users\LENOVO\Desktop\Kk")
OUT = ROOT / "outputs"

FEATURES_PATH = OUT / "motion_features.csv"
EMBEDDINGS_PATH = OUT / "research" / "behaviour_embeddings.csv"
ANNOTATIONS_PATH = OUT / "feeding_annotations_reviewed.csv"

OUTPUT_PATH = OUT / "auto_label_suggestions.csv"


# ---------------------------------------------------------
# FEATURE COLUMNS
# ---------------------------------------------------------

MOTION_COLUMNS = [
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

EMBEDDING_COLUMNS = [
    "embedding_1",
    "embedding_2",
    "embedding_3",
    "embedding_4",
    "embedding_5",
    "embedding_6",
]


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

features = pd.read_csv(FEATURES_PATH)

embeddings = pd.read_csv(
    EMBEDDINGS_PATH
)

annotations = pd.read_csv(
    ANNOTATIONS_PATH,
    dtype=str
).fillna("")


# ---------------------------------------------------------
# MERGE
# ---------------------------------------------------------

data = features.merge(
    embeddings[
        ["clip_uid"]
        + EMBEDDING_COLUMNS
        + ["baseline_reconstruction_mse"]
    ],
    on="clip_uid",
    how="inner"
)

data = data.merge(
    annotations[
        [
            "clip_uid",
            "feeding_intensity",
            "verification_status",
            "training_eligible",
        ]
    ],
    on="clip_uid",
    how="left"
)


# ---------------------------------------------------------
# CHECK
# ---------------------------------------------------------

expected = len(features)

if len(data) != expected:
    raise RuntimeError(
        f"Merge mismatch. Expected {expected}, "
        f"got {len(data)}"
    )


# ---------------------------------------------------------
# MODEL FEATURES
# ---------------------------------------------------------

MODEL_COLUMNS = (
    MOTION_COLUMNS
    + EMBEDDING_COLUMNS
    + ["baseline_reconstruction_mse"]
)

X = data[MODEL_COLUMNS].astype(float)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ---------------------------------------------------------
# BASELINE REFERENCE CENTRE
# ---------------------------------------------------------

baseline_mask = (
    data["phase"] == "baseline"
)

baseline_X = X_scaled[
    baseline_mask.to_numpy()
]

baseline_center = baseline_X.mean(
    axis=0,
    keepdims=True
)

distance_from_baseline = (
    pairwise_distances(
        X_scaled,
        baseline_center
    ).ravel()
)

data["distance_from_baseline"] = (
    distance_from_baseline
)


# ---------------------------------------------------------
# ONLY FEEDING RECORDINGS
# ---------------------------------------------------------

feeding_mask = (
    data["phase"] == "feeding_recording"
)

feeding_data = data.loc[
    feeding_mask
].copy()

feeding_X = X_scaled[
    feeding_mask.to_numpy()
]


# ---------------------------------------------------------
# UNSUPERVISED 4-CLUSTER MODEL
# ---------------------------------------------------------

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=50,
)

cluster_id = kmeans.fit_predict(
    feeding_X
)

feeding_data["cluster_id"] = cluster_id


# ---------------------------------------------------------
# CLUSTER CONFIDENCE
# ---------------------------------------------------------

distances = kmeans.transform(
    feeding_X
)

nearest = np.sort(
    distances,
    axis=1
)

d1 = nearest[:, 0]
d2 = nearest[:, 1]

confidence = (
    1.0 - (d1 / (d2 + 1e-8))
)

confidence = np.clip(
    confidence,
    0,
    1
)

feeding_data[
    "cluster_confidence"
] = confidence


# ---------------------------------------------------------
# CLUSTER BEHAVIOUR SCORE
#
# IMPORTANT:
# This is NOT a hunger score.
# It represents how different the learned behaviour is
# from the baseline behavioural distribution.
# ---------------------------------------------------------

cluster_summary = (
    feeding_data
    .groupby("cluster_id")
    .agg(
        mean_baseline_distance=(
            "distance_from_baseline",
            "mean"
        ),
        mean_reconstruction_mse=(
            "baseline_reconstruction_mse",
            "mean"
        ),
        clips=(
            "clip_uid",
            "count"
        )
    )
    .reset_index()
)


# ---------------------------------------------------------
# CHECK VERIFIED MEDIUM / STRONG LABELS
# ---------------------------------------------------------

verified = feeding_data[
    (
        feeding_data[
            "verification_status"
        ].str.upper()
        == "VERIFIED"
    )
    &
    (
        feeding_data[
            "feeding_intensity"
        ].isin(
            ["Medium", "Strong"]
        )
    )
].copy()


verified_cluster_labels = {}

for cluster in sorted(
    feeding_data["cluster_id"].unique()
):

    subset = verified[
        verified["cluster_id"] == cluster
    ]

    if len(subset) == 0:
        continue

    counts = (
        subset[
            "feeding_intensity"
        ].value_counts()
    )

    verified_cluster_labels[
        cluster
    ] = counts.index[0]


# ---------------------------------------------------------
# ORDER CLUSTERS BY BASELINE DIFFERENCE
# ---------------------------------------------------------

ordered_clusters = (
    cluster_summary
    .sort_values(
        [
            "mean_baseline_distance",
            "mean_reconstruction_mse",
        ]
    )
    ["cluster_id"]
    .tolist()
)


# ---------------------------------------------------------
# SAFE PROVISIONAL NAMES
#
# We do NOT call the first group "None" automatically.
# They are behaviour-response candidates.
# ---------------------------------------------------------

default_names = [
    "LOW_RESPONSE_CANDIDATE",
    "WEAK_RESPONSE_CANDIDATE",
    "MEDIUM_RESPONSE_CANDIDATE",
    "STRONG_RESPONSE_CANDIDATE",
]

cluster_name = {
    cluster: default_names[i]
    for i, cluster
    in enumerate(ordered_clusters)
}


# ---------------------------------------------------------
# USE VERIFIED EVIDENCE WHERE AVAILABLE
# ---------------------------------------------------------

for cluster, label in (
    verified_cluster_labels.items()
):

    if label == "Medium":
        cluster_name[cluster] = (
            "MEDIUM_RESPONSE_CANDIDATE"
        )

    elif label == "Strong":
        cluster_name[cluster] = (
            "STRONG_RESPONSE_CANDIDATE"
        )


feeding_data[
    "auto_suggested_label"
] = feeding_data[
    "cluster_id"
].map(cluster_name)


# ---------------------------------------------------------
# REVIEW PRIORITY
# ---------------------------------------------------------

def review_status(row):

    c = row["cluster_confidence"]

    if c >= 0.60:
        return "HIGH_CONFIDENCE_SUGGESTION"

    elif c >= 0.35:
        return "MEDIUM_CONFIDENCE_REVIEW"

    return "LOW_CONFIDENCE_REVIEW_REQUIRED"


feeding_data[
    "auto_review_status"
] = feeding_data.apply(
    review_status,
    axis=1
)


# ---------------------------------------------------------
# NEVER AUTO VERIFY
# ---------------------------------------------------------

feeding_data[
    "auto_verification_status"
] = "AUTO_SUGGESTED"

feeding_data[
    "auto_training_eligible"
] = "NO"


# ---------------------------------------------------------
# SORT
# ---------------------------------------------------------

feeding_data = feeding_data.sort_values(
    [
        "cluster_id",
        "cluster_confidence",
    ],
    ascending=[
        True,
        False,
    ]
)


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

output_columns = [
    "clip_uid",
    "clip_name",
    "recording_id",
    "session_id",

    "cluster_id",
    "auto_suggested_label",
    "cluster_confidence",

    "distance_from_baseline",
    "baseline_reconstruction_mse",

    "auto_review_status",
    "auto_verification_status",
    "auto_training_eligible",

    "feeding_intensity",
    "verification_status",
]


feeding_data[
    output_columns
].to_csv(
    OUTPUT_PATH,
    index=False
)


# ---------------------------------------------------------
# PRINT RESULTS
# ---------------------------------------------------------

print()
print("=" * 60)
print("AQUAFEEDLAB | AUTOMATIC BEHAVIOUR LABEL SUGGESTION")
print("=" * 60)

print(
    f"Feeding clips analysed : "
    f"{len(feeding_data)}"
)

print()
print("Cluster summary:")
print(cluster_summary.to_string(index=False))

print()
print("Suggested groups:")

for cluster in ordered_clusters:

    n = (
        feeding_data[
            "cluster_id"
        ] == cluster
    ).sum()

    print(
        f"Cluster {cluster}: "
        f"{cluster_name[cluster]} "
        f"({n} clips)"
    )

print()
print(
    "Verified labels were NOT overwritten."
)

print(
    "AUTO_SUGGESTED labels are NOT "
    "ground-truth labels."
)

print()
print(
    "Saved:",
    OUTPUT_PATH
)