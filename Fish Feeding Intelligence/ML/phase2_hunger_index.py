from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans


ROOT = Path(r"C:\Users\LENOVO\Desktop\Kk")
OUT = ROOT / "outputs"

AUTO_LABELS = OUT / "auto_label_suggestions.csv"
EMBEDDINGS = OUT / "research" / "behaviour_embeddings.csv"

OUTPUT = OUT / "hunger_likelihood_predictions.csv"
SESSION_OUTPUT = OUT / "session_hunger_summary.csv"


# ==========================================================
# LOAD
# ==========================================================

auto = pd.read_csv(AUTO_LABELS)
emb = pd.read_csv(EMBEDDINGS)

data = auto.merge(
    emb[
        [
            "clip_uid",
            "embedding_1",
            "embedding_2",
            "embedding_3",
            "embedding_4",
            "embedding_5",
            "embedding_6",
        ]
    ],
    on="clip_uid",
    how="left",
)


# ==========================================================
# INPUT FEATURES
# ==========================================================

feature_columns = [
    "distance_from_baseline",
    "baseline_reconstruction_mse",
    "embedding_1",
    "embedding_2",
    "embedding_3",
    "embedding_4",
    "embedding_5",
    "embedding_6",
]

X = (
    data[feature_columns]
    .astype(float)
    .replace([np.inf, -np.inf], np.nan)
)

X = X.fillna(
    X.median(numeric_only=True)
)


# ==========================================================
# ROBUST NORMALIZATION
# ==========================================================

scaler = RobustScaler()

X_scaled = scaler.fit_transform(X)


# ==========================================================
# DISCOVER TWO HIGH-LEVEL STATES
#
# This is data-driven:
# State A = lower feeding-motivation behaviour
# State B = higher feeding-motivation behaviour
# ==========================================================

state_model = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=50,
)

states = state_model.fit_predict(X_scaled)

data["latent_state"] = states


# ==========================================================
# DETERMINE WHICH STATE IS HIGHER RESPONSE
# ==========================================================

state_statistics = (
    data
    .groupby("latent_state")
    .agg(
        mean_baseline_distance=(
            "distance_from_baseline",
            "mean"
        ),
        mean_reconstruction_error=(
            "baseline_reconstruction_mse",
            "mean"
        ),
    )
)

state_statistics["response_index"] = (
    state_statistics["mean_baseline_distance"]
    +
    state_statistics["mean_reconstruction_error"]
)

high_state = (
    state_statistics["response_index"]
    .idxmax()
)


# ==========================================================
# CONTINUOUS DATA-DRIVEN SCORE
#
# IMPORTANT:
# This is a feeding-motivation / hunger-likelihood proxy.
# It is not a physiological hunger measurement.
# ==========================================================

distance = (
    data["distance_from_baseline"]
    .astype(float)
    .to_numpy()
)

reconstruction = (
    data["baseline_reconstruction_mse"]
    .astype(float)
    .to_numpy()
)


def percentile_score(values):

    values = np.asarray(values)

    order = np.argsort(values)

    ranks = np.empty_like(
        order,
        dtype=float,
    )

    ranks[order] = np.arange(len(values))

    if len(values) == 1:
        return np.array([50.0])

    return (
        ranks /
        (len(values) - 1)
    ) * 100


distance_score = percentile_score(distance)
reconstruction_score = percentile_score(reconstruction)


# Equal weighting.
# No biologically invented weight is introduced.

raw_score = (
    distance_score
    + reconstruction_score
) / 2


# ==========================================================
# STATE ORIENTATION
# ==========================================================

state_high_mask = (
    data["latent_state"].to_numpy()
    == high_state
)

# Keep continuous rank score,
# but use latent state as independent structural evidence.

data["feeding_motivation_score"] = (
    raw_score
)


# ==========================================================
# AUTOMATIC 3-STATE DISCOVERY
#
# Instead of manually choosing thresholds such as 30 or 70,
# let the distribution form three score groups.
# ==========================================================

score_model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=50,
)

score_clusters = score_model.fit_predict(
    raw_score.reshape(-1, 1)
)

centres = (
    score_model.cluster_centers_
    .ravel()
)

order = np.argsort(centres)

label_map = {
    order[0]: "NOT_HUNGRY_LIKELY",
    order[1]: "POSSIBLY_HUNGRY",
    order[2]: "HUNGRY_LIKELY",
}

data["predicted_hunger_state"] = [
    label_map[x]
    for x in score_clusters
]


# ==========================================================
# MODEL CONFIDENCE
# ==========================================================

cluster_distances = score_model.transform(
    raw_score.reshape(-1, 1)
)

sorted_distances = np.sort(
    cluster_distances,
    axis=1,
)

nearest = sorted_distances[:, 0]
second = sorted_distances[:, 1]

confidence = (
    1 -
    nearest /
    (second + 1e-8)
)

confidence = np.clip(
    confidence,
    0,
    1,
)

data["prediction_confidence"] = confidence


# ==========================================================
# RESEARCH LABEL
# ==========================================================

data["prediction_type"] = (
    "VIDEO_FEEDING_MOTIVATION_PROXY"
)

data["physiological_hunger_ground_truth"] = False


# ==========================================================
# SESSION SMOOTHING
#
# Multiple adjacent 30-second clips are not treated as
# independent biological observations.
# ==========================================================

session_summary = (
    data
    .groupby(
        ["recording_id", "session_id"],
        as_index=False,
    )
    .agg(
        clips=(
            "clip_uid",
            "count"
        ),
        hunger_score_mean=(
            "feeding_motivation_score",
            "mean"
        ),
        hunger_score_median=(
            "feeding_motivation_score",
            "median"
        ),
        confidence_mean=(
            "prediction_confidence",
            "mean"
        ),
    )
)


# ==========================================================
# SESSION STATE
#
# Again use distribution-driven clustering.
# ==========================================================

if len(session_summary) >= 3:

    session_model = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=50,
    )

    sc = session_model.fit_predict(
        session_summary[
            ["hunger_score_median"]
        ].to_numpy()
    )

    session_centres = (
        session_model
        .cluster_centers_
        .ravel()
    )

    session_order = np.argsort(
        session_centres
    )

    session_map = {
        session_order[0]:
            "NOT_HUNGRY_LIKELY",

        session_order[1]:
            "POSSIBLY_HUNGRY",

        session_order[2]:
            "HUNGRY_LIKELY",
    }

    session_summary[
        "session_hunger_state"
    ] = [
        session_map[x]
        for x in sc
    ]

else:

    session_summary[
        "session_hunger_state"
    ] = "INSUFFICIENT_SESSIONS"


# ==========================================================
# SAVE
# ==========================================================

result_columns = [
    "clip_uid",
    "clip_name",
    "recording_id",
    "session_id",

    "auto_suggested_label",

    "distance_from_baseline",
    "baseline_reconstruction_mse",

    "feeding_motivation_score",
    "predicted_hunger_state",
    "prediction_confidence",

    "prediction_type",
    "physiological_hunger_ground_truth",
]

data[result_columns].to_csv(
    OUTPUT,
    index=False,
)

session_summary.to_csv(
    SESSION_OUTPUT,
    index=False,
)


# ==========================================================
# PRINT SUMMARY
# ==========================================================

print()
print("=" * 65)
print("AQUAFEEDLAB | PHASE 2 HUNGER-LIKELIHOOD ESTIMATION")
print("=" * 65)

print()
print("Clips analysed:", len(data))

print()
print("Clip predictions:")
print(
    data[
        "predicted_hunger_state"
    ].value_counts()
)

print()
print("Session predictions:")
print(
    session_summary[
        [
            "recording_id",
            "clips",
            "hunger_score_median",
            "session_hunger_state",
        ]
    ].to_string(index=False)
)

print()
print("Saved:")
print(OUTPUT)
print(SESSION_OUTPUT)

print()
print(
    "IMPORTANT: The current output is a video-based "
    "feeding-motivation proxy, not validated physiological "
    "hunger ground truth."
)