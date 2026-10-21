# AquaFeed Vision Intelligence
 
**An experimental computer-vision platform for goldfish behaviour analysis and feeding-motivation research.**
 
> Report version 2.0 | 19 October 2026
 
AquaFeed Vision Intelligence observes goldfish from videos, detects visible fish, measures movement, and investigates whether behaviour may provide useful evidence about feeding motivation. It combines conventional image processing, self-supervised learning, unsupervised clustering, a separately trained YOLO object detector, and a Python web backend.
 
> ⚠️ **Research prototype.** The project currently measures visible movement, learned behaviour difference, and numerical group membership. A high movement score or a high Autoencoder reconstruction error does **not** demonstrate physiological hunger. The "Hungry-Likely" labels are exploratory, unvalidated descriptions, not verified feeding decisions.
 
---
 
## Table of Contents
 
- [Features](#features)
- [Project Status](#project-status)
- [System Architecture](#system-architecture)
- [Dataset](#dataset)
- [Phase 1: Motion Feature Extraction](#phase-1-motion-feature-extraction)
- [Phase 1: Self-Supervised Autoencoder](#phase-1-self-supervised-autoencoder)
- [Phase 2: Behaviour Grouping and Comparative Index](#phase-2-behaviour-grouping-and-comparative-index)
- [YOLO26 Goldfish Detection](#yolo26-goldfish-detection)
- [Web Application](#web-application)
- [User Workflow](#user-workflow)
- [Results](#results)
- [Repository Files](#repository-files)
- [Technology Stack](#technology-stack)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Conclusion](#conclusion)
---
 
## Features
 
- Video upload and approximately 30-second browser camera recording
- YOLO-based detection and counting of visible goldfish, with bounding boxes
- Adjustable YOLO detection-confidence threshold
- Scene-motion analysis using Farneback optical flow and frame differencing (13 motion features)
- Baseline-trained masked denoising Autoencoder (six-dimensional embeddings and reconstruction error)
- Exploratory K-Means behaviour grouping and a rank-based comparative movement index
- Dashboard with fish-count summary, movement-over-time chart, activity (Quickness) indicator, and uncertainty notes
- Feeding schedule reminders and manual feeding history ("Feed Now / Mark as Fed")
- Downloadable JSON and human-readable reports
## Project Status
 
**Completed or prototyped**
 
- Goldfish video collection, FFmpeg segmentation, metadata manifest
- 13-feature optical-flow extraction
- 100-epoch masked denoising Autoencoder training and six-dimensional embedding export
- Exploratory K-Means grouping, comparative deviation index, candidate feeding-response labels
- YOLO training workflow and YOLO-enabled fish-detection web application
- Video upload and browser capture, results dashboard
- Feeding schedule reminders and manual feeding history
**Not yet verified or completed**
 
- Independent four-class feeding-response validation
- Verified Hungry / Not Hungry accuracy
- Calibrated fish swimming speed
- Stable unique fish identities from tracking
- Scientific feeding-dose decisions
- Raspberry Pi optimisation and controlled physical food dispensing
## System Architecture
 
The behaviour pipeline and the YOLO detection pipeline are separate modules joined at the backend and user interface.
 
- **Motion branch:** analyses changes in the overall video and converts them into numerical statistics.
- **YOLO branch:** locates individual visible goldfish in sampled frames and provides bounding boxes and visible counts.
Neither branch alone can establish hunger. Future work can combine YOLO tracks with optical-flow features for more meaningful per-fish measurements.
 
<!-- Add your architecture figure here, e.g.: -->
<!-- ![System architecture](docs/figure1-architecture.png) -->
 
## Dataset
 
| Item | Value |
|---|---|
| Total clips analysed | 288 (approx. 30 seconds each) |
| Baseline / no-feeding clips | 53 |
| Feeding-related clips | 235 |
| Fish in observed aquarium | approx. 6 goldfish |
 
- Feeding-related clips come from recordings `feeding_0` to `feeding_9`; baseline clips were renamed sequentially (`clip001`, `clip002`, …).
- Clips were produced with FFmpeg/FFprobe (H.264, original dimensions retained) and are **not** independent samples.
- `feeding_8` (95 clips) and `feeding_5` (67 clips) contribute about 68.9% of feeding-related clips, so whole recording sessions must be held out during validation.
## Phase 1: Motion Feature Extraction
 
- Targets about five sampled frames per second, resized to 320 px width (aspect ratio preserved), converted to grayscale.
- Farneback optical flow estimates apparent pixel movement. It measures **scene motion**, not confirmed individual fish speed.
- Frame is divided into four quadrants for spatial analysis and spatial entropy.
- Brightness and frame diagnostics are recorded but are not Autoencoder inputs.
**The 13 motion features**
 
| # | Feature | # | Feature |
|---|---|---|---|
| 01 | `flow_mean_norm_s` | 08 | `flow_top_left_norm_s` |
| 02 | `flow_p90_norm_s` | 09 | `flow_top_right_norm_s` |
| 03 | `flow_std_norm_s` | 10 | `flow_bottom_left_norm_s` |
| 04 | `flow_energy_norm_s2` | 11 | `flow_bottom_right_norm_s` |
| 05 | `flow_temporal_std` | 12 | `frame_difference_mean` |
| 06 | `flow_temporal_p90` | 13 | `frame_difference_std` |
| 07 | `spatial_motion_entropy` | | |
 
## Phase 1: Self-Supervised Autoencoder
 
A masked denoising Autoencoder is trained on baseline clips only, because verified Hungry / Not Hungry labels are not available.
 
- Architecture: 13 inputs → encoder (64, 32) → 6-value latent → decoder (32, 64) → 13 outputs
- Training: 100 epochs, batch size 16, learning rate 0.001, masking probability 0.20, latent dimension 6, seed 42 (PyTorch)
- Outputs: `embedding_1` … `embedding_6` and `baseline_reconstruction_mse` per clip
- Reconstruction error is an unfamiliarity / anomaly-related signal, **not** a hunger probability.
## Phase 2: Behaviour Grouping and Comparative Index
 
**Automatic grouping (Phase 2A)**
 
- 13 motion values + 6 embeddings + reconstruction MSE = 20 inputs, scaled with `StandardScaler`
- K-Means with `n_clusters=4`, `random_state=42`, `n_init=50`
- Clusters are ordered by distance from the baseline reference and named `LOW_`, `WEAK_`, `MEDIUM_`, `STRONG_RESPONSE_CANDIDATE`
- Labels are marked `AUTO_SUGGESTED` and are not verified ground truth
**Comparative index (Phase 2B)**
 
- `RobustScaler`, percentile ranks of baseline distance and reconstruction error, averaged with equal weight
- K-Means (k=3) names: `NOT_HUNGRY_LIKELY`, `POSSIBLY_HUNGRY`, `HUNGRY_LIKELY`
- An index of 85 means a high **relative rank**, not that the fish is 85% hungry
## YOLO26 Goldfish Detection
 
- Detects visible goldfish and draws bounding boxes with detection confidence
- Training log: 1,803 training images (including 33 background images) and 36 validation images
- Training script settings: pretrained `yolo26s.pt`, `imgsz=640`, `batch=1`, `epochs=150`, `patience=30`, `workers=0`, mixed precision, GPU device 0
- YOLO26 Large ran out of GPU memory on a 6 GB device, so the Small checkpoint was used
- **Note:** the script is named `train_goldfish_yolo26l.py` but loads `yolo26s.pt`; confirm the architecture inside the final `best.pt`
- Fish count is the median, minimum and maximum visible detections across sampled frames; overlap and occlusion can cause under- or over-counting
## Web Application
 
- **Frontend:** HTML, CSS, JavaScript (browser recording via MediaRecorder API)
- **Backend:** Python FastAPI served with Uvicorn
- A local FFmpeg installation may be needed to convert WebM recordings
- Recorded clips are analysed after capture (near-live capture, not continuous real-time inference)
- The confidence slider changes only the YOLO threshold; it does not affect the Autoencoder or hunger detection
- The Quickness Index is a relative scene-motion percentile compared with baseline, not cm/s and not a hunger percentage
- Health Status remains **Not Assessed**
## User Workflow
 
1. Start the local Python server and open the dashboard
2. Upload a goldfish video or record about 30 seconds from a camera
3. Backend checks duration and format, then samples frames
4. OpenCV extracts the 13 movement features (Farneback optical flow)
5. Compare against baseline statistics; report reconstruction error if an Autoencoder artifact is compatible
6. YOLO draws boxes around visible goldfish
7. Dashboard shows fish count, motion activity, timeline, explanations and uncertainty notes
8. Download the report, adjust the confidence slider, or record that the fish were fed
## Results
 
| Measure | Value |
|---|---|
| Dataset rows / motion-feature rows | 288 / 288, no processing failures |
| Mean normalised scene motion (baseline / feeding) | ≈ 0.009461 / ≈ 0.007702 |
| Mean motion energy (baseline / feeding) | ≈ 0.000664 / ≈ 0.000409 |
| Autoencoder training loss | ≈ 0.9649 (epoch 1) → ≈ 0.1201 (epoch 100), about 87.6% reduction |
| Mean reconstruction MSE (baseline / feeding) | 0.1216 / 0.3894 (about 3.2×) |
| Reviewed annotations | 288 rows; 273 not yet reviewed; 0 verified training labels |
| Supervised response validation | SKIPPED (no verified labels) |
| YOLO precision / recall / mAP | Not reported (no held-out metrics supplied) |
 
These numbers describe processing success and exploratory comparisons. They are **not** classification accuracy, and no hunger-classification accuracy is claimed.
 
## Repository Files
 
**Key research artifacts**
 
| File | Purpose |
|---|---|
| `dataset_manifest.csv` | Clip identifiers, source paths, recording groups |
| `feeding_annotations.csv` | Initial annotation template |
| `motion_features.csv` | 13 motion features and recording diagnostics |
| `processing_errors.csv` | Video-processing errors |
| `motion_preprocessor.joblib` | Fitted preprocessing artifact |
| `motion_autoencoder.pt` | Trained self-supervised model checkpoint |
| `behaviour_embeddings.csv` | Six embeddings and reconstruction error per clip |
| `ssl_training_history.csv` | Autoencoder loss over epochs |
| `response_validation.json` | Optional supervised validation outcome |
| `feeding_annotations_reviewed.csv` | Provisional review details and eligibility fields |
| `auto_label_suggestions.csv` | Cluster membership, suggested labels, separation score |
| `hunger_likelihood_predictions.csv` | Exploratory clip-level comparative index and score group |
| `session_hunger_summary.csv` | Exploratory aggregation of clip scores by recording |
| `best.pt` | YOLO detector checkpoint |
 
**Main Python scripts**
 
| Script | Purpose |
|---|---|
| `dataset_analysis.py` | Segments baseline source recordings into clips |
| `data_analysis_Feeding.py` | Segments feeding recordings and writes a clip manifest |
| `create_review_storyboards.py` | Creates timestamped image grids for visual review |
| `phase1_pipeline.py` | Registers inputs, extracts optical-flow features, trains the Autoencoder, exports embeddings, optional supervised validation |
| `create_research_annotations.py` | Builds review tables and protocol without creating verified labels |
| `auto_label_phase1.py` | Merges features and embeddings, fits K-Means, creates candidate labels |
| `phase2_hunger_index.py` | Exploratory comparative indices and session summaries |
| `train_goldfish_yolo26l.py` | YOLO fine-tuning configuration (loads `yolo26s.pt`) |
 
## Technology Stack
 
| Area | Tools |
|---|---|
| Video preparation | FFmpeg, FFprobe |
| Image processing | OpenCV, NumPy |
| Tabular data | Pandas |
| Self-supervised learning | PyTorch |
| Clustering and preprocessing | scikit-learn, joblib |
| Object detection | Ultralytics YOLO |
| Backend | Python, FastAPI, Uvicorn |
| Frontend | HTML, CSS, JavaScript |
| Data exchange | JSON, CSV |
| Development | Windows, Conda environments |
 
## Limitations
 
- **Baseline is not "Not Hungry":** a no-feeding recording does not establish physiological hunger.
- **Feeding recordings are not always active feeding:** labelling every feeding clip "Strong" would create invalid targets.
- **High difference is not hunger:** reconstruction MSE, baseline distance, image motion and cluster assignment are mathematical measurements of visual behaviour.
- **Confounders:** lighting, reflections, cloudy water, tank geometry, camera viewpoint, frame rate, water flow and fish overlap can affect measurements.
- **Generalisation:** data come from one small group of goldfish; other fish, tanks, cameras or species need independent evaluation.
- **Reproducibility:** the same feature formulas, sampling rate, preprocessing and feature order must be used at training and inference.
- **Welfare and safety:** feeding reminders are schedule-based, not hunger detections. Do not automatically dispense food from the current exploratory score; a hardware feeder needs guardrails, manual override, dose limits and independent testing.
- The Phase 2 clusters and percentile ranks are fitted on the analysed dataset; they are not yet a frozen inference model for arbitrary new videos.
## Roadmap
 
| Phase | Goal |
|---|---|
| 3A | Verify YOLO detections (precision, recall, mAP, count error) and add tracking |
| 3B | Individual movement features: trajectories, speed, acceleration, direction change, spacing; calibrate camera/tank for cm/s |
| 3C | Controlled feeding-response dataset: pre-food, known food introduction, response, post-feeding periods per independent session |
| 3D | Verified target and supervised learning with written rubric and independent session splits |
| 3E | Real-time inference and Raspberry Pi: freeze artifacts, verify feature parity, benchmark latency and memory |
| 3F | Safe feeding recommendations, only after model validation, with human supervision and safe dispensing limits |
 
## Conclusion
 
AquaFeed Vision Intelligence demonstrates a coherent prototype pipeline linking goldfish video collection, optical-flow movement features, baseline Autoencoder learning, exploratory K-Means analysis, goldfish object detection, and a usable web dashboard.
 
The project does **not** yet establish whether an arbitrary goldfish is hungry. That requires experimentally linked pre-feeding recordings, independent feeding-response targets, and robust validation, ideally combining YOLO tracking, controlled response collection and session-level supervised evaluation before any automated feeding decision is made.
 
---
 
*This README describes the project scripts and previously inspected output summaries. No hunger accuracy, YOLO mAP, calibrated swimming speed, or feeder-control results have been added.*
