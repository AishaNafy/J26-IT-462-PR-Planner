# Fish Image Baseline

The prototype trains a two-class Healthy/Abnormal MobileNetV3-Small classifier from exactly 30 fish images in the supplied parquet training split. It uses 20 images for training, 4 for validation, and 6 for testing. It also exports a pretrained ImageNet fish-category gate to reject likely non-fish photos. That gate is best-effort and can reject close-up or poorly framed fish; retake those photos rather than treating a rejection as a health result. With 30 images, metrics are only a pipeline demonstration and must not be presented as clinical performance.

Install the training dependencies:

```bash
python3 -m pip install torch torchvision datasets pillow numpy onnx
```

Train from the local dataset (the dataset is not copied into this repository):

```bash
python3 ml/notebooks/disease/train_fish_disease.py \
	--parquet "/path/to/dataSet/train-00000-of-00001.parquet"
```

The script writes `fish_disease.onnx`, `fish_presence.onnx`, `labels.json`, and `metrics.json` to `ml/trained_models/disease/`. The FastAPI service reads both ONNX models and serves `POST /disease-detection/predict-image`.

The supplied [BD Fish & Shrimp Disease Dataset](https://huggingface.co/datasets/Saon110/bd-fish-disease-dataset) by Saon110 is licensed CC BY-NC-SA 4.0. Retain attribution, follow the share-alike terms for derivatives, and do not use it commercially without the dataset owner's separate permission.

## Seven-class fish model

To train a separate MobileNetV3-Small model on all seven fish classes in the training split, use the local train, valid, and test parquet files:

```bash
python3 ml/notebooks/disease/train_fish_disease_7class.py \
	--dataset-dir "/path/to/dataSet"
```

This fine-tunes from ImageNet weights, selects the best epoch using validation macro F1, reports metrics on the complete held-out fish test split, and then runs 30 stratified test images one at a time. It writes a PyTorch checkpoint, ONNX model, class labels, metrics, and the 30 predictions to `ml/trained_models/disease/fish_7class/`. This is separate from the two-class model used by the current API; it does not replace or change that app model. The 30-image result is a small evaluation sample, not evidence of clinical accuracy.

Install the training packages if they are not already available:

```bash
python3 -m pip install torch torchvision datasets pillow numpy onnx
```

This trained model is derived from the BD Fish & Shrimp Disease Dataset. Preserve the dataset attribution and CC BY-NC-SA 4.0 restrictions, including the non-commercial and share-alike terms.
