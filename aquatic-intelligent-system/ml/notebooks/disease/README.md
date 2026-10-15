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
