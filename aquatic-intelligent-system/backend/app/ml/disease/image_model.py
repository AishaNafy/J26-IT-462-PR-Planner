from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from app.core import config


class FishDiseaseClassifier:
    def __init__(self, model_path: Path, labels_path: Path):
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)
        self._session = None
        self._metadata: dict = {}

    def load(self) -> bool:
        if self._session is not None:
            return True
        if not self.model_path.is_file() or not self.labels_path.is_file():
            return False
        import onnxruntime

        self._metadata = json.loads(self.labels_path.read_text())
        self._session = onnxruntime.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
        return True

    def predict(self, image_bytes: bytes) -> dict:
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image = ImageOps.exif_transpose(image).convert("RGB")
        except Exception as exc:
            raise ValueError("Upload a valid JPEG or PNG image") from exc

        if not self.load():
            raise RuntimeError("Disease model is not trained yet; see ml/notebooks/disease/README.md")

        size = int(self._metadata.get("input_size", 224))
        scale = size * 1.143 / min(image.size)
        image = image.resize((max(size, round(image.width * scale)), max(size, round(image.height * scale))), Image.Resampling.BILINEAR)
        left, top = (image.width - size) // 2, (image.height - size) // 2
        image = image.crop((left, top, left + size, top + size))
        mean = np.asarray(self._metadata.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
        std = np.asarray(self._metadata.get("std", [0.229, 0.224, 0.225]), dtype=np.float32)
        tensor = (np.asarray(image, dtype=np.float32) / 255.0 - mean) / std
        tensor = tensor.transpose(2, 0, 1)[None, ...].astype(np.float32)

        input_name = self._session.get_inputs()[0].name
        logits = self._session.run(None, {input_name: tensor})[0][0]
        logits = logits - np.max(logits)
        probabilities = np.exp(logits) / np.exp(logits).sum()
        labels = self._metadata["labels"]
        index = int(np.argmax(probabilities))
        return {
            "label": labels[index],
            "confidence": round(float(probabilities[index]), 4),
            "probabilities": {label: round(float(prob), 4) for label, prob in zip(labels, probabilities)},
        }


_classifier = FishDiseaseClassifier(config.DISEASE_MODEL_PATH, config.DISEASE_LABELS_PATH)


def get_classifier() -> FishDiseaseClassifier:
    return _classifier