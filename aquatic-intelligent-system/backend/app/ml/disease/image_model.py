from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from app.core import config


class FishDiseaseClassifier:
    def __init__(
        self,
        model_path: Path,
        labels_path: Path,
        seven_class_model_path: Path | None = None,
        seven_class_labels_path: Path | None = None,
    ):
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)
        self.seven_class_model_path = Path(
            seven_class_model_path
            or self.model_path.with_name("fish_7class") / "fish_disease_7class.onnx"
        )
        self.seven_class_labels_path = Path(
            seven_class_labels_path or self.seven_class_model_path.with_name("labels.json")
        )
        self._session = None
        self._fish_gate = None
        self._seven_class_session = None
        self._metadata: dict = {}
        self._seven_class_metadata: dict = {}

    def load(self) -> bool:
        if self._seven_class_session is None:
            if self.seven_class_model_path.is_file() and self.seven_class_labels_path.is_file():
                import onnxruntime

                self._seven_class_metadata = json.loads(self.seven_class_labels_path.read_text())
                self._seven_class_session = onnxruntime.InferenceSession(
                    str(self.seven_class_model_path), providers=["CPUExecutionProvider"]
                )

        fish_gate_path = self.model_path.with_name("fish_presence.onnx")
        if self._session is None and self._fish_gate is None:
            if self.model_path.is_file() and fish_gate_path.is_file() and self.labels_path.is_file():
                import onnxruntime

                self._metadata = json.loads(self.labels_path.read_text())
                self._session = onnxruntime.InferenceSession(
                    str(self.model_path), providers=["CPUExecutionProvider"]
                )
                self._fish_gate = onnxruntime.InferenceSession(
                    str(fish_gate_path), providers=["CPUExecutionProvider"]
                )

        if self._seven_class_session is None and (self._session is None or self._fish_gate is None):
            return False

        return True

    @staticmethod
    def _display_label(label: str) -> str:
        return label.removeprefix("Fish_")

    @staticmethod
    def _is_healthy_label(label: str) -> bool:
        return "healthy" in label.lower()

    def predict(self, image_bytes: bytes) -> dict:
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image = ImageOps.exif_transpose(image).convert("RGB")
        except Exception as exc:
            raise ValueError("Upload a valid JPEG or PNG image") from exc

        if not self.load():
            raise RuntimeError("Image models are not trained yet; see ml/notebooks/disease/README.md")

        metadata = self._seven_class_metadata if self._seven_class_session is not None else self._metadata
        size = int(metadata.get("input_size", 224))
        scale = size * 1.143 / min(image.size)
        image = image.resize((max(size, round(image.width * scale)), max(size, round(image.height * scale))), Image.Resampling.BILINEAR)
        left, top = (image.width - size) // 2, (image.height - size) // 2
        image = image.crop((left, top, left + size, top + size))
        mean = np.asarray(metadata.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
        std = np.asarray(metadata.get("std", [0.229, 0.224, 0.225]), dtype=np.float32)
        tensor = (np.asarray(image, dtype=np.float32) / 255.0 - mean) / std
        tensor = tensor.transpose(2, 0, 1)[None, ...].astype(np.float32)

        fish_confidence = None
        if self._fish_gate is not None:
            fish_input = self._fish_gate.get_inputs()[0].name
            fish_logits = self._fish_gate.run(None, {fish_input: tensor})[0][0]
            fish_logits = fish_logits - np.max(fish_logits)
            fish_probabilities = np.exp(fish_logits) / np.exp(fish_logits).sum()
            fish_indices = set(self._metadata.get("fish_presence_indices", []))
            fish_confidence = float(sum(fish_probabilities[index] for index in fish_indices))
            is_fish = fish_confidence >= float(self._metadata.get("fish_presence_threshold", 0.003))
            if not is_fish:
                return {
                    "is_fish": False,
                    "fish_confidence": round(fish_confidence, 4),
                    "label": None,
                    "disease_name": None,
                    "condition": None,
                    "confidence": None,
                    "probabilities": {},
                }

        if self._seven_class_session is not None:
            input_name = self._seven_class_session.get_inputs()[0].name
            logits = self._seven_class_session.run(None, {input_name: tensor})[0][0]
            logits = logits - np.max(logits)
            probabilities = np.exp(logits) / np.exp(logits).sum()
            labels = self._seven_class_metadata["labels"]
            index = int(np.argmax(probabilities))
            class_label = labels[index]
            healthy = self._is_healthy_label(class_label)
            display_label = self._display_label(class_label)
            return {
                "is_fish": True,
                "fish_confidence": round(fish_confidence, 4) if fish_confidence is not None else None,
                "label": display_label,
                "disease_name": "No visible disease detected" if healthy else display_label,
                "condition": "healthy" if healthy else "disease",
                "confidence": round(float(probabilities[index]), 4),
                "probabilities": {
                    self._display_label(label): round(float(probability), 4)
                    for label, probability in zip(labels, probabilities)
                },
            }

        input_name = self._session.get_inputs()[0].name
        logits = self._session.run(None, {input_name: tensor})[0][0]
        logits = logits - np.max(logits)
        probabilities = np.exp(logits) / np.exp(logits).sum()
        labels = self._metadata["labels"]
        index = int(np.argmax(probabilities))
        label = labels[index]
        healthy = self._is_healthy_label(label)
        return {
            "is_fish": True,
            "fish_confidence": round(fish_confidence, 4) if fish_confidence is not None else None,
            "label": label,
            "disease_name": "No visible disease detected" if healthy else "Abnormal visual pattern",
            "condition": "healthy" if healthy else "abnormal",
            "confidence": round(float(probabilities[index]), 4),
            "probabilities": {label: round(float(prob), 4) for label, prob in zip(labels, probabilities)},
        }


_classifier = FishDiseaseClassifier(
    config.DISEASE_MODEL_PATH,
    config.DISEASE_LABELS_PATH,
    config.DISEASE_7CLASS_MODEL_PATH,
    config.DISEASE_7CLASS_LABELS_PATH,
)


def get_classifier() -> FishDiseaseClassifier:
    return _classifier