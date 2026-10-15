"""Train a small fish-only Healthy/Abnormal image baseline from parquet data.

Uses exactly 30 images from the supplied training split: 15 healthy and 15
abnormal. The split is 20 train, 4 validation, and 6 test images. This is a
pipeline prototype only; its metrics are not evidence of clinical accuracy.
"""

from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class FishImageDataset(Dataset):
    def __init__(self, samples: list[tuple[Image.Image, int]], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image, label = self.samples[index]
        return self.transform(image), label


def image_from_row(value) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes"):
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            return Image.open(value["path"]).convert("RGB")
    raise ValueError("Parquet image field is not a decodable image")


def load_balanced_samples(parquet_path: Path, seed: int):
    from datasets import load_dataset

    data = load_dataset("parquet", data_files=str(parquet_path), split="train")
    candidates = {"Healthy": [], "Abnormal": []}
    for index, row in enumerate(data):
        class_name = str(row.get("label_name", ""))
        if not class_name.lower().startswith("fish_"):
            continue
        label = "Healthy" if "healthy" in class_name.lower() else "Abnormal"
        candidates[label].append(index)

    if any(len(found) < 15 for found in candidates.values()):
        raise ValueError("The training split needs at least 15 healthy and 15 abnormal fish images")

    rng = random.Random(seed)
    samples = {"train": [], "validation": [], "test": []}
    for label_index, label in enumerate(("Healthy", "Abnormal")):
        selected = rng.sample(candidates[label], 15)
        rng.shuffle(selected)
        for split, split_indices in zip(
            samples.values(), (selected[:10], selected[10:12], selected[12:15])
        ):
            for row_index in split_indices:
                row = data[row_index]
                split.append((image_from_row(row["image"]), label_index))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True, type=Path, help="Local fish-disease train parquet file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "trained_models" / "disease",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.parquet.is_file():
        parser.error(f"Dataset file not found: {args.parquet}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = load_balanced_samples(args.parquet, args.seed)
    print("Selected 30 fish images: 15 Healthy and 15 Abnormal")
    print("Split: 20 train, 4 validation, 6 test; test metrics will be highly uncertain")

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    eval_transform = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    )
    loaders = {
        "train": DataLoader(FishImageDataset(samples["train"], train_transform), batch_size=4, shuffle=True),
        "validation": DataLoader(FishImageDataset(samples["validation"], eval_transform), batch_size=4),
        "test": DataLoader(FishImageDataset(samples["test"], eval_transform), batch_size=4),
    }

    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    model.to(device)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=0.001, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    best_state = None
    best_validation_accuracy = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for images, labels in loaders["train"]:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            criterion(model(images), labels).backward()
            optimizer.step()

        model.eval()
        correct = total = 0
        with torch.inference_mode():
            for images, labels in loaders["validation"]:
                predictions = model(images.to(device)).argmax(dim=1).cpu()
                correct += int((predictions == labels).sum())
                total += len(labels)
        validation_accuracy = correct / total
        print(f"Epoch {epoch:02d}/{args.epochs}: validation accuracy {validation_accuracy:.2f}")
        if validation_accuracy >= best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.to(device).eval()
    test_correct = test_total = 0
    with torch.inference_mode():
        for images, labels in loaders["test"]:
            predictions = model(images.to(device)).argmax(dim=1).cpu()
            test_correct += int((predictions == labels).sum())
            test_total += len(labels)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "fish_disease.onnx"
    torch.onnx.export(
        model,
        torch.zeros(1, 3, 224, 224, device=device),
        model_path,
        input_names=["image"],
        output_names=["logits"],
        dynamo=False,
    )
    (args.output_dir / "labels.json").write_text(
        json.dumps(
            {
                "labels": ["Healthy", "Abnormal"],
                "healthy_labels": ["Healthy"],
                "input_size": 224,
                "mean": MEAN,
                "std": STD,
            },
            indent=2,
        )
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "architecture": "MobileNetV3-Small",
                "training_images": 30,
                "class_counts": {"Healthy": 15, "Abnormal": 15},
                "split_counts": {name: len(rows) for name, rows in samples.items()},
                "validation_accuracy": best_validation_accuracy,
                "test_accuracy": test_correct / test_total,
                "warning": "Prototype metrics from 30 images are not clinically reliable.",
            },
            indent=2,
        )
    )
    print(f"Saved model to {model_path}; test accuracy {test_correct / test_total:.2f} ({test_total} images)")


if __name__ == "__main__":
    main()