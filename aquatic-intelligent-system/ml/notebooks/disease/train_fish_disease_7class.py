"""Fine-tune a seven-class fish disease classifier and evaluate held-out images."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
SPLITS = ("train", "valid", "test")
DATASET_LICENSE = "CC BY-NC-SA 4.0"
DATASET_ATTRIBUTION = "Saon110, BD Fish & Shrimp Disease Dataset"


class FishImageDataset(Dataset):
    def __init__(self, data, rows: list[tuple[int, int]], transform):
        self.data = data
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row_index, label_index = self.rows[index]
        image = self.data[row_index]["image"]
        if not isinstance(image, Image.Image):
            raise ValueError(f"Dataset row {row_index} does not contain a decoded image")
        return self.transform(image.convert("RGB")), label_index


def load_splits(dataset_dir: Path):
    from datasets import load_dataset

    data_by_split = {}
    labels_by_split = {}
    for split in SPLITS:
        parquet = dataset_dir / f"{split}-00000-of-00001.parquet"
        if not parquet.is_file():
            raise FileNotFoundError(f"Missing dataset split: {parquet}")
        data = load_dataset("parquet", data_files=str(parquet), split="train")
        data_by_split[split] = data
        labels_by_split[split] = [
            str(row["label_name"]) if str(row["label_name"]).startswith("Fish_") else None
            for row in data
        ]

    classes = sorted({label for label in labels_by_split["train"] if label is not None})
    if len(classes) != 7:
        raise ValueError(f"Expected 7 fish classes in the training split, found {len(classes)}")
    label_to_index = {label: index for index, label in enumerate(classes)}
    rows_by_split = {
        split: [
            (row_index, label_to_index[label])
            for row_index, label in enumerate(labels)
            if label in label_to_index
        ]
        for split, labels in labels_by_split.items()
    }
    if any(not rows_by_split[split] for split in SPLITS):
        raise ValueError("Every train, valid, and test split must contain fish images")
    return data_by_split, rows_by_split, classes


def make_metrics(targets: list[int], predictions: list[int], classes: list[str]) -> dict:
    confusion = [[0 for _ in classes] for _ in classes]
    for target, prediction in zip(targets, predictions):
        confusion[target][prediction] += 1

    per_class = {}
    f1_scores = []
    for index, label in enumerate(classes):
        true_positive = confusion[index][index]
        false_positive = sum(confusion[row][index] for row in range(len(classes))) - true_positive
        false_negative = sum(confusion[index]) - true_positive
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}
        f1_scores.append(f1)

    return {
        "accuracy": sum(target == prediction for target, prediction in zip(targets, predictions)) / len(targets),
        "macro_f1": sum(f1_scores) / len(f1_scores),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def predict_batches(model, loader, device) -> tuple[list[int], list[int]]:
    targets, predictions = [], []
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            targets.extend(labels.tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    return targets, predictions


def select_stratified_rows(rows: list[tuple[int, int]], sample_count: int, seed: int, class_count: int):
    if sample_count > len(rows):
        raise ValueError(f"Cannot select {sample_count} images from {len(rows)} test fish images")
    by_class = {label: [] for label in range(class_count)}
    for row in rows:
        by_class[row[1]].append(row)
    per_class = sample_count // class_count
    if any(len(found) < per_class for found in by_class.values()):
        raise ValueError("The test split does not have enough examples in each fish class")

    rng = random.Random(seed)
    selected, remaining = [], []
    for found in by_class.values():
        rng.shuffle(found)
        selected.extend(found[:per_class])
        remaining.extend(found[per_class:])
    rng.shuffle(remaining)
    selected.extend(remaining[: sample_count - len(selected)])
    rng.shuffle(selected)
    return selected


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path, help="Directory containing train/valid/test parquet files")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "trained_models" / "disease" / "fish_7class",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sample-count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.sample_count < 1:
        parser.error("epochs, patience, batch-size, and sample-count must be positive")
    if not args.dataset_dir.is_dir():
        parser.error(f"Dataset directory not found: {args.dataset_dir}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device()
    data_by_split, rows_by_split, classes = load_splits(args.dataset_dir)
    train_counts = Counter(label for _, label in rows_by_split["train"])
    print(f"Training on {len(rows_by_split['train'])} fish images across {len(classes)} classes using {device}")
    print("Training class counts:", {classes[index]: train_counts[index] for index in range(len(classes))})

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    datasets_by_split = {
        split: FishImageDataset(
            data_by_split[split],
            rows_by_split[split],
            train_transform if split == "train" else eval_transform,
        )
        for split in SPLITS
    }
    loaders = {
        split: DataLoader(
            datasets_by_split[split],
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=0,
        )
        for split in ("train", "valid")
    }

    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
    model.to(device)
    class_weights = torch.tensor(
        [len(rows_by_split["train"]) / (len(classes) * train_counts[index]) for index in range(len(classes))],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    best_f1 = -1.0
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for images, labels in loaders["train"]:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())

        validation_targets, validation_predictions = predict_batches(model, loaders["valid"], device)
        validation_metrics = make_metrics(validation_targets, validation_predictions, classes)
        scheduler.step(validation_metrics["macro_f1"])
        print(
            f"Epoch {epoch:02d}/{args.epochs}: "
            f"loss={total_loss / len(loaders['train']):.4f} "
            f"validation_accuracy={validation_metrics['accuracy']:.4f} "
            f"validation_macro_f1={validation_metrics['macro_f1']:.4f}"
        )
        if validation_metrics["macro_f1"] > best_f1:
            best_f1 = validation_metrics["macro_f1"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after {epoch} epochs")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a best model checkpoint")
    model.load_state_dict(best_state)
    model.to(device).eval()

    test_loader = DataLoader(datasets_by_split["test"], batch_size=args.batch_size, num_workers=0)
    test_targets, test_predictions = predict_batches(model, test_loader, device)
    test_metrics = make_metrics(test_targets, test_predictions, classes)
    selected_rows = select_stratified_rows(
        rows_by_split["test"], args.sample_count, args.seed, len(classes)
    )
    test_position_by_row = {
        row_index: position for position, (row_index, _) in enumerate(rows_by_split["test"])
    }
    predictions_30 = []
    model.eval()
    with torch.inference_mode():
        for number, (row_index, target_index) in enumerate(selected_rows, start=1):
            image_tensor, _ = datasets_by_split["test"][test_position_by_row[row_index]]
            logits = model(image_tensor.unsqueeze(0).to(device))[0]
            probabilities = torch.softmax(logits, dim=0).cpu()
            prediction_index = int(probabilities.argmax())
            predictions_30.append(
                {
                    "sample": number,
                    "test_split_row": row_index,
                    "actual": classes[target_index],
                    "predicted": classes[prediction_index],
                    "confidence": float(probabilities[prediction_index]),
                    "correct": prediction_index == target_index,
                }
            )
            print(
                f"Image {number:02d}/{args.sample_count}: actual={classes[target_index]} "
                f"predicted={classes[prediction_index]} "
                f"confidence={probabilities[prediction_index].item():.3f}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "fish_disease_7class.pth"
    onnx_path = args.output_dir / "fish_disease_7class.onnx"
    torch.save(best_state, model_path)
    model = model.to("cpu").eval()
    torch.onnx.export(
        model,
        torch.zeros(1, 3, 224, 224),
        onnx_path,
        input_names=["image"],
        output_names=["logits"],
        opset_version=12,
        dynamo=False,
    )
    (args.output_dir / "labels.json").write_text(
        json.dumps({"labels": classes, "input_size": 224, "mean": MEAN, "std": STD}, indent=2)
    )
    report = {
        "architecture": "MobileNetV3-Small fine-tuned from ImageNet weights",
        "dataset": DATASET_ATTRIBUTION,
        "dataset_license": DATASET_LICENSE,
        "seed": args.seed,
        "device": str(device),
        "best_epoch": best_epoch,
        "class_counts": {
            split: {
                classes[index]: sum(label == index for _, label in rows_by_split[split])
                for index in range(len(classes))
            }
            for split in SPLITS
        },
        "split_counts": {split: len(rows_by_split[split]) for split in SPLITS},
        "validation_macro_f1_at_best_epoch": best_f1,
        "test_metrics_all_fish": test_metrics,
        "individual_test_predictions": len(predictions_30),
        "individual_test_accuracy": sum(row["correct"] for row in predictions_30) / len(predictions_30),
        "limitations": (
            "Research prototype, not a clinical diagnostic. The 30 individual predictions are a small, "
            "stratified sample; use the full held-out test metrics for a less noisy estimate."
        ),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    with (args.output_dir / "predictions_30.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=predictions_30[0].keys())
        writer.writeheader()
        writer.writerows(predictions_30)
    print(f"All-test accuracy: {test_metrics['accuracy']:.4f}; macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"30-image accuracy: {report['individual_test_accuracy']:.4f}")
    print(f"Saved model and report to {args.output_dir}")


if __name__ == "__main__":
    main()
