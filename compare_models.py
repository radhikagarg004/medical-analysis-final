
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import resnet18
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from src.data_loader import get_loaders, CLASSES
from src.model import create_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)


def load_model(name, path):
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)

    if checkpoint.get("classes") != CLASSES:
        raise ValueError(f"Class order mismatch in {path}")

    if name == "EfficientNet-B0":
        model = create_model(num_classes=len(CLASSES), pretrained=False)
    else:
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(CLASSES))

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()
    return model


def evaluate_model(model, loader):
    actual, predicted = [], []
    elapsed = 0.0
    total_images = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(DEVICE)

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            outputs = model(images)

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            elapsed += time.perf_counter() - start
            total_images += images.size(0)

            predictions = outputs.argmax(dim=1).cpu().tolist()
            actual.extend(labels.tolist())
            predicted.extend(predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(
        actual,
        predicted,
        labels=list(range(len(CLASSES))),
        average="macro",
        zero_division=0
    )

    return {
        "accuracy": accuracy_score(actual, predicted),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "inference_ms_per_image": elapsed * 1000 / total_images,
        "classification_report": classification_report(
            actual,
            predicted,
            labels=list(range(len(CLASSES))),
            target_names=CLASSES,
            output_dict=True,
            zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            actual,
            predicted,
            labels=list(range(len(CLASSES)))
        ).tolist()
    }


def main():
    print("Using device:", DEVICE)
    _, _, test_loader, _ = get_loaders()

    models = {
        "EfficientNet-B0": "models/best_efficientnet.pth",
        "ResNet18": "models/best_resnet18.pth"
    }

    results = {}

    for name, path in models.items():
        if not Path(path).exists():
            raise FileNotFoundError(path)

        print(f"\nEvaluating {name}...")
        model = load_model(name, path)
        results[name] = evaluate_model(model, test_loader)

        metrics = results[name]
        print(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
        print(f"Macro Precision: {metrics['macro_precision']:.4f}")
        print(f"Macro Recall: {metrics['macro_recall']:.4f}")
        print(f"Macro F1: {metrics['macro_f1']:.4f}")
        print(f"Inference: {metrics['inference_ms_per_image']:.2f} ms/image")

        del model

    with open(RESULTS / "model_comparison.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\nComparison completed!")
    print("Saved results/model_comparison.json")
    print("\nSelect the deployment model after reviewing")
    print("validation metrics and class-wise errors.")


if __name__ == "__main__":
    main()
