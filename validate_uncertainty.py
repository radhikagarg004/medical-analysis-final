
import csv
import json
from pathlib import Path

import numpy as np
import torch

from src.data_loader import get_loaders
from src.predict import load_trained_model, DEVICE
from src.uncertainty import enable_dropout

THRESHOLDS = [1.0, 2.0, 3.0, 5.0, 7.0]
NUM_PASSES = 20
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)


def evaluate_threshold(records, threshold):
    incorrect = [r for r in records if not r["correct"]]
    correct = [r for r in records if r["correct"]]

    flagged = [
        r for r in records
        if r["uncertainty_score"] >= threshold
    ]

    flagged_errors = sum(
        not r["correct"] for r in flagged
    )

    flagged_correct = sum(
        r["correct"] for r in flagged
    )

    return {
        "threshold": threshold,
        "total_images": len(records),
        "incorrect_predictions": len(incorrect),
        "correct_predictions": len(correct),
        "flagged_images": len(flagged),
        "flagged_errors": flagged_errors,
        "flagged_correct": flagged_correct,
        "error_detection_rate": round(
            flagged_errors / len(incorrect), 4
        ) if incorrect else None,
        "review_rate": round(
            len(flagged) / len(records), 4
        ),
        "flag_precision": round(
            flagged_errors / len(flagged), 4
        ) if flagged else None,
        "false_review_rate": round(
            flagged_correct / len(correct), 4
        ) if correct else None,
    }


def main():
    print("Loading validation dataset...")
    _, _, val_loader, _ = get_loaders()

    print("Loading EfficientNet-B0...")
    model, classes = load_trained_model()
    enable_dropout(model)

    records = []
    total = len(val_loader.dataset)
    processed = 0

    print(f"Validating {total} images...")

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)

            all_probabilities = []

            for _ in range(NUM_PASSES):
                logits = model(images)
                probabilities = torch.softmax(
                    logits, dim=1
                )
                all_probabilities.append(
                    probabilities.cpu().numpy()
                )

            # Shape: passes x batch x classes
            predictions = np.stack(
                all_probabilities, axis=0
            )

            means = predictions.mean(axis=0)
            stds = predictions.std(axis=0)

            for i, true_label in enumerate(labels.tolist()):
                predicted_index = int(np.argmax(means[i]))

                uncertainty = float(
                    stds[i, predicted_index] * 100
                )

                records.append({
                    "true_class": classes[true_label],
                    "predicted_class": classes[predicted_index],
                    "correct": predicted_index == true_label,
                    "confidence": round(
                        float(means[i, predicted_index] * 100), 4
                    ),
                    "uncertainty_score": round(
                        uncertainty, 6
                    ),
                })

            processed += len(labels)
            print(f"\rProcessed {processed}/{total}", end="")

    print("\n\nThreshold comparison:")

    results = [
        evaluate_threshold(records, threshold)
        for threshold in THRESHOLDS
    ]

    print(
        f"{'Threshold':<12}"
        f"{'Flagged':<12}"
        f"{'Errors caught':<17}"
        f"{'Error detection':<18}"
        f"{'Review rate'}"
    )

    for result in results:
        detection = result["error_detection_rate"]
        detection_text = (
            f"{detection * 100:.2f}%"
            if detection is not None else "N/A"
        )

        print(
            f"{result['threshold']:<12}"
            f"{result['flagged_images']:<12}"
            f"{result['flagged_errors']:<17}"
            f"{detection_text:<18}"
            f"{result['review_rate'] * 100:.2f}%"
        )

    with open(
        OUTPUT_DIR / "uncertainty_test.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=records[0].keys()
        )
        writer.writeheader()
        writer.writerows(records)

    with open(
        OUTPUT_DIR / "threshold_test_comparison.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(results, file, indent=4)

    print("\nSaved:")
    print("results/uncertainty_test.csv")
    print("results/threshold_test_comparison.json")


if __name__ == "__main__":
    main()
