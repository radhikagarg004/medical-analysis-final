
import os
import copy
import json

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from src.data_loader import get_loaders, CLASSES

EPOCHS = 5
LEARNING_RATE = 0.0001
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    actual = []
    predicted = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            predictions = outputs.argmax(dim=1)

            actual.extend(labels.cpu().tolist())
            predicted.extend(predictions.cpu().tolist())

    accuracy = accuracy_score(actual, predicted)
    return total_loss / len(loader), accuracy, actual, predicted


def main():
    print("Device:", DEVICE)
    train_loader, val_loader, test_loader, weights = get_loaders()

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=weights.to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_accuracy = -1
    best_weights = None
    history = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0

        for batch, (images, labels) in enumerate(train_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (batch + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS}, "
                      f"Batch {batch+1}/{len(train_loader)}")

        val_loss, val_accuracy, _, _ = evaluate(
            model, val_loader, criterion
        )

        train_loss = running_loss / len(train_loader)

        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"Train loss: {train_loss:.4f}")
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Validation accuracy: {val_accuracy:.4f}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_weights = copy.deepcopy(model.state_dict())

            torch.save({
                "model_state_dict": best_weights,
                "classes": CLASSES,
                "best_val_accuracy": best_accuracy
            }, "models/best_resnet18.pth")

            print("Best ResNet18 model saved!")

    model.load_state_dict(best_weights)

    _, test_accuracy, y_true, y_pred = evaluate(
        model, test_loader, criterion
    )

    print("\nResNet18 Test Accuracy:", round(test_accuracy * 100, 2), "%")

    report = classification_report(
        y_true, y_pred,
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        zero_division=0
    )

    print(report)

    with open("results/resnet18_report.txt", "w") as f:
        f.write(report)

    with open("results/resnet18_history.json", "w") as f:
        json.dump(history, f, indent=4)

    with open("results/resnet18_metrics.json", "w") as f:
        json.dump({
            "test_accuracy": test_accuracy,
            "classification_report": classification_report(
                y_true, y_pred,
                labels=list(range(len(CLASSES))),
                target_names=CLASSES,
                output_dict=True,
                zero_division=0
            ),
            "confusion_matrix": confusion_matrix(
                y_true, y_pred,
                labels=list(range(len(CLASSES)))
            ).tolist()
        }, f, indent=4)

    print("Saved: models/best_resnet18.pth")
    print("Saved: results/resnet18_metrics.json")


if __name__ == "__main__":
    main()
