
import os
import copy
import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from src.data_loader import get_loaders, CLASSES
from src.model import create_model


# Training configuration
EPOCHS = 5
LEARNING_RATE = 0.0001
DEVICE = torch.device("cpu")

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


def evaluate(model, loader, criterion):

    model.eval()

    total_loss = 0
    predictions = []
    actual_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            predicted = torch.argmax(outputs, dim=1)

            predictions.extend(predicted.cpu().numpy())
            actual_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)

    accuracy = accuracy_score(
        actual_labels,
        predictions
    )

    return avg_loss, accuracy, actual_labels, predictions


def main():

    print("Loading dataset...")

    (
        train_loader,
        val_loader,
        test_loader,
        class_weights
    ) = get_loaders()

    print("\nCreating EfficientNet-B0...")

    # First training run uses pretrained weights.
    model = create_model(
        num_classes=len(CLASSES),
        pretrained=True
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(DEVICE)
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_accuracy = -1

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": []
    }

    best_weights = copy.deepcopy(model.state_dict())

    print("\nStarting training...")

    for epoch in range(EPOCHS):

        model.train()
        running_loss = 0

        for batch_number, (images, labels) in enumerate(train_loader):

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (batch_number + 1) % 50 == 0:
                print(
                    f"Epoch {epoch + 1}/{EPOCHS} | "
                    f"Batch {batch_number + 1}/{len(train_loader)}"
                )

        train_loss = running_loss / len(train_loader)

        val_loss, val_accuracy, _, _ = evaluate(
            model,
            val_loader,
            criterion
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        print(
            f"\nEpoch {epoch + 1}/{EPOCHS}"
            f"\nTraining Loss: {train_loss:.4f}"
            f"\nValidation Loss: {val_loss:.4f}"
            f"\nValidation Accuracy: {val_accuracy:.4f}\n"
        )

        if val_accuracy > best_accuracy:

            best_accuracy = val_accuracy
            best_weights = copy.deepcopy(model.state_dict())

            torch.save(
                {
                    "model_state_dict": best_weights,
                    "classes": CLASSES,
                    "best_val_accuracy": best_accuracy
                },
                "models/best_efficientnet.pth"
            )

            print("Best model checkpoint saved!")

    # Restore best validation model
    model.load_state_dict(best_weights)

    print("\nEvaluating on untouched test dataset...")

    test_loss, test_accuracy, y_true, y_pred = evaluate(
        model,
        test_loader,
        criterion
    )

    print("Test Loss:", round(test_loss, 4))
    print("Test Accuracy:", round(test_accuracy, 4))

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        zero_division=0
    )

    print("\nClassification Report:\n")
    print(report)

    with open(
        "results/classification_report.txt",
        "w"
    ) as file:
        file.write(report)

    with open(
        "results/training_history.json",
        "w"
    ) as file:
        json.dump(history, file, indent=4)

    # Confusion matrix
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(CLASSES)))
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASSES
    )

    display.plot(
        cmap="Blues",
        xticks_rotation=45
    )

    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png")
    plt.close()

    # Loss graph
    plt.figure()

    plt.plot(history["train_loss"], label="Training Loss")
    plt.plot(history["val_loss"], label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.savefig("results/loss_curve.png")
    plt.close()

    print("\nTraining completed successfully!")
    print("Model: models/best_efficientnet.pth")
    print("Reports saved in results/")


if __name__ == "__main__":
    main()
