
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.nn import Dropout, Dropout2d, Dropout3d

from src.predict import load_trained_model, transform, DEVICE


def enable_dropout(model):
    """
    Keep the model in evaluation mode,
    but activate its dropout layers.
    """
    model.eval()

    for module in model.modules():
        if isinstance(module, (Dropout, Dropout2d, Dropout3d)):
            module.train()


def estimate_uncertainty(image_path, num_passes=20):
    """
    Estimate uncertainty using Monte Carlo Dropout.
    """

    if num_passes < 2:
        raise ValueError("num_passes must be at least 2")

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # Load the trained model
    model, classes = load_trained_model()

    # Activate dropout for repeated predictions
    enable_dropout(model)

    # Prepare image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    predictions = []

    # Run the model multiple times
    with torch.no_grad():
        for _ in range(num_passes):
            outputs = model(image_tensor)

            probabilities = torch.softmax(
                outputs, dim=1
            )

            predictions.append(
                probabilities.cpu().numpy()[0]
            )

    predictions = np.array(predictions)

    # Mean probabilities across all passes
    mean_probabilities = predictions.mean(axis=0)

    # Standard deviation of each class probability
    std_probabilities = predictions.std(axis=0)

    # Predicted class
    predicted_index = int(
        np.argmax(mean_probabilities)
    )

    # Variation of the predicted class
    uncertainty_score = float(
        std_probabilities[predicted_index] * 100
    )

    # Normalized predictive entropy
    entropy = -np.sum(
        mean_probabilities *
        np.log(mean_probabilities + 1e-12)
    )

    normalized_entropy = float(
        entropy / np.log(len(classes))
    )

    return {
        "predicted_class": classes[predicted_index],

        "mean_confidence": round(
            float(mean_probabilities[predicted_index] * 100),
            2
        ),

        "uncertainty_score": round(
            uncertainty_score, 4
        ),

        "predictive_entropy": round(
            normalized_entropy, 4
        ),

        "num_passes": num_passes,

        "mean_probabilities": {
            name: round(float(prob * 100), 2)
            for name, prob in zip(
                classes, mean_probabilities
            )
        },

        "standard_deviations": {
            name: round(float(std * 100), 4)
            for name, std in zip(
                classes, std_probabilities
            )
        }
    }


if __name__ == "__main__":

    test_image = (
        "data/classification/test/NORMAL/"
        "IM-0003-0001.jpeg"
    )

    result = estimate_uncertainty(
        test_image,
        num_passes=20
    )

    print("\nMonte Carlo Dropout Results")
    print("---------------------------")

    for key, value in result.items():
        print(f"{key}: {value}")
