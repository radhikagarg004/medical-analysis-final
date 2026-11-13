
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.model import create_model


# Project paths
MODEL_PATH = Path("models/best_efficientnet.pth")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Class names
CLASSES = [
    "COVID19",
    "NORMAL",
    "PNEUMONIA",
    "TURBERCULOSIS"
]

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_trained_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    classes = checkpoint.get("classes", CLASSES)

    model = create_model(
        num_classes=len(classes),
        pretrained=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    return model, classes


def predict_image(image_path):

    model, classes = load_trained_model()

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = Image.open(image_path).convert("RGB")

    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = F.softmax(
            outputs,
            dim=1
        )[0]

        predicted_index = torch.argmax(
            probabilities
        ).item()

    predicted_class = classes[predicted_index]

    confidence = probabilities[predicted_index].item() * 100

    results = {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 2),
        "probabilities": {
            name: round(prob.item() * 100, 2)
            for name, prob in zip(classes, probabilities)
        }
    }

    return results
