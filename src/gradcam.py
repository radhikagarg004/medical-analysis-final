
from pathlib import Path
import torch
import numpy as np
import cv2

from PIL import Image
from torchvision import transforms

from src.model import create_model

MODEL_PATH = Path("models/best_efficientnet.pth")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASSES = [
    "COVID19",
    "NORMAL",
    "PNEUMONIA",
    "TURBERCULOSIS"
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def generate_gradcam(image_path, output_path="results/gradcam.png"):

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    classes = checkpoint.get("classes", CLASSES)

    model = create_model(
        num_classes=len(classes),
        pretrained=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    activations = {}

    def save_activation(module, inputs, output):
        activations["features"] = output
        output.retain_grad()

    target_layer = model.features[-1]
    hook = target_layer.register_forward_hook(save_activation)

    try:
        image = Image.open(image_path).convert("RGB")
        image = image.resize((224, 224))

        image_tensor = transform(image).unsqueeze(0).to(DEVICE)

        model.zero_grad()

        outputs = model(image_tensor)

        predicted_index = outputs.argmax(dim=1).item()
        predicted_class = classes[predicted_index]

        probabilities = torch.softmax(outputs, dim=1)
        confidence = probabilities[0, predicted_index].item() * 100

        outputs[0, predicted_index].backward()

        feature_maps = activations["features"].detach()
        gradients = activations["features"].grad.detach()

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = (weights * feature_maps).sum(dim=1)
        cam = torch.relu(cam)
        cam = cam[0].cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()

        cam = cv2.resize(cam, (224, 224))

        heatmap = np.uint8(255 * cam)
        heatmap = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(
            heatmap,
            cv2.COLOR_BGR2RGB
        )

        original = np.array(image)

        overlay = cv2.addWeighted(
            original, 0.6,
            heatmap, 0.4,
            0
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        Image.fromarray(overlay).save(output_path)

        return {
            "predicted_class": predicted_class,
            "confidence": round(confidence, 2),
            "gradcam_image": str(output_path)
        }

    finally:
        hook.remove()


if __name__ == "__main__":

    result = generate_gradcam(
        "data/classification/test/NORMAL/IM-0003-0001.jpeg"
    )

    print("\nGrad-CAM Result")
    print("----------------")
    print("Predicted class:", result["predicted_class"])
    print("Score:", result["confidence"], "%")
    print("Saved image:", result["gradcam_image"])
