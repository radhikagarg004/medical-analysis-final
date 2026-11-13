
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.predict import load_trained_model, transform

DATA_DIR = Path("data/classification/train")
INDEX_PATH = Path("models/similarity_index.npz")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = ["COVID19", "NORMAL", "PNEUMONIA", "TURBERCULOSIS"]


def extract_features(model, image_path):
    """Extract EfficientNet features before the classifier."""
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        features = model.features(image_tensor)
        features = model.avgpool(features)
        features = torch.flatten(features, 1)
        features = F.normalize(features, p=2, dim=1)

    return features.cpu().numpy()[0]


def build_similarity_index(max_per_class=100):
    """
    Build a searchable index from training images.
    Uses 100 images per class by default to keep CPU processing manageable.
    """
    model, _ = load_trained_model()
    model.eval()

    image_paths = []
    image_labels = []
    feature_vectors = []

    for class_name in CLASSES:
        folder = DATA_DIR / class_name

        if not folder.exists():
            raise FileNotFoundError(f"Missing folder: {folder}")

        images = sorted(
            path for path in folder.iterdir()
            if path.suffix.lower() in [".jpg", ".jpeg", ".png"]
        )

        # Use a reproducible sample
        rng = np.random.default_rng(42)
        if len(images) > max_per_class:
            indices = rng.choice(
                len(images), size=max_per_class, replace=False
            )
            images = [images[i] for i in sorted(indices)]

        print(f"Processing {class_name}: {len(images)} images", flush=True)

        for image_path in images:
            try:
                features = extract_features(model, image_path)
                feature_vectors.append(features)
                image_paths.append(str(image_path))
                image_labels.append(class_name)
            except Exception as error:
                print(f"Skipping {image_path}: {error}")

    if not feature_vectors:
        raise ValueError("No valid images found for indexing.")

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        INDEX_PATH,
        features=np.asarray(feature_vectors, dtype=np.float32),
        paths=np.asarray(image_paths),
        labels=np.asarray(image_labels)
    )

    print(f"\nSimilarity index saved: {INDEX_PATH}")
    print(f"Total indexed images: {len(image_paths)}")


def find_similar_images(image_path, top_k=5):
    """Find the top_k most visually similar training images."""
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            "Similarity index missing. Run: python -m src.similarity"
        )

    model, _ = load_trained_model()
    model.eval()

    query_features = extract_features(model, image_path)

    with np.load(INDEX_PATH, allow_pickle=False) as index:
        features = index["features"]
        paths = index["paths"]
        labels = index["labels"]

    # All feature vectors are L2-normalized.
    # Their dot product equals cosine similarity.
    similarities = features @ query_features

    top_k = min(max(1, int(top_k)), len(similarities))
    best_indices = np.argsort(similarities)[::-1][:top_k]

    results = []

    for rank, idx in enumerate(best_indices, start=1):
        results.append({
            "rank": rank,
            "image_path": str(paths[idx]),
            "dataset_class": str(labels[idx]),
            "similarity_score": round(float(similarities[idx]), 4)
        })

    return results


if __name__ == "__main__":
    build_similarity_index(max_per_class=100)

    test_image = (
        "data/classification/test/NORMAL/IM-0003-0001.jpeg"
    )

    if Path(test_image).exists():
        print("\nTop 5 Similar Images")
        print("--------------------")

        results = find_similar_images(test_image, top_k=5)
        print(json.dumps(results, indent=2))
