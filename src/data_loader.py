
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


# Four classification categories
CLASSES = [
    "COVID19",
    "NORMAL",
    "PNEUMONIA",
    "TURBERCULOSIS"
]

IMAGE_SIZE = 224
BATCH_SIZE = 16


# Custom dataset class
class ChestXrayDataset(Dataset):

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# Read images from class folders
def get_samples(folder):

    samples = []

    for label, class_name in enumerate(CLASSES):

        class_folder = Path(folder) / class_name

        if not class_folder.exists():
            raise FileNotFoundError(
                f"Missing class folder: {class_folder}"
            )

        for image_path in class_folder.iterdir():

            if image_path.suffix.lower() in [
                ".jpg", ".jpeg", ".png", ".bmp"
            ]:
                samples.append((str(image_path), label))

    if not samples:
        raise ValueError(f"No images found in {folder}")

    return samples


# Create training, validation and testing loaders
def get_loaders(data_dir="data/classification"):

    data_dir = Path(data_dir)

    train_samples = get_samples(data_dir / "train")
    test_samples = get_samples(data_dir / "test")

    labels = [label for _, label in train_samples]

    # Create stratified validation split
    train_data, val_data = train_test_split(
        train_samples,
        test_size=0.20,
        random_state=42,
        stratify=labels
    )

    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=7),
        transforms.ColorJitter(
            brightness=0.10,
            contrast=0.10
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Validation and test transformations
    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = ChestXrayDataset(
        train_data, train_transform
    )

    val_dataset = ChestXrayDataset(
        val_data, test_transform
    )

    test_dataset = ChestXrayDataset(
        test_samples, test_transform
    )

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # Calculate class weights for imbalanced data
    train_labels = [label for _, label in train_data]

    class_counts = np.bincount(
        train_labels,
        minlength=len(CLASSES)
    )

    class_weights = len(train_labels) / (
        len(CLASSES) * class_counts
    )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32
    )

    # Display dataset information
    print("\nDataset Summary")
    print("---------------------")

    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print("Testing images:", len(test_dataset))

    for i, name in enumerate(CLASSES):
        print(name, ":", class_counts[i])

    return (
        train_loader,
        val_loader,
        test_loader,
        class_weights
    )
