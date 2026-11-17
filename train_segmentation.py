"""Train U-Net on paired chest X-ray images and binary lung masks.

Expected layout:
data/segmentation/images/<image_id>.png
data/segmentation/masks/<image_id>.png

Matching stems are required; .jpg/.jpeg/.png/.bmp supported. Masks must
mark lungs as white (nonzero) and background as black.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from sklearn.model_selection import train_test_split

from src.segmentation import UNet

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

class LungPairs(Dataset):
    def __init__(self, pairs, augment=False, size=256):
        self.pairs, self.augment, self.size = pairs, augment, size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        image = Image.open(image_path).convert("L").resize((self.size, self.size), Image.Resampling.BILINEAR)
        mask = Image.open(mask_path).convert("L").resize((self.size, self.size), Image.Resampling.NEAREST)
        x = TF.to_tensor(image)
        y = (TF.to_tensor(mask) > 0).float()
        if self.augment and random.random() < 0.5:
            x, y = TF.hflip(x), TF.hflip(y)
        return x, y

def find_pairs(root):
    image_dir, mask_dir = root / "images", root / "masks"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError("Expected data/segmentation/images and data/segmentation/masks")
    masks = {}
    for p in sorted(mask_dir.iterdir()):
        if p.suffix.lower() in EXTENSIONS:
            if p.stem in masks:
                raise ValueError(f"Duplicate mask stem: {p.stem}")
            masks[p.stem] = p
    pairs = []
    missing = []
    for p in sorted(image_dir.iterdir()):
        if p.suffix.lower() in EXTENSIONS:
            if p.stem in masks:
                pairs.append((p, masks[p.stem]))
            else:
                missing.append(p.name)
    if missing:
        raise ValueError(f"Missing masks for {len(missing)} images, e.g. {missing[:5]}")
    if len(pairs) < 20:
        raise ValueError(f"Found {len(pairs)} image-mask pairs; at least 20 required")
    return pairs

def dice_loss(logits, targets):
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (probs * targets).sum(dims)
    denominator = probs.sum(dims) + targets.sum(dims)
    return (1 - (2 * intersection + 1) / (denominator + 1)).mean()

def scores(logits, targets):
    predicted = torch.sigmoid(logits) >= 0.5
    truth = targets >= 0.5
    intersection = (predicted & truth).sum(dim=(1, 2, 3)).float()
    union = (predicted | truth).sum(dim=(1, 2, 3)).float()
    sizes = predicted.sum(dim=(1, 2, 3)) + truth.sum(dim=(1, 2, 3))
    dice = ((2 * intersection + 1) / (sizes + 1)).sum().item()
    iou = ((intersection + 1) / (union + 1)).sum().item()
    return dice, iou

def evaluate(model, loader, device):
    model.eval()
    dice_total = iou_total = count = 0
    with torch.inference_mode():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            dice, iou = scores(model(images), masks)
            dice_total += dice
            iou_total += iou
            count += len(images)
    return {"dice": dice_total / count, "iou": iou_total / count}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/segmentation")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs = find_pairs(Path(args.data))
    # Fixed train/validation/test split; do not train on validation/test images.
    train_pairs, heldout = train_test_split(pairs, test_size=0.30, random_state=42)
    val_pairs, test_pairs = train_test_split(heldout, test_size=0.50, random_state=42)
    train_loader = DataLoader(LungPairs(train_pairs, augment=True), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(LungPairs(val_pairs), batch_size=args.batch_size, num_workers=0)
    test_loader = DataLoader(LungPairs(test_pairs), batch_size=args.batch_size, num_workers=0)
    print(f"Device: {device}; image-mask pairs: {len(pairs)}")
    print(f"Train: {len(train_pairs)}, val: {len(val_pairs)}, test: {len(test_pairs)}")

    model = UNet(base=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = torch.nn.BCEWithLogitsLoss()
    best_dice = -1
    history = []
    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = bce(logits, masks) + dice_loss(logits, masks)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(images)
        validation = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{args.epochs} loss={total/len(train_pairs):.4f} val Dice={validation['dice']:.4f} val IoU={validation['iou']:.4f}", flush=True)
        history.append({"epoch": epoch, "train_loss": total / len(train_pairs), **{"val_" + k: v for k, v in validation.items()}})
        if validation["dice"] > best_dice:
            best_dice = validation["dice"]
            torch.save({"model_state_dict": model.state_dict(), "base": 16, "best_val_dice": best_dice}, "models/best_unet.pth")
            print("Saved best U-Net checkpoint")

    checkpoint = torch.load("models/best_unet.pth", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test = evaluate(model, test_loader, device)
    print(f"Held-out test Dice={test['dice']:.4f}, IoU={test['iou']:.4f}")
    report = {"image_mask_pairs": len(pairs), "train": len(train_pairs), "validation": len(val_pairs), "test": len(test_pairs), "best_val_dice": best_dice, "test_metrics": test, "history": history}
    Path("results/unet_metrics.json").write_text(json.dumps(report, indent=2))
    print("Saved models/best_unet.pth and results/unet_metrics.json")

if __name__ == "__main__":
    main()
