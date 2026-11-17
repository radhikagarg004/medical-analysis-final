"""Lightweight U-Net for binary lung segmentation."""
import torch
from torch import nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, base=16):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.bottleneck = DoubleConv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.up3 = DoubleConv(base * 8 + base * 4, base * 4)
        self.up2 = DoubleConv(base * 4 + base * 2, base * 2)
        self.up1 = DoubleConv(base * 2 + base, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        a = self.enc1(x)
        b = self.enc2(self.pool(a))
        c = self.enc3(self.pool(b))
        z = self.bottleneck(self.pool(c))
        z = self.up3(torch.cat([F.interpolate(z, size=c.shape[-2:], mode="bilinear", align_corners=False), c], dim=1))
        z = self.up2(torch.cat([F.interpolate(z, size=b.shape[-2:], mode="bilinear", align_corners=False), b], dim=1))
        z = self.up1(torch.cat([F.interpolate(z, size=a.shape[-2:], mode="bilinear", align_corners=False), a], dim=1))
        return self.head(z)  # logits, not probabilities

def predict_lung_mask(image_path, checkpoint="models/best_unet.pth", threshold=0.5):
    """Return a binary PIL mask resized to the original image dimensions."""
    from PIL import Image
    from torchvision.transforms import functional as TF
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model = UNet(base=state.get("base", 16)).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    image = Image.open(image_path).convert("L")
    size = image.size
    tensor = TF.to_tensor(image.resize((256, 256))).unsqueeze(0).to(device)
    with torch.inference_mode():
        prob = torch.sigmoid(model(tensor))[0, 0]
    mask = (prob >= threshold).byte().cpu().numpy() * 255
    return Image.fromarray(mask).resize(size, resample=Image.Resampling.NEAREST)
