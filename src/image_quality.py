
import cv2
import numpy as np
from pathlib import Path


def check_image_quality(image_path):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError("Image not found")

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError("Unable to read image")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Blur detection
    blur_score = float(
        cv2.Laplacian(gray, cv2.CV_64F).var()
    )
    is_blurry = blur_score < 100

    # 2. Brightness detection
    brightness = float(np.mean(gray))

    if brightness < 60:
        brightness_status = "Too Dark"
    elif brightness > 200:
        brightness_status = "Too Bright"
    else:
        brightness_status = "Normal"

    # 3. Contrast detection
    contrast = float(np.std(gray))

    if contrast < 40:
        contrast_status = "Low Contrast"
    else:
        contrast_status = "Normal"

    # 4. Noise estimation
    smooth = cv2.GaussianBlur(gray, (5, 5), 0)
    difference = gray.astype(np.float32) - smooth.astype(np.float32)

    noise_score = float(np.std(difference))

    if noise_score > 20:
        noise_status = "Possible High Noise"
    else:
        noise_status = "Normal"

    # Overall quality assessment
    issues = []

    if is_blurry:
        issues.append("Image may be blurry")

    if brightness_status != "Normal":
        issues.append(brightness_status)

    if contrast_status != "Normal":
        issues.append(contrast_status)

    if noise_status != "Normal":
        issues.append(noise_status)

    quality_status = (
        "Needs Review" if issues else "Acceptable"
    )

    return {
        "blur_score": round(blur_score, 2),
        "blur_status": "Blurry" if is_blurry else "Normal",
        "brightness": round(brightness, 2),
        "brightness_status": brightness_status,
        "contrast": round(contrast, 2),
        "contrast_status": contrast_status,
        "noise_score": round(noise_score, 2),
        "noise_status": noise_status,
        "quality_status": quality_status,
        "issues": issues
    }


if __name__ == "__main__":
    result = check_image_quality(
        "data/classification/test/NORMAL/IM-0003-0001.jpeg"
    )

    print("\nImage Quality Assessment")
    print("------------------------")

    for key, value in result.items():
        print(f"{key}: {value}")
